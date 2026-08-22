import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

import config
import distributed
from torch.utils.tensorboard import SummaryWriter
from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.adafactor import Adafactor
from optim.adammini import Adam_mini
from optim.adamuon import AdaMuon
from optim.ademamix import AdEMAMix
from optim.ademamix2 import AdEMAMix2
from optim.adopt import ADOPT
from optim.base import train
from optim.clipped import (AdagradClip, AdaGradClipDelayedEta, AdamClip,
                           AdamClipDelayedEta)
from optim.dion import Dion
from optim.lamb import Lamb
from optim.muonbp import MuonBP
from optim.lion import Lion
from optim.mars import MARS
from optim.muon import CombinedScheduler, Muon
from optim.sign_muon import SignMuon, SignMuonScheduler
from optim.lion_muon import LionMuon, LionMuonScheduler
from optim.prodigy import Prodigy
from optim.rmsspectral import RMSspectral
from optim.schedule import (cos_inf_schedule, cosine_wsd_decay_schedule,
                            dd_schedule, wsd_schedule)
from optim.schedulefree import AdamWScheduleFree, SGDScheduleFree
from optim.sgdf import SGDF
from optim.shampoo import DistributedShampoo
from optim.sign import Signum
from optim.soap import SOAP
from optim.sophia import SophiaG
# from optim.taia import TAIA


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    final_args = config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )

    return final_args, parser


def main(args, parser):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()
    args.datasets_dir = os.path.expanduser(args.datasets_dir)

    if args.full_eval_at is None:
        args.full_eval_at = []

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    if args.experiment_name is not None:
        exp_name = args.experiment_name
    else:
        exp_name = get_exp_name(args, parser, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name
    if (exp_dir / "summary.json").exists():
        print(f"Skipping {exp_name}: already completed (summary.json exists)")
        distributed_backend.finalize()
        return
    args.tb_writer = None
    if distributed_backend.is_master_process() and args.tensorboard:
        tb_log_dir = Path(args.results_base_folder) / args.tensorboard_dir / exp_name
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        args.tb_writer = SummaryWriter(log_dir=str(tb_log_dir), flush_secs=30)
        print(f"TensorBoard logging to: {tb_log_dir}")

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(
        args.device
    )  # todo: take care of initializing the model if args.use_pretrained != 'none'
    print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    if args.tensorboard and args.tb_writer is not None:
        args.tb_writer.add_scalar("model/parameters", params_cnt, 0)
        args.tb_writer.add_scalar("model/optimized_parameters", optimized_params_cnt, 0)

    args.world_size = distributed_backend.get_world_size()

    if args.opt == "adamw":
        device_type = "cuda" if "cuda" in args.device else "cpu"
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            **extra_args,
        )
    elif args.opt == "soap":
        opt = SOAP(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_precond_dim,
            merge_dims=args.merge_dims,
            precondition_1d=args.precondition_1d,
            normalize_grads=args.normalize_grads,
            data_format=args.soap_data_format,
            correct_bias=args.correct_bias,
        )
    elif args.opt == "muon":
        param_list = (
            list(model.parameters())
            if args.distributed_backend is None
            else list(model.module.parameters())
        )
        assert (
            sum(p.numel() for p in param_list) == params_cnt
        ), "number of parameters must be the same"
        opt = Muon(
            muon_params=param_list,
            lr=args.muon_lr_factor,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            weight_decay=args.weight_decay,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(0.8, 0.999),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
        )
    elif args.opt == "lion_muon":
        param_list = (
            list(model.parameters())
            if args.distributed_backend is None
            else list(model.module.parameters())
        )
        assert sum(p.numel() for p in param_list) == params_cnt
        nd_path = getattr(args, "norm_diag_path", None) or None
        if nd_path == "":
            nd_path = None
        if nd_path is None and getattr(args, "norm_diag", False):
            nd_path = str(exp_dir / "norm_diag.json")
        opt = LionMuon(
            muon_params=param_list,
            lr=args.muon_lr_factor,
            lion_lr=args.sign_lr,
            beta1=args.beta1,
            beta2=args.beta2,
            ns_steps=args.muon_ns_steps,
            muon_every_k=args.muon_every_k,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
            srank_alpha=args.srank_alpha,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(0.8, 0.999),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
            norm_diag_path=nd_path,
            norm_diag_every_k=getattr(args, "norm_diag_every_k", 50),
        )
    elif args.opt == "dion":
        param_list = (
            list(model.parameters())
            if args.distributed_backend is None
            else list(model.module.parameters())
        )
        opt = Dion(
            muon_params=param_list,
            lr=args.muon_lr_factor,
            rank_frac=args.dion_rank_frac,
            weight_decay=args.weight_decay,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(0.8, 0.999),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
        )
    elif args.opt == "muonbp":
        param_list = (
            list(model.parameters())
            if args.distributed_backend is None
            else list(model.module.parameters())
        )
        opt = MuonBP(
            muon_params=param_list,
            lr=args.muon_lr_factor,
            momentum=args.momentum,
            n_blocks=args.muonbp_blocks,
            period=args.muonbp_period,
            block_lr_ratio=args.muonbp_block_lr_ratio,
            ns_steps=args.muon_ns_steps,
            weight_decay=args.weight_decay,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(0.8, 0.999),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
        )
    elif args.opt == "sign_muon":
        param_list = (
            list(model.parameters())
            if args.distributed_backend is None
            else list(model.module.parameters())
        )
        assert (
            sum(p.numel() for p in param_list) == params_cnt
        ), "number of parameters must be the same"
        opt = SignMuon(
            muon_params=param_list,
            lr=args.muon_lr_factor,
            cheap_lr=args.sign_lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            muon_every_k=args.muon_every_k,
            cheap_mode=args.cheap_mode,
            cheap_ns_steps=args.cheap_ns_steps,
            sign_scaling=args.sign_scaling,
            weight_decay=args.weight_decay,
            srank_alpha=args.srank_alpha,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(0.8, 0.999),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
        )
    elif args.opt == "adamuon":
        param_list = (
            list(model.parameters())
            if args.distributed_backend is None
            else list(model.module.parameters())
        )
        assert (
            sum(p.numel() for p in param_list) == params_cnt
        ), "number of parameters must be the same"
        opt = AdaMuon(
            params=param_list,
            lr=args.lr,
            momentum=args.momentum,
            ns_steps=args.adamuon_ns_steps,
            weight_decay=args.weight_decay,
            eps=1e-8,
            rms_alignment_factor=args.adamuon_rms_factor,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(0.8, 0.999),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
        )
    elif args.opt == "ademamix":
        opt = AdEMAMix(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.adema_beta3),
            alpha=args.adema_alpha,
            beta3_warmup=args.adema_beta3_warmup,
            alpha_warmup=args.adema_alpha_warmup,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "ademamix2":
        opt = AdEMAMix2(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.adema_beta3),
            alpha=args.adema_alpha,
            beta3_warmup=args.adema_beta3_warmup,
            alpha_warmup=args.adema_alpha_warmup,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lion":
        opt = Lion(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "sf-adamw":
        opt = AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "sf-sgd":
        opt = SGDScheduleFree(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "adam-mini":
        opt = Adam_mini(
            device=args.device,
            world_size=args.world_size,
            named_parameters=model.named_parameters(),  # check
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            model_sharding=args.model_sharding,
            dim=args.n_embd,
            n_heads=args.n_head,
            n_kv_heads=args.n_kv_head,
            verbose=args.adam_mini_verbose,
        )
    elif args.opt == "signsgd":
        opt = Signum(
            group_specs,
            lr=args.lr,
            momentum=0.0,  # always use zero momentum because its signSGD
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "signum":
        opt = Signum(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dampening=args.dampening,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "sgdf":
        opt = SGDF(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "prodigy":
        opt = Prodigy(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.weight_decay,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
            fsdp_in_use=args.prodigy_fsdp_in_use,
        )
    elif args.opt == "sophiag":
        opt = SophiaG(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            rho=args.sophia_rho,
        )
    elif args.opt == "shampoo":
        opt = DistributedShampoo(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            precondition_frequency=args.precondition_frequency,
            weight_decay=args.weight_decay,
            use_decoupled_weight_decay=True,
            # grafting_config=AdamGraftingConfig(
            #     beta2=args.beta2,  # oroginally, the default value is 0.999
            #     epsilon=1e-8,
            # ),
        )
    elif args.opt == "adopt":
        opt = ADOPT(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=1e-6,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "rmsspectral":
        opt = RMSspectral(
            group_specs,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.rmsspectral_eps,
            weight_decay=args.weight_decay,
            ns_steps=args.muon_ns_steps,
            variant=args.rmsspectral_variant,
            adamw_betas=(args.rmsspectral_adamw_beta1, args.rmsspectral_adamw_beta2),
            adamw_eps=args.rmsspectral_adamw_eps,
        )
    elif args.opt == "rmsspectral-sania":
        # Backward-compatible preset kept for old experiment scripts.
        opt = RMSspectral(
            group_specs,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=1e-5,
            weight_decay=args.weight_decay,
            ns_steps=args.muon_ns_steps,
            variant="post",
            adamw_betas=(args.rmsspectral_adamw_beta1, args.rmsspectral_adamw_beta2),
            adamw_eps=args.rmsspectral_adamw_eps,
        )
    elif args.opt in [
        "clip-adagrad",
        "clip-adagrad-delay-eta",
        "clip-adam",
        "clip-adam-delay-eta",
    ]:
        clipped_adagrad_cfg = {
            "lr": args.lr,
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
            "clipping": args.clipping_type,
            "max_grad_norm": 1.0,
        }
        if args.opt == "clip-adagrad":
            opt = AdagradClip(**clipped_adagrad_cfg)
        clipped_adagrad_delay_eta_cfg = {
            **clipped_adagrad_cfg,
            "exp_avg_sq_value": 0.0001,
            "etta": args.clipping_eta,
        }
        if args.opt == "clip-adagrad-delay-eta":
            opt = AdaGradClipDelayedEta(**clipped_adagrad_delay_eta_cfg)
        clipped_adam_cfg = {
            **clipped_adagrad_cfg,
            "betas": (args.beta1, args.beta2),
            "correct_bias": args.correct_bias,
        }
        if args.opt == "clip-adam":
            opt = AdamClip(**clipped_adam_cfg)
        clipped_adam_delay_eta_cfg = {
            **clipped_adam_cfg,
            "exp_avg_sq_value": 0.00001,
            "etta": args.clipping_eta,
        }
        if args.opt == "clip-adam-delay-eta":
            opt = AdamClipDelayedEta(**clipped_adam_delay_eta_cfg)
    elif args.opt == "mars":
        opt = MARS(
            group_specs,
            lr=args.mars_lr,
            betas=(args.mars_beta1, args.mars_beta2),
            weight_decay=args.weight_decay,
            amsgrad=False,
            gamma=args.mars_vr_gamma,
            is_approx=args.mars_is_approx,
            mars_type=args.mars_type,
            optimize_1d=False,  # we set in order to optimize 1D parameters with AdamW
            lr_1d=args.lr,  # AdamW's lr when optimize_1d=False
            betas_1d=(args.beta1, args.beta2),  # AdamW's betas when optimize_1d=False
            weight_decay_1d=0.1,  # AdamW's weight decay
        )
    elif args.opt == "adafactor":
        opt = Adafactor(
            group_specs,
            lr=args.lr,
            decay_rate=args.adafactor_decay_rate,
            beta1=args.beta1,
            clip_threshold=1.0,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lamb":
        opt = Lamb(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            adam=False,
            bias_correction=args.lamb_use_bias_correction,
        )
    else:
        opt = torch.optim.SGD(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    print(f"\nOptimizer:\n{opt}")

    if args.scheduler != "none":
        assert (
            args.warmup_steps < args.iterations
        ), "Warmup steps must be < iterations."  # from schedules-and-scaling
        if args.scheduler in ["cos", "linear"]:
            # Warmup + CosineAnnealingLR (decays to 0)
            if args.opt not in ("muon", "sign_muon", "lion_muon", "dion", "muonbp"):
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-2, total_iters=args.warmup_steps
                )
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=args.iterations - args.warmup_steps, eta_min=0
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    opt, [warmup_sched, cosine_sched], milestones=[args.warmup_steps]
                )
            else:
                scheduler = (
                    LionMuonScheduler(opt, args) if args.opt in ("lion_muon", "dion", "muonbp") else
                    (SignMuonScheduler(opt, args) if args.opt == "sign_muon" else CombinedScheduler(opt, args))
                )
        elif args.scheduler == "cos_inf":
            lambda_schedule = cos_inf_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=1e2,
                final_div_factor=0.1,
            )
            scheduler = (
                torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
                if args.opt not in ("muon", "sign_muon", "lion_muon", "dion", "muonbp")
                else (
                    LionMuonScheduler(opt, args) if args.opt in ("lion_muon", "dion", "muonbp") else
                    (SignMuonScheduler(opt, args) if args.opt == "sign_muon" else CombinedScheduler(opt, args))
                )
            )
        elif args.scheduler == "wsd":
            lambda_schedule = wsd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale,  # should be 0 here
                decay_type=args.decay_type,
            )
            scheduler = (
                torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
                if args.opt not in ("muon", "sign_muon", "lion_muon", "dion", "muonbp")
                else (
                    LionMuonScheduler(opt, args) if args.opt in ("lion_muon", "dion", "muonbp") else
                    (SignMuonScheduler(opt, args) if args.opt == "sign_muon" else CombinedScheduler(opt, args))
                )
            )
        elif args.scheduler == "cos_wsd":
            lambda_schedule = cosine_wsd_decay_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                anneal_end_factor=0.15,  # 0.2
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=0.1,  # should be 0 here
                decay_type=args.decay_type,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        elif args.scheduler == "dd":
            lambda_schedule = dd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_fisrt_decay=args.wsd_fract_decay,  # this will be responsible for the first decay phase
                max_lr=args.lr,  # [group.get("lr", args.lr) for group in group_specs],
                first_final_lr_factor=args.dd_first_lr_factor,
                second_final_lr_factor=0.0,  # stop with zero lr
                div_factor=1e2,
                first_decay_type=args.decay_type,
                second_decay_type=args.dd_second_decay_type,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already exists. "
                + "To resume training, set auto_resume=True. "
                + "Otherwise, specify a different experiment name. "
            )
        else:
            # Auto resume overwrites resume_from
            args.resume_from = str(exp_dir / "ckpts" / "latest")
    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    if args.tb_writer is not None:
        args.tb_writer.close()
    distributed_backend.finalize()


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }


def get_exp_name(
    args,
    parser,
    distributed_backend,
    key_args=["model", "dataset", "opt"],
    ignore_args=[
        "eval_interval",
        "full_eval_at",
        "distributed_backend",
        "latest_ckpt_interval",
        "tensorboard",
        "tensorboard_dir",
        "batch_size",
        "acc_steps",
        "results_base_folder",
        "run_prefix",
    ],
):
    # Get the default values
    defaults = vars(parser.parse_args([]))

    rank = distributed_backend.rank

    # Generate the prefix with key arguments
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            prefix_parts.append(f"{key}-{value}")

    prefix = "_".join(prefix_parts)
    prefix = f"{args.batch_size}x{args.acc_steps}(rank={rank})_" + prefix

    # Generate the rest of the string with non-default arguments
    non_default_parts = []
    for key, value in vars(args).items():
        if key in ignore_args:
            continue
        if key not in defaults:
            print(f"Warning: {key} not in defaults")
            continue
        if key not in key_args and value != defaults[key]:
            non_default_parts.append(f"{key}-{value}")

    non_default_string = "_".join(non_default_parts)

    if args.run_prefix is not None:
        prefix = args.run_prefix + "_" + prefix

    # Combine prefix and non-default string
    if non_default_string:
        return f"{prefix}__{non_default_string}"
    else:
        return prefix


if __name__ == "__main__":
    args, parser = get_args()
    main(args, parser)
