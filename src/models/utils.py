import torch

from .base import GPTBase, LayerNorm
from .qwen import Qwen, RMSNorm

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """Return the right model"""
    if args.model == "base":
        model = GPTBase(args)
        if args.use_pretrained != "none":
            model.from_pretrained(args.use_pretrained)
        return model
    elif args.model == "qwen":
        if args.use_pretrained != "none":
            raise NotImplementedError("no pretrained weights for the Qwen-style model")
        return Qwen(args)
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
