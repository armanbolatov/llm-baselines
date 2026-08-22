DATASETS_DIR=${DATASETS_DIR:-~/datasets/}
ITERATIONS=64000
WARMUP=3000
EVAL_INTERVAL=500
MAX_JOBS=1

# Model architecture
N_LAYER=12
N_HEAD=12
N_EMBD=768
SEQ_LEN=512
BATCH_SIZE=32
ACC_STEPS=1

# Training hyperparams
WEIGHT_DECAY=0.1
GRAD_CLIP=0.5
MUON_NS_STEPS=5
SCHEDULER=cos

# Lion (betas only; LR is per-script)
LION_BETA1=0.9
LION_BETA2=0.99

# Signum (momentum only; LR is per-script)
SIGNUM_MOM=0.95

# Muon (betas + momentum; LR is per-script)
MUON_MOM=0.95

# SignMuon (betas + momentum; LRs are per-script per-k)
SM_MOM=0.95

# LionMuon (betas only; LRs are per-script per-k)
LM_BETA1=0.9
LM_BETA2=0.99

# AdamW backup lr for 1D params in hybrid optimizers
MUON_ADAMW_LR=1e-3
SM_ADAMW_LR=1e-3
LM_ADAMW_LR=1e-3