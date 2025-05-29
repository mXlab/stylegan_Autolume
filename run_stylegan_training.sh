#!/bin/bash

# Load variables from config
set -a
source "$CONFIG_FILE"
set +a

# Paths
SOURCEDIR=$HOME/projects/def-sofian/sofian/stylegan_Autolume
DATADIR=$SOURCEDIR/data
SCRATCH_CHECKPOINTS=$HOME/scratch/stylegan_checkpoints/$EXPERIMENT_NAME
mkdir -p "$SCRATCH_CHECKPOINTS"

# Find latest checkpoint
LAST_CHECKPOINT=$(find "$SCRATCH_CHECKPOINTS" -maxdepth 1 -name "network-snapshot-*.pkl" -print0 | xargs -r -0 ls -t | head -n 1)

if [ -n "$LAST_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $LAST_CHECKPOINT"
    RESUME_ARG="--resume $LAST_CHECKPOINT"
else
    echo "Starting from scratch"
    RESUME_ARG=""
fi

# Print configuration summary
echo "========================================"
echo "      LAUNCHING STYLEGAN2 TRAINING      "
echo "========================================"
echo "Experiment name     : $EXPERIMENT_NAME"
echo "Config file         : $CONFIG_FILE"
echo "GPUs requested      : $NGPUS"
echo "Batch size          : $BATCH"
echo "Gamma               : $GAMMA"
echo "kimg (train length) : $KIMG"
echo "Memory              : $MEM"
echo "CPUs per task       : $NWORKERS"
echo "Dataset             : $DATASET"
echo "Source code dir     : $SOURCEDIR"
echo "Data directory      : $DATADIR"
echo "Checkpoint dir      : $SCRATCH_CHECKPOINTS"
if [ -n "$LAST_CHECKPOINT" ]; then
echo "Resuming from       : $LAST_CHECKPOINT"
fi
echo "========================================"

# Load modules and create environment
module load python/3.11 cuda cudnn

echo "Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Installing dependencies..."
pip install -r "$SOURCEDIR/requirements.txt" --no-index

echo "Copying dataset..."
cp "$DATADIR/$DATASET" "$SLURM_TMPDIR/$DATASET"

echo "Starting training..."
python "$SOURCEDIR/train.py" \
  --cfg stylegan2 \
  --gpus "$NGPUS" \
  --batch "$BATCH" \
  --gamma "$GAMMA" \
  --kimg "$KIMG" \
  --outdir "$SCRATCH_CHECKPOINTS" \
  --data "$SLURM_TMPDIR/$DATASET" \
  --workers "$NWORKERS" \
  --metrics none \
  --snap 10 \
  $RESUME_ARG

