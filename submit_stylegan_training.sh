# Usage:
# ./submit_stylegan_training.sh CONFIG_FILE KIMG TIME EXPERIMENT_NAME

CONFIG_FILE=$1
KIMG=$2
TIME=$3
EXPERIMENT_NAME=$4

if [ -z "$CONFIG_FILE" ] || [ -z "$KIMG" ] || [ -z "$TIME" ] || [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 CONFIG_FILE KIMG TIME EXPERIMENT_NAME"
    exit 1
fi

# Load config
set -a
source "$CONFIG_FILE"
set +a

# Submit the job
sbatch \
  --job-name=${EXPERIMENT_NAME} \
  --output="%x-%N-%j.out" \
  --nodes=1 \
  --gres=gpu:$NGPUS \
  --mem=$MEM \
  --tasks-per-node=1 \
  --cpus-per-task=$NWORKERS \
  --time=$TIME \
  --export=ALL,CONFIG_FILE=$CONFIG_FILE,KIMG=$KIMG,EXPERIMENT_NAME=$EXPERIMENT_NAME \
  run_stylegan_training.sh
