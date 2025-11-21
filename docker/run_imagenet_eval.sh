#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-dinov3-imagenet-eval}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

usage() {
    cat <<'EOF'
Usage: docker/run_imagenet_eval.sh [docker-run-args ...] -- [evaluation-script-args]

Examples:
bash  docker/run_imagenet_eval.sh  \
  --gpus all \
  -v /projects3/datasets/imagenet/imagenet-1k:/datasets/imagenet:ro \
  -- \
  --train-dir /datasets/imagenet/train \
  --val-dir /datasets/imagenet/val \
  --device cuda


bash  docker/run_imagenet_eval.sh  \
  --gpus all \
    --shm-size=24g \
  -v /projects3/datasets/imagenet/imagenet-1k:/datasets/imagenet:ro \
  -- \
    --batch-size 1024 \
  --val-dir /datasets/imagenet/val \
  --device cuda

  docker/run_imagenet_eval.sh -- \
    --gpus all \ 
    -v /path/to/imagenet:/datasets/imagenet:ro \
      --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val --device cuda

All arguments after the first "--" are forwarded to tools/eval_imagenet_accuracy.py.
Any arguments before the "--" are passed to docker run (for example, to expose GPUs
or mount volumes).
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

if [[ " $* " != *" -- "* ]]; then
    echo "[ERROR] Please separate docker run args and evaluation args with --" >&2
    usage
    exit 1
fi

docker_args=()
eval_args=()
seen_separator=0
for arg in "$@"; do
    if [[ $seen_separator -eq 0 ]]; then
        if [[ "$arg" == "--" ]]; then
            seen_separator=1
            continue
        fi
        docker_args+=("$arg")
    else
        eval_args+=("$arg")
    fi
done

if [[ ${#eval_args[@]} -eq 0 ]]; then
    echo "[ERROR] Missing arguments for tools/eval_imagenet_accuracy.py" >&2
    usage
    exit 1
fi

echo "docker_args: ${docker_args[@]}"
echo "eval_args  : ${eval_args[@]}"


docker build -f "${REPO_ROOT}/docker/imagenet_eval.Dockerfile" -t "${IMAGE_NAME}" "${REPO_ROOT}"

docker run --rm "${docker_args[@]}" "${IMAGE_NAME}" "${eval_args[@]}"
