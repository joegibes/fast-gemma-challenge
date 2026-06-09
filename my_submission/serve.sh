#!/bin/bash
set -e

MODEL_DIR="/tmp/int4-g128-chanhead"
hf buckets sync hf://buckets/gemma-challenge/gemma-ml-intern/weights/int4-g128-chanhead/ "$MODEL_DIR"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "gemma-4-e4b-it" \
  --host "0.0.0.0" \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --trust-remote-code \
  --no-enable-log-requests \
  --num-speculative-tokens 4 \
  --speculative-model "$MODEL_DIR"
