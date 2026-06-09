#!/usr/bin/env python
"""Serve the fastest safe int4 checkpoint with vLLM nightly MTP speculation.

The target weights are ml-intern's PPL-safe Pareto checkpoint:
full-body group-size 128 int4 plus a channel-wise untied lm_head. This submission
keeps the full multimodal model intact and changes only the serving runtime by
adding Gemma MTP speculative decoding on a vLLM nightly that postdates the
0.22.0 mixed-head attention-group crash.
"""
from __future__ import annotations

import os
import subprocess
import sys


WEIGHTS_BUCKET = os.environ.get(
    "WEIGHTS_BUCKET",
    "hf://buckets/gemma-challenge/gemma-ml-intern/weights/int4-g128-chanhead",
)
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/tmp/int4-g128-chanhead")


def ensure_weights() -> None:
    config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    if os.path.isdir(LOCAL_MODEL_DIR) and os.path.exists(config_path):
        return
    print(f"[serve] syncing weights {WEIGHTS_BUCKET} -> {LOCAL_MODEL_DIR}", flush=True)
    subprocess.run(["hf", "buckets", "sync", WEIGHTS_BUCKET, LOCAL_MODEL_DIR], check=True)


def main() -> None:
    ensure_weights()

    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        LOCAL_MODEL_DIR,
        "--served-model-name",
        os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it"),
        "--host",
        os.environ.get("HOST", "0.0.0.0"),
        "--port",
        os.environ.get("PORT", "8000"),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-num-seqs",
        os.environ.get("MAX_NUM_SEQS", "1"),
        "--performance-mode",
        os.environ.get("PERFORMANCE_MODE", "interactivity"),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]

    max_num_batched_tokens = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    if max_num_batched_tokens:
        args += ["--max-num-batched-tokens", max_num_batched_tokens]

    speculative_config = os.environ.get("SPECULATIVE_CONFIG")
    if speculative_config:
        args += ["--speculative-config", speculative_config]

    print("[serve] launching:", " ".join(args), flush=True)
    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
