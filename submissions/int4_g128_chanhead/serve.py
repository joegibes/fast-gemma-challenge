#!/usr/bin/env python
from __future__ import annotations

import os
import sys


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def append_optional(args: list[str], flag: str, env_name: str) -> None:
    value = os.environ.get(env_name)
    if value:
        args.extend([flag, value])


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "gemma-ml-intern/weights/int4-g128-chanhead")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")

    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        port,
        "--dtype",
        os.environ.get("DTYPE", "bfloat16"),
        "--max-model-len",
        os.environ.get("MAX_MODEL_LEN", "4096"),
        "--gpu-memory-utilization",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"),
        "--max-num-seqs",
        os.environ.get("MAX_NUM_SEQS", "1"),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]

    append_optional(args, "--max-num-batched-tokens", "MAX_NUM_BATCHED_TOKENS")
    append_optional(args, "--kv-cache-dtype", "KV_CACHE_DTYPE")
    append_optional(args, "--quantization", "QUANTIZATION")

    if env_flag("ENABLE_CHUNKED_PREFILL", True):
        args.append("--enable-chunked-prefill")
    if env_flag("DISABLE_CUSTOM_ALL_REDUCE", True):
        args.append("--disable-custom-all-reduce")

    os.execvpe(args[0], args, os.environ)


if __name__ == "__main__":
    main()
