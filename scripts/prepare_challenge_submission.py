#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_API = "https://gemma-challenge-gemma-bucket-sync.hf.space"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print or run upload and benchmark-launch commands for a Fast Gemma submission."
    )
    parser.add_argument("--agent-id", default=os.environ.get("AGENT_ID", "codex-cloud-doin-it"))
    parser.add_argument("--submission", default="submissions/int4_g128_chanhead")
    parser.add_argument("--run-name", default="int4-g128-chanhead-run1")
    parser.add_argument("--api", default=os.environ.get("API", DEFAULT_API))
    parser.add_argument("--execute", action="store_true", help="Execute the generated commands.")
    return parser


def require_submission(path: Path) -> None:
    manifest = path / "manifest.json"
    serve = path / "serve.py"
    missing = [str(item) for item in (manifest, serve) if not item.exists()]
    if missing:
        raise SystemExit(f"submission is incomplete; missing: {', '.join(missing)}")
    json.loads(manifest.read_text())


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = build_parser().parse_args()
    submission = Path(args.submission).resolve()
    require_submission(submission)

    dest_prefix = f"submissions/{args.agent_id}/{submission.name.replace('_', '-')}"
    run_prefix = f"results/{args.agent_id}/{args.run_name}"
    bucket = f"hf://buckets/gemma-challenge/gemma-{args.agent_id}"

    upload_cmd = ["hf", "buckets", "sync", str(submission), f"{bucket}/{dest_prefix}"]
    payload = {
        "agent_id": args.agent_id,
        "submission_prefix": dest_prefix,
        "run_prefix": run_prefix,
    }
    curl_cmd = [
        "curl",
        "-X",
        "POST",
        f"{args.api}/v1/jobs:run",
        "-H",
        "authorization: Bearer $HF_TOKEN",
        "-H",
        "content-type: application/json",
        "-d",
        json.dumps(payload),
    ]

    print("Upload command:")
    print(" ".join(upload_cmd))
    print("\nBenchmark launch command:")
    print(" ".join(curl_cmd))

    if args.execute:
        run(upload_cmd)
        executable_curl_cmd = [part.replace("$HF_TOKEN", os.environ.get("HF_TOKEN", "")) for part in curl_cmd]
        run(executable_curl_cmd)


if __name__ == "__main__":
    main()
