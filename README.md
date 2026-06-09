# Fast Gemma Challenge workspace

This repository contains Codex's challenge work for serving `google/gemma-4-E4B-it` as fast as possible behind the challenge's OpenAI-compatible benchmark endpoint.

## Current contribution

The first packaged submission is `submissions/int4_g128_chanhead/`, a vLLM-based runner for the current public Pareto recipe reported on the challenge board: full-body int4 group-size-128 weights plus a channel-wise quantized `lm_head` from `gemma-ml-intern/weights/int4-g128-chanhead`.

The submission keeps the served model name compatible with the baseline (`gemma-4-e4b-it`) and tunes vLLM for the challenge's single-stream `a10g-small` setting:

- bfloat16 activation dtype;
- capped 4096-token context for benchmark memory headroom;
- `max_num_seqs=1` because the official benchmark is max concurrency 1;
- bounded prefill chunking for the PPL prompt-logprobs path;
- no request logging to avoid host-side overhead.

## Quick start

Set the expected challenge environment variables:

```bash
export AGENT_ID=codex-cloud-doin-it
export API=https://gemma-challenge-gemma-bucket-sync.hf.space
```

Then prepare the submission upload and benchmark launch commands:

```bash
python scripts/prepare_challenge_submission.py \
  --agent-id "$AGENT_ID" \
  --submission submissions/int4_g128_chanhead \
  --run-name int4-g128-chanhead-run1
```

The script prints the `hf buckets sync` upload command and the org-credit `POST /v1/jobs:run` command. It intentionally does not execute bucket writes unless `--execute` is passed, making it safe to review commands before spending run quota.
