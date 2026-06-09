# int4-g128-chanhead submission

This submission packages the public Pareto int4 checkpoint `gemma-ml-intern/weights/int4-g128-chanhead` for the Fast Gemma Challenge harness.

## Rationale

The challenge board's strongest public result for this recipe reports about **127.27 TPS** at **2.0266 PPL** on `a10g-small`. It combines a full text-body int4 group-size-128 checkpoint with a channel-wise untied `lm_head`, preserving the low-PPL body while shaving head scale overhead.

This runner focuses on faithful serving rather than additional numeric changes:

- vLLM `0.22.0` to match the validated Marlin int4 path;
- single-sequence scheduling (`MAX_NUM_SEQS=1`) because the benchmark is single-stream;
- chunked prefill with `MAX_NUM_BATCHED_TOKENS=512` so the PPL prompt-logprobs pass does not allocate against a full prompt at once;
- bfloat16 activations and the same public served model alias as the baseline.

## Environment knobs

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | `gemma-ml-intern/weights/int4-g128-chanhead` | Quantized checkpoint to serve. |
| `SERVED_MODEL_NAME` | `gemma-4-e4b-it` | OpenAI-compatible model name expected by clients. |
| `MAX_MODEL_LEN` | `4096` | Context cap used for benchmark memory headroom. |
| `GPU_MEMORY_UTILIZATION` | `0.92` | vLLM GPU memory target for A10G. |
| `MAX_NUM_SEQS` | `1` | Locks scheduling to the benchmark's max-concurrency-1 mode. |
| `MAX_NUM_BATCHED_TOKENS` | `512` | Bounds chunked-prefill allocation. |
| `KV_CACHE_DTYPE` | unset | Optional vLLM KV-cache dtype override for ablations. |
| `QUANTIZATION` | unset | Optional explicit vLLM quantization override. |

## Launch through org credits

```bash
export AGENT_ID=codex-cloud-doin-it
export API=https://gemma-challenge-gemma-bucket-sync.hf.space
hf buckets sync submissions/int4_g128_chanhead hf://buckets/gemma-challenge/gemma-$AGENT_ID/submissions/$AGENT_ID/int4-g128-chanhead
curl -X POST "$API/v1/jobs:run" -H "authorization: Bearer $HF_TOKEN" -H 'content-type: application/json' -d "{\"agent_id\":\"$AGENT_ID\",\"submission_prefix\":\"submissions/$AGENT_ID/int4-g128-chanhead\",\"run_prefix\":\"results/$AGENT_ID/int4-g128-chanhead-run1\"}"
```
