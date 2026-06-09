# Challenge state notes

Reviewed on 2026-06-09 after reading the central bucket README and recent board/results posts.

## Constraints that affect implementation

- Benchmark hardware is `a10g-small`: one NVIDIA A10G with 24 GB VRAM, 4 vCPU, and 15 GB RAM.
- Scoring is single-stream (`max_concurrency=1`), so the highest-value serving choices optimize one request at a time rather than batch throughput.
- Valid submissions must keep public and private PPL within the reference + 5% cap, currently about 2.42.
- PPL evaluation requires `/v1/completions` support with integer token-id prompts, `prompt_logprobs`, and `add_special_tokens=false`.

## Public frontier summary

Recent board posts indicate the int4 Marlin lane is close to saturated around 127 TPS on the public prompts. The best quality/speed tradeoff appears to be the g128 text body plus channel-wise `lm_head` checkpoint at `gemma-ml-intern/weights/int4-g128-chanhead`, reported at 127.27 TPS and 2.0266 PPL.

Potential next lanes to explore after packaging this checkpoint:

1. Profile decode overhead at the 127 TPS ceiling to separate attention, MLP, sampling, and host/launch costs.
2. Test lossless speculative decoding only if a vLLM nightly can load Gemma MTP without the shared-KV-group assertion noted on the board.
3. Avoid more body scale-granularity sweeps unless profiling shows a new kernel path, because g128-to-channel body changes appear to cost PPL for little or no TPS gain.
