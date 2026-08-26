# Forced Alignment

Shared lazy-loading adapter for `Qwen/Qwen3-ForcedAligner-0.6B`. It aligns each
generated TTS WAV against worker-derived plain text before any silence or track
assembly is applied. The adapter returns segment-relative word spans and then
merges the original inline audio tags as zero-duration alignment items.

The model snapshot, CUDA device, and dtype are supplied by each task's TOML
policy. Production defaults use a pinned Hugging Face revision, `cuda:0`, and
`bfloat16`.
