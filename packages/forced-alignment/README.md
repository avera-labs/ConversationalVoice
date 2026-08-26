# Forced Alignment

Shared lazy-loading adapter for `Qwen/Qwen3-ForcedAligner-0.6B`. It aligns each
generated TTS WAV against worker-derived plain text before any silence or track
assembly is applied. The adapter returns segment-relative word spans and then
merges the original inline audio tags as zero-duration alignment items.

Language capability is enforced only at this model boundary. The adapter maps
`zh`, `yue`, `en`, `de`, `es`, `fr`, `it`, `ja`, `ko`, `pt`, and `ru` to the
11 language names supported by the pinned model. Other identifiers can flow
through the pipeline but fail when this adapter is invoked.

The model snapshot, CUDA device, and dtype are supplied by each task's TOML
policy. Production defaults use a pinned Hugging Face revision, `cuda:0`, and
`bfloat16`.
