# score_chunk

`score_chunk(chunk_id)` evaluates one `completed` chunk without changing its pipeline
status or publishing a successor. It is an idempotent, CPU-only terminal evaluation
task.

The worker calculates acoustic quality, speaker identity, reconstruction fidelity,
expansion interaction metrics, and utterance-level audio-tag alignment. CER/WER
transcription uses the OpenRouter speech-to-text endpoint; no ASR model weights are
loaded locally. The default model is `qwen/qwen3-asr-1.7b`. Tagged reconstruction and
expansion utterances are evaluated with Gemini on a 1--5 scale, where 1 means that the
declared tag is not expressed in the audio and 5 means that it is expressed perfectly.

Full artifacts are uploaded under the chunk's deterministic `results/evaluation/v2`
prefix. Only a compact artifact descriptor is stored at
`chunks.final_results.evaluation`.

The artifact set is `score-report.json`, `metric-records.jsonl`,
`score-summary.csv`, `event-matches.jsonl`, `asr-transcripts.jsonl`,
`run-manifest.json`, `audio-tag-scores.jsonl`, `audio-tag-summary.json`, and
`failures.jsonl`. Metric objects expose their value, unit, direction, status,
support, and missing-value reason. The Audio-tag Alignment Score is the arithmetic mean
over tagged utterances only; untagged utterances are recorded as `not_applicable`.

Required environment variables:

- `DATABASE_URL`
- `CELERY_BROKER_URL`
- `S3_BUCKET`
- `S3_REGION`
- `OPENROUTER_API_KEY`

Optional variables:

- `S3_ENDPOINT_URL`
- `SCORE_CHUNK_CONFIG_FILE`

Run with `uv run --project tasks/score-chunk score-chunk-worker`.

Dispatch an already completed chunk with a JSON UUID argument, for example:

```bash
uv run --project tasks/score-chunk celery call score_chunk \
  --queue score_chunk \
  --args='["00000000-0000-0000-0000-000000000000"]'
```
