# Score Completed Chunks

Read-only evaluation for every `chunks.status = 'completed'` row. Each
chunk is scored three times—once each for `separation`, `reconstruction`, and
`expansion`—using NISQA, DNSMOS, and WavLM speaker similarity. Reconstruction
and expansion utterances are additionally evaluated for audio-tag accuracy by
`google/gemini-3.7-flash` through OpenRouter.

The tool has its own uv environment and does not add metric dependencies to any
pipeline worker.

An opt-in Gemini evaluation also compares each Expansion directly with its paired
Reconstruction for content coherence and scores whether the Expansion is a natural
two-person conversation. It sends the mixed Reconstruction and Expansion WAVs in one
request per chunk and returns both scores on a 1.0--5.0 scale in 0.1 increments.

## Run

From this directory:

```bash
uv sync
uv run score-completed-chunks
```

With no arguments, the command discovers the repository root, loads the root
`.env`, reads every completed chunk, runs local models on CPU, and writes
a timestamped run under `outputs/chunk-quality/`.

Useful optional arguments:

```bash
uv run score-completed-chunks --limit 10
uv run score-completed-chunks --chunk-id <uuid>
uv run score-completed-chunks --device cpu --output-dir /tmp/chunk-scores
uv run score-completed-chunks --model-cache-dir /models/chunk-quality
uv run score-completed-chunks --output-dir /tmp/chunk-scores --resume
uv run score-completed-chunks --audio-tag-workers 8
uv run score-completed-chunks --audio-tag-inline-only
uv run score-completed-chunks --skip-audio-tag-evaluation
uv run score-completed-chunks --interaction-coverage
uv run score-completed-chunks --interaction-coverage --skip-interaction-nonverbal
```

Run only the paired conversation-quality evaluation for an explicit set of chunks:

```bash
uv run score-conversation-quality \
  --output-dir ../../outputs/chunk-quality/conversation-quality-run \
  --chunk-id <uuid> \
  --chunk-id <uuid>
```

The dedicated command requires at least one explicit `--chunk-id`, so it cannot
accidentally submit every completed chunk. It produces
`conversation-quality-scores.jsonl`, `conversation-quality-summary.json`,
`run-manifest.json`, and `failures.jsonl`. `--resume` reuses a successful row only
when both pairs of source-track hashes and the complete evaluator fingerprint match.

`--model-cache-dir` controls the verified NISQA and DNSMOS weights as well as
the Hugging Face cache used by WavLM.

The root `.env` needs the same database and object-storage variables used by
the pipeline plus an OpenRouter key for the default audio-tag evaluation:

```dotenv
DATABASE_URL=postgresql+psycopg://pipeline:password@localhost:5432/voice_pipeline
S3_BUCKET=voice-pipeline
S3_REGION=us-east-1
# S3_ENDPOINT_URL=http://localhost:9000
OPENROUTER_API_KEY=...
```

AWS credentials use the standard boto3 credential chain. The database session
is explicitly read-only. No Celery or Redis connection is used. Use
`--skip-audio-tag-evaluation` when only the local acoustic and speaker metrics
are needed; in that mode `OPENROUTER_API_KEY` is optional. The model can be
overridden with `--audio-tag-model`, although results from different evaluator
models or prompt fingerprints should not be combined without qualification.

## Outputs

Each run produces:

```text
run-manifest.json
speaker-scores.jsonl
chunk-group-scores.jsonl
audio-tag-scores.jsonl
summary.json
summary.csv
audio-tag-summary.json
audio-tag-summary.csv
failures.jsonl
```

With `--interaction-coverage`, the run additionally produces:

```text
interaction-events.jsonl
chunk-interaction-scores.jsonl
interaction-declared-observed.jsonl
interaction-summary.json
interaction-summary.csv
interaction-paired-deltas.csv
interaction-bootstrap.json
```

Interaction analysis is opt-in and its outputs are marked as automatic-only.
It computes full-track speaker activity independently of transcript timing,
then uses transcript intervals only to attribute detected activity to source
utterances. The reconstruction output includes one-to-one
Turn/Overlap/Backchannel event precision, recall, and F1. Activity IoU is intentionally excluded
because source recordings and TTS reconstruction do not share a duration-aligned
timebase. Three-stage outputs include transition rates, event densities, category
support, effective category count, and Jensen--Shannon distance. Expansion
uses the paired reconstruction as its primary distributional reference; the
separation comparison remains available as a secondary diagnostic. Expansion
events are never matched one-to-one to reference events.

A distinct cross-speaker overlap event requires at least 60 ms of simultaneous
activity, corresponding to two 30 ms VAD frames. Qualifying fragments separated
by at most 500 ms are counted as one event; the intervening gap is not added to
simultaneous-speech duration. The same protocol is applied to every evaluated
stage and to overlap-based transition classification.

The default VAD is the versioned deterministic adaptive frame-RMS protocol in
the run manifest. It is suitable for reproducible implementation tests but is
not automatically claim-grade. Nonverbal analysis uses the AudioSet AST model
`MIT/ast-finetuned-audioset-10-10-0.4593` at frozen revision
`f826b80d28226b62986cc218e5cec390b1096902`; its resolved weights SHA-256 is
recorded in the manifest. Use `--skip-interaction-nonverbal` to run the primary
reconstruction protocol without loading that model.

Interaction outputs remain marked `provisional_automatic_only=true`. Bootstrap
resampling clusters chunks by their originating raw audio and is stratified by
language for the pooled result.

Quality metrics are computed on transcript-derived active speech rather than
the full dual tracks, so scheduled silence does not dominate the scores. The
separation group uses the original transcription result; reconstruction and
expansion use their generated transcripts. All three groups use the exact same
canonical speaker references recorded by the completed extension result.
NISQA inputs longer than 50 seconds are scored in contiguous windows and
combined with a duration-weighted mean.

For audio-tag accuracy, the tool slices each reconstruction and expansion
utterance from its corresponding single-speaker track and sends that WAV plus
its annotation record to OpenRouter. Current transcripts send inline
`text_with_audio_tags`, including tag position and order. Legacy transcripts
send structured `text`, utterance-level `tone`, and unordered `audio_tags`;
legacy evaluation does not infer positional accuracy that the source contract
does not encode. Every row records its `annotation_representation`. The
evaluator returns a structured score:

Use `--audio-tag-inline-only` to evaluate only utterances carrying
`text_with_audio_tags`. Legacy rows are then recorded as `not_applicable` and
excluded from the denominator.

- `1`: not expressed at all
- `2`: weakly expressed
- `3`: partially expressed
- `4`: well expressed
- `5`: perfectly expressed

`audio-tag-scores.jsonl` preserves the per-utterance score, short rationale,
tags, model fingerprint, token/cost metadata when returned by OpenRouter, and a
resume key. `audio-tag-summary.json` and `.csv` report score counts and
proportions, mean, median, score-at-least-4 well-expressed rate, and score-5 perfect-match
rate overall and by language and group. Utterances without any audio tag are
recorded as `not_applicable` and excluded from score denominators; provider or
contract failures are recorded separately and are also excluded. `--resume`
reuses successful and not-applicable rows when their audio, text, timing, model,
prompt, and schema fingerprint are unchanged.

Audio-tag evaluation uploads each utterance audio clip and tagged transcript to
OpenRouter. Requests require parameter support, deny provider data collection,
require zero-data-retention routing, and use a fixed evaluator seed. Confirm
that the source data's privacy, consent, and data-processing requirements permit
third-party inference before running the default evaluation. Gemini reasoning is
fixed to `low` and excluded from stored responses; its reasoning and final JSON
share a bounded output-token budget. The client strictly validates the returned
JSON fields and 1--5 score locally because the current Google Vertex endpoint
does not reliably populate JSON Schema output when audio input is present.

Conversation-quality evaluation uploads only the mixed Reconstruction and Expansion
WAVs; it does not upload transcripts. Each chunk uses one Gemini request that returns
`content_coherence` and `dialogue_naturalness`. For both dimensions, 1.0 is the
weakest and 5.0 is the strongest judgment. The prompt prefers calibrated decimal
scores such as 4.8 when the evidence lies between rubric anchors and requires one
concise explanatory sentence per dimension. The score file preserves these
rationales, submitted audio hashes, evaluator fingerprint, and token/cost metadata
when provided. As with audio-tag evaluation, zero-data-retention routing is required
and provider data collection is denied.

When an extension had no diarization reference and fell back to a
`separated_track_slice`, separation speaker similarity is partly circular
because its reference is derived from the same separated track. The
`reference_source` field makes these rows identifiable; do not use them for an
unqualified cross-group speaker-similarity claim.

NISQA model weights are licensed separately by their authors under
CC BY-NC-SA 4.0. Confirm that the intended use is compatible before running
this tool in a commercial workflow.
