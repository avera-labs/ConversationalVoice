# Chunk contracts

This package defines the strict, versioned JSON contracts shared by chunk
processing tasks. It has no database, storage, Celery, or model-runtime
dependencies.

The package validates the exact Parakeet v1 (`en`) and Paraformer v1 (`zh`)
utterance artifacts, word-alignment artifacts, and minimal durable
transcription results. Empty per-speaker word and utterance arrays are valid,
while present items require canonical chunk-relative timestamps, finite
confidence, and the persisted speaker mapping. Language parsers accept only
the canonical `en` and `zh` codes. Reconstruction and dialogue-extension
validation preserve and verify the same language identity.

`speaker-0.wav` and `speaker-1.wav` identify fixed, chunk-local output slots.
The `diarization_speaker_id` stored with each slot is the only canonical
mapping to the speaker identifiers in the chunk diarization snapshot.
