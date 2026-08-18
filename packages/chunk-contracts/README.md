# Chunk contracts

This package defines the strict, versioned JSON contracts shared by chunk
processing tasks. It has no database, storage, Celery, or model-runtime
dependencies.

The package also validates the exact Parakeet v1 utterance artifact, word
alignment artifact, and minimal durable transcription result. Empty per-speaker
word and utterance arrays are valid, while present items require canonical
chunk-relative timestamps, finite confidence, and the persisted speaker
mapping.

`speaker-0.wav` and `speaker-1.wav` identify fixed, chunk-local output slots.
The `diarization_speaker_id` stored with each slot is the only canonical
mapping to the speaker identifiers in the chunk diarization snapshot.
