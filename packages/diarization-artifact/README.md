# Diarization artifact contract

This package owns the version 1 JSON contract shared by the diarization writer
and quality-filter reader. It has no model-runtime or service dependencies.

The reader converts all accepted timestamps to integer milliseconds before
returning them to consumers.
