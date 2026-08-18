# Quality-Filter Flow

The `quality_filter_audio_part` task turns one diarized audio part into zero or
more clean two-speaker chunk WAV files. All time values become integer
milliseconds as soon as the diarization artifact is parsed.

## Overview

```text
audio part WAV                         diarization JSON
      |                                      |
      |                                speaker turns
      |                                      |
      |                              union overlapping or
      |                              touching turns
      |                                      |
      |                               speech intervals
      |
whole-part music detection
      |
global music intervals
      |                                      |
      +------------------+-------------------+
                         |
                 per-speech-interval
                   WADA SNR + music
                    overlap decision
                         |
                    good/bad labels
                         |
              group consecutive labels
              absorb short internal bad
               split at music-filled gaps
                         |
                    good regions
                         |
              align to overlapping turns
                         |
              Sidon raw-window planning
                 inside each region
                         |
                greedy region-local merge
                         |
             cut and upload chunk WAV files
                         |
          atomically create chunks and mark
                the audio part completed
```

Music detection and diarization are parallel views of the same audio timeline.
Music is detected once over the complete audio part. It does not remove audio
before speech intervals are derived.

## Quality decisions

For each speech interval:

```text
music_overlap_ratio = music_overlap_ms / speech_interval_duration_ms

good = WADA_SNR >= 10.0 dB
       and music_overlap_ratio <= 0.30
```

Equality passes both thresholds. A music interval that overlaps only a small
portion of a speech interval does not automatically make that interval bad.

Consecutive decisions are converted into good regions as follows:

1. Group neighboring speech intervals by their good/bad label.
2. Absorb an internal bad group only when its wall-clock span is less than
   `3,000 ms` and it has a good group on both sides.
3. Split an accepted group when music overlaps a gap between consecutive speech
   intervals.
4. Keep only regions whose wall-clock duration is at least `20,000 ms`.
5. Align each region to the first and last diarization turns that overlap it.

A good region may therefore contain several speech intervals, ordinary silence
gaps, and an absorbed short bad group.

## Sidon raw windows

Each good region is planned independently. A raw candidate must:

- be between `20,000 ms` and `60,000 ms`;
- contain exactly two overlapping speakers;
- contain, for each speaker, a fully contained turn of at least `4,000 ms`;
- contain at least `8,000 ms` of fully contained turns for each speaker; and
- contain no effective monologue longer than `40,000 ms`.

A third speaker invalidates a candidate even when the third speaker appears only
in a short backchannel. Turns shorter than `1,500 ms` are ignored only when
constructing effective monologue runs.

The `60,000 ms` value limits one raw-window planning attempt. It is not a final
chunk duration limit.

## Greedy merge

Raw windows are sorted and greedily merged from left to right, but only within
their original good region. A tentative merge is accepted when the combined
span still:

- contains exactly two speakers; and
- has no effective monologue longer than `40,000 ms`.

The merge does not reapply the `60,000 ms` raw-window limit. It keeps extending
the current result while these two conditions hold, so a final chunk may be
longer than 60 seconds. Windows from different good regions are never merged.

## Persistence

The task claims an audio part with one compare-and-set update from `diarized` or
`failed` to `filtering`. Duplicate deliveries for an already active or completed
part are no-ops.

Every final window is cut from the decoded source waveform as a 16 kHz mono
16-bit PCM WAV and uploaded to the deterministic path
`chunks/<chunk_index>/audio.wav`. After all uploads succeed, one database
transaction creates all chunk rows and changes the audio part to `completed`.
Producing zero chunks is also a successful result.

After commit, the handler publishes registered UUID-only `separate_chunk`
messages in chunk-index order. This version has no outbox or missing-delivery
recovery. A completed redelivery is an immediate no-op.

Policy values and exact comparison boundaries are defined in
[`resources/default.toml`](src/voice_pipeline_quality_filter_audio_part/resources/default.toml).
