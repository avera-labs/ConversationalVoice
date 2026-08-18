-- ============================================================================
-- Voice Pipeline · database schema
--
-- Entity relationships (1:N down the chain; every FK is ON DELETE CASCADE,
-- so deleting a raw_audio removes everything derived from it):
--
--   ┌──────────────────┐        ┌──────────────────────┐        ┌─────────────────────┐
--   │ raw_audios       │ 1    N │ audio_parts          │ 1    N │ chunks              │
--   │ one ingested     │───────<│ conversation windows │───────<│ clean two-speaker   │
--   │ podcast audio    │        │ cut from a raw_audio │        │ dialogue segments   │
--   │ (16 kHz WAV)     │        │ by VAD               │        │ (one per good_item) │
--   └──────────────────┘        └──────────────────────┘        └─────────────────────┘
--                 FK: raw_audio_id              FK: audio_part_id
--                 UNIQUE (raw_audio_id,         UNIQUE (audio_part_id,
--                         part_index)                   chunk_index)
--
-- How the pipeline writes these tables:
--   ingest         → creates raw_audios (transcode + upload, then audio_uri set)
--   vad-window     → cuts windows, creates audio_parts
--   diarization    → speaker labels per part (JSON on S3), writes
--                    audio_parts.diarization_uri, advances status
--   quality-filter → keeps clean two-speaker spans, creates one chunks row each
--   processing     → per chunk: separate → transcribe → persona → extend; fills
--                    diarizations / persona / final_results
-- ============================================================================

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Ingest owns only pending -> failed when downstream task publication fails.
-- raw_audios.status transitions owned by the split task:
--   pending         -> splitting
--   failed          -> splitting
--   splitting       -> split_completed
--   splitting       -> failed
--   split_completed -> failed only when downstream task publication fails
CREATE TABLE raw_audios (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status           TEXT NOT NULL DEFAULT 'pending',
    audio_uri        TEXT,
    content_sha1     TEXT UNIQUE,                -- SHA-1 of the original uploaded audio bytes, not the normalized WAV
    title            TEXT,
    source_url       TEXT,
    lang             TEXT NOT NULL DEFAULT 'en',
    meta             JSONB NOT NULL DEFAULT '{}'::jsonb,
    duration_ms      INTEGER,
    size_bytes       BIGINT,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_raw_audios_status CHECK (
        status IN ('pending', 'splitting', 'split_completed', 'failed')
    )
);

CREATE INDEX idx_raw_audios_status_created
    ON raw_audios (status, created_at DESC);

CREATE TRIGGER trg_raw_audios_updated_at
    BEFORE UPDATE ON raw_audios
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- audio_parts.status transitions owned by diarization and quality-filter:
--   pending    -> diarizing
--   failed     -> diarizing or filtering, selected by the retrying task
--   diarizing  -> diarized or failed
--   diarized   -> filtering, or failed on downstream publication failure
--   filtering  -> completed or failed
CREATE TABLE audio_parts (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_audio_id      UUID NOT NULL REFERENCES raw_audios(id) ON DELETE CASCADE,
    part_index        INTEGER NOT NULL CHECK (part_index >= 0),
    status            TEXT NOT NULL DEFAULT 'pending',
    audio_uri         TEXT NOT NULL,
    diarization_uri   TEXT,
    lang              TEXT NOT NULL DEFAULT 'en',
    relative_start_ms INTEGER NOT NULL,           -- relative to raw_audios audio
    relative_end_ms   INTEGER NOT NULL,           -- relative to raw_audios audio
    duration_ms       INTEGER NOT NULL,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (raw_audio_id, part_index),
    CONSTRAINT ck_audio_parts_status CHECK (
        status IN ('pending', 'diarizing', 'diarized', 'filtering', 'completed', 'failed')
    ),
    CHECK (relative_end_ms > relative_start_ms)
);

CREATE INDEX idx_audio_parts_status_created
    ON audio_parts (status, created_at DESC);

CREATE TRIGGER trg_audio_parts_updated_at
    BEFORE UPDATE ON audio_parts
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- chunks.status transitions owned by separation, transcription, persona, and extension:
--   pending      -> separating
--   failed       -> separating, transcribing, persona_generating, or extending, selected
--                   by the retrying task from durable result namespaces
--   separating   -> separated, rejected, or failed
--   separated    -> transcribing
--   transcribing -> transcribed or failed
--   transcribed  -> persona_generating
--   persona_generating -> persona_generated or failed
--   persona_generated -> extending
--   extending    -> completed, rejected, or failed
--   rejected and completed are terminal
CREATE TABLE chunks (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audio_part_id     UUID NOT NULL REFERENCES audio_parts(id) ON DELETE CASCADE,
    chunk_index       INTEGER NOT NULL CHECK (chunk_index >= 0),
    status            TEXT NOT NULL DEFAULT 'pending',
    audio_uri         TEXT NOT NULL,
    lang              TEXT NOT NULL DEFAULT 'en',
    duration_ms       INTEGER NOT NULL,
    relative_start_ms INTEGER NOT NULL,           -- relative to audio_parts audio
    relative_end_ms   INTEGER NOT NULL,           -- relative to audio_parts audio
    diarization_model TEXT,
    diarizations      JSONB,
    persona           JSONB,
    final_results     JSONB,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (audio_part_id, chunk_index),
    CHECK (relative_start_ms >= 0),
    CHECK (relative_end_ms > relative_start_ms),
    CHECK (duration_ms > 0),
    CHECK (duration_ms = relative_end_ms - relative_start_ms)
);

CREATE INDEX idx_chunks_status_created
    ON chunks (status, created_at DESC);

CREATE TRIGGER trg_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
