from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable
from voice_pipeline_models import AudioPart, Base, Chunk, RawAudio

EXPECTED_COLUMNS = {
    "raw_audios": [
        "id",
        "status",
        "audio_uri",
        "content_sha1",
        "title",
        "source_url",
        "lang",
        "meta",
        "duration_ms",
        "size_bytes",
        "error",
        "created_at",
        "updated_at",
    ],
    "audio_parts": [
        "id",
        "raw_audio_id",
        "part_index",
        "status",
        "audio_uri",
        "diarization_uri",
        "lang",
        "relative_start_ms",
        "relative_end_ms",
        "duration_ms",
        "error",
        "created_at",
        "updated_at",
    ],
    "chunks": [
        "id",
        "audio_part_id",
        "chunk_index",
        "status",
        "audio_uri",
        "lang",
        "duration_ms",
        "relative_start_ms",
        "relative_end_ms",
        "diarization_model",
        "diarizations",
        "persona",
        "final_results",
        "error",
        "created_at",
        "updated_at",
    ],
}


def _constraint_sql(constraints: Iterable[object], kind: type[object]) -> set[str]:
    return {
        str(constraint.sqltext)
        for constraint in constraints
        if isinstance(constraint, kind)
    }


def test_metadata_contains_only_authoritative_tables() -> None:
    assert set(Base.metadata.tables) == set(EXPECTED_COLUMNS)


def test_model_columns_match_schema_order() -> None:
    for table_name, expected_columns in EXPECTED_COLUMNS.items():
        assert list(Base.metadata.tables[table_name].columns.keys()) == expected_columns


def test_foreign_keys_cascade_on_delete() -> None:
    raw_audio_fk = next(iter(AudioPart.__table__.c.raw_audio_id.foreign_keys))
    audio_part_fk = next(iter(Chunk.__table__.c.audio_part_id.foreign_keys))

    assert raw_audio_fk.target_fullname == "raw_audios.id"
    assert raw_audio_fk.ondelete == "CASCADE"
    assert audio_part_fk.target_fullname == "audio_parts.id"
    assert audio_part_fk.ondelete == "CASCADE"


def test_unique_constraints_match_schema() -> None:
    raw_audio_uniques = {
        tuple(column.name for column in constraint.columns)
        for constraint in RawAudio.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    audio_part_uniques = {
        tuple(column.name for column in constraint.columns)
        for constraint in AudioPart.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    chunk_uniques = {
        tuple(column.name for column in constraint.columns)
        for constraint in Chunk.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert raw_audio_uniques == {("content_sha1",)}
    assert audio_part_uniques == {("raw_audio_id", "part_index")}
    assert chunk_uniques == {("audio_part_id", "chunk_index")}


def test_check_constraints_match_schema() -> None:
    assert _constraint_sql(RawAudio.__table__.constraints, CheckConstraint) == {
        "status IN ('pending', 'splitting', 'split_completed', 'failed')",
    }
    assert _constraint_sql(AudioPart.__table__.constraints, CheckConstraint) == {
        "part_index >= 0",
        (
            "status IN ('pending', 'diarizing', 'diarized', "
            "'filtering', 'completed', 'failed')"
        ),
        "relative_end_ms > relative_start_ms",
    }
    assert _constraint_sql(Chunk.__table__.constraints, CheckConstraint) == {
        "chunk_index >= 0",
        "relative_start_ms >= 0",
        "relative_end_ms > relative_start_ms",
        "duration_ms > 0",
        "duration_ms = relative_end_ms - relative_start_ms",
    }


def test_authoritative_schema_contains_status_contracts() -> None:
    schema_path = Path(__file__).resolve().parents[3] / "schema" / "schema.sql"
    normalized_schema = " ".join(schema_path.read_text(encoding="utf-8").split())

    assert (
        "CONSTRAINT ck_raw_audios_status CHECK "
        "( status IN ('pending', 'splitting', 'split_completed', 'failed') )"
        in normalized_schema
    )
    assert (
        "CONSTRAINT ck_audio_parts_status CHECK "
        "( status IN ('pending', 'diarizing', 'diarized', 'filtering', "
        "'completed', 'failed') )" in normalized_schema
    )
    assert "CONSTRAINT ck_chunks_status" not in normalized_schema
    assert (
        "Ingest owns only pending -> failed when downstream task publication fails."
        in normalized_schema
    )
    for transition in (
        "pending -> splitting",
        "failed -> splitting",
        "splitting -> split_completed",
        "splitting -> failed",
        "split_completed -> failed only when downstream task publication fails",
        "pending -> diarizing",
        "failed -> diarizing or filtering, selected by the retrying task",
        "diarizing -> diarized or failed",
        "diarized -> filtering, or failed on downstream publication failure",
        "filtering -> completed or failed",
    ):
        assert transition in normalized_schema
    assert (
        "chunks.status transitions owned by separation, transcription, persona, and extension:"
        in normalized_schema
    )
    for transition in (
        "pending -> separating",
        "failed -> separating, transcribing, persona_generating, or extending, selected",
        "separating -> separated, rejected, or failed",
        "separated -> transcribing",
        "transcribing -> transcribed or failed",
        "transcribed -> persona_generating",
        "persona_generating -> persona_generated or failed",
        "persona_generated -> extending",
        "extending -> completed, rejected, or failed",
        "rejected and completed are terminal",
    ):
        assert transition in normalized_schema


def test_status_created_indexes_match_schema() -> None:
    expected_indexes = {
        "raw_audios": "idx_raw_audios_status_created",
        "audio_parts": "idx_audio_parts_status_created",
        "chunks": "idx_chunks_status_created",
    }

    for table_name, index_name in expected_indexes.items():
        table = Base.metadata.tables[table_name]
        assert {index.name for index in table.indexes} == {index_name}

        index = next(iter(table.indexes))
        compiled = str(CreateIndex(index).compile(dialect=postgresql.dialect()))
        assert "(status, created_at DESC)" in compiled


def test_postgresql_ddl_contains_database_defaults() -> None:
    raw_audio_ddl = str(
        CreateTable(RawAudio.__table__).compile(dialect=postgresql.dialect())
    )
    audio_part_ddl = str(
        CreateTable(AudioPart.__table__).compile(dialect=postgresql.dialect())
    )
    chunk_ddl = str(CreateTable(Chunk.__table__).compile(dialect=postgresql.dialect()))

    for ddl in (raw_audio_ddl, audio_part_ddl, chunk_ddl):
        assert "DEFAULT gen_random_uuid()" in ddl
        assert "status TEXT DEFAULT 'pending' NOT NULL" in ddl
        assert "lang TEXT DEFAULT 'en' NOT NULL" in ddl
        assert "TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL" in ddl

    assert "meta JSONB DEFAULT '{}'::jsonb NOT NULL" in raw_audio_ddl
    assert "diarization_uri TEXT" in audio_part_ddl
    assert "CONSTRAINT ck_audio_parts_status" in audio_part_ddl
    assert "audio_uri TEXT NOT NULL" in chunk_ddl
    assert "duration_ms INTEGER NOT NULL" in chunk_ddl
