from voice_pipeline_ingest_api.dependencies import normalize_database_url


def test_generic_postgresql_urls_select_psycopg_three() -> None:
    assert normalize_database_url("postgresql://user@db/voice") == (
        "postgresql+psycopg://user@db/voice"
    )
    assert normalize_database_url("postgres://user@db/voice") == (
        "postgresql+psycopg://user@db/voice"
    )


def test_explicit_sqlalchemy_driver_is_preserved() -> None:
    url = "postgresql+psycopg://user@db/voice"

    assert normalize_database_url(url) == url
