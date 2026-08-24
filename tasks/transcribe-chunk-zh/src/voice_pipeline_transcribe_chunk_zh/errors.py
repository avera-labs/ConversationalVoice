def safe_error(code: str, maximum: int) -> str:
    value = code if isinstance(code, str) and code else "transcription_failed"
    return value[:maximum]
