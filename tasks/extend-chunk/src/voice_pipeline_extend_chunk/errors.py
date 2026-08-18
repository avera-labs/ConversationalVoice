def safe_error(code: str, maximum: int) -> str:
    """Return a bounded stable error code without exception details."""

    return code[:maximum]
