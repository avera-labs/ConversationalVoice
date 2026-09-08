from __future__ import annotations


class ScoringError(RuntimeError):
    """A stable, user-reportable scoring failure."""

    def __init__(self, code: str, detail: str | None = None):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


def error_code(error: BaseException) -> str:
    if isinstance(error, ScoringError):
        return error.code
    return "unexpected_error"
