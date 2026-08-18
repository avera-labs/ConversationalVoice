"""Safe task failure messages."""


def safe_error(code: str, maximum: int) -> str:
    message = f"{code}: task failed; inspect worker logs using the chunk id"
    return message[:maximum]
