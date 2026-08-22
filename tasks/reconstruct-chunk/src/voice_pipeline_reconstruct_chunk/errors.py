def safe_error(code: str, maximum_length: int) -> str:
    return code[:maximum_length]


class OpenRouterProviderError(RuntimeError):
    """Safe provider error without request or response content."""
