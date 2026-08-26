"""Compatibility imports for the provider clients.

New code should import from ``openrouter`` or ``fish_audio`` directly.
"""

from .errors import OpenRouterProviderError
from .fish_audio import FishAudioClient
from .openrouter import AudioTagsClient

__all__ = [
    "AudioTagsClient",
    "FishAudioClient",
    "OpenRouterProviderError",
]
