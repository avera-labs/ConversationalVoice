from uuid import UUID

import pytest
from voice_pipeline_task_contracts import TRANSCRIBE_CHUNK, TRANSCRIBE_CHUNK_ZH
from voice_pipeline_separate_chunk.publisher import TranscribeChunkPublisher


IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")


class Publisher:
    def publish(self, contract, identifier):
        self.value = (contract, identifier)
        return "task-id"

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("language", "contract"),
    [
        ("en", TRANSCRIBE_CHUNK),
        ("zh", TRANSCRIBE_CHUNK_ZH),
        ("zh-CN", TRANSCRIBE_CHUNK_ZH),
        ("zh-Hant-TW", TRANSCRIBE_CHUNK_ZH),
        ("es", TRANSCRIBE_CHUNK),
    ],
)
def test_routes_chinese_language_family_to_the_specialized_worker(language, contract):
    inner = Publisher()
    publisher = TranscribeChunkPublisher(inner)
    assert publisher.publish(IDENTIFIER, language) == "task-id"
    assert inner.value == (contract, IDENTIFIER)
