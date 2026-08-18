from uuid import UUID

from voice_pipeline_task_contracts import QUALITY_FILTER_AUDIO_PART

from voice_pipeline_diarize_audio_part.publisher import QualityFilterPublisher


def test_publisher_uses_shared_contract() -> None:
    calls: list[tuple] = []

    class SharedPublisher:
        def publish(self, contract, identifier):
            calls.append((contract, identifier))
            return "message-id"

    identifier = UUID("11111111-1111-1111-1111-111111111111")
    assert QualityFilterPublisher(SharedPublisher()).publish(identifier) == "message-id"
    assert calls == [(QUALITY_FILTER_AUDIO_PART, identifier)]
