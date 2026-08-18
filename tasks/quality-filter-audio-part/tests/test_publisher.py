from uuid import UUID

from voice_pipeline_quality_filter_audio_part.publisher import SeparateChunkPublisher
from voice_pipeline_task_contracts import SEPARATE_CHUNK


class Client:
    def publish(self, contract, identifier):
        self.call = (contract, identifier)
        return "task-id"

    def close(self):
        self.closed = True


def test_publisher_uses_registered_chunk_contract():
    client = Client()
    publisher = SeparateChunkPublisher(client)
    identifier = UUID("11111111-1111-1111-1111-111111111111")
    assert publisher.publish(identifier) == "task-id"
    assert client.call == (SEPARATE_CHUNK, identifier)
