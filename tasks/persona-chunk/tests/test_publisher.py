from uuid import UUID

from voice_pipeline_persona_chunk.publisher import ExtendChunkPublisher
from voice_pipeline_task_contracts import EXTEND_CHUNK


class Publisher:
    def publish(self, contract, identifier):
        self.call = contract, identifier
        return "message-id"

    def close(self):
        self.closed = True


def test_publisher_uses_registered_extension_contract():
    backend = Publisher()
    publisher = ExtendChunkPublisher(backend)
    identifier = UUID("11111111-1111-1111-1111-111111111111")
    assert publisher.publish(identifier) == "message-id"
    assert backend.call == (EXTEND_CHUNK, identifier)
    publisher.close()
    assert backend.closed is True
