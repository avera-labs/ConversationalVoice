from uuid import UUID

from voice_pipeline_persona_chunk.publisher import ReconstructChunkPublisher
from voice_pipeline_task_contracts import RECONSTRUCT_CHUNK


class Publisher:
    def publish(self, contract, identifier):
        self.call = contract, identifier
        return "message-id"

    def close(self):
        self.closed = True


def test_publisher_uses_registered_reconstruction_contract():
    backend = Publisher()
    publisher = ReconstructChunkPublisher(backend)
    identifier = UUID("11111111-1111-1111-1111-111111111111")
    assert publisher.publish(identifier) == "message-id"
    assert backend.call == (RECONSTRUCT_CHUNK, identifier)
    publisher.close()
    assert backend.closed is True
