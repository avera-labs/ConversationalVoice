from uuid import UUID

from voice_pipeline_reconstruct_chunk.publisher import ExtendChunkPublisher
from voice_pipeline_task_contracts import EXTEND_CHUNK


class Backend:
    def publish(self, contract, identifier):
        self.call = contract, identifier
        return "task-id"

    def close(self):
        self.closed = True


def test_publisher_routes_to_extend_chunk():
    backend = Backend()
    publisher = ExtendChunkPublisher(backend)
    identifier = UUID("11111111-1111-1111-1111-111111111111")
    assert publisher.publish(identifier) == "task-id"
    assert backend.call == (EXTEND_CHUNK, identifier)
    publisher.close()
    assert backend.closed is True
