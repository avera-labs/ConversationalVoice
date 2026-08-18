from uuid import UUID

from voice_pipeline_task_contracts import PERSONA_CHUNK

from voice_pipeline_transcribe_chunk.publisher import PersonaChunkPublisher


class Publisher:
    def publish(self, contract, identifier):
        self.call = contract, identifier
        return "message-id"

    def close(self):
        self.closed = True


def test_publisher_uses_registered_persona_contract():
    backend = Publisher()
    publisher = PersonaChunkPublisher(backend)
    identifier = UUID("11111111-1111-1111-1111-111111111111")
    assert publisher.publish(identifier) == "message-id"
    assert backend.call == (PERSONA_CHUNK, identifier)
    publisher.close()
    assert backend.closed is True
