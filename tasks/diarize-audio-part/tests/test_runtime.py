from voice_pipeline_diarize_audio_part.runtime import TaskRuntime


class Closeable:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def close(self) -> None:
        self.events.append(self.name)


def test_runtime_closes_model_and_clients_once() -> None:
    events: list[str] = []
    runtime = TaskRuntime(
        app=Closeable("app", events),
        repository=Closeable("repository", events),
        storage=Closeable("storage", events),
        publisher=Closeable("publisher", events),
        diarization=Closeable("diarization", events),
        task=object(),
    )
    runtime.close()
    runtime.close()
    assert events == ["diarization", "publisher", "storage", "repository", "app"]
