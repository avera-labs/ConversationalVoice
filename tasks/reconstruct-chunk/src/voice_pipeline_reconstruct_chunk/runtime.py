from dataclasses import dataclass

from voice_pipeline_forced_alignment import Qwen3SegmentAligner

from .celery_app import create_app
from .fish_audio import FishAudioClient
from .openrouter import AudioTagsClient
from .publisher import ExtendChunkPublisher
from .repository import Repository
from .storage import ObjectStorage
from .task import Handler, register


@dataclass(slots=True)
class Runtime:
    app: object
    repository: Repository
    storage: ObjectStorage
    tags_client: AudioTagsClient
    tts_client: FishAudioClient
    forced_aligner: Qwen3SegmentAligner
    publisher: ExtendChunkPublisher
    task: object

    @classmethod
    def create(cls, settings):
        app = create_app(settings.environment)
        repository = Repository.create(settings.environment)
        storage = ObjectStorage.create(settings.environment)
        api_key = settings.environment.openrouter_api_key.get_secret_value()
        tags_client = AudioTagsClient(settings.policy.audio_tags, api_key)
        tts_client = FishAudioClient(settings.policy.tts, api_key)
        forced_aligner = Qwen3SegmentAligner(settings.policy.forced_alignment)
        publisher = ExtendChunkPublisher.create(settings.environment)
        task = register(
            app,
            Handler(
                repository,
                storage,
                tags_client,
                tts_client,
                publisher,
                settings.policy,
                forced_aligner=forced_aligner,
            ),
        )
        return cls(
            app,
            repository,
            storage,
            tags_client,
            tts_client,
            forced_aligner,
            publisher,
            task,
        )

    def close(self):
        self.forced_aligner.close()
        self.publisher.close()
        self.tts_client.close()
        self.tags_client.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
