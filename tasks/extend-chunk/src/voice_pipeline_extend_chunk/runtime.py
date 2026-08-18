from dataclasses import dataclass

from .celery_app import create_app
from .fish_audio import OpenRouterFishAudioClient
from .openrouter import OpenRouterClient
from .repository import Repository
from .storage import ObjectStorage
from .task import Handler, register


@dataclass(slots=True)
class Runtime:
    app: object
    repository: Repository
    storage: ObjectStorage
    dialogue_client: OpenRouterClient
    fish_client: OpenRouterFishAudioClient
    task: object

    @classmethod
    def create(cls, settings):
        app = create_app(settings.environment)
        repository = Repository.create(settings.environment)
        storage = ObjectStorage.create(settings.environment)
        dialogue_client = OpenRouterClient(
            settings.policy.openrouter,
            settings.environment.openrouter_api_key.get_secret_value(),
        )
        fish_client = OpenRouterFishAudioClient(
            settings.policy.fish_audio,
            settings.environment.openrouter_api_key.get_secret_value(),
        )
        task = register(
            app,
            Handler(
                repository,
                storage,
                dialogue_client,
                fish_client,
                settings.policy,
            ),
        )
        return cls(app, repository, storage, dialogue_client, fish_client, task)

    def close(self):
        self.fish_client.close()
        self.dialogue_client.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
