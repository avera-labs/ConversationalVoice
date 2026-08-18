from dataclasses import dataclass

from .celery_app import create_app
from .openrouter import OpenRouterClient
from .publisher import ExtendChunkPublisher
from .repository import Repository
from .storage import ObjectStorage
from .task import Handler, register


@dataclass(slots=True)
class Runtime:
    app: object
    repository: Repository
    storage: ObjectStorage
    client: OpenRouterClient
    publisher: ExtendChunkPublisher
    task: object

    @classmethod
    def create(cls, settings):
        app = create_app(settings.environment)
        repository = Repository.create(settings.environment)
        storage = ObjectStorage.create(settings.environment)
        client = OpenRouterClient(
            settings.policy.openrouter,
            settings.environment.openrouter_api_key.get_secret_value(),
        )
        publisher = ExtendChunkPublisher.create(settings.environment)
        task = register(
            app, Handler(repository, storage, client, publisher, settings.policy)
        )
        return cls(app, repository, storage, client, publisher, task)

    def close(self):
        self.publisher.close()
        self.client.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
