from dataclasses import dataclass

from .celery_app import create_app
from .model import ParakeetModel
from .publisher import PersonaChunkPublisher
from .repository import Repository
from .storage import ObjectStorage
from .task import Handler, register


@dataclass(slots=True)
class Runtime:
    app: object
    repository: Repository
    storage: ObjectStorage
    model: ParakeetModel
    task: object
    publisher: PersonaChunkPublisher

    @classmethod
    def create(cls, settings):
        app = create_app(settings.environment)
        repository = Repository.create(settings.environment)
        storage = ObjectStorage.create(settings.environment)
        model = ParakeetModel(settings.policy, settings.environment.hf_token)
        publisher = PersonaChunkPublisher.create(settings.environment)
        task = register(
            app,
            Handler(repository, storage, model, settings.policy, publisher=publisher),
        )
        return cls(app, repository, storage, model, task, publisher)

    def close(self):
        self.model.close()
        self.publisher.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
