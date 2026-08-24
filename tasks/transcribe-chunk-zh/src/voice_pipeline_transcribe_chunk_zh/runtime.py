from dataclasses import dataclass

from .celery_app import create_app
from .model import ParaformerModel, PunctuationModel
from .publisher import PersonaChunkPublisher
from .repository import Repository
from .storage import ObjectStorage
from .task import Handler, register


@dataclass(slots=True)
class Runtime:
    app: object
    repository: Repository
    storage: ObjectStorage
    model: ParaformerModel
    punctuation: PunctuationModel
    task: object
    publisher: PersonaChunkPublisher

    @classmethod
    def create(cls, settings):
        app = create_app(settings.environment)
        repository = Repository.create(settings.environment)
        storage = ObjectStorage.create(settings.environment)
        model = ParaformerModel(
            settings.policy, settings.environment.paraformer_model_dir
        )
        punctuation = PunctuationModel(
            settings.policy,
            settings.policy.model.device,
            settings.environment.punctuation_model_dir,
        )
        publisher = PersonaChunkPublisher.create(settings.environment)
        task = register(
            app,
            Handler(
                repository,
                storage,
                model,
                punctuation,
                settings.policy,
                publisher=publisher,
            ),
        )
        return cls(app, repository, storage, model, punctuation, task, publisher)

    def close(self):
        self.model.close()
        self.punctuation.close()
        self.publisher.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
