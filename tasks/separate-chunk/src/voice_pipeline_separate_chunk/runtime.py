from dataclasses import dataclass

from .alignment import WavLMAligner
from .celery_app import create_app
from .model import DialogueSidon
from .publisher import TranscribeChunkPublisher
from .repository import Repository
from .storage import ObjectStorage
from .task import Handler, register


@dataclass(slots=True)
class Runtime:
    app: object
    repository: Repository
    storage: ObjectStorage
    model: DialogueSidon
    aligner: WavLMAligner
    task: object
    publisher: TranscribeChunkPublisher

    @classmethod
    def create(cls, settings):
        app = create_app(settings.environment)
        repository = Repository.create(settings.environment)
        storage = ObjectStorage.create(settings.environment)
        model = DialogueSidon(settings.policy.model, settings.environment.hf_token)
        aligner = WavLMAligner(settings.policy.alignment)
        publisher = TranscribeChunkPublisher.create(settings.environment)
        task = register(
            app,
            Handler(
                repository,
                storage,
                model,
                aligner,
                settings.policy,
                publisher=publisher,
            ),
        )
        return cls(app, repository, storage, model, aligner, task, publisher)

    def close(self):
        self.aligner.close()
        self.model.close()
        self.publisher.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
