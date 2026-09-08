from dataclasses import dataclass
from pathlib import Path

from voice_pipeline_score_completed_chunks.asr import OpenRouterAsrClient
from voice_pipeline_score_completed_chunks.audio_tag_accuracy import (
    AudioTagEvaluator,
    AudioTagScoreEngine,
)
from voice_pipeline_score_completed_chunks.dnsmos import DnsmosScorer
from voice_pipeline_score_completed_chunks.nisqa import NisqaScorer
from voice_pipeline_score_completed_chunks.service import ChunkScoreService
from voice_pipeline_score_completed_chunks.speaker_similarity import (
    SpeakerSimilarityScorer,
)

from .celery_app import create_app
from .repository import Repository
from .storage import ObjectStorage
from .task import Handler, register


@dataclass(slots=True)
class Runtime:
    app: object
    repository: Repository
    storage: ObjectStorage
    asr: OpenRouterAsrClient
    audio_tag: AudioTagScoreEngine
    speaker: SpeakerSimilarityScorer
    service: ChunkScoreService
    task: object

    @classmethod
    def create(cls, settings):
        app = create_app(settings.environment)
        repository = Repository.create(settings.environment)
        storage = ObjectStorage.create(settings.environment)
        cache = Path(settings.policy.task.model_cache_dir).expanduser().resolve()
        nisqa = NisqaScorer(cache)
        dnsmos = DnsmosScorer(cache)
        speaker = SpeakerSimilarityScorer("cpu", cache)
        asr = OpenRouterAsrClient(
            settings.environment.openrouter_api_key.get_secret_value(),
            model=settings.policy.asr.model,
            timeout_seconds=settings.policy.asr.timeout_seconds,
            max_attempts=settings.policy.asr.max_attempts,
        )
        audio_tag_evaluator = AudioTagEvaluator(
            settings.environment.openrouter_api_key.get_secret_value(),
            model=settings.policy.audio_tag.model,
            timeout_seconds=settings.policy.audio_tag.timeout_seconds,
        )
        audio_tag = AudioTagScoreEngine(
            storage=storage,
            evaluator=audio_tag_evaluator,
            workers=settings.policy.audio_tag.workers,
        )
        service = ChunkScoreService(
            storage=storage,
            nisqa=nisqa,
            dnsmos=dnsmos,
            speaker=speaker,
            asr=asr,
            audio_tag=audio_tag,
        )
        task = register(app, Handler(repository, storage, service))
        return cls(app, repository, storage, asr, audio_tag, speaker, service, task)

    def close(self):
        self.asr.close()
        self.speaker.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
