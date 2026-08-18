import pytest

from voice_pipeline_quality_filter_audio_part.config import PlannerPolicy, QualityPolicy, TaskPolicy


@pytest.fixture
def quality_policy() -> QualityPolicy:
    return QualityPolicy(
        min_snr_db=10.0,
        music_probability_threshold=0.2,
        min_music_interval_ms=2000,
        music_gap_fill_ms=600,
        max_music_overlap_ratio=0.3,
        max_absorbable_bad_group_ms=3000,
        min_good_region_ms=20000,
    )


@pytest.fixture
def planner_policy() -> PlannerPolicy:
    return PlannerPolicy(
        min_planning_window_ms=20000,
        max_planning_window_ms=60000,
        min_speaker_turn_ms=4000,
        min_speaker_total_ms=8000,
        backchannel_threshold_ms=1500,
        max_monologue_ms=40000,
        max_inner_iterations=200,
    )


@pytest.fixture
def task_policy() -> TaskPolicy:
    return TaskPolicy(
        error_max_length=512,
        workspace_prefix="quality-test-",
        max_diarization_bytes=1024 * 1024,
    )
