from voice_pipeline_extend_chunk.interaction import derive_interaction_targets


def _utterance(index, speaker, start, end, text):
    return {
        "utterance_index": index,
        "speaker_id": speaker,
        "start_ms": start,
        "end_ms": end,
        "text": text,
    }


def test_targets_are_derived_from_reconstruction_timing_and_safely_capped(policy):
    transcript = {
        "utterances": [
            _utterance(0, 0, 0, 4000, "I can explain the first part now."),
            _utterance(1, 1, 3000, 3600, "Yeah"),
            _utterance(2, 0, 4200, 7000, "Then the second part follows."),
            _utterance(3, 1, 6500, 8000, "That makes sense to me."),
        ]
    }

    targets = derive_interaction_targets(transcript, policy.dialogue)

    assert targets.reconstruction_effective_duration_ms == 8000
    assert targets.reconstruction_turn_count == 4
    assert targets.reconstruction_backchannel_count == 1
    assert targets.reconstruction_overlap_event_count == 2
    assert targets.target_turn_count == policy.dialogue.max_utterances
    assert targets.target_backchannel_count <= targets.target_turn_count // 3
    assert targets.target_overlap_event_count <= (targets.target_turn_count - 1) // 2
    prompt_payload = targets.prompt_payload()
    assert prompt_payload["target_overlap_spacing_seconds"] == round(
        policy.dialogue.target_duration_ms
        / targets.target_overlap_event_count
        / 1000,
        2,
    )
    assert prompt_payload["target_overlap_anchor_interval"] == round(
        targets.target_turn_count / targets.target_overlap_event_count,
        2,
    )


def test_same_speaker_fragments_with_tiny_gap_form_one_turn(policy):
    transcript = {
        "utterances": [
            _utterance(0, 0, 0, 1000, "One."),
            _utterance(1, 0, 1050, 2000, "Two."),
            _utterance(2, 1, 2300, 3000, "Three."),
        ]
    }

    targets = derive_interaction_targets(transcript, policy.dialogue)

    assert targets.reconstruction_turn_count == 2


def test_transcript_overlap_target_uses_sixty_millisecond_threshold(policy):
    included = {
        "utterances": [
            _utterance(0, 0, 0, 1000, "One substantive anchor."),
            _utterance(1, 1, 940, 1100, "Brief response."),
        ]
    }
    excluded = {
        "utterances": [
            _utterance(0, 0, 0, 1000, "One substantive anchor."),
            _utterance(1, 1, 941, 1100, "Brief response."),
        ]
    }

    assert (
        derive_interaction_targets(included, policy.dialogue)
        .reconstruction_overlap_event_count
        == 1
    )
    assert (
        derive_interaction_targets(excluded, policy.dialogue)
        .reconstruction_overlap_event_count
        == 0
    )


def test_transcript_overlap_target_matches_evaluator_500_ms_event_merge(policy):
    merged = {
        "utterances": [
            _utterance(0, 0, 0, 3000, "One substantive anchor."),
            _utterance(1, 1, 500, 600, "First response."),
            _utterance(2, 1, 1100, 1200, "Second response."),
        ]
    }
    separate = {
        "utterances": [
            _utterance(0, 0, 0, 3000, "One substantive anchor."),
            _utterance(1, 1, 500, 600, "First response."),
            _utterance(2, 1, 1101, 1201, "Second response."),
        ]
    }

    assert (
        derive_interaction_targets(merged, policy.dialogue)
        .reconstruction_overlap_event_count
        == 1
    )
    assert (
        derive_interaction_targets(separate, policy.dialogue)
        .reconstruction_overlap_event_count
        == 2
    )
