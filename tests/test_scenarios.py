"""Scenario consistency tests."""

from __future__ import annotations

from scenarios import SCENARIOS

REQUIRED_FIELDS = [
    "task_id",
    "name",
    "difficulty",
    "description",
    "max_steps",
    "initial_alert",
    "service_topology",
    "root_cause",
    "correct_fix_command",
    "correct_fix_target",
    "key_clues",
    "response_map",
]


def test_all_scenarios_have_required_fields() -> None:
    for scenario in SCENARIOS.values():
        for field in REQUIRED_FIELDS:
            assert hasattr(scenario, field)
        assert scenario.max_steps > 0
        assert scenario.root_cause
        assert scenario.correct_fix_target in scenario.service_topology


def test_response_maps_cover_expected_path_minimum() -> None:
    assert len(SCENARIOS["easy_crashed_service"].response_map) >= 10
    assert len(SCENARIOS["medium_intermittent_ghost"].response_map) >= 20
    assert len(SCENARIOS["hard_cascading_failure"].response_map) >= 25


def test_key_clues_are_findable_in_map_text() -> None:
    # This is a pragmatic check to ensure each scenario contains evidence text
    # likely to trigger clue discovery heuristics.
    expected_markers = {
        "easy_crashed_service": [
            "missing config key",
            "deploy_id",
            "status: DOWN",
        ],
        "hard_cascading_failure": [
            "memory=98%",
            "eviction",
            "depends on",
        ],
        "medium_intermittent_ghost": [
            "replica-3",
            "OOMKilled",
            "Recent deploys for api-gateway",
        ],
    }

    for task_id, markers in expected_markers.items():
        scenario = SCENARIOS[task_id]
        all_text = "\n".join(scenario.response_map.values())
        all_text += "\n" + "\n".join(scenario.parameter_response_map.values())
        for marker in markers:
            assert marker in all_text
