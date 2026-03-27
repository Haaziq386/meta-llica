"""Tests for deterministic grader behavior."""

from __future__ import annotations

from models import IncidentState
from scenarios import SCENARIOS
from server.grader import grade_episode


def test_grader_returns_score_in_range() -> None:
    state = IncidentState(
        task_id="easy_crashed_service",
        step_count=5,
        diagnosis_submitted=True,
        diagnosis="bad_deployment",
        fix_applied=True,
        correct_fix=True,
        clues_discovered=["a", "b", "c"],
        collateral_damage=False,
    )
    score, _ = grade_episode(state, SCENARIOS["easy_crashed_service"])
    assert 0.0 <= score <= 1.0


def test_perfect_score_scenario() -> None:
    scenario = SCENARIOS["easy_crashed_service"]
    state = IncidentState(
        task_id=scenario.task_id,
        step_count=5,
        diagnosis_submitted=True,
        diagnosis=scenario.root_cause,
        fix_applied=True,
        correct_fix=True,
        clues_discovered=list(scenario.key_clues),
        collateral_damage=False,
    )
    score, breakdown = grade_episode(state, scenario)
    assert score == 1.0
    assert breakdown["diagnosis_accuracy"] == 0.4
    assert breakdown["fix_applied"] == 0.3
    assert breakdown["no_collateral_damage"] == 0.15


def test_zero_score_like_scenario() -> None:
    scenario = SCENARIOS["medium_cascading_failure"]
    state = IncidentState(
        task_id=scenario.task_id,
        step_count=8,
        diagnosis_submitted=False,
        diagnosis=None,
        fix_applied=False,
        correct_fix=False,
        clues_discovered=[],
        collateral_damage=True,
    )
    score, breakdown = grade_episode(state, scenario)
    assert score == 0.0
    assert breakdown["diagnosis_accuracy"] == 0.0
    assert breakdown["fix_applied"] == 0.0
    assert breakdown["efficiency"] == 0.0
    assert breakdown["no_collateral_damage"] == 0.0


def test_partial_score_scenario() -> None:
    scenario = SCENARIOS["hard_intermittent_ghost"]
    state = IncidentState(
        task_id=scenario.task_id,
        step_count=10,
        diagnosis_submitted=True,
        diagnosis="memory leak",
        fix_applied=True,
        correct_fix=False,
        clues_discovered=["intermittent_503_errors", "replica_3_identified"],
        collateral_damage=True,
    )
    score, _ = grade_episode(state, scenario)
    assert 0.0 < score < 1.0


def test_grader_is_deterministic() -> None:
    scenario = SCENARIOS["easy_crashed_service"]
    state = IncidentState(
        task_id=scenario.task_id,
        step_count=6,
        diagnosis_submitted=True,
        diagnosis=scenario.root_cause,
        fix_applied=True,
        correct_fix=True,
        clues_discovered=list(scenario.key_clues),
        collateral_damage=False,
    )
    score_a, breakdown_a = grade_episode(state, scenario)
    score_b, breakdown_b = grade_episode(state, scenario)
    assert score_a == score_b
    assert breakdown_a == breakdown_b
