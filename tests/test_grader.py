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
    scenario = SCENARIOS["hard_cascading_failure"]
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
    scenario = SCENARIOS["medium_intermittent_ghost"]
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


def test_gradual_fix_scoring_perfect() -> None:
    """Test that perfect fix gets full credit."""
    scenario = SCENARIOS["easy_crashed_service"]
    state = IncidentState(
        task_id=scenario.task_id,
        step_count=5,
        diagnosis_submitted=True,
        diagnosis=scenario.root_cause,
        fix_applied=True,
        correct_fix=True,
        fix_quality=1.0,  # Perfect fix
        clues_discovered=list(scenario.key_clues),
        collateral_damage=False,
    )
    score, breakdown = grade_episode(state, scenario)
    assert breakdown["fix_applied"] == 0.3  # Full credit
    assert score == 1.0


def test_gradual_fix_scoring_partial() -> None:
    """Test that reasonable but suboptimal fix gets partial credit."""
    scenario = SCENARIOS["hard_cascading_failure"]
    state = IncidentState(
        task_id=scenario.task_id,
        step_count=10,
        diagnosis_submitted=True,
        diagnosis="cache memory exhaustion",
        fix_applied=True,
        correct_fix=False,  # Wrong fix
        fix_quality=0.5,  # But reasonable alternative (e.g., restart instead of scale)
        clues_discovered=["api_gateway_high_latency", "cache_redis_high_memory"],
        collateral_damage=False,
    )
    score, breakdown = grade_episode(state, scenario)
    # Should get 0.15 for fix (0.3 * 0.5)
    assert breakdown["fix_applied"] == 0.15
    # Perfect diagnosis (0.4) + partial fix (0.15) + efficiency + collateral = > 0.5
    assert score > 0.5
    assert score < 1.0


def test_independent_diagnosis_scoring() -> None:
    """Test that diagnosis is scored independently of fix."""
    scenario = SCENARIOS["easy_crashed_service"]
    # Perfect diagnosis but no fix
    state = IncidentState(
        task_id=scenario.task_id,
        step_count=4,
        diagnosis_submitted=True,
        diagnosis=scenario.root_cause,  # Correct diagnosis
        fix_applied=False,  # But no fix taken
        correct_fix=False,
        clues_discovered=["payment_service_down", "crash_error_in_logs", "recent_deploy_found"],
        collateral_damage=False,
    )
    score, breakdown = grade_episode(state, scenario)
    # Should still get diagnosis credit even without fix
    assert breakdown["diagnosis_accuracy"] == 0.4  # Full diagnosis credit
    # But no fix component
    assert breakdown["fix_applied"] == 0.0
    # Score should be diagnosis (0.4) + efficiency + collateral (0.15) = 0.55+
    assert score >= 0.55


def test_collateral_damage_flag_independent() -> None:
    """Test that collateral damage is independent of fix correctness."""
    scenario = SCENARIOS["hard_cascading_failure"]
    # Wrong fix but marked as not harmful (exploratory)
    state = IncidentState(
        task_id=scenario.task_id,
        step_count=8,
        diagnosis_submitted=True,
        diagnosis="cache memory exhaustion",
        fix_applied=True,
        correct_fix=False,
        fix_quality=0.3,  # Exploratory
        clues_discovered=["api_gateway_high_latency", "database_overloaded"],
        collateral_damage=False,  # Explicitly false - fix was exploratory but helped
    )
    score, breakdown = grade_episode(state, scenario)
    # Should get collateral credit even with wrong fix
    assert breakdown["no_collateral_damage"] == 0.15
    # Total should reflect the correctness, not auto-penalized
    assert score > 0.4  # At least diagnosis + collateral
