"""Deterministic episode grader for IncidentEnv.

The grader converts final episode state into a normalized score in [0.0, 1.0].
Unlike step rewards (used for learning), this score is used for evaluation and
leaderboard comparisons.
"""

from __future__ import annotations

from models import IncidentState
from scenarios.base import Scenario
from server.reward import grade_diagnosis


def grade_episode(
    state: IncidentState, scenario: Scenario
) -> tuple[float, dict[str, float]]:
    """Grade a completed episode and return (score, breakdown).

    Formula from the design specification:
    score = diagnosis_accuracy * 0.4 + fix_correct * 0.3 +
            efficiency * 0.15 + no_collateral * 0.15
    """

    diagnosis_component = 0.0
    if state.diagnosis_submitted:
        diagnosis_component = 0.4 * grade_diagnosis(
            {"reason": state.diagnosis or ""}, scenario
        )

    fix_component = 0.0
    if state.fix_applied:
        fix_component = 0.3 if state.correct_fix else 0.0

    useful_steps = len(state.clues_discovered)
    total_steps = state.step_count
    efficiency_raw = min(1.0, useful_steps / max(1, total_steps - 2))
    efficiency_component = 0.15 * efficiency_raw

    no_collateral_component = 0.15 if not state.collateral_damage else 0.0

    score = round(
        diagnosis_component
        + fix_component
        + efficiency_component
        + no_collateral_component,
        4,
    )

    breakdown = {
        "diagnosis_accuracy": round(diagnosis_component, 4),
        "fix_applied": round(fix_component, 4),
        "efficiency": round(efficiency_component, 4),
        "no_collateral_damage": round(no_collateral_component, 4),
    }
    return score, breakdown
