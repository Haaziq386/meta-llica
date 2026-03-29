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

    Improved formula with gradual credit:
    score = diagnosis_accuracy * 0.4 + fix_quality * 0.3 +
            investigation_quality * 0.15 + no_collateral * 0.15

    BACKWARD COMPATIBLE with original formula while adding partial credit.
    """

    # 1. Diagnosis accuracy (independent of fix)
    diagnosis_component = 0.0
    if state.diagnosis_submitted:
        diagnosis_accuracy = grade_diagnosis(
            {"reason": state.diagnosis or ""}, scenario
        )
        diagnosis_component = 0.4 * diagnosis_accuracy

    # 2. Fix quality (gradual scoring instead of binary)
    # Backward compatible: if fix_quality not set, infer from correct_fix
    fix_component = 0.0
    if state.fix_applied:
        fix_quality = state.fix_quality
        # Backward compat: if fix_quality is 0 (default), infer from correct_fix
        if fix_quality == 0.0 and state.correct_fix:
            fix_quality = 1.0
        elif fix_quality == 0.0 and not state.correct_fix:
            fix_quality = 0.0
        fix_component = 0.3 * fix_quality

    # 3. Investigation quality (improved from raw clues/steps)
    # Rewards systematic discovery while allowing for necessary steps
    useful_steps = len(state.clues_discovered)
    total_steps = state.step_count
    
    # Old formula: min(1.0, clues / (steps - 2)) * 0.15
    # New formula: no baseline penalty, rewards actual discovery
    if useful_steps > 0 and total_steps > 2:
        # Coverage: percentage of key clues discovered
        key_clue_coverage = (
            useful_steps / max(1, len(scenario.key_clues))
            if scenario.key_clues
            else 0.0
        )
        
        # Efficiency: clue discovery rate
        clue_efficiency = min(useful_steps / (total_steps - 2), 1.0)
        
        # Combined: 60% coverage, 40% efficiency
        investigation_quality = (key_clue_coverage * 0.6) + (clue_efficiency * 0.4)
        investigation_quality = min(1.0, investigation_quality)
    else:
        # No clues discovered -> no investigation credit
        investigation_quality = 0.0
    
    investigation_component = 0.15 * investigation_quality

    # 4. No collateral damage
    no_collateral_component = 0.15 if not state.collateral_damage else 0.0

    # 5. Calculate total
    score = round(
        diagnosis_component
        + fix_component
        + investigation_component
        + no_collateral_component,
        4,
    )
    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

    # Breakdown for API/tests (maintains backward compatibility)
    breakdown = {
        "diagnosis_accuracy": round(diagnosis_component, 4),
        "fix_applied": round(fix_component, 4),  # Now uses fix_quality
        "efficiency": round(investigation_component, 4),  # Improved calculation
        "no_collateral_damage": round(no_collateral_component, 4),
    }
    return score, breakdown
