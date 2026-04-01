"""Scenario registry for all IncidentEnv tasks."""

from __future__ import annotations

from scenarios.base import Scenario
from scenarios.easy_crashed_service import build_easy_crashed_service
from scenarios.medium_intermittent import build_medium_intermittent_ghost
from scenarios.hard_cascading import build_hard_cascading_failure

SCENARIOS: dict[str, Scenario] = {
    "easy_crashed_service": build_easy_crashed_service(),
    "medium_intermittent_ghost": build_medium_intermittent_ghost(),
    "hard_cascading_failure": build_hard_cascading_failure(),
}


def get_scenario(task_id: str) -> Scenario:
    """Return a scenario instance by task id.

    Raises:
            KeyError: If task_id is not defined in SCENARIOS.
    """

    if task_id not in SCENARIOS:
        known = ", ".join(sorted(SCENARIOS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Known tasks: {known}")
    return SCENARIOS[task_id]


__all__ = ["SCENARIOS", "Scenario", "get_scenario"]
