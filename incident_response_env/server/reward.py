"""Step-level reward functions for the incident response environment.

This module contains reward shaping logic. Reward shaping means giving the agent
small feedback at each step instead of only rewarding the final outcome. That
helps learning because the model can detect which intermediate behaviors are
useful (for example finding a key clue) and which are wasteful.
"""

from __future__ import annotations

from typing import Iterable

from models import IncidentAction
from scenarios.base import Scenario


def _normalize(text: str) -> str:
    """Normalize text for deterministic keyword matching."""

    return " ".join(text.lower().strip().replace("_", " ").split())


def _clue_patterns(clue: str) -> list[str]:
    """Return clue-specific keyword patterns.

    These patterns are intentionally simple and deterministic.
    """

    patterns: dict[str, list[str]] = {
        "payment_service_down": ["payment-service status: down", "error rate: 100%"],
        "crash_error_in_logs": ["runtimeerror", "startup failed", "process terminated"],
        "recent_deploy_found": ["recent deploys", "deploy_id", "dev-bot"],
        "config_key_missing": ["missing config key", "stripe_api_version"],
        "api_gateway_high_latency": ["api-gateway", "p99_latency", "error_rate"],
        "database_overloaded": ["database metrics", "cpu=95%", "slow_query_count"],
        "cache_redis_high_memory": ["cache-redis metrics", "memory=98%"],
        "cache_eviction_warnings": ["eviction", "oom pressure", "evicting"],
        "dependency_chain_traced": ["depends on"],
        "intermittent_503_errors": ["intermittent", "status=503", "error rate 18%"],
        "replica_3_identified": ["replica-3"],
        "replica_3_high_memory": ["replica-3", "memory=95%", "probable memory leak"],
        "oom_killed_events": ["oomkilled", "kubelet", "restarted"],
        "recent_deploy_to_api_gateway": ["recent deploys for api-gateway", "gateway-"],
        "other_replicas_healthy": ["replica-1", "replica-2", "replica-4"],
    }
    return patterns.get(clue, [])


def find_new_clues(
    action: IncidentAction,
    result: str,
    scenario: Scenario,
    existing_clues: Iterable[str] | None = None,
) -> list[str]:
    """Find newly discovered clue IDs from an action result.

    Example:
        If logs contain "missing config key" in the easy task, this function can
        return ["config_key_missing"] if that clue has not already been found.
    """

    seen = set(existing_clues or [])
    result_n = _normalize(result)
    action_n = _normalize(f"{action.command} {action.target}")
    found: list[str] = []

    for clue in scenario.key_clues:
        if clue in seen:
            continue
        patterns = _clue_patterns(clue)
        if not patterns:
            continue
        if all(
            _normalize(p) in result_n or _normalize(p) in action_n for p in patterns[:1]
        ):
            # Primary match: first pattern must appear. Additional hints are optional
            # because many realistic logs vary in exact wording.
            found.append(clue)
            continue

        # Fallback heuristic: any pattern match for robust detection.
        if any(_normalize(p) in result_n for p in patterns):
            found.append(clue)

    # Scenario-specific guardrails to reduce false positives.
    if (
        "dependency_chain_traced" in scenario.key_clues
        and "dependency_chain_traced" not in seen
        and action.command == "trace_dependency"
        and "depends on" in result_n
    ):
        if "dependency_chain_traced" not in found:
            found.append("dependency_chain_traced")

    return found


def _relevant_service_set(scenario: Scenario) -> set[str]:
    """Compute services that are relevant for diagnosis within this scenario."""

    relevant = set(scenario.affected_services)
    frontier = list(scenario.affected_services)

    while frontier:
        service = frontier.pop()
        for dep in scenario.service_topology.get(service, []):
            if dep not in relevant:
                relevant.add(dep)
                frontier.append(dep)

    return relevant


def is_relevant_service(target: str, scenario: Scenario) -> bool:
    """Return whether a target is relevant to the active incident chain."""

    if target == "root_cause":
        return True
    if target in scenario.escalation_targets:
        return scenario.should_escalate
    return target in _relevant_service_set(scenario)


def is_correct_fix(action: IncidentAction, scenario: Scenario) -> bool:
    """Check if a corrective action exactly matches the scenario ground truth."""

    return (
        action.command == scenario.correct_fix_command
        and action.target == scenario.correct_fix_target
    )


def grade_diagnosis(parameters: dict | None, scenario: Scenario) -> float:
    """Grade diagnosis reason in range [0.0, 1.0].

    The baseline and humans may submit synonyms, so we support deterministic
    synonym matching while keeping strictness high.
    """

    if not parameters:
        return 0.0

    reason_raw = str(parameters.get("reason", ""))
    reason = _normalize(reason_raw)
    if not reason:
        return 0.0

    expected = _normalize(scenario.root_cause)
    if reason == expected:
        return 1.0

    synonyms: dict[str, set[str]] = {
        "bad deployment": {
            "bad deployment",
            "faulty deploy",
            "broken deploy",
            "regression from deploy",
        },
        "cache memory exhaustion": {
            "cache memory exhaustion",
            "redis memory exhaustion",
            "redis oom",
            "cache oom",
            "cache eviction storm",
        },
        "memory leak single replica": {
            "memory leak single replica",
            "single replica memory leak",
            "replica-3 memory leak",
            "api gateway replica leak",
            "one bad replica",
        },
    }

    expected_bucket = expected.replace("_", " ")
    candidates = synonyms.get(expected_bucket, set())
    if reason in candidates:
        return 1.0

    # Partial credit when the major concept is present but specificity is missing.
    if "deployment" in reason and "deployment" in expected_bucket:
        return 0.7
    if "cache" in reason and "memory" in reason and "cache" in expected_bucket:
        return 0.7
    if "memory leak" in reason and "memory leak" in expected_bucket:
        return 0.8

    # Minimal partial credit for mentioning impacted subsystem only.
    if "redis" in reason and "cache" in expected_bucket:
        return 0.4
    if "api-gateway" in reason and "replica" in expected_bucket:
        return 0.4

    return 0.0


def compute_step_reward(
    action: IncidentAction,
    result: str,
    existing_clues: Iterable[str],
    scenario: Scenario,
) -> tuple[float, list[str]]:
    """Compute shaped reward for one environment step.

    Returns:
        Tuple of (reward, new_clues)

    Reward components follow the design spec:
    - Discovery bonus
    - Relevance bonus/penalty
    - Corrective action bonus/penalty
    - Diagnosis bonus
    - Time pressure penalty
    - Escalation shaping
    """

    reward = 0.0
    new_clues = find_new_clues(action, result, scenario, existing_clues)

    # 1) Discovery bonus.
    reward += 0.1 * len(new_clues)

    # 2) Efficiency bonus for relevant targets.
    if is_relevant_service(action.target, scenario):
        reward += 0.02
    else:
        reward -= 0.02

    # 3) Corrective action shaping.
    if action.command in {"restart_service", "rollback_deploy", "scale_service"}:
        reward += 0.3 if is_correct_fix(action, scenario) else -0.15

    # 4) Diagnosis bonus.
    if action.command == "submit_diagnosis":
        reward += 0.3 * grade_diagnosis(action.parameters, scenario)

    # 5) Time-pressure penalty.
    reward -= 0.01

    # 6) Escalation shaping.
    if action.command == "escalate":
        reward += 0.05 if scenario.should_escalate else -0.05

    return (reward, new_clues)
