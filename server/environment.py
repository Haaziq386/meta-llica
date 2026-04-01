"""Core simulation engine for the incident response environment.

This module implements the reset/step/state loop used by OpenEnv-like systems:
- reset(task_id): starts a new incident episode.
- step(action): applies one action and returns an observation.
- state: returns internal state for debugging and grading.

An episode is one full incident from alert to resolution (or timeout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from models import IncidentAction, IncidentObservation, IncidentState
from scenarios import get_scenario
from scenarios.base import Scenario
from server.reward import (
    compute_step_reward,
    evaluate_fix_quality,
    grade_diagnosis,
    is_correct_fix,
)

# Import compatibility for different openenv-core versions.
try:  # Try 1: newer path
    from openenv.core.env_server.interfaces import Environment as OpenEnvEnvironment
except Exception:  # noqa: BLE001
    try:  # Try 2: older path
        from openenv.core.env_server import Environment as OpenEnvEnvironment
    except Exception:  # noqa: BLE001

        class OpenEnvEnvironment:  # type: ignore[override]
            """Fallback interface used when openenv-core import is unavailable."""

            def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise NotImplementedError

            def step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise NotImplementedError

            @property
            def state(self) -> dict[str, Any]:
                raise NotImplementedError


@dataclass(slots=True)
class EpisodeRuntime:
    """Mutable runtime metadata not directly part of IncidentState."""

    scenario: Scenario | None = None
    done: bool = False
    actions_taken: list[str] = field(default_factory=list)
    services_status: dict[str, str] = field(default_factory=dict)
    results_history: list[str] = field(default_factory=list)
    query_logs_history: list[str] = field(default_factory=list)
    service_history: list[str] = field(default_factory=list)


class IncidentEnvironment(OpenEnvEnvironment):
    """Deterministic environment implementing incident triage interactions."""

    def __init__(self) -> None:
        self._runtime = EpisodeRuntime()
        self._state = IncidentState()

    @property
    def state(self) -> IncidentState:
        """Return current internal state for /state endpoint consumers."""

        return self._state

    @property
    def current_scenario(self) -> Scenario | None:
        """Expose active scenario metadata for API handlers."""

        return self._runtime.scenario

    @property
    def done(self) -> bool:
        """Expose whether current episode has completed."""

        return self._runtime.done

    def reset(self, task_id: str = "easy_crashed_service") -> IncidentObservation:
        """Reset environment to a task and return initial observation."""

        scenario = get_scenario(task_id)
        self._runtime = EpisodeRuntime(
            scenario=scenario,
            done=False,
            actions_taken=[],
            services_status={
                service: "unknown" for service in scenario.service_topology
            },
        )
        self._state = IncidentState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=scenario.task_id,
            task_difficulty=scenario.difficulty,
            root_cause_found=False,
            fix_applied=False,
            diagnosis_submitted=False,
            clues_discovered=[],
            services_affected=list(scenario.affected_services),
            collateral_damage=False,
            score=0.0,
            diagnosis=None,
            correct_fix=False,
        )
        return IncidentObservation(
            done=False,
            reward=0.0,
            alert_message=scenario.initial_alert,
            action_result="Environment reset. Awaiting first action.",
            services_status=dict(self._runtime.services_status),
            step_number=0,
            max_steps=scenario.max_steps,
            available_services=scenario.available_services,
            clues_found=[],
            actions_taken=[],
            previous_action_results=[],
            previous_logs=[],
            dependency_chain=self._current_dependency_chain(scenario),
            hypothesis=self._build_hypothesis(scenario),
            metadata={"task_id": scenario.task_id, "difficulty": scenario.difficulty},
        )

    def _scenario_or_raise(self) -> Scenario:
        scenario = self._runtime.scenario
        if scenario is None:
            raise RuntimeError("Environment has not been reset. Call /reset first.")
        return scenario

    def _status_from_result(self, target: str, result: str) -> None:
        """Extract coarse status labels from text responses.

        This is intentionally simple. We only need user-facing progress hints,
        not full parser fidelity.
        """

        text = result.lower()
        if "status: down" in text:
            self._runtime.services_status[target] = "down"
        elif "status: degraded" in text or "degraded" in text:
            self._runtime.services_status[target] = "degraded"
        elif "status: healthy" in text or "ready" in text:
            self._runtime.services_status[target] = "healthy"

    def _current_dependency_chain(self, scenario: Scenario) -> list[str]:
        """Choose the most relevant service and return its dependency chain."""

        if self._runtime.service_history:
            root_service = self._runtime.service_history[-1]
        elif scenario.affected_services:
            root_service = scenario.affected_services[0]
        else:
            root_service = next(iter(scenario.service_topology), "")

        if root_service and root_service in scenario.service_topology:
            return scenario.trace_chain(root_service)
        return []

    def _describe_clue(self, clue: str) -> str:
        mapping = {
            "payment_service_down": "payment-service is down",
            "crash_error_in_logs": "payment-service is crashing on startup",
            "config_key_missing": "a required config key is missing",
            "recent_deploy_found": "a recent deployment changed payment-service behavior",
            "dependency_chain_traced": "dependency relationships have been explored",
        }
        return mapping.get(clue, clue.replace("_", " "))

    def _build_hypothesis(self, scenario: Scenario) -> str:
        """Create a concise hypothesis from known clues and service status."""

        clues = self._state.clues_discovered
        clue_set = set(clues)
        if clue_set:
            if {"config_key_missing", "recent_deploy_found"}.issubset(clue_set):
                return (
                    "Working hypothesis: a recent payment-service deployment removed a required "
                    "config fallback, causing startup failure and service outage."
                )
            if {"crash_error_in_logs", "payment_service_down"}.issubset(clue_set):
                return (
                    "Working hypothesis: payment-service is down because it crashes on startup. "
                    "Investigate the service config and recent deploys."
                )
            descriptions = ", ".join(self._describe_clue(clue) for clue in clues)
            return f"Working hypothesis: {descriptions}."

        down_services = [
            service
            for service, status in self._runtime.services_status.items()
            if status == "down"
        ]
        degraded_services = [
            service
            for service, status in self._runtime.services_status.items()
            if status == "degraded"
        ]

        if down_services:
            return f"Working hypothesis: {', '.join(down_services)} are currently down."
        if degraded_services:
            return f"Working hypothesis: {', '.join(degraded_services)} are degraded."

        if scenario.affected_services:
            service = scenario.affected_services[0]
            return (
                f"No strong hypothesis yet; continue investigating {service} and its immediate dependencies."
            )

        return "No strong hypothesis yet; continue investigating the incident."

    def _record_action_result(self, action: IncidentAction, action_result: str) -> None:
        """Persist the action result into runtime history for future observations."""

        self._runtime.results_history.append(
            f"{self._state.step_count}. {action.command} {action.target} => {action_result}"
        )
        if action.command == "query_logs":
            self._runtime.query_logs_history.append(action_result)
        if action.target in self._runtime.services_status:
            self._runtime.service_history.append(action.target)

    def _build_observation(
        self,
        scenario: Scenario,
        reward: float,
        action_result: str,
    ) -> IncidentObservation:
        """Create the canonical observation payload."""

        return IncidentObservation(
            done=self._runtime.done,
            reward=float(round(reward, 4)),
            alert_message=scenario.initial_alert,
            action_result=action_result,
            services_status=dict(self._runtime.services_status),
            step_number=self._state.step_count,
            max_steps=scenario.max_steps,
            available_services=scenario.available_services,
            clues_found=list(self._state.clues_discovered),
            actions_taken=list(self._runtime.actions_taken),
            previous_action_results=list(self._runtime.results_history),
            previous_logs=list(self._runtime.query_logs_history),
            dependency_chain=self._current_dependency_chain(scenario),
            hypothesis=self._build_hypothesis(scenario),
            metadata={
                "task_id": scenario.task_id,
                "difficulty": scenario.difficulty,
                "episode_id": self._state.episode_id,
            },
        )

    def step(self, action: IncidentAction) -> IncidentObservation:
        """Execute one action and return resulting observation.

        Steps in this method map directly to the design doc loop:
        - validate action
        - lookup scenario response
        - discover clues
        - update fix/diagnosis state
        - compute reward
        - mark done if solved or timed out
        """

        scenario = self._scenario_or_raise()

        if self._runtime.done:
            obs = self._build_observation(
                scenario=scenario,
                reward=0.0,
                action_result=(
                    "Episode already completed. Call /reset to start a new incident."
                ),
            )
            return obs

        # Validate action command.
        if not scenario.is_valid_command(action.command):
            self._state.step_count += 1
            self._runtime.actions_taken.append(
                f"{self._state.step_count}. invalid_command:{action.command}"
            )
            reward = -0.1
            if self._state.step_count >= scenario.max_steps:
                self._runtime.done = True
            obs = self._build_observation(
                scenario=scenario,
                reward=reward,
                action_result=(
                    f"Invalid command '{action.command}'. See /tasks action schema."
                ),
            )
            self._record_action_result(action, obs.action_result)
            return obs

        # Validate target for this specific command.
        if not scenario.is_valid_target(action.target, action.command):
            self._state.step_count += 1
            self._runtime.actions_taken.append(
                f"{self._state.step_count}. {action.command} {action.target} (invalid target)"
            )
            reward = -0.1
            if self._state.step_count >= scenario.max_steps:
                self._runtime.done = True
            obs = self._build_observation(
                scenario=scenario,
                reward=reward,
                action_result=(
                    f"Invalid target '{action.target}' for command '{action.command}'."
                ),
            )
            self._record_action_result(action, obs.action_result)
            return obs

        # Action is valid and counts as a normal step.
        self._state.step_count += 1
        self._runtime.actions_taken.append(
            f"{self._state.step_count}. {action.command} {action.target}"
        )

        action_result = scenario.get_response(action)
        self._status_from_result(action.target, action_result)

        # Corrective actions can help or harm production state.
        if action.command in {"restart_service", "rollback_deploy", "scale_service"}:
            self._state.fix_applied = True
            self._state.correct_fix = is_correct_fix(action, scenario)
            if not self._state.correct_fix:
                self._state.collateral_damage = True
            else:
                # Correct fix applied — clear any prior collateral damage from exploratory attempts
                self._state.collateral_damage = False

        # Diagnosis captures whether the root cause was identified correctly.
        if action.command == "submit_diagnosis":
            self._state.diagnosis_submitted = True
            reason = ""
            if action.parameters:
                reason = str(action.parameters.get("reason", ""))
            self._state.diagnosis = reason.strip() or None
            self._state.root_cause_found = (
                grade_diagnosis({"reason": reason}, scenario) >= 0.8
            )

        reward, new_clues = compute_step_reward(
            action=action,
            result=action_result,
            existing_clues=self._state.clues_discovered,
            scenario=scenario,
            actions_taken=self._runtime.actions_taken[:-1],  # exclude current step
        )
        if new_clues:
            self._state.clues_discovered.extend(new_clues)

        # Done conditions: solved or max-step timeout.
        solved = self._state.fix_applied and self._state.diagnosis_submitted
        timed_out = self._state.step_count >= scenario.max_steps
        self._runtime.done = solved or timed_out

        if timed_out and not solved:
            action_result = f"{action_result}\nEpisode ended: max steps reached ({scenario.max_steps})."

        obs = self._build_observation(
            scenario=scenario,
            reward=reward,
            action_result=action_result,
        )
        self._record_action_result(action, action_result)
        return obs
