"""Base scenario abstraction for deterministic incident simulation.

A scenario is a pre-scripted incident story with a known root cause, known
correct fix, and deterministic responses to actions. Instead of connecting to
real services, we use response maps that translate an action into a canned
result. This gives us three major advantages:

1. Determinism: the same action sequence always produces the same observations.
2. Reproducibility: graders can score results consistently across runs.
3. Safety and speed: no infrastructure setup, no flaky network dependencies.

The core pattern is:
    (command, target) -> response string

For actions that depend on parameters (for example per-replica metrics), we use
an additional map:
    (command, target, parameter_signature) -> response string

This design keeps scenario logic easy to read for beginners while still
supporting nuanced behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models import IncidentAction

ALLOWED_COMMANDS: set[str] = {
    "query_logs",
    "check_metrics",
    "check_service_status",
    "trace_dependency",
    "check_recent_deploys",
    "restart_service",
    "rollback_deploy",
    "scale_service",
    "escalate",
    "submit_diagnosis",
}

CORRECTIVE_COMMANDS: set[str] = {
    "restart_service",
    "rollback_deploy",
    "scale_service",
}


@dataclass(slots=True)
class Scenario:
    """Container for a single incident task definition.

    Attributes:
        task_id: Stable unique task identifier.
        name: Human-readable task name.
        difficulty: Task difficulty label.
        description: Short scenario description.
        max_steps: Maximum actions before timeout.
        initial_alert: Alert shown at reset.
        service_topology: Dependency graph where each service maps to its
            downstream dependencies.
        root_cause: Ground-truth diagnosis reason string.
        correct_fix_command: Ground-truth corrective command.
        correct_fix_target: Ground-truth target service for the fix.
        key_clues: Clue identifiers used by reward and grader logic.
        response_map: Main deterministic action-response map.
        parameter_response_map: Optional parameter-aware responses.
        default_unknown_response: Fallback response for unsupported combinations.
        escalation_targets: Valid escalation targets (teams/aliases).
        affected_services: Services that are part of the incident blast radius.
        should_escalate: Whether escalation is expected for this scenario.
    """

    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    initial_alert: str
    service_topology: dict[str, list[str]]
    root_cause: str
    correct_fix_command: str
    correct_fix_target: str
    key_clues: list[str]
    response_map: dict[tuple[str, str], str] = field(default_factory=dict)
    parameter_response_map: dict[tuple[str, str, str], str] = field(
        default_factory=dict
    )
    default_unknown_response: str = (
        "No useful signal found for that action-target combination."
    )
    escalation_targets: set[str] = field(default_factory=set)
    affected_services: list[str] = field(default_factory=list)
    should_escalate: bool = False

    @property
    def available_services(self) -> list[str]:
        """Return sorted interactive targets.

        We include service names and escalation aliases because both are valid
        action targets in this environment.
        """

        merged = set(self.service_topology.keys())
        merged.update(self.escalation_targets)
        merged.add("root_cause")
        return sorted(merged)

    def is_valid_command(self, command: str) -> bool:
        """Check whether the command is part of the action space."""

        return command in ALLOWED_COMMANDS

    def is_valid_target(self, target: str, command: str) -> bool:
        """Validate action targets by command type.

        submit_diagnosis uses the synthetic target 'root_cause'.
        escalate uses team aliases from escalation_targets.
        All other commands should target known services.
        """

        if command == "submit_diagnosis":
            return target == "root_cause"
        if command == "escalate":
            return target in self.escalation_targets
        return target in self.service_topology

    @staticmethod
    def parameter_signature(parameters: dict | None) -> str:
        """Create a deterministic parameter signature for map lookups."""

        if not parameters:
            return ""
        items = sorted((str(k), str(v)) for k, v in parameters.items())
        return "|".join(f"{k}={v}" for k, v in items)

    def get_response(self, action: IncidentAction) -> str:
        """Resolve an action to a deterministic textual response.

        Lookup order:
        1. Exact parameterized response (command, target, parameter_signature)
        2. Generic response (command, target)
        3. Fallback unknown response
        """

        signature = self.parameter_signature(action.parameters)
        if signature:
            key_with_params = (action.command, action.target, signature)
            if key_with_params in self.parameter_response_map:
                return self.parameter_response_map[key_with_params]

        key = (action.command, action.target)
        return self.response_map.get(key, self.default_unknown_response)

    def trace_chain(self, service: str) -> list[str]:
        """Return a flattened dependency chain reachable from a service."""

        seen: set[str] = set()
        order: list[str] = []

        def dfs(node: str) -> None:
            for dep in self.service_topology.get(node, []):
                if dep in seen:
                    continue
                seen.add(dep)
                order.append(dep)
                dfs(dep)

        dfs(service)
        return order

    def task_summary(self) -> dict[str, str | int]:
        """Return lightweight metadata used by the /tasks endpoint."""

        return {
            "id": self.task_id,
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "max_steps": self.max_steps,
        }
