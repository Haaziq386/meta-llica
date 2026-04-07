"""Typed data models for the Incident Response OpenEnv environment.

This module defines the action, observation, and state contracts used by the
entire environment. In OpenEnv-style systems, these models are important
because every part of the stack (server, client, tests, and baseline agent)
must agree on the exact JSON shape.

Usage examples:

    from models import IncidentAction, IncidentObservation

    action = IncidentAction(command="query_logs", target="payment-service")

    obs = IncidentObservation(
        done=False,
        reward=0.0,
        alert_message="CRITICAL: payment-service returning 503 errors",
        action_result="No action yet",
    )

These are Pydantic v2 models, so they validate incoming data and also produce
clear JSON schemas for API documentation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IncidentAction(BaseModel):
    """Action the on-call agent can take in one environment step.

    The agent sends exactly one action on each call to step().
    """

    command: str = Field(
        ...,
        description=(
            "One of: query_logs, check_metrics, check_service_status, "
            "trace_dependency, check_recent_deploys, restart_service, "
            "rollback_deploy, scale_service, escalate, submit_diagnosis"
        ),
    )
    target: str = Field(
        ...,
        description=(
            "Service, component, or resource to act on. Examples: "
            "'api-gateway', 'database', 'auth-service', 'database-team'."
        ),
    )
    parameters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional command-specific parameters. Example: {'time_range': '1h'} "
            "for query_logs or {'reason': 'cache_memory_exhaustion'} for "
            "submit_diagnosis."
        ),
    )


class IncidentObservation(BaseModel):
    """Observation returned after each action.

    In reinforcement learning terms, this is what the agent sees after taking
    an action. The agent uses this to decide its next move.
    """

    done: bool = Field(
        default=False,
        description="Whether the current episode has ended.",
    )
    reward: float = Field(
        default=0.0,
        description="Step-level reward signal for the action that was just taken.",
    )
    alert_message: str = Field(
        ...,
        description="Current incident alert shown to the on-call agent.",
    )
    action_result: str = Field(
        ...,
        description=(
            "Result of the action taken in this step, such as logs, metrics, "
            "service status, or an operation outcome."
        ),
    )
    services_status: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Known service statuses. Example: {'api-gateway': 'degraded', "
            "'payment-service': 'down'}."
        ),
    )
    step_number: int = Field(
        default=0,
        description="Current step number in the episode, starting at 0 after reset.",
    )
    max_steps: int = Field(
        default=15,
        description="Maximum number of steps allowed before timeout.",
    )
    available_services: list[str] = Field(
        default_factory=list,
        description="All services/resources the agent can interact with.",
    )
    clues_found: list[str] = Field(
        default_factory=list,
        description="Key clues discovered so far in this episode.",
    )
    actions_taken: list[str] = Field(
        default_factory=list,
        description="Human-readable action history for this episode.",
    )
    previous_action_results: list[str] = Field(
        default_factory=list,
        description=(
            "Readable history of past action results, excluding the current "
            "action_result field."
        ),
    )
    previous_logs: list[str] = Field(
        default_factory=list,
        description=(
            "Text output from all query_logs actions executed so far in the episode."
        ),
    )
    dependency_chain: list[str] = Field(
        default_factory=list,
        description=(
            "Dependency chain for the service currently under investigation."
        ),
    )
    hypothesis: str = Field(
        default="",
        description=(
            "A concise working hypothesis based on discovered clues and known service status."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra structured data for debugging/analysis. Not required for agent "
            "logic, but useful for introspection."
        ),
    )


class IncidentState(BaseModel):
    """Internal environment state returned by the /state endpoint.

    This state is richer than the observation and is mainly for debugging,
    grading, and reproducibility checks.
    """

    episode_id: str | None = Field(
        default=None,
        description="Unique identifier for the current episode.",
    )
    step_count: int = Field(
        default=0,
        description="Total number of actions taken in the current episode.",
    )
    task_id: str = Field(
        default="",
        description="Identifier of the active scenario task.",
    )
    task_difficulty: str = Field(
        default="",
        description="Difficulty label for the task: easy, medium, or hard.",
    )
    root_cause_found: bool = Field(
        default=False,
        description="Whether the agent has identified the root cause with enough confidence.",
    )
    fix_applied: bool = Field(
        default=False,
        description="Whether a corrective action has been applied.",
    )
    diagnosis_submitted: bool = Field(
        default=False,
        description="Whether submit_diagnosis has been called.",
    )
    clues_discovered: list[str] = Field(
        default_factory=list,
        description="All key clues discovered during the episode.",
    )
    services_affected: list[str] = Field(
        default_factory=list,
        description="Services known to be affected by the incident.",
    )
    collateral_damage: bool = Field(
        default=False,
        description=(
            "Whether the agent caused extra harm through incorrect corrective actions."
        ),
    )
    score: float = Field(
        default=0.0001,
        description="Final grader score for the episode in range (0.0, 1.0).",
    )
    diagnosis: str | None = Field(
        default=None,
        description="Normalized diagnosis reason submitted by the agent.",
    )
    correct_fix: bool = Field(
        default=False,
        description="Whether the applied fix matched the scenario ground truth.",
    )
    fix_quality: float = Field(
        default=0.0,
        description=(
            "Quality score of the applied fix in range [0.0, 1.0], where 1.0 is perfect, "
            "0.5 is reasonable alternative, 0.3 is exploratory, 0.0 is wrong."
        ),
    )
