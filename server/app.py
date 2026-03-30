"""FastAPI application exposing IncidentEnv over HTTP.

This module maps OpenEnv-style environment operations to API endpoints.
The endpoint contract is intentionally simple:
- /reset starts an episode
- /step advances one action
- /state inspects internal state
- /grader returns deterministic final score
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import run_baseline
from models import IncidentAction, IncidentObservation, IncidentState
from scenarios import SCENARIOS
from server.environment import IncidentEnvironment
from server.grader import grade_episode


class ResetRequest(BaseModel):
    """Request body for resetting the environment."""

    task_id: str = Field(..., description="Task identifier to run.")


class BaselineRequest(BaseModel):
    """Optional controls for baseline execution."""

    env_url: str | None = Field(
        default=None,
        description="Environment base URL. Defaults to INCIDENT_ENV_URL or localhost.",
    )
    model: str | None = Field(
        default=None,
        description="Groq model name override for baseline execution.",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Per-request timeout during baseline inference.",
        ge=5.0,
        le=120.0,
    )


app = FastAPI(
    title="Incident Response OpenEnv",
    version="1.0.0",
    description=(
        "On-call incident response triage environment with deterministic tasks, "
        "step rewards, and final grading."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

environment = IncidentEnvironment()


@app.get("/")
def root() -> dict[str, Any]:
    """Basic info endpoint for quick discovery."""

    return {
        "name": "incident_response_env",
        "version": "1.0.0",
        "status": "ready",
        "tasks_available": list(SCENARIOS.keys()),
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Healthcheck endpoint required for deployment probes."""

    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    """Return task metadata and action schema."""

    return {
        "tasks": [scenario.task_summary() for scenario in SCENARIOS.values()],
        "action_schema": {
            "command": {
                "type": "string",
                "enum": [
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
                ],
            },
            "target": {"type": "string"},
            "parameters": {"type": "object", "optional": True},
        },
    }


@app.post("/reset", response_model=IncidentObservation)
def reset(
    payload: ResetRequest | None = None,
    task_id: str | None = None,
) -> IncidentObservation:
    """
    Reset environment to a selected task.
    Accept either JSON body or query param and tolerate missing payload.
    """
    if payload is None:
        if task_id:
            payload = ResetRequest(task_id=task_id)
        else:
            # default for hackathon check flow
            payload = ResetRequest(task_id="easy_crashed_service")

    try:
        return environment.reset(task_id=payload.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=IncidentObservation)
def step(action: IncidentAction) -> IncidentObservation:
    """Execute one environment action."""

    try:
        return environment.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=IncidentState)
def state() -> IncidentState:
    """Return internal episode state."""

    return environment.state


@app.get("/grader")
def grader() -> dict[str, Any]:
    """Return deterministic episode score after completion."""

    scenario = environment.current_scenario
    if scenario is None:
        raise HTTPException(status_code=400, detail="Environment has not been reset.")

    if not environment.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is not complete yet. Finish incident before grading.",
        )

    score, breakdown = grade_episode(environment.state, scenario)
    environment.state.score = score

    return {
        "task_id": environment.state.task_id,
        "score": score,
        "breakdown": breakdown,
        "episode_id": environment.state.episode_id,
        "steps_taken": environment.state.step_count,
        "diagnosis_submitted": environment.state.diagnosis_submitted,
        "correct_fix": environment.state.correct_fix,
    }


@app.post("/baseline")
def baseline(payload: BaselineRequest | None = None) -> dict[str, Any]:
    """Run baseline inference and return scores for all tasks."""

    payload = payload or BaselineRequest()
    env_url = payload.env_url or os.getenv("INCIDENT_ENV_URL", "http://localhost:7860")

    try:
        return run_baseline(
            env_url=env_url,
            model=payload.model,
            timeout_seconds=payload.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Baseline execution failed: {exc}",
        ) from exc


def main() -> None:
    """Server entry point required for OpenEnv multi-mode deployment."""
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )

if __name__ == "__main__":
    main()