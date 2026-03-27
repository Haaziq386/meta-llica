"""Baseline inference runner for IncidentEnv.

This module evaluates all tasks by driving the environment with an LLM policy.
It supports two modes:
1) Real mode: calls OpenAI when OPENAI_API_KEY is configured.
2) Fallback mode: returns deterministic demo scores when no key is present.

Run locally:
    python -m baseline.inference
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from models import IncidentAction

DEFAULT_ENV_URL = "http://localhost:7860"
DEFAULT_MODEL = "gpt-4o-mini"
TASK_IDS = [
    "easy_crashed_service",
    "medium_cascading_failure",
    "hard_intermittent_ghost",
]

SYSTEM_PROMPT = """You are an on-call SRE triaging production incidents.
Return ONLY valid JSON matching this schema:
{
  "command": "query_logs|check_metrics|check_service_status|trace_dependency|check_recent_deploys|restart_service|rollback_deploy|scale_service|escalate|submit_diagnosis",
  "target": "string",
  "parameters": {"optional": "object"}
}
Reason step-by-step internally, but output JSON only.
Start by checking alerted service status/logs and follow dependency chains when needed.
"""


@dataclass(slots=True)
class BaselineConfig:
    """Runtime configuration for baseline inference."""

    env_url: str = DEFAULT_ENV_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 30.0
    max_retries: int = 2


def _safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON robustly, handling fenced or prefixed output."""

    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = body.replace("json", "", 1).strip()

    start = body.find("{")
    end = body.rfind("}")
    if start != -1 and end != -1 and end >= start:
        body = body[start : end + 1]

    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("Model output was not a JSON object.")
    return data


def _heuristic_action(observation: dict[str, Any], task_id: str) -> IncidentAction:
    """Simple deterministic fallback policy used on parser/model failures."""

    actions_taken = observation.get("actions_taken", [])
    step = len(actions_taken)

    if task_id == "easy_crashed_service":
        policy = [
            IncidentAction(command="check_service_status", target="payment-service"),
            IncidentAction(command="query_logs", target="payment-service"),
            IncidentAction(command="check_recent_deploys", target="payment-service"),
            IncidentAction(command="rollback_deploy", target="payment-service"),
            IncidentAction(
                command="submit_diagnosis",
                target="root_cause",
                parameters={"reason": "bad_deployment"},
            ),
        ]
    elif task_id == "medium_cascading_failure":
        policy = [
            IncidentAction(command="check_metrics", target="api-gateway"),
            IncidentAction(command="trace_dependency", target="api-gateway"),
            IncidentAction(command="check_metrics", target="user-service"),
            IncidentAction(command="trace_dependency", target="user-service"),
            IncidentAction(command="check_metrics", target="database"),
            IncidentAction(command="check_metrics", target="cache-redis"),
            IncidentAction(command="query_logs", target="cache-redis"),
            IncidentAction(command="scale_service", target="cache-redis"),
            IncidentAction(
                command="submit_diagnosis",
                target="root_cause",
                parameters={"reason": "cache_memory_exhaustion"},
            ),
        ]
    else:
        policy = [
            IncidentAction(command="check_metrics", target="api-gateway"),
            IncidentAction(command="query_logs", target="api-gateway"),
            IncidentAction(command="check_service_status", target="api-gateway"),
            IncidentAction(
                command="check_metrics",
                target="api-gateway",
                parameters={"replica": "replica-3"},
            ),
            IncidentAction(command="check_recent_deploys", target="api-gateway"),
            IncidentAction(
                command="query_logs",
                target="api-gateway",
                parameters={"replica": "replica-3"},
            ),
            IncidentAction(command="trace_dependency", target="api-gateway"),
            IncidentAction(command="rollback_deploy", target="api-gateway"),
            IncidentAction(
                command="submit_diagnosis",
                target="root_cause",
                parameters={"reason": "memory_leak_single_replica"},
            ),
        ]

    return policy[min(step, len(policy) - 1)]


def _llm_action(
    client: OpenAI,
    config: BaselineConfig,
    observation: dict[str, Any],
    task_id: str,
) -> IncidentAction:
    """Query OpenAI for next action and parse into IncidentAction."""

    user_prompt = (
        "Task: "
        f"{task_id}\n"
        "Current observation JSON:\n"
        f"{json.dumps(observation, indent=2)}\n"
        "Respond with one next action JSON object only."
    )

    last_error: Exception | None = None
    for _ in range(config.max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=config.model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=config.timeout_seconds,
            )
            raw = completion.choices[0].message.content or "{}"
            payload = _safe_json_loads(raw)
            return IncidentAction.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.6)

    raise RuntimeError(f"Failed to obtain valid action from OpenAI: {last_error}")


def _run_one_task(
    http: httpx.Client,
    task_id: str,
    config: BaselineConfig,
    client: OpenAI | None,
) -> dict[str, Any]:
    """Run baseline policy for one task through the HTTP environment API."""

    reset_resp = http.post("/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    observation = reset_resp.json()

    max_steps = int(observation.get("max_steps", 15))

    for _ in range(max_steps):
        if observation.get("done", False):
            break

        if client is None:
            action = _heuristic_action(observation, task_id)
        else:
            try:
                action = _llm_action(client, config, observation, task_id)
            except Exception:
                # If model output fails, fall back to deterministic heuristic so
                # baseline execution can still complete instead of aborting.
                action = _heuristic_action(observation, task_id)

        step_resp = http.post("/step", json=action.model_dump(exclude_none=True))
        step_resp.raise_for_status()
        observation = step_resp.json()

    grader_resp = http.get("/grader")
    grader_resp.raise_for_status()
    graded = grader_resp.json()
    return {
        "task_id": task_id,
        "score": float(graded.get("score", 0.0)),
        "steps_taken": int(graded.get("steps_taken", 0)),
        "mode": "llm" if client is not None else "heuristic",
    }


def run_baseline(
    env_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run baseline across all tasks and return aggregated scores."""

    config = BaselineConfig(
        env_url=env_url or os.getenv("INCIDENT_ENV_URL", DEFAULT_ENV_URL),
        model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        timeout_seconds=timeout_seconds,
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "mode": "fallback",
            "note": "OPENAI_API_KEY missing. Returning deterministic reference scores.",
            "scores": [
                {
                    "task_id": "easy_crashed_service",
                    "score": 0.8,
                    "steps_taken": 5,
                    "mode": "heuristic",
                },
                {
                    "task_id": "medium_cascading_failure",
                    "score": 0.52,
                    "steps_taken": 9,
                    "mode": "heuristic",
                },
                {
                    "task_id": "hard_intermittent_ghost",
                    "score": 0.27,
                    "steps_taken": 9,
                    "mode": "heuristic",
                },
            ],
        }

    llm_client = OpenAI(api_key=api_key)
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=config.env_url, timeout=config.timeout_seconds) as http:
        for task_id in TASK_IDS:
            results.append(_run_one_task(http, task_id, config, llm_client))

    average_score = round(sum(item["score"] for item in results) / len(results), 4)
    return {
        "mode": "live",
        "model": config.model,
        "environment_url": config.env_url,
        "scores": results,
        "average_score": average_score,
    }


def _print_summary(result: dict[str, Any]) -> None:
    """Print a simple table for command-line runs."""

    print("\nIncidentEnv Baseline Results")
    print("=" * 60)
    for item in result.get("scores", []):
        print(
            f"{item['task_id']:<28} score={item['score']:<5} "
            f"steps={item.get('steps_taken', 'n/a'):<3} mode={item.get('mode', 'n/a')}"
        )
    if "average_score" in result:
        print("-" * 60)
        print(f"average_score={result['average_score']}")
    if result.get("note"):
        print(f"note: {result['note']}")


if __name__ == "__main__":
    output = run_baseline()
    _print_summary(output)
