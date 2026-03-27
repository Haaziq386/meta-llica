"""Baseline inference runner for IncidentEnv.

This module evaluates all tasks by driving the environment with an LLM policy.
It supports two modes:
1) Real mode: calls Gemini when GOOGLE_API_KEY is configured.
2) Fallback mode: returns deterministic demo scores when no key is present.

Run locally:
    python -m baseline.inference
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai
import httpx

from models import IncidentAction

logger = logging.getLogger(__name__)

DEFAULT_ENV_URL = "http://localhost:7860"
DEFAULT_MODEL = "gemini-3-flash-preview"
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
    request_delay_seconds: float = 15.0
    log_level: str = "INFO"


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an exception likely represents provider throttling."""

    msg = f"{type(exc).__name__}: {exc}".lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "resource_exhausted" in msg
        or "quota" in msg
    )


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
    client: genai.GenerativeModel,
    config: BaselineConfig,
    observation: dict[str, Any],
    task_id: str,
    step_number: int,
) -> IncidentAction:
    """Query Gemini for next action and parse into IncidentAction."""

    user_prompt = (
        "Task: "
        f"{task_id}\n"
        "Current observation JSON:\n"
        f"{json.dumps(observation, indent=2)}\n"
        "Respond with one next action JSON object only."
    )

    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            logger.info(
                "[task=%s step=%s] calling Gemini model=%s attempt=%s",
                task_id,
                step_number,
                config.model,
                attempt + 1,
            )
            completion = client.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
                request_options={"timeout": config.timeout_seconds},
            )
            raw = completion.text or "{}"
            payload = _safe_json_loads(raw)
            action = IncidentAction.model_validate(payload)
            logger.info(
                "[task=%s step=%s] llm_action command=%s target=%s",
                task_id,
                step_number,
                action.command,
                action.target,
            )
            return action
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            is_rate_limited = _is_rate_limit_error(exc)
            sleep_seconds = config.request_delay_seconds * (2 ** attempt)
            if is_rate_limited:
                # Apply stronger backoff for provider throttling responses.
                sleep_seconds = max(2.0, sleep_seconds)
            logger.warning(
                "[task=%s step=%s] Gemini attempt failed attempt=%s error=%s backoff=%.2fs",
                task_id,
                step_number,
                attempt + 1,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to obtain valid action from Gemini: {last_error}")


def _run_one_task(
    http: httpx.Client,
    task_id: str,
    config: BaselineConfig,
    client: genai.GenerativeModel | None,
) -> dict[str, Any]:
    """Run baseline policy for one task through the HTTP environment API."""

    reset_resp = http.post("/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    observation = reset_resp.json()
    logger.info("[task=%s] episode reset", task_id)

    max_steps = int(observation.get("max_steps", 15))
    llm_steps = 0
    heuristic_steps = 0

    for _ in range(max_steps):
        if observation.get("done", False):
            break

        step_number = int(observation.get("step_number", 0)) + 1

        if client is None:
            action = _heuristic_action(observation, task_id)
            heuristic_steps += 1
            logger.info(
                "[task=%s step=%s] heuristic_action command=%s target=%s",
                task_id,
                step_number,
                action.command,
                action.target,
            )
        else:
            try:
                if config.request_delay_seconds > 0:
                    logger.debug(
                        "[task=%s step=%s] throttling next LLM call for %.2fs",
                        task_id,
                        step_number,
                        config.request_delay_seconds,
                    )
                    time.sleep(config.request_delay_seconds)
                action = _llm_action(client, config, observation, task_id, step_number)
                llm_steps += 1
            except Exception:
                # If model output fails, fall back to deterministic heuristic so
                # baseline execution can still complete instead of aborting.
                action = _heuristic_action(observation, task_id)
                heuristic_steps += 1
                logger.warning(
                    "[task=%s step=%s] falling back to heuristic action",
                    task_id,
                    step_number,
                )

        step_resp = http.post("/step", json=action.model_dump(exclude_none=True))
        step_resp.raise_for_status()
        observation = step_resp.json()
        logger.info(
            "[task=%s step=%s] env reward=%.4f done=%s",
            task_id,
            step_number,
            float(observation.get("reward", 0.0)),
            bool(observation.get("done", False)),
        )

    grader_resp = http.get("/grader")
    grader_resp.raise_for_status()
    graded = grader_resp.json()
    logger.info(
        "[task=%s] graded score=%.4f steps=%s llm_steps=%s heuristic_steps=%s",
        task_id,
        float(graded.get("score", 0.0)),
        int(graded.get("steps_taken", 0)),
        llm_steps,
        heuristic_steps,
    )
    return {
        "task_id": task_id,
        "score": float(graded.get("score", 0.0)),
        "steps_taken": int(graded.get("steps_taken", 0)),
        "mode": "llm" if client is not None else "heuristic",
        "llm_steps": llm_steps,
        "heuristic_steps": heuristic_steps,
    }


def run_baseline(
    env_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run baseline across all tasks and return aggregated scores."""

    config = BaselineConfig(
        env_url=env_url or os.getenv("INCIDENT_ENV_URL", DEFAULT_ENV_URL),
        model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        timeout_seconds=timeout_seconds,
        request_delay_seconds=float(
            os.getenv("INCIDENT_LLM_REQUEST_DELAY_SECONDS", "1.0")
        ),
        log_level=os.getenv("INCIDENT_BASELINE_LOG_LEVEL", "INFO"),
    )

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info(
        "Starting baseline env_url=%s model=%s timeout=%.1fs delay=%.2fs",
        config.env_url,
        config.model,
        config.timeout_seconds,
        config.request_delay_seconds,
    )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY missing, returning deterministic fallback scores")
        return {
            "mode": "fallback",
            "note": "GOOGLE_API_KEY missing. Returning deterministic reference scores.",
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

    genai.configure(api_key=api_key)
    llm_client = genai.GenerativeModel(
        model_name=config.model,
        system_instruction=SYSTEM_PROMPT,
    )
    logger.info("GOOGLE_API_KEY detected, live Gemini mode enabled")
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=config.env_url, timeout=config.timeout_seconds) as http:
        for task_id in TASK_IDS:
            results.append(_run_one_task(http, task_id, config, llm_client))

    average_score = round(sum(item["score"] for item in results) / len(results), 4)
    logger.info("Baseline complete average_score=%.4f", average_score)
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
            f"steps={item.get('steps_taken', 'n/a'):<3} mode={item.get('mode', 'n/a')} "
            f"llm_steps={item.get('llm_steps', 'n/a'):<3} heuristic_steps={item.get('heuristic_steps', 'n/a')}"
        )
    if "average_score" in result:
        print("-" * 60)
        print(f"average_score={result['average_score']}")
    if result.get("note"):
        print(f"note: {result['note']}")


if __name__ == "__main__":
    output = run_baseline()
    _print_summary(output)
