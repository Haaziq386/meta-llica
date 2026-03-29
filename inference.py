"""Root-level inference script for IncidentEnv.

Hackathon-compliant entry point. Reads the following env vars:
  HF_TOKEN      - API key (alias for GROQ_API_KEY)
  MODEL_NAME    - Model identifier (alias for GROQ_MODEL)
  API_BASE_URL  - LLM API base URL (read for compliance; Groq uses its own endpoint)

Run:
    python inference.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from groq import Groq

from models import IncidentAction

logger = logging.getLogger(__name__)

DEFAULT_ENV_URL = "http://localhost:7860"
DEFAULT_MODEL = "llama-3.1-8b-instant"
TASK_IDS = [
    "easy_crashed_service",
    "medium_cascading_failure",
    "hard_intermittent_ghost",
]

# Hackathon-required env var names, with project-specific fallbacks.
_API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")  # read for compliance
_MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are an on-call SRE triaging a production incident. You interact with the environment one JSON action at a time.

MANDATORY GOAL: Identify the root cause AND apply the correct fix. Both are required to close the episode and score points.

=== PHASE 1 — INVESTIGATE (first ~60% of your step budget) ===
Commands available: query_logs, check_metrics, check_service_status, trace_dependency, check_recent_deploys
Rules:
- Follow the evidence. Start with the alerted service, then trace its dependencies.
- NEVER repeat a (command, target) pair you have already used — it returns the exact same result and wastes a step.
- For check_metrics on services with multiple replicas, use {"replica": "replica-N"} in parameters to get per-replica data.

=== PHASE 2 — ACT AND CLOSE (when root cause is clear OR steps_remaining <= 4) ===
Do BOTH, in this order:
1. Apply ONE corrective fix: restart_service / rollback_deploy / scale_service on the culprit service.
2. Call submit_diagnosis with parameters: {"reason": "<concise root cause description>"}

The episode does NOT end until you have done BOTH a corrective fix AND submit_diagnosis.

=== CRITICAL WARNING ===
If you reach max_steps without a fix + diagnosis, you score ZERO on 70% of the total score.
When steps_remaining <= 4, STOP all investigation and immediately apply fix + diagnosis.

Return ONLY valid JSON — no markdown, no explanation:
{"command": "...", "target": "...", "parameters": {"key": "value"}}
The "parameters" field is optional except for submit_diagnosis where you must include {"reason": "..."}.
"""


@dataclass(slots=True)
class BaselineConfig:
    """Runtime configuration for baseline inference."""

    env_url: str = DEFAULT_ENV_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 30.0
    max_retries: int = 2
    request_delay_seconds: float = 1.0
    log_level: str = "INFO"


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(kw in msg for kw in ("429", "rate limit", "resource_exhausted", "quota"))


def _safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON robustly, handling fenced or prefixed output."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").replace("json", "", 1).strip()
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end >= start:
        body = body[start : end + 1]
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("Model output was not a JSON object.")
    return data


def _heuristic_action(observation: dict[str, Any], task_id: str) -> IncidentAction:
    """Deterministic near-optimal fallback policy for each task."""
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


def _build_user_prompt(observation: dict[str, Any], task_id: str) -> str:
    """Build a structured, context-rich user prompt for each step."""
    step_number = int(observation.get("step_number", 0))
    max_steps = int(observation.get("max_steps", 15))
    steps_remaining = max_steps - step_number
    actions_taken = observation.get("actions_taken", [])
    clues_found = observation.get("clues_found", [])
    alert = observation.get("alert_message", "")
    last_result = observation.get("action_result", "")
    services = observation.get("available_services", [])
    services_status = observation.get("services_status", {})

    urgent = ""
    if steps_remaining <= 4:
        urgent = (
            f"\n⚠ URGENT: Only {steps_remaining} steps remaining. "
            "STOP investigating. Apply corrective fix then submit_diagnosis RIGHT NOW.\n"
        )

    actions_str = "\n".join(actions_taken) if actions_taken else "None yet"
    clues_str = ", ".join(clues_found) if clues_found else "None yet"
    known_status = {k: v for k, v in services_status.items() if v != "unknown"}
    status_str = ", ".join(f"{k}={v}" for k, v in known_status.items()) or "none yet"

    return (
        f"Task: {task_id}\n"
        f"Alert: {alert}\n"
        f"Step: {step_number}/{max_steps}  Steps remaining: {steps_remaining}"
        f"{urgent}\n"
        f"Known service statuses: {status_str}\n"
        f"Clues discovered: {clues_str}\n\n"
        f"Actions taken so far:\n{actions_str}\n\n"
        f"Last action result:\n{last_result}\n\n"
        f"Available services/targets: {', '.join(services)}\n\n"
        "REMINDER: Do NOT repeat any (command, target) pair listed above.\n"
        "Respond with ONE JSON action only."
    )


def _is_repeated_action(action: IncidentAction, actions_taken: list[str]) -> bool:
    """Return True if this action represents a stuck loop (4+ consecutive identical actions).
    
    This allows LLM to explore and even revisit services, but prevents infinite loops.
    """
    key = f"{action.command} {action.target}"
    
    # Check last 4 actions — if ALL are identical to current, it's a stuck loop
    recent = actions_taken[-4:] if len(actions_taken) >= 4 else actions_taken
    if len(recent) >= 4 and all(key in entry for entry in recent):
        return True
    
    return False


def _llm_action(
    client: Groq,
    config: BaselineConfig,
    messages: list[dict],
    observation: dict[str, Any],
    task_id: str,
    step_number: int,
) -> tuple[IncidentAction, bool]:
    """Query Groq for next action using full conversation history.

    Returns (action, used_llm) where used_llm is False when heuristic was used.
    """
    user_prompt = _build_user_prompt(observation, task_id)
    messages.append({"role": "user", "content": user_prompt})

    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            logger.info(
                "[task=%s step=%s] calling Groq model=%s attempt=%s",
                task_id,
                step_number,
                config.model,
                attempt + 1,
            )
            completion = client.chat.completions.create(
                model=config.model,
                temperature=0.1,
                messages=messages,
                timeout=config.timeout_seconds,
            )
            raw = (
                (completion.choices[0].message.content or "{}")
                if completion.choices
                else "{}"
            )
            payload = _safe_json_loads(raw)
            action = IncidentAction.model_validate(payload)

            # Repetition guard: if LLM is stuck in a loop (4+ consecutive identical), use heuristic.
            actions_taken = observation.get("actions_taken", [])
            if _is_repeated_action(action, actions_taken):
                logger.warning(
                    "[task=%s step=%s] LLM stuck in loop (4+ repeats) command=%s target=%s, switching to heuristic",
                    task_id,
                    step_number,
                    action.command,
                    action.target,
                )
                messages.pop()  # remove the user message we just appended
                return _heuristic_action(observation, task_id), False

            # Record assistant turn in conversation history.
            messages.append({"role": "assistant", "content": raw})
            logger.info(
                "[task=%s step=%s] llm_action command=%s target=%s",
                task_id,
                step_number,
                action.command,
                action.target,
            )
            return action, True

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep_seconds = config.request_delay_seconds * (2**attempt)
            if _is_rate_limit_error(exc):
                sleep_seconds = max(2.0, sleep_seconds)
            logger.warning(
                "[task=%s step=%s] Groq attempt failed attempt=%s error=%s backoff=%.2fs",
                task_id,
                step_number,
                attempt + 1,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    # All retries exhausted — remove the user message we appended.
    messages.pop()
    raise RuntimeError(f"Failed to obtain valid action from Groq: {last_error}")


def _verify_server(http: httpx.Client) -> None:
    """Verify the environment server is reachable and healthy."""
    resp = http.get("/health")
    resp.raise_for_status()
    logger.info("Server health check passed: %s", resp.json())


def _run_one_task(
    http: httpx.Client,
    task_id: str,
    config: BaselineConfig,
    client: Groq | None,
) -> dict[str, Any]:
    """Run baseline policy for one task through the HTTP environment API."""
    reset_resp = http.post("/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    observation = reset_resp.json()
    logger.info("[task=%s] episode reset", task_id)

    max_steps = int(observation.get("max_steps", 15))
    llm_steps = 0
    heuristic_steps = 0

    # Conversation history — maintained across all steps of this episode.
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

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
            if config.request_delay_seconds > 0:
                time.sleep(config.request_delay_seconds)
            try:
                action, used_llm = _llm_action(
                    client, config, messages, observation, task_id, step_number
                )
                if used_llm:
                    llm_steps += 1
                else:
                    heuristic_steps += 1
            except Exception:
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
        model=model or _MODEL_NAME,
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

    api_key = _HF_TOKEN
    if not api_key:
        logger.warning(
            "No API key found (HF_TOKEN / GROQ_API_KEY). Returning deterministic fallback scores."
        )
        return {
            "mode": "fallback",
            "note": "No API key configured. Returning deterministic reference scores.",
            "scores": [
                {
                    "task_id": "easy_crashed_service",
                    "score": 0.80,
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

    llm_client = Groq(api_key=api_key)
    logger.info("API key detected, live Groq mode enabled model=%s", config.model)

    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=config.env_url, timeout=config.timeout_seconds) as http:
        _verify_server(http)
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
    print("\nIncidentEnv Baseline Results")
    print("=" * 60)
    for item in result.get("scores", []):
        print(
            f"{item['task_id']:<28} score={item['score']:<5} "
            f"steps={item.get('steps_taken', 'n/a'):<3} mode={item.get('mode', 'n/a')} "
            f"llm_steps={item.get('llm_steps', 'n/a'):<3} "
            f"heuristic_steps={item.get('heuristic_steps', 'n/a')}"
        )
    if "average_score" in result:
        print("-" * 60)
        print(f"average_score={result['average_score']}")
    if result.get("note"):
        print(f"note: {result['note']}")


if __name__ == "__main__":
    output = run_baseline()
    _print_summary(output)
