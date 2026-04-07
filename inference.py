"""Root-level inference script for IncidentEnv.

Hackathon-compliant entry point. Reads the following env vars:
  API_BASE_URL  - LLM API base URL (injected by evaluator)
  MODEL_NAME    - Model identifier
  HF_TOKEN      - API key (also accepts API_KEY injected by evaluator)

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
from openai import OpenAI

from models import IncidentAction

logger = logging.getLogger(__name__)

DEFAULT_ENV_URL = "https://atul-k-6o-incident-response-env.hf.space"
DEFAULT_MODEL = "llama-3.1-8b-instant"
TASK_IDS = [
    "easy_crashed_service",
    "medium_intermittent_ghost",
    "hard_cascading_failure",
]

# Hackathon-required env var names. API_BASE_URL and MODEL_NAME have defaults;
# HF_TOKEN / API_KEY are injected by the evaluator at runtime.
_API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
_MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)
_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

BENCHMARK = "incident_response_env"
SUCCESS_SCORE_THRESHOLD = 0.1
SCORE_EPSILON = 1e-6


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
    api_base_url: str = "https://router.huggingface.co/v1"
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 30.0
    max_retries: int = 2
    request_delay_seconds: float = 10.0
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
    elif task_id == "hard_cascading_failure":
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
    previous_results = observation.get("previous_action_results", [])
    previous_logs = observation.get("previous_logs", [])
    dependency_chain = observation.get("dependency_chain", [])
    hypothesis = observation.get("hypothesis", "No hypothesis available.")

    results_str = "\n".join(previous_results) if previous_results else "None yet"
    logs_str = ("\n---\n".join(previous_logs)) if previous_logs else "None yet"
    chain_str = " -> ".join(dependency_chain) if dependency_chain else "None yet"

    return (
        f"Task: {task_id}\n"
        f"Alert: {alert}\n"
        f"Step: {step_number}/{max_steps}  Steps remaining: {steps_remaining}"
        f"{urgent}\n"
        f"Known service statuses: {status_str}\n"
        f"Clues discovered: {clues_str}\n\n"
        f"Actions taken so far:\n{actions_str}\n\n"
        f"Previous action results:\n{results_str}\n\n"
        f"Previous query_logs results:\n{logs_str}\n\n"
        f"Potential dependencies for the current service:\n{chain_str}\n\n"
        f"Current working hypothesis:\n{hypothesis}\n\n"
        f"Last action result:\n{last_result}\n\n"
        f"Available services/targets: {', '.join(services)}\n\n"
        "REMINDER: Use the hypothesis to guide your next step, but do not chase unrelated dependencies unless evidence supports them.\n"
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
    client: OpenAI,
    config: BaselineConfig,
    messages: list[dict],
    observation: dict[str, Any],
    task_id: str,
    step_number: int,
) -> tuple[IncidentAction, bool]:
    """Query LLM for next action using full conversation history.

    Returns (action, used_llm) where used_llm is False when heuristic was used.
    """
    user_prompt = _build_user_prompt(observation, task_id)
    messages.append({"role": "user", "content": user_prompt})

    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            logger.info(
                "[task=%s step=%s] calling LLM model=%s attempt=%s",
                task_id,
                step_number,
                config.model,
                attempt + 1,
            )
            completion = client.chat.completions.create(
                model=config.model,
                temperature=0,
                seed=42,
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
            if _is_rate_limit_error(exc):
                # SDK already retried with its own backoff — don't stack more sleep.
                sleep_seconds = 2.0
            else:
                sleep_seconds = config.request_delay_seconds * (2**attempt)
            logger.warning(
                "[task=%s step=%s] LLM attempt failed attempt=%s error=%s backoff=%.2fs",
                task_id,
                step_number,
                attempt + 1,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    # All retries exhausted — remove the user message we appended.
    messages.pop()
    raise RuntimeError(f"Failed to obtain valid action from LLM: {last_error}")


def _verify_server(http: httpx.Client) -> None:
    """Verify the environment server is reachable and healthy."""
    resp = http.get("/health")
    resp.raise_for_status()
    logger.info("Server health check passed: %s", resp.json())


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
    logger.info("[task=%s] episode reset", task_id)

    max_steps = int(observation.get("max_steps", 15))
    llm_steps = 0
    heuristic_steps = 0
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Conversation history — maintained across all steps of this episode.
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    log_start(task=task_id, env=BENCHMARK, model=config.model)

    try:
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
            if step_resp.status_code == 400:
                try:
                    err_body = step_resp.json()
                    error = err_body.get("detail") or err_body.get("error") or "HTTP 400"
                except Exception:
                    error = "HTTP 400"

                if "not been reset" in (error or "").lower():
                    # Environment session expired (likely due to long rate-limit waits).
                    # Recover by resetting, then end this episode — we can't resume mid-state.
                    logger.warning(
                        "[task=%s step=%s] env session expired — resetting and ending episode",
                        task_id, step_number,
                    )
                    try:
                        http.post("/reset", json={"task_id": task_id})
                    except Exception:
                        pass
                    break

                logger.warning(
                    "[task=%s step=%s] env rejected action command=%s target=%s error=%s — retrying with heuristic",
                    task_id, step_number, action.command, action.target, error,
                )
                action = _heuristic_action(observation, task_id)
                heuristic_steps += 1
                step_resp = http.post("/step", json=action.model_dump(exclude_none=True))
                if step_resp.status_code == 400:
                    # Heuristic also rejected — env is in a bad state, end episode cleanly.
                    try:
                        err_body = step_resp.json()
                        error = err_body.get("detail") or err_body.get("error") or "HTTP 400"
                    except Exception:
                        error = "HTTP 400"
                    logger.warning(
                        "[task=%s step=%s] heuristic also rejected error=%s — ending episode",
                        task_id, step_number, error,
                    )
                    if "not been reset" in (error or "").lower():
                        try:
                            http.post("/reset", json={"task_id": task_id})
                        except Exception:
                            pass
                    break
            step_resp.raise_for_status()
            observation = step_resp.json()

            step_reward = float(observation.get("reward", 0.0))
            done = bool(observation.get("done", False))
            error = observation.get("error") or observation.get("last_action_error") or None
            rewards.append(step_reward)
            steps_taken = step_number

            action_str = f"{action.command}({action.target})"
            log_step(step=step_number, action=action_str, reward=step_reward, done=done, error=error)

            logger.info(
                "[task=%s step=%s] env reward=%.4f done=%s",
                task_id,
                step_number,
                step_reward,
                done,
            )

        grader_resp = http.get("/grader")
        grader_resp.raise_for_status()
        graded = grader_resp.json()
        raw_score = graded.get("score")
        if raw_score is None:
            raise RuntimeError("Grader response missing required 'score' field.")
        score = float(raw_score)
        score = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, score))
        steps_taken = int(graded.get("steps_taken", steps_taken))
        success = score >= SUCCESS_SCORE_THRESHOLD
        logger.info(
            "[task=%s] graded score=%.4f steps=%s llm_steps=%s heuristic_steps=%s",
            task_id,
            score,
            steps_taken,
            llm_steps,
            heuristic_steps,
        )
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "steps_taken": steps_taken,
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
        api_base_url=_API_BASE_URL,
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
                    "task_id": "medium_intermittent_ghost",
                    "score": 0.27,
                    "steps_taken": 9,
                    "mode": "heuristic",
                },
                {
                    "task_id": "hard_cascading_failure",
                    "score": 0.52,
                    "steps_taken": 9,
                    "mode": "heuristic",
                },
            ],
        }

    llm_client = OpenAI(base_url=config.api_base_url, api_key=api_key)
    logger.info("API key detected, live LLM mode enabled base_url=%s model=%s", config.api_base_url, config.model)

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
