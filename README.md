---
title: Incident Response Environment
emoji: 🚨
colorFrom: blue
colorTo: red
sdk: docker
---

# IncidentEnv: On-Call Incident Response Triage Environment

IncidentEnv simulates the real workflow of an on-call SRE responding to production alerts. An agent receives an alert, investigates logs and metrics, follows service dependencies, applies fixes, and submits a diagnosis. The environment is deterministic and multi-step, making it suitable for both reinforcement learning and benchmark-style evaluation.

This project is built for the Meta x Hugging Face OpenEnv Hackathon. It focuses on practical incident triage tasks that production teams handle daily, from obvious crashed services to intermittent, hard-to-debug failures in one unhealthy replica.

## Why this environment matters

Incident response is one of the highest-leverage activities in software engineering. When incidents happen, engineers must reason under uncertainty, separate symptoms from causes, and avoid making the outage worse. IncidentEnv creates a reproducible sandbox for training and evaluating those skills in AI agents.

Unlike toy tasks, this environment includes realistic noise, dependency chains, red herrings, and state-changing corrective actions. It gives signal both per-step (for training) and per-episode (for grading).

## Architecture

```text
+--------------------+        POST /step (IncidentAction)        +----------------------+
| Agent / Baseline   | -----------------------------------------> | FastAPI server/app.py |
| (LLM or heuristic) |                                            +----------+-----------+
+---------+----------+                                                       |
          |                              reset/state/grader                 |
          |<----------------------------------------------------------------+
          |
          v
+---------------------------+     uses     +------------------------------+
| server/environment.py     | -----------> | scenarios/*.py                |
| reset/step/state loop     |              | deterministic response maps   |
+-------------+-------------+              +------------------------------+
              |
              +----> server/reward.py (step reward)
              +----> server/grader.py (final score)
```

## Action Space

| Command | Description | Target Example |
|---|---|---|
| query_logs | Read recent service logs | payment-service |
| check_metrics | Read CPU/memory/latency/error metrics | cache-redis |
| check_service_status | Read health and readiness | api-gateway |
| trace_dependency | Inspect dependency graph | user-service |
| check_recent_deploys | Inspect recent deploy history | auth-service |
| restart_service | Restart service workloads | api-gateway |
| rollback_deploy | Roll back to previous deploy | payment-service |
| scale_service | Scale service replicas/capacity | cache-redis |
| escalate | Escalate to a team alias | platform-oncall |
| submit_diagnosis | Submit root-cause reason and end diagnosis | root_cause |

## Observation Space

| Field | Type | Description |
|---|---|---|
| done | bool | Episode ended flag |
| reward | float | Step reward signal |
| alert_message | str | Current incident alert |
| action_result | str | Result from last action |
| services_status | dict[str, str] | Known service status labels |
| step_number | int | Current episode step |
| max_steps | int | Step budget |
| available_services | list[str] | Allowed service/team targets |
| clues_found | list[str] | Discovered clue IDs |
| actions_taken | list[str] | Action history |
| metadata | dict | Extra debug context |

## Tasks

### 1) Easy: The Crashed Service
- Alert: payment-service 503, 100% errors.
- Root cause: bad deployment.
- Correct fix: rollback_deploy on payment-service.
- Typical path: check_service_status -> query_logs -> check_recent_deploys -> rollback_deploy -> submit_diagnosis.

### 2) Medium: The Cascading Failure
- Alert: broad p99 latency spike across api-gateway/user-service/order-service.
- Root cause: cache-redis memory exhaustion -> cache misses -> database overload -> latency cascade.
- Correct fix: scale_service on cache-redis (restart can be temporary but not ideal).
- Includes red herring: recent deploy to user-service that is not the root cause.

### 3) Hard: The Intermittent Ghost
- Alert: api-gateway intermittent 503 (~18%).
- Root cause: memory leak in one replica (replica-3) after recent deploy.
- Correct fix: rollback_deploy on api-gateway.
- Challenge: aggregate metrics hide outlier; must inspect per-replica metrics/logs via parameters.

## Reward Function (Reward Shaping)

Reward shaping gives intermediate feedback each step so the agent can learn useful behavior without waiting until episode end.

Step reward components:
- Discovery bonus: +0.1 per new clue.
- Relevance bonus: +0.02 for relevant targets, -0.02 for irrelevant.
- Corrective action: +0.3 for correct fix, -0.15 for wrong fix.
- Diagnosis bonus: up to +0.3 based on diagnosis accuracy.
- Time pressure: -0.01 every step.
- Escalation shaping: +0.05 only when escalation is scenario-appropriate, else -0.05.

## Episode Grader

Final score in [0.0, 1.0]:
- Diagnosis accuracy: 40%
- Correct fix: 30%
- Efficiency: 15%
- No collateral damage: 15%

Deterministic grading means same state and actions always produce the same score.

## Setup

### Local Python

```bash
cd incident_response_env
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
pip install pytest black
```

### Run Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Tests

```bash
python -m pytest
```

## Docker

```bash
cd incident_response_env
docker build -f server/Dockerfile -t incident-env .
docker run --rm -p 7860:7860 incident-env
```

Then test health:

```bash
curl http://localhost:7860/health
```

## Baseline Inference

Use an env file for live mode configuration:

```bash
cp .env.example .env
# edit .env with your values
set -a
source .env
set +a
python -m baseline.inference
```

Without a key, baseline returns deterministic fallback demo scores.

### Expected baseline score ranges

| Task | Expected Range |
|---|---|
| easy_crashed_service | 0.70 - 0.85 |
| medium_cascading_failure | 0.40 - 0.60 |
| hard_intermittent_ghost | 0.15 - 0.35 |

## Endpoint Summary

- POST /reset with {"task_id": "..."}
- POST /step with IncidentAction JSON
- GET /state
- GET /health
- GET /tasks
- GET /grader
- POST /baseline

## Glossary

- SRE: Site Reliability Engineer, responsible for availability and reliability.
- Incident response: process of triaging and resolving production outages.
- Root cause: fundamental underlying problem, not just symptoms.
- Cascading failure: one failing component overloads downstream services.
- Replica: one instance/pod of a service.
- p99 latency: response time that 99% of requests are faster than.
- Episode: one full environment run from alert to completion.
- Reward signal: numeric feedback at each step guiding learning.
- Grader: final deterministic evaluator that computes an overall score.
