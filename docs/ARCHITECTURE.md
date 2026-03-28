# Architecture Guide

## File Responsibilities

- models.py: Pydantic contracts for action, observation, and state.
- scenarios/base.py: Scenario abstraction and response-map lookup logic.
- scenarios/*.py: Deterministic task content (alerts, logs, metrics, deploy history).
- server/environment.py: reset/step/state execution engine.
- server/reward.py: step reward shaping.
- server/grader.py: deterministic final scoring.
- server/app.py: FastAPI endpoints.
- baseline/inference.py: baseline policy runner.
- client.py: Python HTTP client wrapper.

## How one step() flows

```text
Client sends IncidentAction
  -> environment validates command/target
  -> scenario resolves deterministic action_result via response map
  -> reward module detects new clues and computes step reward
  -> environment updates state (fix, diagnosis, collateral, step_count)
  -> done condition checked
  -> IncidentObservation returned
```

## Scenario response-map pattern

Instead of real infrastructure calls, each scenario stores text responses in maps:
- (command, target) -> response
- (command, target, parameter_signature) -> response

This design keeps behavior deterministic and easy to author.

## Reward computation flow

1. Identify newly discovered clues from action result text.
2. Add relevance bonus or penalty.
3. Score corrective action correctness.
4. Score diagnosis quality if submitted.
5. Apply small per-step time penalty.
6. Apply escalation shaping.

## Episode grading flow

At episode end, grader computes weighted components:
- diagnosis_accuracy (40%)
- fix_applied correctness (30%)
- efficiency (15%)
- no_collateral_damage (15%)

The total is rounded to 4 decimals for stable deterministic output.
