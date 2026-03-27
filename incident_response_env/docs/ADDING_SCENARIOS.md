# Adding Scenarios Guide

## Goal

This guide explains how to add a new deterministic task (Task 4, Task 5, etc.).

## Step-by-step

1. Create a new file in scenarios/, for example scenarios/hard_network_partition.py.
2. Define service_topology as a dependency graph.
3. Fill scenario metadata (task_id, name, difficulty, max_steps, initial_alert).
4. Define ground truth:
   - root_cause
   - correct_fix_command
   - correct_fix_target
5. Define key_clues list with 4-8 clue IDs.
6. Fill response_map with realistic text for expected and off-path actions.
7. Add parameter_response_map when behavior depends on parameters.
8. Register the scenario in scenarios/__init__.py SCENARIOS.
9. Add tests in tests/test_scenarios.py for clue discoverability and coverage.

## Template

```python
from scenarios.base import Scenario


def build_new_scenario() -> Scenario:
    service_topology = {
        "service-a": ["service-b"],
        "service-b": [],
    }

    response_map = {
        ("check_service_status", "service-a"): "service-a status: DEGRADED",
        ("query_logs", "service-a"): "...",
        ("rollback_deploy", "service-a"): "...",
        ("submit_diagnosis", "root_cause"): "Diagnosis received.",
    }

    return Scenario(
        task_id="new_task_id",
        name="New Task",
        difficulty="medium",
        description="Describe incident.",
        max_steps=12,
        initial_alert="WARNING: ...",
        service_topology=service_topology,
        root_cause="your_root_cause",
        correct_fix_command="rollback_deploy",
        correct_fix_target="service-a",
        key_clues=["clue_one", "clue_two"],
        response_map=response_map,
        affected_services=["service-a"],
        escalation_targets={"platform-oncall"},
        should_escalate=False,
    )
```

## Designing good response maps

- Include expected path responses for all likely diagnostic steps.
- Include at least a few realistic wrong-path responses.
- Keep logs and metrics plausible and consistent over the story.
- Add red herrings only when they serve the intended difficulty.

## Setting key clues

A key clue should represent a meaningful diagnostic milestone, not just any text.
Examples:
- cache_redis_high_memory
- dependency_chain_traced
- replica_3_high_memory

## Testing your new scenario

- Ensure get_scenario(task_id) resolves correctly.
- Ensure key clues appear in response map text.
- Run full tests:

```bash
pytest
```
