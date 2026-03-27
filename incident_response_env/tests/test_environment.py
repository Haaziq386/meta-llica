"""Environment integration tests for reset/step behavior."""

from __future__ import annotations

from models import IncidentAction
from server.environment import IncidentEnvironment
from server.grader import grade_episode


def test_reset_works_for_each_task() -> None:
    env = IncidentEnvironment()
    for task_id in [
        "easy_crashed_service",
        "medium_cascading_failure",
        "hard_intermittent_ghost",
    ]:
        obs = env.reset(task_id=task_id)
        assert obs.done is False
        assert obs.step_number == 0
        assert env.state.task_id == task_id


def test_step_with_valid_and_invalid_actions() -> None:
    env = IncidentEnvironment()
    env.reset(task_id="easy_crashed_service")

    valid_obs = env.step(
        IncidentAction(command="check_service_status", target="payment-service")
    )
    assert valid_obs.step_number == 1

    invalid_obs = env.step(
        IncidentAction(command="not_a_command", target="payment-service")
    )
    assert invalid_obs.reward < 0
    assert "Invalid command" in invalid_obs.action_result


def test_episode_ends_after_fix_and_diagnosis() -> None:
    env = IncidentEnvironment()
    env.reset(task_id="easy_crashed_service")

    env.step(IncidentAction(command="check_service_status", target="payment-service"))
    env.step(IncidentAction(command="query_logs", target="payment-service"))
    env.step(IncidentAction(command="check_recent_deploys", target="payment-service"))
    obs = env.step(IncidentAction(command="rollback_deploy", target="payment-service"))
    assert obs.done is False

    final_obs = env.step(
        IncidentAction(
            command="submit_diagnosis",
            target="root_cause",
            parameters={"reason": "bad_deployment"},
        )
    )
    assert final_obs.done is True


def test_reward_computation_is_exposed_in_observation() -> None:
    env = IncidentEnvironment()
    env.reset(task_id="medium_cascading_failure")

    obs = env.step(IncidentAction(command="check_metrics", target="api-gateway"))
    assert isinstance(obs.reward, float)


def test_grader_scoring_after_completion() -> None:
    env = IncidentEnvironment()
    env.reset(task_id="easy_crashed_service")
    env.step(IncidentAction(command="check_service_status", target="payment-service"))
    env.step(IncidentAction(command="query_logs", target="payment-service"))
    env.step(IncidentAction(command="check_recent_deploys", target="payment-service"))
    env.step(IncidentAction(command="rollback_deploy", target="payment-service"))
    env.step(
        IncidentAction(
            command="submit_diagnosis",
            target="root_cause",
            parameters={"reason": "bad_deployment"},
        )
    )

    scenario = env.current_scenario
    assert scenario is not None
    score, _ = grade_episode(env.state, scenario)
    assert 0.0 <= score <= 1.0


def test_actions_after_done_return_terminal_message() -> None:
    env = IncidentEnvironment()
    env.reset(task_id="easy_crashed_service")
    env.step(IncidentAction(command="rollback_deploy", target="payment-service"))
    env.step(
        IncidentAction(
            command="submit_diagnosis",
            target="root_cause",
            parameters={"reason": "bad_deployment"},
        )
    )
    obs = env.step(IncidentAction(command="query_logs", target="payment-service"))
    assert "already completed" in obs.action_result
