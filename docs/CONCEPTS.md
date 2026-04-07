# Concepts Guide

## What is Reinforcement Learning?

Reinforcement learning (RL) is a way to train an agent by letting it act in an environment and receive feedback. The feedback is a reward signal. Over many episodes, the agent learns which action sequences produce better outcomes.

In IncidentEnv, RL is like training a junior on-call engineer. The agent sees alerts and chooses actions (check logs, inspect metrics, roll back deploys). Good decisions earn reward, wasted or harmful decisions lose reward.

## What is OpenEnv?

OpenEnv is a standard interface for building RL-compatible environments. It gives a consistent loop and endpoint contracts so agents can interact with many tasks in the same way.

This standardization is useful because training code can be reused across different domains, from games to incident response.

## What is an Episode?

An episode is one complete incident. It starts at reset() and ends when the incident is resolved, the diagnosis is submitted, or max steps are reached.

In this project, each episode is one deterministic scenario like easy_crashed_service.

## What is a Reward Signal?

A reward signal is immediate numerical feedback after each action. IncidentEnv uses reward shaping:
- Positive reward for discovering clues and applying correct fixes.
- Negative reward for irrelevant actions, wrong fixes, and time spent.

This helps the agent learn useful behavior faster than only giving reward at the very end.

## What is a Grader?

The grader is an episode-end evaluator. It computes one final score in (0, 1) based on diagnosis accuracy, fix correctness, efficiency, and collateral damage.

Difference from reward: reward is per-step training feedback; grader is final evaluation.

## What is an Action Space?

The action space is the set of commands the agent is allowed to execute.

In IncidentEnv this includes query_logs, check_metrics, rollback_deploy, submit_diagnosis, and others.

## What is an Observation Space?

The observation space is what the agent receives after each step. It includes done, reward, action_result, clues_found, and other context needed for the next decision.

## What is the reset()/step()/state() loop?

```text
reset(task_id)
  -> returns initial observation with alert

loop:
  action = agent_policy(observation)
  observation = step(action)
  if observation.done:
      break

state() -> inspect internal state for debugging or grading
```

This loop is the core interaction pattern in OpenEnv-style environments.
