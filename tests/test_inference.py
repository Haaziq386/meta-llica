"""Tests for baseline task discovery and family mapping."""

from __future__ import annotations

import httpx

from inference import _discover_tasks, _task_family_from_metadata


def test_task_family_mapping_handles_current_and_legacy_ids() -> None:
    assert _task_family_from_metadata("easy_crashed_service") == "easy"
    assert _task_family_from_metadata("medium_cascading") == "medium"
    assert _task_family_from_metadata("hard_intermittent_ghost") == "hard"


def test_discover_tasks_orders_by_family() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tasks"
        return httpx.Response(
            200,
            json={
                "tasks": [
                    {"id": "hard_intermittent", "difficulty": "hard"},
                    {"id": "easy_crashed_service", "difficulty": "easy"},
                    {"id": "medium_cascading", "difficulty": "medium"},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url="https://example.test", transport=transport) as client:
        tasks = _discover_tasks(client)

    assert [task["id"] for task in tasks] == [
        "easy_crashed_service",
        "medium_cascading",
        "hard_intermittent",
    ]
