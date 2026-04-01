"""Python client for interacting with the IncidentEnv HTTP server.

This client supports both synchronous and asynchronous usage and works even when
openenv-core client imports are unavailable.

Examples:
    from client import IncidentEnv

    env = IncidentEnv("https://atul-k-6o-incident-response-env.hf.space")
    obs = env.reset("easy_crashed_service")
    obs = env.step({"command": "query_logs", "target": "payment-service"})
    print(env.state())

    # Async usage
    # obs = await env.areset("easy_crashed_service")
"""

from __future__ import annotations

from typing import Any

import httpx

from models import IncidentAction, IncidentObservation, IncidentState

try:
    from openenv.core.client import EnvClient as OpenEnvClient  # type: ignore
except Exception:  # noqa: BLE001
    OpenEnvClient = object  # type: ignore[misc,assignment]


class IncidentEnv(OpenEnvClient):
    """HTTP client helper for the incident response environment."""

    def __init__(self, base_url: str = "https://atul-k-6o-incident-response-env.hf.space", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _sync_client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _action_payload(
        self, action: IncidentAction | dict[str, Any]
    ) -> dict[str, Any]:
        if isinstance(action, IncidentAction):
            return action.model_dump(exclude_none=True)
        return IncidentAction.model_validate(action).model_dump(exclude_none=True)

    def reset(self, task_id: str) -> IncidentObservation:
        """Reset environment to a task and return initial observation."""

        with self._sync_client() as client:
            response = client.post("/reset", json={"task_id": task_id})
            response.raise_for_status()
            return IncidentObservation.model_validate(response.json())

    def step(self, action: IncidentAction | dict[str, Any]) -> IncidentObservation:
        """Send one action and return resulting observation."""

        payload = self._action_payload(action)
        with self._sync_client() as client:
            response = client.post("/step", json=payload)
            response.raise_for_status()
            return IncidentObservation.model_validate(response.json())

    def state(self) -> IncidentState:
        """Fetch internal state from /state endpoint."""

        with self._sync_client() as client:
            response = client.get("/state")
            response.raise_for_status()
            return IncidentState.model_validate(response.json())

    def grader(self) -> dict[str, Any]:
        """Fetch final grader output from /grader endpoint."""

        with self._sync_client() as client:
            response = client.get("/grader")
            response.raise_for_status()
            return response.json()

    def baseline(self, model: str | None = None) -> dict[str, Any]:
        """Trigger /baseline endpoint and return results."""

        payload: dict[str, Any] = {}
        if model:
            payload["model"] = model
        with self._sync_client() as client:
            response = client.post("/baseline", json=payload)
            response.raise_for_status()
            return response.json()

    async def areset(self, task_id: str) -> IncidentObservation:
        """Async reset equivalent."""

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        ) as client:
            response = await client.post("/reset", json={"task_id": task_id})
            response.raise_for_status()
            return IncidentObservation.model_validate(response.json())

    async def astep(
        self, action: IncidentAction | dict[str, Any]
    ) -> IncidentObservation:
        """Async step equivalent."""

        payload = self._action_payload(action)
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        ) as client:
            response = await client.post("/step", json=payload)
            response.raise_for_status()
            return IncidentObservation.model_validate(response.json())

    async def astate(self) -> IncidentState:
        """Async state fetch."""

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        ) as client:
            response = await client.get("/state")
            response.raise_for_status()
            return IncidentState.model_validate(response.json())
