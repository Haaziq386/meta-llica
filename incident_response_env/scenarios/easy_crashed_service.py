"""Easy scenario: a single crashed service caused by a bad deploy.

This task is intentionally straightforward so baseline agents can learn the
interaction loop: check status, inspect logs, inspect recent deploys, fix, and
submit diagnosis.
"""

from __future__ import annotations

from scenarios.base import Scenario


def build_easy_crashed_service() -> Scenario:
    """Construct the easy deterministic scenario."""

    service_topology = {
        "api-gateway": ["user-service", "payment-service"],
        "payment-service": ["database", "cache-redis"],
        "user-service": ["database", "cache-redis"],
        "database": [],
        "cache-redis": [],
    }

    response_map: dict[tuple[str, str], str] = {
        (
            "check_service_status",
            "payment-service",
        ): (
            "payment-service status: DOWN\n"
            "healthcheck: failing\n"
            "pods ready: 0/3\n"
            "error rate: 100%\n"
            "impact: checkout requests returning HTTP 503"
        ),
        (
            "query_logs",
            "payment-service",
        ): (
            "2026-03-27T01:18:34Z ERROR payment-service RuntimeError: missing config "
            "key 'STRIPE_API_VERSION'\n"
            "2026-03-27T01:18:34Z ERROR payment-service startup failed; exiting\n"
            "2026-03-27T01:18:35Z WARN  payment-service container restart backoff\n"
            "2026-03-27T01:18:36Z ERROR payment-service process terminated code=1"
        ),
        (
            "check_recent_deploys",
            "payment-service",
        ): (
            "Recent deploys for payment-service:\n"
            "- 2026-03-27T00:58:12Z deploy_id=pay-4421 by=dev-bot status=completed\n"
            "- 2026-03-24T18:09:42Z deploy_id=pay-4394 by=alice status=completed\n"
            "Diff note: removed fallback for STRIPE_API_VERSION"
        ),
        (
            "rollback_deploy",
            "payment-service",
        ): (
            "Rollback initiated to deploy_id=pay-4394\n"
            "New pod state: ready 3/3\n"
            "error rate dropped from 100% to 0.8%\n"
            "checkout success rate recovering"
        ),
        (
            "restart_service",
            "payment-service",
        ): (
            "Restart attempted but startup still fails with missing STRIPE_API_VERSION.\n"
            "pods ready: 0/3"
        ),
        (
            "scale_service",
            "payment-service",
        ): (
            "Scaled replicas from 3 to 6, but all replicas crash on startup.\n"
            "pods ready: 0/6"
        ),
        (
            "check_metrics",
            "payment-service",
        ): (
            "payment-service metrics (last 5m):\n"
            "cpu=12% memory=44% p95_latency=12ms p99_latency=23ms\n"
            "request_volume=very_low error_rate=100%\n"
            "note: process crashes before serving traffic"
        ),
        (
            "trace_dependency",
            "payment-service",
        ): "payment-service depends on: database, cache-redis",
        (
            "check_service_status",
            "api-gateway",
        ): (
            "api-gateway status: DEGRADED\n"
            "reason: downstream payment-service unavailable\n"
            "5xx spike on /checkout"
        ),
        (
            "query_logs",
            "api-gateway",
        ): (
            "2026-03-27T01:19:06Z WARN api-gateway upstream payment-service returned 503\n"
            "2026-03-27T01:19:07Z WARN api-gateway route=/checkout status=503"
        ),
        (
            "check_metrics",
            "database",
        ): (
            "database metrics: cpu=41% memory=63% p99_query_latency=29ms\n"
            "error_rate=0.2% connections=normal"
        ),
        (
            "check_metrics",
            "cache-redis",
        ): (
            "cache-redis metrics: cpu=28% memory=57% hit_rate=93% evictions=0\n"
            "health=normal"
        ),
        (
            "check_recent_deploys",
            "api-gateway",
        ): (
            "Recent deploys for api-gateway:\n"
            "- 2026-03-20T12:03:10Z deploy_id=gateway-883 status=completed\n"
            "No deploys in last 24h"
        ),
        (
            "trace_dependency",
            "api-gateway",
        ): "api-gateway depends on: user-service, payment-service",
        (
            "escalate",
            "database-team",
        ): "Escalation sent to database-team. They report no DB anomalies.",
        (
            "submit_diagnosis",
            "root_cause",
        ): (
            "Diagnosis received. Grader will evaluate submitted reason against root cause."
        ),
    }

    return Scenario(
        task_id="easy_crashed_service",
        name="The Crashed Service",
        difficulty="easy",
        description="A payment service is down due to a bad deployment.",
        max_steps=10,
        initial_alert=(
            "CRITICAL: payment-service returning 503 errors. 100% error rate. "
            "Users cannot complete checkout."
        ),
        service_topology=service_topology,
        root_cause="bad_deployment",
        correct_fix_command="rollback_deploy",
        correct_fix_target="payment-service",
        key_clues=[
            "payment_service_down",
            "crash_error_in_logs",
            "recent_deploy_found",
            "config_key_missing",
        ],
        response_map=response_map,
        escalation_targets={"database-team", "payments-oncall"},
        affected_services=["payment-service", "api-gateway"],
        should_escalate=False,
    )
