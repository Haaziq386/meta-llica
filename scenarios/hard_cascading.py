"""Hard scenario: cascading latency originating from cache memory exhaustion.

This scenario teaches upstream root-cause reasoning. Multiple services look bad,
but only cache-redis is the primary failure source.
"""

from __future__ import annotations

from scenarios.base import Scenario


def build_hard_cascading_failure() -> Scenario:
    """Construct the hard deterministic scenario."""

    service_topology = {
        "api-gateway": ["user-service", "order-service", "auth-service"],
        "user-service": ["database", "cache-redis", "auth-service"],
        "order-service": ["database", "user-service", "payment-service"],
        "payment-service": ["database", "cache-redis"],
        "auth-service": ["database", "cache-redis"],
        "database": [],
        "cache-redis": [],
    }

    response_map: dict[tuple[str, str], str] = {
        (
            "check_metrics",
            "api-gateway",
        ): (
            "api-gateway metrics (5m): cpu=46% memory=58% p99_latency=910ms "
            "error_rate=4.8%\n"
            "note: host resources normal; downstream timings elevated"
        ),
        (
            "trace_dependency",
            "api-gateway",
        ): "api-gateway depends on: user-service, order-service, auth-service",
        (
            "check_metrics",
            "user-service",
        ): (
            "user-service metrics: cpu=71% memory=67% p99_latency=840ms "
            "error_rate=3.1%\n"
            "db_wait_time elevated"
        ),
        (
            "trace_dependency",
            "user-service",
        ): "user-service depends on: database, cache-redis, auth-service",
        (
            "check_metrics",
            "database",
        ): (
            "database metrics: cpu=95% memory=78% p99_query_latency=680ms "
            "connections=1890 (high)\n"
            "slow_query_count increased 5.3x"
        ),
        (
            "check_metrics",
            "cache-redis",
        ): (
            "cache-redis metrics: cpu=82% memory=98% hit_rate=41% "
            "evictions=12,803/min\n"
            "cache_miss_amplification detected"
        ),
        (
            "query_logs",
            "cache-redis",
        ): (
            "2026-03-27T00:58:10Z WARN cache-redis memory limit near capacity: 97.8%\n"
            "2026-03-27T00:58:21Z WARN cache-redis OOM pressure, evicting hot keys\n"
            "2026-03-27T00:58:45Z WARN cache-redis eviction storm: 2112 keys/s"
        ),
        (
            "scale_service",
            "cache-redis",
        ): (
            "Scaled cache-redis from 2 to 4 replicas with larger memory class.\n"
            "memory dropped to 61%, hit_rate recovered to 90%, db cpu trending down."
        ),
        (
            "restart_service",
            "cache-redis",
        ): (
            "cache-redis restarted. Brief improvement observed, but memory climbed "
            "back above 95% within minutes."
        ),
        (
            "check_recent_deploys",
            "user-service",
        ): (
            "Recent deploys for user-service:\n"
            "- 2026-03-27T00:44:00Z deploy_id=user-3312 by=dev-bot status=completed\n"
            "notes: updated logging formatter only (no business logic changes)"
        ),
        (
            "query_logs",
            "user-service",
        ): (
            "2026-03-27T01:02:08Z WARN user-service cache timeout -> fallback to DB\n"
            "2026-03-27T01:02:09Z WARN user-service query latency high (612ms)"
        ),
        (
            "check_metrics",
            "order-service",
        ): (
            "order-service metrics: cpu=66% memory=64% p99_latency=1012ms "
            "timeout_rate=3.9%\n"
            "dependency wait concentrated in user-service"
        ),
        (
            "trace_dependency",
            "order-service",
        ): "order-service depends on: database, user-service, payment-service",
        (
            "query_logs",
            "order-service",
        ): (
            "2026-03-27T01:04:22Z ERROR order-service timed out waiting for user profile\n"
            "2026-03-27T01:04:23Z WARN order-service degraded path invoked"
        ),
        (
            "check_service_status",
            "auth-service",
        ): (
            "auth-service status: DEGRADED\n"
            "healthcheck: passing with warning\n"
            "dependency latency elevated"
        ),
        (
            "check_metrics",
            "auth-service",
        ): (
            "auth-service metrics: cpu=57% memory=55% p99_latency=490ms\n"
            "cache lookup time increased"
        ),
        (
            "query_logs",
            "database",
        ): (
            "2026-03-27T01:03:11Z WARN database connection pool saturation at 94%\n"
            "2026-03-27T01:03:12Z INFO database top query source=user-service"
        ),
        (
            "trace_dependency",
            "database",
        ): "database dependencies: none (leaf service)",
        (
            "check_recent_deploys",
            "cache-redis",
        ): (
            "Recent deploys for cache-redis:\n"
            "- 2026-03-10T14:22:52Z redis-config-890 status=completed\n"
            "No deployments in last 7 days"
        ),
        (
            "check_service_status",
            "cache-redis",
        ): (
            "cache-redis status: DEGRADED\n"
            "replicas healthy=2/2\n"
            "warning: memory pressure and heavy eviction"
        ),
        (
            "rollback_deploy",
            "user-service",
        ): (
            "Rolled back user-service deploy. No measurable improvement in global latency."
        ),
        (
            "scale_service",
            "database",
        ): (
            "Scaled database read replicas. Query queue reduced slightly but p99 remains high "
            "because cache misses persist."
        ),
        (
            "escalate",
            "database-team",
        ): (
            "Escalation sent to database-team. Response: DB load appears secondary to "
            "upstream cache miss storm."
        ),
        (
            "submit_diagnosis",
            "root_cause",
        ): "Diagnosis received. Grader will evaluate the submitted reason.",
    }

    return Scenario(
        task_id="hard_cascading_failure",
        name="The Cascading Failure",
        difficulty="hard",
        description=(
            "Elevated p99 latency across multiple services caused by an upstream "
            "cache-redis memory exhaustion event."
        ),
        max_steps=15,
        initial_alert=(
            "WARNING: Elevated p99 latency across api-gateway, user-service, "
            "order-service."
        ),
        service_topology=service_topology,
        root_cause="cache_memory_exhaustion",
        correct_fix_command="scale_service",
        correct_fix_target="cache-redis",
        key_clues=[
            "api_gateway_high_latency",
            "database_overloaded",
            "cache_redis_high_memory",
            "cache_eviction_warnings",
            "dependency_chain_traced",
        ],
        response_map=response_map,
        escalation_targets={"database-team", "platform-oncall"},
        affected_services=[
            "api-gateway",
            "user-service",
            "order-service",
            "database",
            "cache-redis",
        ],
        should_escalate=False,
    )
