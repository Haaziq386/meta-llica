"""Hard scenario: intermittent failures caused by one leaking replica.

This task is intentionally deceptive. Aggregate service-level metrics look only
slightly degraded, but one replica is repeatedly OOM-killed.
"""

from __future__ import annotations

from scenarios.base import Scenario


def build_hard_intermittent_ghost() -> Scenario:
    """Construct the hard deterministic scenario."""

    service_topology = {
        "api-gateway": ["user-service", "order-service", "auth-service"],
        "user-service": ["database", "cache-redis", "auth-service"],
        "order-service": ["database", "user-service", "payment-service"],
        "auth-service": ["database", "cache-redis"],
        "payment-service": ["database", "cache-redis"],
        "database": [],
        "cache-redis": [],
    }

    response_map: dict[tuple[str, str], str] = {
        (
            "check_metrics",
            "api-gateway",
        ): (
            "api-gateway aggregate metrics (all replicas): cpu=52% memory=68% "
            "p99_latency=420ms error_rate=18%\n"
            "note: averages may hide per-replica outliers"
        ),
        (
            "query_logs",
            "api-gateway",
        ): (
            "2026-03-27T00:41:11Z INFO api-gateway replica-1 status=200 route=/checkout\n"
            "2026-03-27T00:41:12Z WARN api-gateway replica-3 status=503 upstream reset\n"
            "2026-03-27T00:41:13Z INFO api-gateway replica-2 status=200 route=/profile\n"
            "2026-03-27T00:41:15Z WARN api-gateway replica-3 restarting after readiness fail"
        ),
        (
            "check_service_status",
            "api-gateway",
        ): (
            "api-gateway status: DEGRADED\n"
            "replicas healthy: 3/4\n"
            "replica-3 state: restarting (CrashLoopBackOff events observed)"
        ),
        (
            "check_recent_deploys",
            "api-gateway",
        ): (
            "Recent deploys for api-gateway:\n"
            "- 2026-03-26T18:02:09Z deploy_id=gateway-941 by=dev-bot status=completed\n"
            "notes: optimized request middleware"
        ),
        (
            "trace_dependency",
            "api-gateway",
        ): "api-gateway depends on: user-service, order-service, auth-service",
        (
            "check_metrics",
            "user-service",
        ): (
            "user-service metrics: cpu=44% memory=53% p99_latency=122ms error_rate=0.5%\n"
            "healthy"
        ),
        (
            "check_metrics",
            "order-service",
        ): (
            "order-service metrics: cpu=46% memory=59% p99_latency=138ms error_rate=0.6%\n"
            "healthy"
        ),
        (
            "check_metrics",
            "auth-service",
        ): (
            "auth-service metrics: cpu=38% memory=47% p99_latency=95ms error_rate=0.2%\n"
            "healthy"
        ),
        (
            "check_metrics",
            "database",
        ): (
            "database metrics: cpu=58% memory=70% p99_query_latency=55ms error_rate=0.2%\n"
            "minor spike 2h ago, now stable"
        ),
        (
            "query_logs",
            "database",
        ): (
            "2026-03-26T22:09:30Z WARN database slow query 882ms (transient)\n"
            "2026-03-26T22:11:01Z INFO database latency normalized"
        ),
        (
            "check_recent_deploys",
            "user-service",
        ): (
            "Recent deploys for user-service:\n"
            "- 2026-03-27T00:05:44Z deploy_id=user-3388 status=completed\n"
            "notes: minor config change (request timeout +5ms)"
        ),
        (
            "check_recent_deploys",
            "database",
        ): (
            "Recent deploys for database:\n"
            "- 2026-03-18T13:04:28Z schema-migration-772 status=completed"
        ),
        (
            "query_logs",
            "user-service",
        ): (
            "2026-03-27T00:42:11Z INFO user-service all upstream checks passing\n"
            "2026-03-27T00:42:12Z INFO user-service p99 latency within SLO"
        ),
        (
            "check_service_status",
            "user-service",
        ): "user-service status: HEALTHY, replicas ready 4/4",
        (
            "check_service_status",
            "order-service",
        ): "order-service status: HEALTHY, replicas ready 3/3",
        (
            "check_service_status",
            "auth-service",
        ): "auth-service status: HEALTHY, replicas ready 2/2",
        (
            "check_metrics",
            "cache-redis",
        ): "cache-redis metrics: cpu=33% memory=49% hit_rate=94% evictions=0",
        (
            "query_logs",
            "cache-redis",
        ): "cache-redis logs: healthy heartbeat, no OOM, no evictions",
        (
            "check_recent_deploys",
            "payment-service",
        ): (
            "Recent deploys for payment-service:\n"
            "- 2026-03-23T12:21:10Z deploy_id=pay-4403 status=completed"
        ),
        (
            "restart_service",
            "api-gateway",
        ): (
            "api-gateway restarted. Error rate drops to 4% briefly, then climbs back to "
            "~16% as replica-3 memory grows again."
        ),
        (
            "rollback_deploy",
            "api-gateway",
        ): (
            "Rolled back api-gateway to deploy_id=gateway-938.\n"
            "replica-3 memory stabilized, replicas healthy 4/4, error_rate now 0.7%."
        ),
        (
            "scale_service",
            "api-gateway",
        ): (
            "Scaled api-gateway replicas from 4 to 6. Aggregate error rate improves "
            "slightly, but intermittent 503s continue from one unstable replica."
        ),
        (
            "escalate",
            "platform-oncall",
        ): (
            "Escalation sent to platform-oncall. They recommend checking per-replica "
            "metrics because aggregate looks misleading."
        ),
        (
            "submit_diagnosis",
            "root_cause",
        ): "Diagnosis received. Grader will evaluate submitted reason.",
        (
            "trace_dependency",
            "user-service",
        ): "user-service depends on: database, cache-redis, auth-service",
        (
            "trace_dependency",
            "order-service",
        ): "order-service depends on: database, user-service, payment-service",
        (
            "trace_dependency",
            "auth-service",
        ): "auth-service depends on: database, cache-redis",
    }

    parameter_response_map: dict[tuple[str, str, str], str] = {
        (
            "check_metrics",
            "api-gateway",
            "replica=replica-1",
        ): (
            "api-gateway replica-1 metrics: cpu=49% memory=56% p99_latency=180ms "
            "error_rate=0.4%"
        ),
        (
            "check_metrics",
            "api-gateway",
            "replica=replica-2",
        ): (
            "api-gateway replica-2 metrics: cpu=51% memory=60% p99_latency=205ms "
            "error_rate=0.6%"
        ),
        (
            "check_metrics",
            "api-gateway",
            "replica=replica-3",
        ): (
            "api-gateway replica-3 metrics: cpu=78% memory=95% (climbing) "
            "p99_latency=1880ms error_rate=67%\n"
            "warning: probable memory leak"
        ),
        (
            "check_metrics",
            "api-gateway",
            "replica=replica-4",
        ): (
            "api-gateway replica-4 metrics: cpu=46% memory=58% p99_latency=172ms "
            "error_rate=0.5%"
        ),
        (
            "query_logs",
            "api-gateway",
            "replica=replica-3",
        ): (
            "2026-03-27T00:44:01Z WARN api-gateway replica-3 memory usage 93%\n"
            "2026-03-27T00:44:12Z ERROR api-gateway replica-3 OOMKilled by kubelet\n"
            "2026-03-27T00:44:23Z INFO api-gateway replica-3 container restarted\n"
            "2026-03-27T00:46:10Z WARN api-gateway replica-3 memory usage 95%"
        ),
        (
            "query_logs",
            "api-gateway",
            "replica=replica-2",
        ): "replica-2 logs: normal request handling, no restarts.",
    }

    return Scenario(
        task_id="hard_intermittent_ghost",
        name="The Intermittent Ghost",
        difficulty="hard",
        description=(
            "Intermittent api-gateway 503s caused by a memory leak in one replica; "
            "aggregate metrics initially mask the issue."
        ),
        max_steps=20,
        initial_alert=(
            "WARNING: api-gateway intermittent 503s. Error rate 18%. Affecting "
            "approximately 1 in 5 requests."
        ),
        service_topology=service_topology,
        root_cause="memory_leak_single_replica",
        correct_fix_command="rollback_deploy",
        correct_fix_target="api-gateway",
        key_clues=[
            "intermittent_503_errors",
            "replica_3_identified",
            "replica_3_high_memory",
            "oom_killed_events",
            "recent_deploy_to_api_gateway",
            "other_replicas_healthy",
        ],
        response_map=response_map,
        parameter_response_map=parameter_response_map,
        escalation_targets={"platform-oncall", "sre-primary"},
        affected_services=["api-gateway"],
        should_escalate=False,
    )
