"""Trace metrics are exported and the dashboard asks for names that exist.

The failure this guards against is specific: a trace counter that is defined in
Python but never appears on /metrics, or a dashboard panel querying a metric
whose name was changed. Both fail silently — the panel just draws "No data",
which looks identical to "nothing has gone wrong yet". Since the entire point
of these metrics is to make silent data loss visible, a silently broken panel
defeats them completely.
"""

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest

DASHBOARD = Path(__file__).resolve().parents[2] / "metrics/grafana/dashboards/vllm.json"

# Metric names this service is responsible for. The vllm: ones come from the
# model server and are checked on the box, not here.
TRACE_METRICS = [
    "traces_written_total",
    "trace_write_failures_total",
    "trace_drops_total",
    "trace_queue_depth",
]


def test_trace_metrics_are_exposed(client: TestClient) -> None:
    """/metrics is what Prometheus scrapes; being in the registry is not enough."""
    body = client.get("/metrics").text

    for name in TRACE_METRICS:
        assert name in body, f"{name} is defined but not exported"


def test_every_failure_reason_is_exported_before_it_ever_happens() -> None:
    """A healthy process must still export the loss counters, at zero.

    A labelled metric has no series until a label combination is used, so
    without pre-declaring them a process that has never failed exports nothing
    for these families and their panels read "No data" — indistinguishable from
    "nothing has gone wrong yet", which is the confusion they exist to remove.

    Found by running the documented dashboard command against a real container
    and seeing two of the four metrics missing.
    """
    exported = generate_latest(REGISTRY).decode()

    for reason in ("timeout", "database", "unexpected"):
        assert f'trace_write_failures_total{{reason="{reason}"}}' in exported
    for reason in ("disabled", "queue_full"):
        assert f'trace_drops_total{{reason="{reason}"}}' in exported


def test_metrics_endpoint_needs_no_api_key(client: TestClient) -> None:
    """Prometheus holds no credentials; a 401 here is a dashboard that never fills."""
    assert client.get("/metrics").status_code == 200


def test_dashboard_only_queries_metrics_this_service_exports() -> None:
    """Catches a renamed counter before the panel quietly goes blank."""
    dashboard = json.loads(DASHBOARD.read_text())
    exprs = [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]

    queried = {
        name
        for expr in exprs
        for name in re.findall(r"\b(trace[a-z_]*|traces[a-z_]*)\b", expr)
        if not name.startswith("trace_id")
    }

    assert queried, "no trace panels found in the dashboard"
    unknown = queried - set(TRACE_METRICS)
    assert not unknown, f"dashboard queries metrics that do not exist: {sorted(unknown)}"


def test_every_trace_metric_appears_on_the_dashboard() -> None:
    """A metric nobody plots is a metric nobody notices."""
    dashboard = json.loads(DASHBOARD.read_text())
    exprs = " ".join(t["expr"] for p in dashboard["panels"] for t in p.get("targets", []))

    for name in TRACE_METRICS:
        assert name in exprs, f"{name} is exported but on no panel"
