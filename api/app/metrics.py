"""Application metrics beyond the HTTP defaults.

`prometheus_fastapi_instrumentator` already exposes request rate, latency, and
status codes on /metrics. These are the ones specific to the training pipeline:
they answer "is the signal capture actually capturing?", which the HTTP metrics
cannot, because a request that traces nothing is still a perfectly healthy 200.

Registered on the default collector registry, which is the one the
instrumentator serves, so defining them is all that is needed to have them
scraped by the existing `api` job in `metrics/prometheus.yml`.

Counters are deliberately split by outcome rather than lumped into one
`trace_events_total{result=...}`: on the dashboard, "written" is a rate you
want to see climb and the other two are rates you want to see flat at zero, and
they are read at a glance rather than by reading label values.
"""

from prometheus_client import Counter, Gauge

TRACES_WRITTEN = Counter(
    "traces_written_total",
    "Traces durably written to the trace store.",
)

# Failed *after* reaching the store. Split from drops because the fixes differ:
# a failure means the store is unhealthy, a drop means it is unreachable or
# too slow to keep up.
TRACE_FAILURES = Counter(
    "trace_write_failures_total",
    "Traces that reached the writer but could not be stored.",
    ["reason"],
)

# Never reached the store at all.
TRACE_DROPS = Counter(
    "trace_drops_total",
    "Traces discarded before a write was attempted.",
    ["reason"],
)

# The early warning for both of the above: a queue that is not near zero means
# the writer is falling behind and drops are coming.
TRACE_QUEUE_DEPTH = Gauge(
    "trace_queue_depth",
    "Traces waiting to be written.",
)
