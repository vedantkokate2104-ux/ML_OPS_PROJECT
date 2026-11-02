"""
app_metrics.py
EventOracle – Prometheus instrumentation for the Streamlit dashboard

Streamlit owns its HTTP server, so metrics are exposed on a separate
port (METRICS_PORT, default 9100) via prometheus_client. Enabled only
when ENABLE_METRICS=true (Docker / Kubernetes); off on Streamlit Cloud.
"""

import logging
import os
import time
from contextlib import contextmanager

log = logging.getLogger("eventoracle.metrics")

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _AVAILABLE = True
except ImportError:  # metrics are optional
    _AVAILABLE = False

ENABLED = _AVAILABLE and os.getenv("ENABLE_METRICS", "false").lower() == "true"

if _AVAILABLE:
    PAGE_VIEWS   = Counter("eventoracle_page_views_total", "Dashboard page renders", ["page"])
    DATA_LATENCY = Histogram("eventoracle_data_load_seconds", "MongoDB / model call latency", ["operation"])
    ERRORS       = Counter("eventoracle_errors_total", "Application errors", ["operation"])
    START_TIME   = Gauge("eventoracle_start_time_seconds", "Unix time the app process started")
    MONGO_UP     = Gauge("eventoracle_mongo_up", "1 if the last MongoDB ping succeeded")

_started = False


def start_server() -> bool:
    """Start the /metrics endpoint once per process."""
    global _started
    if not ENABLED or _started:
        return _started
    port = int(os.getenv("METRICS_PORT", "9100"))
    try:
        start_http_server(port)
        START_TIME.set(time.time())
        _started = True
        log.info(f"Prometheus metrics on :{port}/metrics")
    except OSError as e:  # already bound by another Streamlit session/thread
        log.warning(f"Metrics server not started: {e}")
    return _started


def page_view(page: str) -> None:
    if _AVAILABLE:
        PAGE_VIEWS.labels(page=page).inc()


def mongo_status(up: bool) -> None:
    if _AVAILABLE:
        MONGO_UP.set(1 if up else 0)


@contextmanager
def track(operation: str):
    """Time a block and count it as an error if it raises."""
    if not _AVAILABLE:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    except Exception:
        ERRORS.labels(operation=operation).inc()
        raise
    finally:
        DATA_LATENCY.labels(operation=operation).observe(time.perf_counter() - start)
