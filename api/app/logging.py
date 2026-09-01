"""Structured JSON logging.

One line per request, written to stdout for `docker compose logs` to collect.
No file handling or rotation: the container runtime owns that.

Each line carries a `request_id` generated per request. The upstream server
does not echo inbound headers, so cross-service correlation is done by logging
this id alongside the upstream response id rather than by header passthrough.
"""

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        # Anything attached via `extra=` on the call site.
        payload.update(getattr(record, "context", {}))
        return json.dumps(payload)


def configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("api")
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]
    logger.propagate = False
    return logger


logger = configure_logging()
