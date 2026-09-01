"""API key check.

A single shared secret between one trusted UI client and this service — not
multi-user auth. Applied per-route rather than globally on purpose: `/health`
and `/metrics` must stay open, because the Compose healthcheck and Prometheus
have no reason to hold the application's key, and 401ing them would break
startup ordering and leave the dashboard silently empty.
"""

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import settings


async def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    # compare_digest, not ==, so a wrong key cannot be recovered by timing.
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )


RequireApiKey = Depends(require_api_key)
