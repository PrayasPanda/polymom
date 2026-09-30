"""Outbound completion webhooks with HMAC-SHA256 signing and SSRF protection.

Deliveries retry on transient errors (network, 429, 5xx) with exponential backoff.
The URL is validated before every attempt: only http/https, and only public hosts
unless ``WEBHOOK_ALLOW_PRIVATE_HOSTS`` is on (for local development).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import socket
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.core.metrics import WEBHOOKS

logger = get_logger(__name__)

MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 60.0
DELIVERY_TTL_SECONDS = 7 * 24 * 3600
PRIVATE_HOST_NAMES = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
    }
)


def validate_callback_url(url: str, *, allow_private: bool) -> str:
    """Reject anything but http(s) and (unless ``allow_private``) private/loopback hosts."""
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValidationError("callback_url is not a valid URL.", details={"url": url}) from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValidationError(
            "callback_url must be http:// or https:// with a host.",
            details={"url": url, "scheme": parsed.scheme},
        )
    if allow_private:
        return url
    host = parsed.hostname.lower()
    if host in PRIVATE_HOST_NAMES:
        raise ValidationError(
            "callback_url points to a local host.", details={"url": url, "host": host}
        )
    for family_addr in _resolve(host):
        try:
            ip = ipaddress.ip_address(family_addr)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValidationError(
                "callback_url resolves to a private IP.",
                details={"url": url, "host": host, "ip": str(ip)},
            )
    return url


def _resolve(host: str) -> list[str]:
    try:
        return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]
    except OSError:
        raise ValidationError(
            "callback_url host could not be resolved.", details={"host": host}
        ) from None


def sign(secret: str, body: bytes, timestamp: int) -> str:
    """``sha256=<hex>``, over ``{timestamp}.{body}`` (Stripe-style)."""
    payload = f"{timestamp}.".encode() + body
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def signature_ok(secret: str, body: bytes, timestamp: int, header: str) -> bool:
    return hmac.compare_digest(sign(secret, body, timestamp), header)


def _backoff(attempt: int) -> float:
    return float(min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS))


async def deliver_webhook(
    settings: Settings,
    redis: Any,
    meeting_id: uuid.UUID,
    status: str,
    callback_url: str,
    *,
    request_id: str | None = None,
    now_fn: Any = time.time,
) -> bool:
    """POST the final status; returns True on 2xx, False after exhausting retries."""
    if settings.webhook_secret is None:
        logger.warning("webhook_skipped_no_secret", meeting_id=str(meeting_id))
        WEBHOOKS.labels("skipped").inc()
        return False
    try:
        url = validate_callback_url(
            callback_url, allow_private=settings.webhook_allow_private_hosts
        )
    except ValidationError as exc:
        logger.warning("webhook_ssrf_blocked", url=callback_url, reason=exc.message)
        WEBHOOKS.labels("ssrf_blocked").inc()
        return False
    body = json.dumps({"meeting_id": str(meeting_id), "status": status}).encode()
    secret = settings.webhook_secret.get_secret_value()
    max_attempts = min(settings.webhook_max_attempts, MAX_ATTEMPTS)
    for attempt in range(1, max_attempts + 1):
        timestamp = int(now_fn())
        headers = {
            "Content-Type": "application/json",
            "X-Polymom-Event": "meeting.completed",
            "X-Polymom-Timestamp": str(timestamp),
            "X-Polymom-Signature": sign(secret, body, timestamp),
            "X-Polymom-Attempt": str(attempt),
            "User-Agent": "polymom-webhook/1",
        }
        if request_id:
            headers["X-Request-ID"] = request_id
        try:
            async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
                response = await client.post(url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("webhook_attempt_failed", url=url, attempt=attempt, error=str(exc))
        else:
            if 200 <= response.status_code < 300:
                logger.info(
                    "webhook_delivered", url=url, attempt=attempt, status=response.status_code
                )
                WEBHOOKS.labels("delivered").inc()
                return True
            logger.warning(
                "webhook_attempt_failed",
                url=url,
                attempt=attempt,
                status=response.status_code,
            )
            if response.status_code < 500 and response.status_code != 429:
                WEBHOOKS.labels("client_error").inc()
                return False
        if attempt < max_attempts:
            await asyncio.sleep(_backoff(attempt))
    WEBHOOKS.labels("exhausted").inc()
    return False
