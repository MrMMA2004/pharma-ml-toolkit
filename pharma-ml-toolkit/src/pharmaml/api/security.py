"""Local-only API security for the PharmaML Studio backend service.

Part10 (§9.4 "Local API security") — the FastAPI companion is a local
service, never a network server:

* bind only to 127.0.0.1 (enforced by the server runner, asserted by
  tests via the app state);
* a cryptographically secure session token is generated at startup
  with :func:`secrets.token_urlsafe`, never persisted, never logged;
* every request must carry it (``Authorization: Bearer`` or the
  ``X-Session-Token`` header) or be the health probe with a valid
  Host/Origin pair;
* ``Host`` must be loopback; ``Origin``/``Referer`` when present must
  be loopback too — anything else is rejected before routing.

The token travels to the parent process (the WinUI host) only through
the structured startup handshake (see :mod:`pharmaml.api.server`) — a
private machine-readable line, distinct from diagnostic logs.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, status

__all__ = [
    "LOOPBACK_HOSTS",
    "SessionSecurity",
    "reset_session_security",
    "session_security",
]


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _host_is_loopback(host_header: str) -> bool:
    """True when the Host header (``host[:port]``) is loopback."""
    if not host_header:
        return False
    host = host_header.split(":", 1)[0].strip()
    return host.lower() in LOOPBACK_HOSTS


def _origin_is_loopback(origin_header: str) -> bool:
    """True when an Origin/Referer URL's hostname is loopback.

    An absent header is fine (non-browser clients); a *present* header
    must be loopback.
    """
    if not origin_header:
        return True
    try:
        host = urlsplit(origin_header).hostname or ""
    except ValueError:
        return False
    return host.lower() in LOOPBACK_HOSTS


class SessionSecurity:
    """Per-session token + loopback enforcement.

    ``token`` is generated with ``secrets.token_urlsafe(32)`` and kept
    in memory only.  ``token_sha256`` (a hex digest, not the token) is
    what the startup handshake exposes for diagnostics.
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token or secrets.token_urlsafe(32)
        self._sha256 = hashlib.sha256(self._token.encode()).hexdigest()

    @property
    def token(self) -> str:
        return self._token

    @property
    def token_sha256(self) -> str:
        """Safe-to-print digest of the session token."""
        return self._sha256

    def verify(self, presented: str | None) -> bool:
        """Constant-time check of a presented session token."""
        if not presented:
            return False
        return hmac.compare_digest(presented, self._token)

    def guard_request(self, request: Request) -> None:
        """Reject non-loopback hosts/origins and unauthenticated calls.

        Reads everything from the live request (works identically
        as a FastAPI dependency or a direct call).  Raises
        HTTPException (403/401) — the FastAPI dependency runs before
        routing, so no endpoint sees the request.
        """
        host = request.headers.get("host", "")
        if not _host_is_loopback(host):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Non-loopback Host rejected.",
            )
        for header in ("origin", "referer"):
            if not _origin_is_loopback(request.headers.get(header, "")):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Non-loopback Origin rejected.",
                )
        authorization = request.headers.get("authorization", "")
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization.split(" ", 1)[1].strip()
        presented = request.headers.get("x-session-token") or bearer
        if not self.verify(presented):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing session token.",
            )


_session: SessionSecurity | None = None


def session_security() -> SessionSecurity:
    """Process-wide SessionSecurity (created lazily, once)."""
    global _session
    if _session is None:
        _session = SessionSecurity()
    return _session


def reset_session_security(token: str | None = None) -> SessionSecurity:
    """Replace the process-wide session (tests use fixed tokens)."""
    global _session
    _session = SessionSecurity(token=token)
    return _session
