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
* ``Host`` must be loopback; ``Origin``/``Referer`` when present
  must be loopback **or an exact trusted application origin**
  (part14 §3.2's narrowly scoped trusted-origin policy — see
  :data:`TRUSTED_APP_ORIGINS`) — anything else is rejected before
  routing.

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
    "TRUSTED_APP_ORIGINS",
    "SessionSecurity",
    "origin_is_trusted",
    "reset_session_security",
    "session_security",
]


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

#: The packaged Tauri 2 desktop application's WebView2 origins — the
#: narrowly scoped trusted-origin policy of part14 §3.2 (D-1).
#:
#: Tauri 2 on Windows serves the packaged frontend from
#: ``tauri.localhost`` (``tauri.conf.json`` declares no origin
#: override, so the production and development origins differ — dev
#: is the loopback Vite origin, production is this virtual host).
#: Both entries are **scheme-strict and host-exact**: subdomains
#: (``evil.tauri.localhost``), lookalike hosts
#: (``tauri.localhost.evil.example``), port-bearing variants, and any
#: other scheme stay untrusted.  The ``Host`` header policy is
#: deliberately UNCHANGED — it remains loopback-only, because the
#: packaged app's requests target ``http://127.0.0.1:<port>`` (the
#: handshake's URL), never the virtual host itself.
TRUSTED_APP_ORIGINS = frozenset(
    {
        "http://tauri.localhost",
        "https://tauri.localhost",
    }
)


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


def _origin_is_trusted_app(origin_header: str) -> bool:
    """True when a present Origin/Referer is exactly a trusted app origin.

    The check is scheme-strict and host-exact against
    :data:`TRUSTED_APP_ORIGINS` (part14 §3.2 — never a wildcard, never
    a suffix match): ``Referer`` values, which carry paths, are parsed
    so only scheme + host (+ no port) are compared.  An absent header
    stays the non-browser client's lane (handled by the caller's
    falsy short-circuit).
    """
    if not origin_header:
        return False
    try:
        parts = urlsplit(origin_header)
        port = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or port is not None:
        return False
    origin = f"{parts.scheme}://{(parts.hostname or '').lower()}"
    return origin in TRUSTED_APP_ORIGINS


def origin_is_trusted(origin_header: str) -> bool:
    """True when a present Origin/Referer is loopback or a trusted app origin.

    The exact REST-lane policy (part14 §3.2, D-1/D-2), exported for
    the WebSocket handshake guard (part15 §4.2, D-6): a *present*
    header must be a loopback hostname or an exact trusted
    application origin (:data:`TRUSTED_APP_ORIGINS`), while an
    *absent* header is the non-browser client's lane and returns
    ``True`` — the caller decides what that means for its protocol.
    """
    return _origin_is_loopback(origin_header) or _origin_is_trusted_app(
        origin_header
    )


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
            if not origin_is_trusted(request.headers.get(header, "")):
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
