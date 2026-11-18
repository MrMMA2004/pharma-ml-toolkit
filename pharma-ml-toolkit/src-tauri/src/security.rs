//! The host-side security boundary (part11 §15 / §9.4).
//!
//! The WebView frontend and the FastAPI backend are separate trust
//! boundaries; the ONLY privileged bridge between them is the
//! startup handshake this host brokers:
//!
//! * the session token reaches the WebView through the
//!   `backend_endpoint` IPC command (a protected, in-process Tauri
//!   invoke — never a query string, never a log, never persisted);
//! * the CSP in `tauri.conf.json` scopes `connect-src` to the
//!   packaged self origin plus the LOOPBACK API/WebSocket endpoint
//!   — no remote script, style, or fetch can execute;
//! * no shell/FS/dialog capability is granted to the window
//!   (`capabilities/default.json` is `core:default` only) — the
//!   sidecar is spawned by the Rust host, not the WebView;
//! * `assert_loopback` guards the endpoint this module hands out:
//!   a handshake naming anything but 127.0.0.1 is refused loudly
//!   (the backend binds loopback-only by design — a regression in
//!   that contract must fail HERE, not in the browser).

use crate::backend_manager::BackendEndpoint;

/// Refuse a non-loopback handshake endpoint (§15: bind only to
/// 127.0.0.1 — never 0.0.0.0).
pub fn assert_loopback(endpoint: &BackendEndpoint) -> Result<(), String> {
    let prefix = "http://127.0.0.1:";
    if endpoint.base_url.starts_with(prefix) && endpoint.base_url[16..].parse::<u16>().is_ok() {
        Ok(())
    } else {
        Err(format!(
            "the backend handshake named a NON-LOOPBACK endpoint \
             ({}), refusing to hand it to the WebView — the local API \
             contract is 127.0.0.1-only (part11 §15)",
            endpoint.base_url
        ))
    }
}

/// The token-handoff guard: the session token is memory-only in the
/// host (never written to disk, never a log line).  This is a
/// compile-time documentation point enforced by review + the
/// command's single consumer; the runtime guards live in the
/// backend's own security layer (constant-time verification,
/// Host/Origin validation — pinned by `tests/api/`).
pub const TOKEN_HANDOFF_CONTRACT: &str =
    "the session token travels: backend handshake line -> Rust state \
     -> the backend_endpoint IPC invoke -> the frontend's in-memory \
     configureBackend.  No argv, no environment, no query string, no \
     persistence, no log.";
