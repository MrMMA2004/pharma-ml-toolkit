//! The minimal IPC command surface (part11 §9.1: "Minimal IPC
//! commands" — the ONE bridge the frontend needs: the startup
//! handshake's endpoint + session token, delivered in-process).

use tauri::State;

use crate::backend_manager::{BackendEndpoint, SharedBackend};

/// `invoke("backend_endpoint")` — the §14.1 step-11 bridge: the
/// frontend's typed clients (`configureBackend`) receive the
/// loopback base URL and the per-startup session token.  This is the
/// ONLY application command: no filesystem passthrough, no shell,
/// no scientific operations (those ride the backend's own REST/WS
/// surface — §9.2's rule).
#[tauri::command]
pub fn backend_endpoint(
    backend: State<'_, SharedBackend>,
) -> Result<BackendEndpoint, String> {
    Ok(backend.endpoint())
}
