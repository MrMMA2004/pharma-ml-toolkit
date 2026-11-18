//! The PharmaML Studio Tauri 2 host (part11 Phase Four / §5.2).
//!
//! Responsibilities — exactly the §5.2 "Tauri 2 desktop host" list
//! and nothing else:
//!
//! * the desktop window + the offline WebView2 frontend load
//!   (frontendDist — the Vite-built static assets; no dev server,
//!   no remote URL in production);
//! * the Python backend sidecar lifecycle (`backend_manager.rs`:
//!   spawn -> PHARMAML-READY handshake -> supervision -> graceful
//!   termination);
//! * the one protected IPC bridge (`commands::backend_endpoint`)
//!   that hands the WebView the loopback endpoint + session token;
//! * no scientific domain logic (§3: Python stays the science; Rust
//!   stays the host).

mod backend_manager;
mod commands;
mod security;

use std::sync::Arc;

use tauri::{RunEvent, Manager};

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // §14.1's steps 4–10: locate the sidecar, spawn it, parse
            // the handshake — a startup failure is a LOUD setup error
            // with the full diagnostic in the message (§14.2: clear
            // startup error messages; the app exits rather than
            // showing a dead window with a silently-missing backend).
            let handle = app.handle().clone();
            let backend = backend_manager::spawn_backend(&handle)?;

            // §15: the loopback assertion — the endpoint handed to
            // the WebView is 127.0.0.1-only or the host refuses to
            // start.  security::TOKEN_HANDOFF_CONTRACT documents the
            // token's single handoff path (handshake -> state -> the
            // backend_endpoint invoke -> configureBackend).
            security::assert_loopback(&backend.endpoint())?;

            app.manage(Arc::new(backend) as backend_manager::SharedBackend);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![commands::backend_endpoint])
        .build(tauri::generate_context!())
        .expect("the Tauri context builds (tauri.conf.json is validated by the workflow's schema step)")
        .run(|app, event| {
            // §14.3: on app exit the sidecar terminates through the
            // graceful window + forced fallback — never orphaned.
            if let RunEvent::Exit = event {
                if let Some(backend) = app.try_state::<backend_manager::SharedBackend>() {
                    backend_manager::terminate(&*backend);
                }
            }
        });
}
