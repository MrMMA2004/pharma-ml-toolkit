// Prevents the additional console window on Windows in release
// builds (the sidecar console is suppressed separately through
// CREATE_NO_WINDOW in backend_manager.rs).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    pharmaml_studio_lib::run()
}
