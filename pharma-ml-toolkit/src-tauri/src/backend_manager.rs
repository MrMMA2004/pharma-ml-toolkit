//! The Python backend sidecar supervisor — part11 §14 (Phase Nine:
//! Local Backend Process Lifecycle) and §31.1 (the sidecar contract).
//!
//! The wire contract this module implements is NOT invented here: it
//! is the backend's own tested startup protocol
//! (`pharmaml.api.server`, pinned by `tests/api/test_backend_service.py`
//! — `TestStartupHandshake`):
//!
//! * the sidecar is spawned with NO arguments — the session token is
//!   generated inside the backend (`secrets.token_urlsafe(32)`), so
//!   no secret ever rides argv, the environment, or a log (§15);
//! * stdout carries exactly ONE machine-readable line:
//!   `PHARMAML-READY {"port":…,"token":…,"pid":…,"version":…}`;
//! * the backend binds loopback-only on a port-0 socket it owns
//!   itself (no check-then-bind race — §14.1);
//! * diagnostics go to stderr only — this supervisor pipes stderr
//!   through to the host's stderr, never mixing it with stdout;
//! * the backend's graceful shutdown publishes the shutdown envelope
//!   to connected WebSocket clients BEFORE the sockets close
//!   (`TestGracefulShutdownPush`); the host's `terminate` gives that
//!   path a bounded window, then forces termination as the
//!   documented fallback (§14.3's last step).
//!
//! Startup timeout, crash detection and a clear-termination policy
//! are the §14.2 requirements.  The restart policy is deliberately
//! single-attempt-per-session: the frontend's WS liveness + REST
//! tracker reconcile a dead backend (the tested r280 client
//! contract), and a silently respawning backend would hide real
//! failures — §14.2's "clear startup error messages" wins over
//! silent retry.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tauri::AppHandle;

/// The parsed `PHARMAML-READY` handshake line (§14.1's step 8).
#[derive(Clone, Debug, Deserialize)]
pub struct BackendReady {
    pub port: u16,
    pub token: String,
    pub pid: u32,
    pub version: String,
}

/// The endpoint + session token the frontend's typed REST/WS clients
/// consume (the `configureBackend` bridge — part11 §5.2).
#[derive(Clone, Debug, serde::Serialize)]
pub struct BackendEndpoint {
    /// e.g. `http://127.0.0.1:54127` — the handshake's actual port.
    pub base_url: String,
    /// The per-startup session token (memory-only; never persisted).
    pub token: String,
    /// The backend's reported version (displayed by the shell).
    pub version: String,
}

/// The supervised sidecar: the parsed endpoint + the live child.
pub struct BackendHandle {
    endpoint: BackendEndpoint,
    child: Mutex<Child>,
}

impl BackendHandle {
    /// The endpoint the one IPC command serves (commands.rs).
    pub fn endpoint(&self) -> BackendEndpoint {
        self.endpoint.clone()
    }
}

/// One supervised spawn attempt's outcome.
enum SpawnOutcome {
    Ready(Box<BackendHandle>),
    /// §14.2's startup timeout — a clear, bounded failure.
    TimedOut { reason: String },
    /// Spawn/handshake failure — a clear, immediate failure.
    Failed { reason: String },
}

/// The single machine-readable prefix the backend prints (§14.1).
const HANDSHAKE_PREFIX: &str = "PHARMAML-READY ";

/// §14.2: a bounded startup timeout. The scientific stack imports
/// (torch/rdkit) can take tens of seconds cold — 180s is the honest
/// ceiling, surfaced as a clear error when exceeded.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);

/// §14.3: the graceful-shutdown window before forced termination.
const GRACEFUL_WINDOW: Duration = Duration::from_secs(10);

/// Windows `CREATE_NO_WINDOW` — the console sidecar must never flash
/// a terminal window in the packaged desktop app.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Locate the frozen backend sidecar inside the packaged resources.
///
/// PyInstaller's onedir layout (pharmaml_backend.spec's `COLLECT`)
/// puts the executable at `<resources>/backend/pharmaml-backend/
/// pharmaml-backend.exe` with its `_internal` tree beside it — the
/// whole directory rides the installer through tauri.conf.json's
/// `bundle.resources`.
fn backend_exe(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    use tauri::Manager;
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|error| format!("the resource dir did not resolve: {error}"))?;
    let exe = resource_dir
        .join("backend")
        .join("pharmaml-backend")
        .join("pharmaml-backend.exe");
    if exe.is_file() {
        Ok(exe)
    } else {
        Err(format!(
            "the bundled backend sidecar is missing at {} — the installer's \
             resources/backend payload did not ship (part11_2 §5)",
            exe.display()
        ))
    }
}

/// Spawn the sidecar and wait for the handshake (§14.1's steps 6–10).
fn spawn_and_handshake(exe: &std::path::Path) -> SpawnOutcome {
    let mut command = Command::new(exe);
    // NO arguments: the session token is generated inside the backend
    // (never argv — §15).  stdout carries the handshake; stderr the
    // diagnostics — piped separately (§14.1's separation rule).
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return SpawnOutcome::Failed {
                reason: format!("the backend sidecar did not spawn: {error}"),
            };
        }
    };

    let stdout = child.stdout.take().expect("stdout is piped above");
    let stderr = child.stderr.take().expect("stderr is piped above");

    // stderr streams through to the host's stderr (diagnostics, never
    // parsed as machine-readable — §14.1).
    let stderr_pump = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("[backend] {line}");
        }
    });

    // The handshake read runs on its own thread so the timeout can be
    // enforced even when the backend produces no output at all (a
    // hung import must surface as TimedOut, not a frozen host).
    let (line_tx, line_rx) = mpsc::channel::<String>();
    let stdout_reader = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if line.starts_with(HANDSHAKE_PREFIX) {
                let _ = line_tx.send(line);
                return; // exactly one machine-readable line — done
            }
            // stdout noise before the handshake is logged, not parsed
            eprintln!("[backend:stdout] {line}");
        }
    });

    // r297: recv_timeout takes a Duration, not an Instant (the
    // authored-era file had never compiled anywhere -- the Blocker
    // Register's C-2, caught by the first real rustc pass, run
    // 44528322869's E0308).
    let ready_line = match line_rx.recv_timeout(STARTUP_TIMEOUT) {
        Ok(line) => line,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let _ = child.kill();
            let _ = child.wait();
            return SpawnOutcome::TimedOut {
                reason: String::from(
                    "no PHARMAML-READY line within the 180s startup window",
                ),
            };
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            // the reader ended without a handshake line — the backend
            // died during startup (§14.2's crash detection)
            let _ = stdout_reader.join();
            let _ = stderr_pump.join();
            let _ = child.wait();
            return SpawnOutcome::Failed {
                reason: String::from(
                    "the backend exited before printing its PHARMAML-READY \
                     handshake line — see the diagnostics above",
                ),
            };
        }
    };

    let payload = &ready_line[HANDSHAKE_PREFIX.len()..];
    let parsed: BackendReady = match serde_json::from_str(payload) {
        Ok(parsed) => parsed,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return SpawnOutcome::Failed {
                reason: format!(
                    "the handshake line did not parse as the documented JSON \
                     shape: {error} (line: {ready_line})"
                ),
            };
        }
    };

    SpawnOutcome::Ready(Box::new(BackendHandle {
        endpoint: BackendEndpoint {
            base_url: format!("http://127.0.0.1:{}", parsed.port),
            token: parsed.token,
            version: parsed.version,
        },
        child: Mutex::new(child),
    }))
}

/// The setup-time entry: spawn, handshake, surface failures loudly.
pub fn spawn_backend(app: &AppHandle) -> Result<BackendHandle, String> {
    let exe = backend_exe(app)?;
    match spawn_and_handshake(&exe) {
        SpawnOutcome::Ready(handle) => Ok(*handle),
        SpawnOutcome::TimedOut { reason } => Err(format!(
            "PharmaML Studio backend startup TIMED OUT: {reason}. The \
             application cannot start its local scientific backend — see \
             the diagnostics above (part11 §14.2)."
        )),
        SpawnOutcome::Failed { reason } => Err(format!(
            "PharmaML Studio backend FAILED during startup: {reason}. The \
             application cannot start its local scientific backend (part11 \
             §14.2)."
        )),
    }
}

/// The shutdown path (§14.3): the backend's own graceful-shutdown
/// handling publishes the shutdown envelope to WebSocket clients
/// before exit (the tested backend contract); the host gives that
/// path the bounded `GRACEFUL_WINDOW` via a wait-with-timeout, then
/// forces termination as the documented fallback.
pub fn terminate(handle: &BackendHandle) {
    if let Ok(mut child) = handle.child.lock() {
        // A bounded wait for a self-exiting child: poll try_wait for
        // the graceful window, then kill (§14.3's documented ladder).
        let deadline = Instant::now() + GRACEFUL_WINDOW;
        while let Ok(None) = child.try_wait() {
            if Instant::now() > deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// The shared-state type managed on the Tauri app (lib.rs).
pub type SharedBackend = Arc<BackendHandle>;
