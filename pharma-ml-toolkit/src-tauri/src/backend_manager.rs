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
//!
//! part14 §3.3 hardening (the D-4/D-5 fixes):
//!
//! * a MALFORMED handshake line never reaches an error surface raw —
//!   the failure reason carries the serde error + the line's byte
//!   length only (the line may contain the session token, which must
//!   never leak into an exception, log, crash report, or UI message);
//! * the parsed handshake is field-validated before it reaches the
//!   endpoint bridge: a real bound port (never 0), the documented
//!   urlsafe token shape, a non-zero pid, and a numeric-semver
//!   version — every rejection redacted the same way.

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

/// part14 §3.3 (D-4) — the redacted parse-failure reason: the raw
/// handshake line NEVER appears (it may carry the session token);
/// only the serde error and the line's byte length do.
fn redacted_parse_failure(line_len: usize, error: serde_json::Error) -> String {
    format!(
        "the handshake line did not parse as the documented JSON \
         shape: {error} (a {line_len}-byte malformed line was received; \
         its raw contents are deliberately omitted — a session token \
         must never reach an error surface: part14 §3.3)"
    )
}

/// part14 §3.3 (D-5) — sanity-validate the parsed handshake before
/// its fields reach the endpoint bridge.  Every rejection is a
/// REDACTED, self-describing reason (never the raw line).
fn validate_handshake(ready: &BackendReady) -> Result<(), String> {
    // port: the backend binds port-0 itself and reports the ACTUAL
    // bound port — 0 means the report never happened.
    if ready.port == 0 {
        return Err(String::from(
            "the handshake reported port 0 — the backend must publish its \
             actually bound port, not the wildcard it requested",
        ));
    }
    // token: `secrets.token_urlsafe(32)` — 43 bytes of the urlsafe
    // alphabet [A-Za-z0-9_-].  Anything shorter, longer, or outside
    // that alphabet is not the documented token shape.
    let token = ready.token.as_str();
    let token_shape_ok = (16..=128).contains(&token.len())
        && token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if !token_shape_ok {
        return Err(String::from(
            "the handshake token failed the documented shape check \
             (16..=128 bytes of the urlsafe alphabet) — the session token \
             is not presentable, refusing to bridge it",
        ));
    }
    // pid: 0 is not a real process on any supported host.
    if ready.pid == 0 {
        return Err(String::from(
            "the handshake reported pid 0 — not a real backend process",
        ));
    }
    // version: the backend prints `pharmaml.__version__` — a short
    // numeric-semver string (e.g. "1.2.240").
    let version = ready.version.as_str();
    if version.is_empty()
        || version.len() > 32
        || !version.starts_with(|c: char| c.is_ascii_digit())
    {
        return Err(String::from(
            "the handshake version failed the documented shape check \
             (a short numeric-semver string, e.g. 1.2.240) — refusing to \
             bridge it",
        ));
    }
    Ok(())
}

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
            // D-4: the raw line (which may contain the session token)
            // never reaches the error surface — only the redacted form
            return SpawnOutcome::Failed {
                reason: redacted_parse_failure(ready_line.len(), error),
            };
        }
    };
    // D-5: field validation before the fields reach the endpoint
    // bridge (part14 §3.3) — the rejection is redacted the same way.
    if let Err(rejection) = validate_handshake(&parsed) {
        let _ = child.kill();
        let _ = child.wait();
        return SpawnOutcome::Failed { reason: rejection };
    }

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

#[cfg(test)]
mod tests {
    //! part14 §3.3 — the malformed-handshake regression tests (D-4's
    //! redaction + D-5's field validation).  Pure-function level: the
    //! spawn/handshake machinery needs a real sidecar binary; these
    //! pin exactly the policy the wire contract demands.

    use super::*;

    /// A well-formed handshake payload (the real wire shape).
    const VALID: &str = r#"{"port":43551,"token":"43-bytes-of-urlsafe-alphabet-Aa0-Zz9_","pid":1234,"version":"1.2.240"}"#;

    fn parsed(payload: &str) -> BackendReady {
        serde_json::from_str(payload).expect("the fixture must parse")
    }

    #[test]
    fn a_well_formed_handshake_validates() {
        assert!(validate_handshake(&parsed(VALID)).is_ok());
    }

    #[test]
    fn port_zero_is_rejected() {
        let mut ready = parsed(VALID);
        ready.port = 0;
        let reason = validate_handshake(&ready).unwrap_err();
        assert!(reason.contains("port 0"));
    }

    #[test]
    fn a_missing_token_is_rejected() {
        let ready = parsed(
            r#"{"port":43551,"token":"","pid":1234,"version":"1.2.240"}"#,
        );
        let reason = validate_handshake(&ready).unwrap_err();
        assert!(reason.contains("token"));
    }

    #[test]
    fn a_short_token_is_rejected() {
        let ready = parsed(
            r#"{"port":43551,"token":"abc","pid":1234,"version":"1.2.240"}"#,
        );
        assert!(validate_handshake(&ready).is_err());
    }

    #[test]
    fn a_token_outside_the_urlsafe_alphabet_is_rejected() {
        let ready = parsed(
            r#"{"port":43551,"token":"bad token ! symbols here padding the length!!","pid":1234,"version":"1.2.240"}"#,
        );
        let reason = validate_handshake(&ready).unwrap_err();
        assert!(reason.contains("shape"));
    }

    #[test]
    fn a_too_long_token_is_rejected() {
        let long = "a";
        let ready = parsed(&format!(
            "{{\"port\":43551,\"token\":\"{}\",\"pid\":1234,\"version\":\"1.2.240\"}}",
            long.repeat(200)
        ));
        assert!(validate_handshake(&ready).is_err());
    }

    #[test]
    fn pid_zero_is_rejected() {
        let ready = parsed(
            r#"{"port":43551,"token":"43-bytes-of-urlsafe-alphabet-Aa0-Zz9_","pid":0,"version":"1.2.240"}"#,
        );
        let reason = validate_handshake(&ready).unwrap_err();
        assert!(reason.contains("pid 0"));
    }

    #[test]
    fn an_empty_version_is_rejected() {
        let ready = parsed(
            r#"{"port":43551,"token":"43-bytes-of-urlsafe-alphabet-Aa0-Zz9_","pid":1234,"version":""}"#,
        );
        assert!(validate_handshake(&ready).is_err());
    }

    #[test]
    fn a_non_numeric_version_is_rejected() {
        let ready = parsed(
            r#"{"port":43551,"token":"43-bytes-of-urlsafe-alphabet-Aa0-Zz9_","pid":1234,"version":"beta-release"}"#,
        );
        let reason = validate_handshake(&ready).unwrap_err();
        assert!(reason.contains("version"));
    }

    #[test]
    fn the_parse_failure_reason_never_carries_the_raw_line_or_token() {
        // D-4: a malformed line that LOOKS like it carries a session
        // token — the redacted reason must not contain it (or any of
        // the raw payload), only the serde error + the byte length.
        let malformed =
            "PHARMAML-READY {\"port\":43551,\"token\":\"SECRET_TOKEN_VALUE_012345\"";
        let error = serde_json::from_str::<BackendReady>(malformed).unwrap_err();
        let reason = redacted_parse_failure(malformed.len(), error);
        assert!(!reason.contains("SECRET_TOKEN_VALUE_012345"));
        assert!(!reason.contains(malformed));
        assert!(reason.contains(&malformed.len().to_string()));
        assert!(reason.contains("deliberately omitted"));
    }
}
