"""Part14 round-326 §18.5 reproduction — the Tauri/WebView2 <-> FastAPI
origin incident, re-run against the CURRENT source tree (the post-r325
disk state) with the real create_app() exactly as the Tauri host
launches it.

Every leg uses browser-shaped headers exactly as the packaged WebView2
frontend (origin http(s)://tauri.localhost) and the dev frontend
(origin http://localhost:5174) would send them.  Evidence file:
scratch/p14_r326_origin_probe.txt
"""

from fastapi.testclient import TestClient

from pharmaml.api import create_app

TOKEN = "r326-fixed-probe-token"
app = create_app(token=TOKEN)
client = TestClient(app)

LEGS = [
    (
        "P1 packaged WebView2 origin http://tauri.localhost + token, GET health",
        "get",
        "/api/v1/health",
        {
            "Host": "127.0.0.1:54127",
            "Origin": "http://tauri.localhost",
            "X-Session-Token": TOKEN,
        },
    ),
    (
        "P2 packaged WebView2 origin https://tauri.localhost + token, GET health",
        "get",
        "/api/v1/health",
        {
            "Host": "127.0.0.1:54127",
            "Origin": "https://tauri.localhost",
            "X-Session-Token": TOKEN,
        },
    ),
    (
        "P3 CORS preflight OPTIONS from tauri.localhost (JSON + custom header)",
        "options",
        "/api/v1/health",
        {
            "Host": "127.0.0.1:54127",
            "Origin": "http://tauri.localhost",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type,x-session-token",
        },
    ),
    (
        "P4 CORS preflight OPTIONS from the dev origin localhost:5174",
        "options",
        "/api/v1/health",
        {
            "Host": "127.0.0.1:54127",
            "Origin": "http://localhost:5174",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type,x-session-token",
        },
    ),
    (
        "P5 loopback origin, NO token (the auth surface)",
        "get",
        "/api/v1/health",
        {"Host": "127.0.0.1:54127", "Origin": "http://localhost:5174"},
    ),
    (
        "P6 loopback origin + valid token (the lane that must work)",
        "get",
        "/api/v1/health",
        {
            "Host": "127.0.0.1:54127",
            "Origin": "http://localhost:5174",
            "X-Session-Token": TOKEN,
        },
    ),
    (
        "P7 non-loopback Host (tauri.localhost as the Host itself)",
        "get",
        "/api/v1/health",
        {
            "Host": "tauri.localhost:54127",
            "Origin": "http://localhost:5174",
            "X-Session-Token": TOKEN,
        },
    ),
    (
        "P8 LAN Host 10.0.0.5 (the must-reject lane)",
        "get",
        "/api/v1/health",
        {"Host": "10.0.0.5:54127", "X-Session-Token": TOKEN},
    ),
]

lines = []
for name, method, path, headers in LEGS:
    fn = getattr(client, method)
    r = fn(path, headers=headers)
    detail = ""
    try:
        detail = r.json().get("detail", "")
    except Exception:
        detail = (r.text or "")[:60]
    lines.append(
        f"{name}\n  -> HTTP {r.status_code} | detail={detail!r} "
        f"| allow={r.headers.get('allow', '-')} "
        f"| acao={r.headers.get('access-control-allow-origin', '-')}\n"
    )

report = "".join(lines)
print(report)
with open("/home/z/my-project/scratch/p14_r326_origin_probe.txt", "w") as f:
    f.write(report)
