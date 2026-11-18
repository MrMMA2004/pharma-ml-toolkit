#!/usr/bin/env python3
"""r289 firing: finish the Phase Sixteen startup-gate wiring.

Three edits, each pinned by tests/test_frontend_startup.py:
1. App.tsx — mount the shell inside <StartupGate> (header + nav + main).
2. vite.config.ts — the /api proxy forwards the WS lane (ws: true).
3. package.json — @tauri-apps/api is the runtime invoke dependency.
"""
import json
import re
from pathlib import Path

FRONTEND = Path("/home/z/my-project/pharma-ml-toolkit/frontend")


def edit_app() -> None:
    p = FRONTEND / "src" / "app" / "App.tsx"
    src = p.read_text(encoding="utf-8")

    # 1. the import (sorted: "@/app/StartupGate" before "@/app/workspaces")
    anchor = 'import { INITIAL_WORKSPACE, WORKSPACES, type WorkspaceId } from "@/app/workspaces";'
    assert src.count(anchor) == 1, "workspaces import anchor not found"
    src = src.replace(
        anchor,
        'import { StartupGate } from "@/app/StartupGate";\n' + anchor,
    )

    # 2. the docstring gains the §14.1 line
    doc_anchor = (
        " * One page instance per workspace (state preserved across switches;\n"
        " * no duplicate mounts), Introduction selected at startup, the exact\n"
        " * §10.1 navigation order."
    )
    assert src.count(doc_anchor) == 1, "docstring anchor not found"
    src = src.replace(
        doc_anchor,
        " * The whole shell mounts inside the §14.1 StartupGate: no page\n"
        " * renders before the backend handshake completes, health is\n"
        " * verified, and the event stream opens.\n"
        " *\n"
        " * One page instance per workspace (state preserved across switches;\n"
        " * no duplicate mounts), Introduction selected at startup, the exact\n"
        " * §10.1 navigation order.",
    )

    # 3. wrap the return block: re-indent the inner JSX one level
    ret_marker = '  return (\n    <div className="flex min-h-screen flex-col bg-[var(--color-background)]">'
    assert src.count(ret_marker) == 1, "return marker not found"
    i = src.index(ret_marker)
    j = src.index("\n  );\n}\n", i)
    inner = src[i + len("  return (\n") : j]  # '<div ...' .. '</div>'
    reindented = "\n".join(
        ("  " + ln) if ln.strip() else ln for ln in inner.split("\n")
    )
    new_block = (
        "  return (\n"
        "    <StartupGate>\n"
        + reindented  # already ends with the re-indented closing </div>
        + "\n    </StartupGate>"
    )
    src = src[:i] + new_block + src[j:]

    p.write_text(src, encoding="utf-8")
    print("App.tsx wired")


def edit_vite() -> None:
    p = FRONTEND / "vite.config.ts"
    src = p.read_text(encoding="utf-8")
    assert "ws: false," in src, "expected ws: false in the dev proxy"
    src = src.replace(
        """            changeOrigin: false,
            ws: false,""",
        """            changeOrigin: false,
            ws: true,""",
    )
    old_comment = (
        "// proxied server-side to the real backend's loopback port, so a\n"
        "    // browser dev session rides the actual service same-origin (the\n"
        "    // CORS gate never applies server-side; the X-Session-Token header\n"
        "    // forwards as-is; the WS lane stays direct — loopback origins pass\n"
        "    // the backend's own guard). The production frontend never uses\n"
    )
    new_comment = (
        "// proxied server-side to the real backend's loopback port, so a\n"
        "    // browser dev session rides the actual service same-origin (the\n"
        "    // CORS gate never applies server-side; the X-Session-Token header\n"
        "    // forwards as-is; ws: true carries the event stream through the\n"
        "    // same proxy so the dev fallback's WS lane opens same-origin).\n"
        "    // The production frontend never uses\n"
    )
    assert old_comment in src, "proxy comment anchor not found"
    src = src.replace(old_comment, new_comment)
    p.write_text(src, encoding="utf-8")
    print("vite.config.ts ws lane on")


def edit_pkg() -> None:
    p = FRONTEND / "package.json"
    pkg = json.loads(p.read_text(encoding="utf-8"))
    deps = pkg["dependencies"]
    deps["@tauri-apps/api"] = "^2.12.5"
    pkg["dependencies"] = dict(sorted(deps.items()))
    p.write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")
    print("package.json: @tauri-apps/api ^2.12.5 added")


if __name__ == "__main__":
    edit_app()
    edit_vite()
    edit_pkg()
