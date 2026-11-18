"""``/api/v1/capabilities`` — capability discovery (part10 §9.1).

The single source of truth is the existing
``pharmaml.studio.capability_registry`` (62 rows, test-referenced);
the API layer projects it — never re-declares it.  The WinUI host
uses this to build its workspace menus from *live* data.
"""

from __future__ import annotations

from fastapi import APIRouter

from pharmaml.studio import capability_registry as registry

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


@router.get("")
def capabilities() -> dict:
    return {
        "total": registry.summary().total,
        "by_card": registry.summary().by_card,
        "by_status": registry.summary().by_status,
        "capabilities": [
            {
                "capability_id": c.capability_id,
                "name": c.name,
                "backend_module": c.backend_module,
                "public_api": c.public_api,
                "card": c.card,
                "workspace": c.route,
                "status": c.status,
                "tests": list(c.tests),
                "optional_deps": list(c.optional_deps),
                "engines": list(c.engines),
            }
            for c in registry.CAPABILITIES
        ],
    }


@router.get("/{capability_id}")
def capability(capability_id: str) -> dict | None:
    found = registry.find(capability_id)
    if found is None:
        return None
    return {
        "capability_id": found.capability_id,
        "name": found.name,
        "backend_module": found.backend_module,
        "public_api": found.public_api,
        "card": found.card,
        "workspace": found.route,
        "status": found.status,
        "tests": list(found.tests),
        "optional_deps": list(found.optional_deps),
        "engines": list(found.engines),
    }
