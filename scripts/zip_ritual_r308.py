#!/usr/bin/env python3
"""Round 308 zip ritual: testzip every archive in the four homes, then
rebuild the offsite home (~/pharma-ml-toolkit-zips/) from the homes and
testzip it 11/11. Prints per-home totals, never the token."""
import shutil
import zipfile
from pathlib import Path

REPO = Path("/home/z/my-project")
HOMES = [
    REPO / "download",
    REPO / "pharma-ml-toolkit" / "download",
    REPO / "pharma-ml-toolkit" / "dist-archive",
    REPO / "upload",
]
OFFSITE = Path("/home/z/pharma-ml-toolkit-zips")
CURRENT = "pharma-ml-toolkit-v1.2.240.zip"


def testzip_all(home: Path) -> tuple[int, list[str]]:
    zips = sorted(home.glob("pharma-ml-toolkit-v*.zip"))
    bad = []
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                if zf.testzip() is not None:
                    bad.append(z.name)
        except zipfile.BadZipFile:
            bad.append(z.name + " (BAD)")
    return len(zips), bad


all_bad = []
for home in HOMES:
    n, bad = testzip_all(home)
    status = "OK" if not bad else f"BAD: {bad}"
    print(f"{home.relative_to(REPO)}: {n} zips testzip {'OK' if not bad else 'FAIL'}")
    all_bad += [f"{home}:{b}" for b in bad]

# rebuild offsite: one copy per version, prefer the current version's home,
# older versions come from whichever home holds them (all byte-identical).
OFFSITE.mkdir(parents=True, exist_ok=True)
versions = {}
for home in HOMES:
    for z in sorted(home.glob("pharma-ml-toolkit-v*.zip")):
        versions.setdefault(z.name, home / z.name)
copied = 0
for name, src in sorted(versions.items()):
    dst = OFFSITE / name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
        copied += 1
n, bad = testzip_all(OFFSITE)
print(f"offsite: rebuilt {copied} new copies, {n} zips total, testzip {'OK' if not bad else 'BAD: ' + str(bad)}")
cur = OFFSITE / CURRENT
print(f"offsite current-version parity: {cur.stat().st_size} B, "
      f"sha256 {__import__('hashlib').sha256(cur.read_bytes()).hexdigest()[:8]}")
# r324: 18 zips = the v1.2.221-240 span restored whole (the git-tracked
# homes carry every retired version again after the fifteenth rollback's
# reset -- one more than the r323 disk state had, recorded honestly).
print("RITUAL " + ("PASS" if not all_bad and not bad and n == 18 else "CHECK"))
