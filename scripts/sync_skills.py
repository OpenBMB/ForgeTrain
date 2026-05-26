#!/usr/bin/env python3
"""Sync .rules/skills/<name>/** → multi-platform skill directories.

The canonical source for every project skill lives under
``.rules/skills/<name>/``. This script mirrors each skill into the
platform-specific skill directories so Cursor and Claude Code can
discover them with their native conventions:

    .rules/skills/<name>/...   (canonical, edit here)
        ├─→ .cursor/skills/<name>/...
        └─→ .claude/skills/<name>/...

Every mirrored file is prefixed with an AUTO-GENERATED banner inside
the SKILL.md frontmatter's body (after the closing ``---``); other
files are mirrored byte-for-byte. The banner reminds humans to edit
the canonical source instead of the generated peer.

Usage:
    python scripts/sync_skills.py --check   # check only (CI)
    python scripts/sync_skills.py --sync    # force regenerate
    python scripts/sync_skills.py           # auto-fix + fail if dirty
"""
from __future__ import annotations

import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = HARNESS_ROOT / ".rules" / "skills"

TARGET_ROOTS: dict[str, Path] = {
    "Cursor": HARNESS_ROOT / ".cursor" / "skills",
    "Claude Code": HARNESS_ROOT / ".claude" / "skills",
}

BANNER = (
    "<!-- AUTO-GENERATED from .rules/skills/{name}/SKILL.md — DO NOT EDIT -->\n"
    "<!-- Regenerate: python scripts/sync_skills.py --sync -->\n"
)


def _inject_banner(skill_name: str, text: str) -> str:
    """Insert the AUTO-GENERATED banner after the YAML frontmatter.

    Frontmatter is delimited by ``---`` lines at the top of the file.
    If no frontmatter is present (defensive fallback), prepend the
    banner directly.
    """
    banner = BANNER.format(name=skill_name)
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("---"):
        return banner + "\n" + text
    # Find the closing frontmatter delimiter.
    for idx in range(1, len(lines)):
        if lines[idx].startswith("---"):
            head = "".join(lines[: idx + 1])
            tail = "".join(lines[idx + 1 :])
            sep = "" if tail.startswith("\n") else "\n"
            return head + sep + banner + tail
    # Malformed frontmatter: prepend defensively.
    return banner + "\n" + text


def _expected_content(skill_name: str, src: Path) -> bytes:
    """Return the bytes the mirror file should contain."""
    raw = src.read_bytes()
    if src.name == "SKILL.md":
        text = raw.decode("utf-8")
        return _inject_banner(skill_name, text).encode("utf-8")
    return raw


def _iter_skills() -> list[Path]:
    if not SOURCE_ROOT.exists():
        return []
    return sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir())


def _iter_skill_files(skill_dir: Path) -> list[Path]:
    return sorted(p for p in skill_dir.rglob("*") if p.is_file())


def _plan() -> list[tuple[str, Path, Path, bytes]]:
    """Return [(label, src, target, expected_bytes), ...]."""
    plan: list[tuple[str, Path, Path, bytes]] = []
    for skill_dir in _iter_skills():
        skill_name = skill_dir.name
        for src in _iter_skill_files(skill_dir):
            rel = src.relative_to(skill_dir)
            expected = _expected_content(skill_name, src)
            for label, root in TARGET_ROOTS.items():
                target = root / skill_name / rel
                plan.append((label, src, target, expected))
    return plan


def sync_all() -> list[str]:
    """Write all targets. Return list of changed targets."""
    changed: list[str] = []
    expected_targets: set[Path] = set()
    for label, _src, target, expected in _plan():
        expected_targets.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_bytes() if target.exists() else b""
        if existing != expected:
            target.write_bytes(expected)
            changed.append(f"  {label}: {target.relative_to(HARNESS_ROOT)}")

    # Prune mirrored files that no longer have a canonical source.
    for label, root in TARGET_ROOTS.items():
        for skill_dir in (p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
            for path in (p for p in skill_dir.rglob("*") if p.is_file()):
                if path not in expected_targets:
                    path.unlink()
                    changed.append(f"  {label} (pruned): {path.relative_to(HARNESS_ROOT)}")
            # Drop emptied directories.
            for sub in sorted(
                (p for p in skill_dir.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                try:
                    sub.rmdir()
                except OSError:
                    pass
            if not any(skill_dir.iterdir()) and not (SOURCE_ROOT / skill_dir.name).exists():
                skill_dir.rmdir()
                changed.append(f"  {label} (pruned): {skill_dir.relative_to(HARNESS_ROOT)}")
    return changed


def check_all() -> list[str]:
    """Return list of drifted targets (no writes)."""
    drifted: list[str] = []
    expected_targets: set[Path] = set()
    for label, _src, target, expected in _plan():
        expected_targets.add(target)
        existing = target.read_bytes() if target.exists() else b""
        if existing != expected:
            drifted.append(f"  {label}: {target.relative_to(HARNESS_ROOT)}")

    for label, root in TARGET_ROOTS.items():
        if not root.exists():
            continue
        for path in (p for p in root.rglob("*") if p.is_file()):
            if path not in expected_targets:
                drifted.append(f"  {label} (orphan): {path.relative_to(HARNESS_ROOT)}")
    return drifted


def main() -> int:
    args = set(sys.argv[1:])
    if "--check" in args:
        drifted = check_all()
        if drifted:
            print("skills drift detected:\n" + "\n".join(drifted))
            print("\nRun: python scripts/sync_skills.py --sync")
            return 1
        return 0

    changed = sync_all()
    if "--sync" in args:
        if changed:
            print("Synced:\n" + "\n".join(changed))
        else:
            print("All skills up to date.")
        return 0

    # pre-commit mode: auto-fix then fail if anything changed.
    if changed:
        print("sync-skills: fixed drift, stage the updated files:")
        print("\n".join(changed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
