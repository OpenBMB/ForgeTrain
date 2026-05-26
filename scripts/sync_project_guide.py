#!/usr/bin/env python3
"""Sync .rules/project-guide.md → multi-platform discovery files.

Ensures every supported agent platform sees identical project guidance.
Runs as a pre-commit hook; exits non-zero on drift (so the developer
stages the auto-fixed peers).

Usage:
    python scripts/sync_project_guide.py --check   # check only (CI)
    python scripts/sync_project_guide.py --sync    # force regenerate
    python scripts/sync_project_guide.py           # auto-fix + fail if dirty
"""
from __future__ import annotations

import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
SOURCE = HARNESS_ROOT / ".rules" / "bootstrap-guide.md"

TARGETS: dict[str, Path] = {
    "Cursor": HARNESS_ROOT / ".cursor" / "rules" / "bootstrap-guide.mdc",
    "Claude Code": HARNESS_ROOT / "CLAUDE.md",
    "Codex / OpenCode": HARNESS_ROOT / "AGENTS.md",
    "GitHub Copilot": HARNESS_ROOT / ".github" / "copilot-instructions.md",
}

HEADER = (
    "<!-- AUTO-GENERATED from .rules/bootstrap-guide.md — DO NOT EDIT -->\n"
    "<!-- Regenerate: python scripts/sync_project_guide.py --sync -->\n\n"
)


def generate(source_text: str) -> str:
    return HEADER + source_text


def sync_all() -> list[str]:
    """Write all targets. Return list of targets that changed."""
    if not SOURCE.exists():
        print(f"ERROR: source not found: {SOURCE}", file=sys.stderr)
        sys.exit(2)
    source_text = SOURCE.read_text(encoding="utf-8")
    content = generate(source_text)
    changed: list[str] = []
    for label, target in TARGETS.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if existing != content:
            target.write_text(content, encoding="utf-8")
            changed.append(f"  {label}: {target.relative_to(HARNESS_ROOT)}")
    return changed


def check_all() -> list[str]:
    """Return list of targets with drift (no writes)."""
    if not SOURCE.exists():
        print(f"ERROR: source not found: {SOURCE}", file=sys.stderr)
        sys.exit(2)
    source_text = SOURCE.read_text(encoding="utf-8")
    content = generate(source_text)
    drifted: list[str] = []
    for label, target in TARGETS.items():
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if existing != content:
            drifted.append(f"  {label}: {target.relative_to(HARNESS_ROOT)}")
    return drifted


def main() -> int:
    args = set(sys.argv[1:])
    if "--check" in args:
        drifted = check_all()
        if drifted:
            print("project-guide drift detected:\n" + "\n".join(drifted))
            print("\nRun: python scripts/sync_project_guide.py --sync")
            return 1
        return 0

    changed = sync_all()
    if "--sync" in args:
        if changed:
            print("Synced:\n" + "\n".join(changed))
        else:
            print("All targets up to date.")
        return 0

    # pre-commit mode: auto-fix then fail if anything changed
    if changed:
        print("sync-project-guide: fixed drift, stage the updated files:")
        print("\n".join(changed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
