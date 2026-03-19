#!/usr/bin/env python3
"""Lint all skills in a directory for structural and stylistic compliance.

Usage:
    python3 lint-skills.py <skills-directory>
    python3 lint-skills.py ~/.claude/skills/
    python3 lint-skills.py ~/.claude/skills/browser/

Checks:
    - SKILL.md exists
    - YAML frontmatter valid with required fields (name, description)
    - Description includes both "what" and "when"
    - Required sections present: Prerequisites, Quick Start, Troubleshooting
    - No TODO placeholders
    - UK English check (flags US spellings in prose)
    - File size reasonable (SKILL.md under 50KB)
    - Code blocks specify a language
    - Exactly one H1 heading

Produces:
    - stdout: human-readable report
    - reports/lint-report.json: machine-readable results
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SKILL_SIZE = 50 * 1024  # 50 KB

REQUIRED_SECTIONS = [
    "prerequisites",
    "quick start",
    "troubleshooting",
]

RECOMMENDED_SECTIONS = [
    "usage",
    "when to use",
    "when not to use",
]

# TODO-like markers to flag
TODO_MARKERS = re.compile(
    r"\b(TODO|FIXME|HACK|XXX)\b",
    re.IGNORECASE,
)

# US English spellings to flag (word boundary match, case-insensitive)
# Only flag in prose, not in code blocks.
US_SPELLINGS = {
    "color": "colour",
    "colors": "colours",
    "behavior": "behaviour",
    "behaviors": "behaviours",
    "optimize": "optimise",
    "optimized": "optimised",
    "optimizes": "optimises",
    "optimization": "optimisation",
    "optimizations": "optimisations",
    "organize": "organise",
    "organized": "organised",
    "organizes": "organises",
    "organization": "organisation",
    "organizations": "organisations",
    "analyze": "analyse",
    "analyzed": "analysed",
    "analyzes": "analyses",
    "analyzing": "analysing",
    "center": "centre",
    "centers": "centres",
    "defense": "defence",
    "defenses": "defences",
    "catalog": "catalogue",
    "catalogs": "catalogues",
    "favor": "favour",
    "favors": "favours",
    "favorable": "favourable",
    "honor": "honour",
    "honors": "honours",
    "labor": "labour",
    "labors": "labours",
    "neighbor": "neighbour",
    "neighbors": "neighbours",
    "program": "programme",  # Note: only non-computing uses
    "realize": "realise",
    "realized": "realised",
    "realizes": "realises",
    "recognize": "recognise",
    "recognized": "recognised",
    "recognizes": "recognises",
    "specialize": "specialise",
    "specialized": "specialised",
    "specializes": "specialises",
    "standardize": "standardise",
    "standardized": "standardised",
    "standardizes": "standardises",
    "customize": "customise",
    "customized": "customised",
    "customizes": "customises",
    "license": "licence",  # noun form; verb is 'license' in both
    "licenses": "licences",
    "characterize": "characterise",
    "characterized": "characterised",
    "characterizes": "characterises",
    "minimize": "minimise",
    "minimized": "minimised",
    "minimizes": "minimises",
    "maximize": "maximise",
    "maximized": "maximised",
    "maximizes": "maximises",
    "prioritize": "prioritise",
    "prioritized": "prioritised",
    "prioritizes": "prioritises",
    "summarize": "summarise",
    "summarized": "summarised",
    "summarizes": "summarises",
    "utilize": "utilise",
    "utilized": "utilised",
    "utilizes": "utilises",
}

# Words to skip US English checks on (appear in code contexts often)
US_SPELLING_EXCEPTIONS = {
    "color",  # CSS, CLI flags
    "optimize",  # compiler flags, function names
    "analyze",  # tool names
    "center",  # CSS property
    "program",  # computing context
}

# Unlanguaged code block pattern
UNLANGUAGED_CODEBLOCK = re.compile(r"^```\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# YAML extraction (reused from validate-frontmatter.py)
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract YAML frontmatter. Returns (dict, error)."""
    text = text.strip()
    if not text.startswith("---"):
        return None, "No YAML frontmatter found"
    end_idx = text.find("---", 3)
    if end_idx == -1:
        return None, "No closing frontmatter delimiter"
    yaml_block = text[3:end_idx].strip()
    if not yaml_block:
        return None, "Empty frontmatter"

    try:
        import yaml

        parsed = yaml.safe_load(yaml_block)
        if not isinstance(parsed, dict):
            return None, f"Frontmatter is {type(parsed).__name__}, expected dict"
        return parsed, None
    except ImportError:
        pass
    except Exception as exc:
        return None, f"YAML parse error: {exc}"

    # Basic fallback
    result = {}
    for line in yaml_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            value = [item.strip().strip("\"'") for item in items if item.strip()]
        result[key] = value
    return result, None


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------


def strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks from text, returning only prose."""
    in_block = False
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        if not in_block:
            lines.append(line)
    return "\n".join(lines)


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from text."""
    text = text.strip()
    if not text.startswith("---"):
        return text
    end_idx = text.find("---", 3)
    if end_idx == -1:
        return text
    return text[end_idx + 3:].strip()


def extract_headings(text: str) -> list[tuple[int, str]]:
    """Extract markdown headings as (level, text) tuples."""
    headings = []
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((level, title))
    return headings


# ---------------------------------------------------------------------------
# Lint checks
# ---------------------------------------------------------------------------


def lint_skill(skill_dir: Path) -> dict[str, Any]:
    """Lint a single skill directory. Returns a result dict."""
    skill_file = skill_dir / "SKILL.md"
    result = {
        "skill": skill_dir.name,
        "path": str(skill_file),
        "errors": [],
        "warnings": [],
        "info": [],
    }

    # Check SKILL.md exists
    if not skill_file.exists():
        result["errors"].append("SKILL.md not found")
        return result

    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception as exc:
        result["errors"].append(f"Could not read SKILL.md: {exc}")
        return result

    # File size check
    size = skill_file.stat().st_size
    if size > MAX_SKILL_SIZE:
        result["errors"].append(
            f"SKILL.md is {size:,} bytes, exceeds {MAX_SKILL_SIZE:,} byte limit"
        )

    # Frontmatter check
    frontmatter, fm_error = parse_frontmatter(content)
    if fm_error:
        result["errors"].append(f"Frontmatter: {fm_error}")
    else:
        if "name" not in frontmatter:
            result["errors"].append("Frontmatter missing required field: name")
        if "description" not in frontmatter:
            result["errors"].append("Frontmatter missing required field: description")
        else:
            desc = str(frontmatter.get("description", "")).lower()
            # Heuristic what/when check
            what_kws = [
                "automate", "manage", "coordinate", "process", "generate",
                "analyse", "analyze", "create", "build", "deploy", "monitor",
                "validate", "transform", "convert", "compile", "execute",
                "run", "scan", "detect", "implement", "orchestrate", "perform",
                "provide", "enable", "support",
            ]
            when_kws = [
                "when", "use for", "use to", "invoke when", "trigger",
                "activate", "suitable for", "designed for", "intended for",
                "use if", "apply when", "reach for",
            ]
            if not any(kw in desc for kw in what_kws):
                result["warnings"].append(
                    "Description may lack a 'what' capability statement"
                )
            if not any(kw in desc for kw in when_kws):
                result["warnings"].append(
                    "Description may lack a 'when' usage condition"
                )

    # Body content (after frontmatter)
    body = strip_frontmatter(content)
    headings = extract_headings(body)

    # Exactly one H1
    h1_count = sum(1 for level, _ in headings if level == 1)
    if h1_count == 0:
        result["errors"].append("No H1 heading found")
    elif h1_count > 1:
        result["warnings"].append(f"Multiple H1 headings found ({h1_count})")

    # Required sections (H2)
    h2_titles = [title.lower() for level, title in headings if level == 2]
    for section in REQUIRED_SECTIONS:
        if not any(section in t for t in h2_titles):
            result["errors"].append(f"Missing required section: ## {section.title()}")

    # Recommended sections
    for section in RECOMMENDED_SECTIONS:
        if not any(section in t for t in h2_titles):
            result["warnings"].append(
                f"Missing recommended section: ## {section.title()}"
            )

    # TODO markers
    prose = strip_code_blocks(body)
    for match in TODO_MARKERS.finditer(prose):
        # Find line number
        line_start = prose.count("\n", 0, match.start()) + 1
        result["errors"].append(
            f"TODO marker found on line ~{line_start}: '{match.group()}'"
        )

    # Also check code blocks for TODOs (as warning, not error)
    for match in TODO_MARKERS.finditer(body):
        if match.start() >= len(prose):
            line_start = body.count("\n", 0, match.start()) + 1
            result["warnings"].append(
                f"TODO marker in code block on line ~{line_start}: '{match.group()}'"
            )

    # UK English spot-check (prose only, skip code blocks)
    # Build a set of words from prose
    prose_words = re.findall(r"\b[a-zA-Z]+\b", prose.lower())
    for us_word, uk_word in US_SPELLINGS.items():
        if us_word in US_SPELLING_EXCEPTIONS:
            continue
        if us_word in prose_words:
            result["warnings"].append(
                f"US English '{us_word}' found in prose; prefer UK '{uk_word}'"
            )

    # Unlanguaged code blocks
    unlanguaged = UNLANGUAGED_CODEBLOCK.findall(content)
    if unlanguaged:
        result["warnings"].append(
            f"{len(unlanguaged)} code block(s) without language specifier"
        )

    return result


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


def find_skills(root: Path) -> list[Path]:
    """Find skill directories under root."""
    root = root.resolve()
    if (root / "SKILL.md").exists():
        return [root]
    skills = []
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                skills.append(entry)
    return skills


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_report(results: list[dict], reports_dir: Path) -> None:
    """Write JSON report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "lint-report.json"
    report = {
        "total": len(results),
        "with_errors": sum(1 for r in results if r["errors"]),
        "with_warnings": sum(1 for r in results if r["warnings"]),
        "clean": sum(
            1 for r in results if not r["errors"] and not r["warnings"]
        ),
        "total_errors": sum(len(r["errors"]) for r in results),
        "total_warnings": sum(len(r["warnings"]) for r in results),
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report written to: {report_path}")


def print_summary(results: list[dict]) -> None:
    """Print human-readable summary."""
    with_errors = [r for r in results if r["errors"]]
    with_warnings = [r for r in results if r["warnings"] and not r["errors"]]
    clean = [
        r for r in results if not r["errors"] and not r["warnings"]
    ]
    total_errors = sum(len(r["errors"]) for r in results)
    total_warnings = sum(len(r["warnings"]) for r in results)

    print("=" * 70)
    print("SKILLS LINT REPORT")
    print("=" * 70)
    print(f"Skills scanned:    {len(results)}")
    print(f"With errors:       {len(with_errors)}")
    print(f"With warnings:     {len(with_warnings)}")
    print(f"Clean:             {len(clean)}")
    print(f"Total errors:      {total_errors}")
    print(f"Total warnings:    {total_warnings}")
    print("=" * 70)

    if with_errors:
        print(f"\nERRORS ({len(with_errors)} skills):")
        print("-" * 70)
        for r in with_errors:
            print(f"\n  {r['skill']}:")
            for err in r["errors"]:
                print(f"    ERROR: {err}")
            for warn in r["warnings"]:
                print(f"    WARNING: {warn}")

    if with_warnings:
        print(f"\nWARNINGS ONLY ({len(with_warnings)} skills):")
        print("-" * 70)
        for r in with_warnings:
            print(f"\n  {r['skill']}:")
            for warn in r["warnings"]:
                print(f"    WARNING: {warn}")

    if clean:
        print(f"\nCLEAN ({len(clean)} skills):")
        print("-" * 70)
        for r in clean:
            print(f"  {r['skill']}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <skills-directory>", file=sys.stderr)
        print(f"Example: {sys.argv[0]} ~/.claude/skills/", file=sys.stderr)
        return 1

    target = Path(sys.argv[1]).expanduser().resolve()
    if not target.exists():
        print(f"Error: path does not exist: {target}", file=sys.stderr)
        return 1

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    for i, arg in enumerate(sys.argv):
        if arg == "--reports-dir" and i + 1 < len(sys.argv):
            reports_dir = Path(sys.argv[i + 1]).resolve()

    skills = find_skills(target)
    if not skills:
        print(f"No skill directories found under: {target}", file=sys.stderr)
        return 1

    results = [lint_skill(s) for s in skills]

    print_summary(results)
    write_report(results, reports_dir)

    return 1 if any(r["errors"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
