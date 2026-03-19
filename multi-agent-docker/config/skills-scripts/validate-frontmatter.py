#!/usr/bin/env python3
"""Validate YAML frontmatter in SKILL.md files against the canonical schema.

Usage:
    python3 validate-frontmatter.py <skills-directory>
    python3 validate-frontmatter.py ~/.claude/skills/
    python3 validate-frontmatter.py ~/.claude/skills/browser/

Checks every skill's frontmatter against the schema defined in
references/frontmatter-schema.md. Outputs pass/fail per skill with
specific violations.

Produces:
    - stdout: human-readable summary
    - reports/frontmatter-report.json: machine-readable results
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"name", "description"}

OPTIONAL_FIELDS = {
    "version",
    "status",
    "upstream",
    "tags",
    "depends_on",
    "supersedes",
}

DISALLOWED_FIELDS = {
    "triggers",
    "tools",
    "capabilities",
    "author",
    "category",
    "requires",
}

VALID_STATUSES = {"active", "deprecated", "experimental"}
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
URL_RE = re.compile(r"^https?://")
KEBAB_RE = re.compile(r"^[a-z0-9-]+$")
NAME_MAX_LEN = 64
DESC_MAX_LEN = 1024
TAGS_MAX_COUNT = 10

# Heuristic keywords that suggest the description says "when" to use it
WHEN_KEYWORDS = [
    "when",
    "use for",
    "use to",
    "invoke when",
    "trigger",
    "activate",
    "suitable for",
    "designed for",
    "intended for",
    "use if",
    "apply when",
    "reach for",
]

# Heuristic keywords that suggest the description says "what" it does
WHAT_KEYWORDS = [
    "automate",
    "manage",
    "coordinate",
    "process",
    "generate",
    "analyse",
    "analyze",
    "create",
    "build",
    "deploy",
    "monitor",
    "validate",
    "transform",
    "convert",
    "compile",
    "execute",
    "run",
    "scan",
    "detect",
    "implement",
    "orchestrate",
    "perform",
    "provide",
    "enable",
    "support",
]


# ---------------------------------------------------------------------------
# YAML parsing (no external dependency)
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract YAML frontmatter from markdown text.

    Returns (parsed_dict, error_message). If parsing fails, parsed_dict is
    None and error_message explains why.
    """
    text = text.strip()
    if not text.startswith("---"):
        return None, "File does not begin with YAML frontmatter delimiter (---)"

    # Find closing delimiter
    end_idx = text.find("---", 3)
    if end_idx == -1:
        return None, "No closing YAML frontmatter delimiter (---) found"

    yaml_block = text[3:end_idx].strip()
    if not yaml_block:
        return None, "Frontmatter block is empty"

    # Use PyYAML if available, otherwise fall back to basic parsing
    try:
        import yaml

        parsed = yaml.safe_load(yaml_block)
        if not isinstance(parsed, dict):
            return None, f"Frontmatter parsed as {type(parsed).__name__}, expected dict"
        return parsed, None
    except ImportError:
        pass
    except Exception as exc:
        return None, f"YAML parse error: {exc}"

    # Fallback: simple key-value parser for basic YAML
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

        # Handle quoted strings
        if value and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]

        # Handle lists (basic)
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            value = [item.strip().strip("\"'") for item in items if item.strip()]

        result[key] = value

    return result, None


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def validate_skill(skill_dir: Path) -> dict[str, Any]:
    """Validate a single skill directory. Returns a result dict."""
    skill_file = skill_dir / "SKILL.md"
    result = {
        "skill": skill_dir.name,
        "path": str(skill_file),
        "passed": True,
        "errors": [],
        "warnings": [],
    }

    if not skill_file.exists():
        result["passed"] = False
        result["errors"].append("SKILL.md not found")
        return result

    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception as exc:
        result["passed"] = False
        result["errors"].append(f"Could not read SKILL.md: {exc}")
        return result

    frontmatter, parse_error = parse_frontmatter(content)
    if parse_error:
        result["passed"] = False
        result["errors"].append(parse_error)
        return result

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in frontmatter:
            result["passed"] = False
            result["errors"].append(f"Missing required field: {field}")

    # Check disallowed fields
    for field in DISALLOWED_FIELDS:
        if field in frontmatter:
            result["warnings"].append(
                f"Disallowed field present: {field} (migrate per schema)"
            )

    # Check unknown fields
    known_fields = REQUIRED_FIELDS | OPTIONAL_FIELDS | DISALLOWED_FIELDS
    for field in frontmatter:
        if field not in known_fields:
            result["warnings"].append(f"Unknown field: {field}")

    # Validate 'name'
    name = frontmatter.get("name")
    if name is not None:
        if not isinstance(name, str):
            result["passed"] = False
            result["errors"].append(f"'name' must be a string, got {type(name).__name__}")
        elif len(name) > NAME_MAX_LEN:
            result["passed"] = False
            result["errors"].append(
                f"'name' exceeds {NAME_MAX_LEN} chars ({len(name)} chars)"
            )
        elif len(name) == 0:
            result["passed"] = False
            result["errors"].append("'name' is empty")

    # Validate 'description'
    desc = frontmatter.get("description")
    if desc is not None:
        if not isinstance(desc, str):
            result["passed"] = False
            result["errors"].append(
                f"'description' must be a string, got {type(desc).__name__}"
            )
        elif len(desc) > DESC_MAX_LEN:
            result["passed"] = False
            result["errors"].append(
                f"'description' exceeds {DESC_MAX_LEN} chars ({len(desc)} chars)"
            )
        elif len(desc) == 0:
            result["passed"] = False
            result["errors"].append("'description' is empty")
        else:
            desc_lower = desc.lower()
            has_what = any(kw in desc_lower for kw in WHAT_KEYWORDS)
            has_when = any(kw in desc_lower for kw in WHEN_KEYWORDS)

            if not has_what:
                result["warnings"].append(
                    "'description' may lack a 'what' capability statement"
                )
            if not has_when:
                result["warnings"].append(
                    "'description' may lack a 'when' usage condition"
                )

    # Validate 'version'
    version = frontmatter.get("version")
    if version is not None:
        version_str = str(version)
        if not SEMVER_RE.match(version_str):
            result["passed"] = False
            result["errors"].append(
                f"'version' is not valid semver: '{version_str}'"
            )

    # Validate 'status'
    status = frontmatter.get("status")
    if status is not None:
        if not isinstance(status, str) or status not in VALID_STATUSES:
            result["passed"] = False
            result["errors"].append(
                f"'status' must be one of {VALID_STATUSES}, got '{status}'"
            )

    # Validate 'upstream'
    upstream = frontmatter.get("upstream")
    if upstream is not None:
        if not isinstance(upstream, str) or not URL_RE.match(upstream):
            result["passed"] = False
            result["errors"].append(
                f"'upstream' must be a valid URL starting with http(s)://, got '{upstream}'"
            )

    # Validate 'tags'
    tags = frontmatter.get("tags")
    if tags is not None:
        if not isinstance(tags, list):
            result["passed"] = False
            result["errors"].append(
                f"'tags' must be a list, got {type(tags).__name__}"
            )
        else:
            if len(tags) > TAGS_MAX_COUNT:
                result["passed"] = False
                result["errors"].append(
                    f"'tags' has {len(tags)} items, max is {TAGS_MAX_COUNT}"
                )
            for tag in tags:
                tag_str = str(tag)
                if not KEBAB_RE.match(tag_str):
                    result["warnings"].append(
                        f"Tag '{tag_str}' is not lowercase kebab-case"
                    )

    # Validate 'depends_on'
    depends = frontmatter.get("depends_on")
    if depends is not None:
        if not isinstance(depends, list):
            result["passed"] = False
            result["errors"].append(
                f"'depends_on' must be a list, got {type(depends).__name__}"
            )
        else:
            for dep in depends:
                dep_str = str(dep)
                if not KEBAB_RE.match(dep_str):
                    result["warnings"].append(
                        f"depends_on entry '{dep_str}' is not lowercase kebab-case"
                    )

    # Validate 'supersedes'
    supersedes = frontmatter.get("supersedes")
    if supersedes is not None:
        if not isinstance(supersedes, list):
            result["passed"] = False
            result["errors"].append(
                f"'supersedes' must be a list, got {type(supersedes).__name__}"
            )
        else:
            for sup in supersedes:
                sup_str = str(sup)
                if not KEBAB_RE.match(sup_str):
                    result["warnings"].append(
                        f"supersedes entry '{sup_str}' is not lowercase kebab-case"
                    )

    return result


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


def find_skills(root: Path) -> list[Path]:
    """Find skill directories under root.

    A skill directory is one that contains SKILL.md, or (for tolerance)
    any immediate subdirectory of root.
    """
    root = root.resolve()

    # If root itself contains SKILL.md, it is a single skill
    if (root / "SKILL.md").exists():
        return [root]

    # Otherwise, scan subdirectories
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
    """Write JSON report to the reports directory."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "frontmatter-report.json"
    report = {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report written to: {report_path}")


def print_summary(results: list[dict]) -> None:
    """Print human-readable summary to stdout."""
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    total_errors = sum(len(r["errors"]) for r in results)
    total_warnings = sum(len(r["warnings"]) for r in results)

    print("=" * 70)
    print("FRONTMATTER VALIDATION REPORT")
    print("=" * 70)
    print(f"Skills scanned: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    print("=" * 70)

    # Show failures first
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        print("-" * 70)
        for r in failures:
            print(f"\n  {r['skill']}:")
            for err in r["errors"]:
                print(f"    ERROR: {err}")
            for warn in r["warnings"]:
                print(f"    WARNING: {warn}")

    # Show warnings for passing skills
    warned = [r for r in results if r["passed"] and r["warnings"]]
    if warned:
        print(f"\nPASSED WITH WARNINGS ({len(warned)}):")
        print("-" * 70)
        for r in warned:
            print(f"\n  {r['skill']}:")
            for warn in r["warnings"]:
                print(f"    WARNING: {warn}")

    # Show clean passes
    clean = [r for r in results if r["passed"] and not r["warnings"]]
    if clean:
        print(f"\nPASSED CLEAN ({len(clean)}):")
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

    # Determine reports directory
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    for i, arg in enumerate(sys.argv):
        if arg == "--reports-dir" and i + 1 < len(sys.argv):
            reports_dir = Path(sys.argv[i + 1]).resolve()

    skills = find_skills(target)
    if not skills:
        print(f"No skill directories found under: {target}", file=sys.stderr)
        return 1

    results = [validate_skill(s) for s in skills]

    print_summary(results)
    write_report(results, reports_dir)

    # Exit with error if any skill failed
    return 1 if any(not r["passed"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
