#!/usr/bin/env python3
"""Validate internal links in skill markdown files.

Usage:
    python3 check-links.py <skills-directory>
    python3 check-links.py ~/.claude/skills/
    python3 check-links.py ~/.claude/skills/browser/

Finds all internal references (markdown links, cross-skill references,
anchor links, image references) and verifies they resolve to existing
files or headings.

Produces:
    - stdout: human-readable summary
    - reports/links-report.json: machine-readable results
"""

import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Link extraction patterns
# ---------------------------------------------------------------------------

# Standard markdown links: [text](path) and [text](path#anchor)
MARKDOWN_LINK_RE = re.compile(
    r"\[([^\]]*)\]\(([^)]+)\)"
)

# Image references: ![alt](path)
IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)]+)\)"
)


# ---------------------------------------------------------------------------
# Heading anchor generation
# ---------------------------------------------------------------------------


def heading_to_anchor(heading: str) -> str:
    """Convert a markdown heading to its GitHub-style anchor.

    Rules:
    - Lowercase
    - Replace spaces with hyphens
    - Remove non-alphanumeric characters except hyphens
    - Collapse multiple hyphens
    """
    anchor = heading.lower().strip()
    # Remove markdown formatting
    anchor = re.sub(r"[*_`]", "", anchor)
    # Replace spaces with hyphens
    anchor = re.sub(r"\s+", "-", anchor)
    # Remove non-alphanumeric except hyphens
    anchor = re.sub(r"[^a-z0-9-]", "", anchor)
    # Collapse hyphens
    anchor = re.sub(r"-+", "-", anchor)
    anchor = anchor.strip("-")
    return anchor


def extract_headings_from_file(filepath: Path) -> set[str]:
    """Extract all heading anchors from a markdown file."""
    if not filepath.exists():
        return set()
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return set()

    anchors = set()
    in_code = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if match:
            heading_text = match.group(1).strip()
            anchors.add(heading_to_anchor(heading_text))
    return anchors


# ---------------------------------------------------------------------------
# Link validation
# ---------------------------------------------------------------------------


def is_external_url(href: str) -> bool:
    """Check if a link is an external URL."""
    return href.startswith("http://") or href.startswith("https://") or href.startswith("mailto:")


def validate_links_in_file(
    filepath: Path,
    skills_root: Path,
) -> list[dict[str, Any]]:
    """Validate all internal links in a single markdown file.

    Returns a list of issue dicts.
    """
    issues: list[dict[str, Any]] = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as exc:
        issues.append({
            "file": str(filepath),
            "line": 0,
            "link": "(file)",
            "type": "read_error",
            "severity": "error",
            "message": f"Could not read file: {exc}",
        })
        return issues

    lines = content.splitlines()
    in_code = False

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        # Find all markdown links on this line
        for match in MARKDOWN_LINK_RE.finditer(line):
            link_text = match.group(1)
            href = match.group(2).strip()

            # Skip external URLs
            if is_external_url(href):
                continue

            # Skip empty links
            if not href:
                issues.append({
                    "file": str(filepath),
                    "line": line_num,
                    "link": f"[{link_text}]()",
                    "type": "empty_link",
                    "severity": "error",
                    "message": "Empty link target",
                })
                continue

            # Parse path and anchor
            if "#" in href:
                path_part, anchor_part = href.split("#", 1)
            else:
                path_part = href
                anchor_part = None

            # Resolve the file path
            if path_part:
                # Relative to the file's directory
                target_path = (filepath.parent / path_part).resolve()

                if not target_path.exists():
                    issues.append({
                        "file": str(filepath),
                        "line": line_num,
                        "link": href,
                        "type": "broken_path",
                        "severity": "error",
                        "message": f"File not found: {target_path}",
                    })
                    continue

                # If there's an anchor, check it against the target file
                if anchor_part and target_path.suffix in (".md", ".markdown"):
                    target_anchors = extract_headings_from_file(target_path)
                    if anchor_part not in target_anchors:
                        issues.append({
                            "file": str(filepath),
                            "line": line_num,
                            "link": href,
                            "type": "broken_anchor",
                            "severity": "warning",
                            "message": (
                                f"Anchor '#{anchor_part}' not found in "
                                f"{target_path.name}. Available: "
                                f"{', '.join(sorted(target_anchors)[:10]) or '(none)'}"
                            ),
                        })
            elif anchor_part:
                # Anchor-only link (#section) -- check within the same file
                self_anchors = extract_headings_from_file(filepath)
                if anchor_part not in self_anchors:
                    issues.append({
                        "file": str(filepath),
                        "line": line_num,
                        "link": href,
                        "type": "broken_anchor",
                        "severity": "warning",
                        "message": (
                            f"Anchor '#{anchor_part}' not found in current file. "
                            f"Available: {', '.join(sorted(self_anchors)[:10]) or '(none)'}"
                        ),
                    })

    return issues


def validate_frontmatter_refs(
    skill_dir: Path,
    skills_root: Path,
) -> list[dict[str, Any]]:
    """Validate depends_on and supersedes references in frontmatter."""
    issues: list[dict[str, Any]] = []
    skill_file = skill_dir / "SKILL.md"

    if not skill_file.exists():
        return issues

    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception:
        return issues

    # Parse frontmatter
    content_stripped = content.strip()
    if not content_stripped.startswith("---"):
        return issues
    end_idx = content_stripped.find("---", 3)
    if end_idx == -1:
        return issues
    yaml_block = content_stripped[3:end_idx].strip()

    try:
        import yaml
        fm = yaml.safe_load(yaml_block)
        if not isinstance(fm, dict):
            return issues
    except ImportError:
        # Basic parsing fallback
        fm = {}
        for line in yaml_block.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                value = [item.strip().strip("\"'") for item in items if item.strip()]
            fm[key] = value
    except Exception:
        return issues

    # Check depends_on
    depends = fm.get("depends_on", [])
    if isinstance(depends, list):
        for dep in depends:
            dep_str = str(dep)
            dep_path = skills_root / dep_str
            if not dep_path.exists():
                issues.append({
                    "file": str(skill_file),
                    "line": 0,
                    "link": f"depends_on: {dep_str}",
                    "type": "broken_dependency",
                    "severity": "warning",
                    "message": f"depends_on skill not found: {dep_str}",
                })

    # Check supersedes
    supersedes = fm.get("supersedes", [])
    if isinstance(supersedes, list):
        for sup in supersedes:
            sup_str = str(sup)
            sup_path = skills_root / sup_str
            # Superseded skills might have been removed, so this is info, not error
            if not sup_path.exists():
                issues.append({
                    "file": str(skill_file),
                    "line": 0,
                    "link": f"supersedes: {sup_str}",
                    "type": "missing_superseded",
                    "severity": "info",
                    "message": f"Superseded skill not found (may have been removed): {sup_str}",
                })

    return issues


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


def find_markdown_files(skill_dir: Path) -> list[Path]:
    """Find all markdown files in a skill directory."""
    md_files = []
    for f in skill_dir.rglob("*.md"):
        # Skip hidden directories
        if any(part.startswith(".") for part in f.relative_to(skill_dir).parts):
            continue
        md_files.append(f)
    return sorted(md_files)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_report(
    all_issues: dict[str, list[dict]],
    skill_count: int,
    reports_dir: Path,
) -> None:
    """Write JSON report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "links-report.json"

    total_errors = 0
    total_warnings = 0
    total_info = 0
    for issues in all_issues.values():
        for issue in issues:
            if issue["severity"] == "error":
                total_errors += 1
            elif issue["severity"] == "warning":
                total_warnings += 1
            else:
                total_info += 1

    report = {
        "skills_analysed": skill_count,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "total_info": total_info,
        "skills": {
            name: issues for name, issues in all_issues.items() if issues
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report written to: {report_path}")


def print_summary(
    all_issues: dict[str, list[dict]],
    skill_count: int,
) -> None:
    """Print human-readable summary."""
    total_errors = 0
    total_warnings = 0
    total_info = 0
    skills_with_issues = 0

    for issues in all_issues.values():
        has_issues = False
        for issue in issues:
            has_issues = True
            if issue["severity"] == "error":
                total_errors += 1
            elif issue["severity"] == "warning":
                total_warnings += 1
            else:
                total_info += 1
        if has_issues:
            skills_with_issues += 1

    print("=" * 70)
    print("LINK VALIDATION REPORT")
    print("=" * 70)
    print(f"Skills analysed:     {skill_count}")
    print(f"Skills with issues:  {skills_with_issues}")
    print(f"Total errors:        {total_errors}")
    print(f"Total warnings:      {total_warnings}")
    print(f"Total info:          {total_info}")
    print("=" * 70)

    if total_errors == 0 and total_warnings == 0 and total_info == 0:
        print("\nAll links validated successfully.")
        print()
        return

    # Group by severity
    for severity, label in [("error", "ERRORS"), ("warning", "WARNINGS"), ("info", "INFO")]:
        severity_issues = {
            name: [i for i in issues if i["severity"] == severity]
            for name, issues in all_issues.items()
        }
        severity_issues = {k: v for k, v in severity_issues.items() if v}

        if not severity_issues:
            continue

        total = sum(len(v) for v in severity_issues.values())
        print(f"\n{label} ({total}):")
        print("-" * 70)
        for name, issues in sorted(severity_issues.items()):
            print(f"\n  {name}:")
            for issue in issues:
                loc = f"line {issue['line']}" if issue["line"] else "frontmatter"
                print(f"    [{loc}] {issue['link']}")
                print(f"      {issue['message']}")

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

    # Determine the skills root (for cross-skill reference checking)
    if (target / "SKILL.md").exists():
        skills_root = target.parent
    else:
        skills_root = target

    all_issues: dict[str, list[dict]] = {}

    for skill_dir in skills:
        skill_name = skill_dir.name
        issues: list[dict] = []

        # Check links in all markdown files
        md_files = find_markdown_files(skill_dir)
        for md_file in md_files:
            issues.extend(validate_links_in_file(md_file, skills_root))

        # Check frontmatter cross-references
        issues.extend(validate_frontmatter_refs(skill_dir, skills_root))

        all_issues[skill_name] = issues

    print_summary(all_issues, len(skills))
    write_report(all_issues, len(skills), reports_dir)

    # Exit with error if any errors found
    has_errors = any(
        any(i["severity"] == "error" for i in issues)
        for issues in all_issues.values()
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
