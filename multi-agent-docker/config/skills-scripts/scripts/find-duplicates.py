#!/usr/bin/env python3
"""Detect duplicate or highly similar skills by comparing text content.

Usage:
    python3 find-duplicates.py <skills-directory>
    python3 find-duplicates.py ~/.claude/skills/

Reads all SKILL.md files, strips markdown formatting, computes pairwise
Jaccard similarity on 3-gram shingles, and flags pairs exceeding 30%
similarity. Groups overlapping skills into clusters.

Produces:
    - stdout: human-readable summary
    - reports/duplicates-report.json: machine-readable results
"""

import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.30  # 30%
SHINGLE_SIZE = 3


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter."""
    text = text.strip()
    if not text.startswith("---"):
        return text
    end_idx = text.find("---", 3)
    if end_idx == -1:
        return text
    return text[end_idx + 3:].strip()


def strip_markdown(text: str) -> str:
    """Strip markdown formatting to get plain text content."""
    # Remove code blocks entirely
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove markdown headings markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove markdown links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    # Remove bold/italic markers
    text = re.sub(r"[*_]{1,3}", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove table formatting
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"-{3,}", "", text)
    # Remove blockquotes
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def compute_shingles(text: str, n: int = SHINGLE_SIZE) -> set[str]:
    """Compute n-gram character shingles from text."""
    words = text.split()
    if len(words) < n:
        return {text} if text else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def build_clusters(
    pairs: list[tuple[str, str, float]],
) -> list[list[str]]:
    """Build connected-component clusters from similarity pairs.

    Uses union-find for efficiency.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, _ in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    # Group by root
    groups: dict[str, list[str]] = {}
    for node in parent:
        root = find(node)
        groups.setdefault(root, []).append(node)

    return [sorted(members) for members in groups.values() if len(members) > 1]


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


def load_skill_text(skill_dir: Path) -> str | None:
    """Load and preprocess SKILL.md text."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None
    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception:
        return None
    content = strip_frontmatter(content)
    content = strip_markdown(content)
    return content


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_report(
    pairs: list[tuple[str, str, float]],
    clusters: list[list[str]],
    skill_count: int,
    reports_dir: Path,
) -> None:
    """Write JSON report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "duplicates-report.json"

    report = {
        "skills_analysed": skill_count,
        "threshold": SIMILARITY_THRESHOLD,
        "shingle_size": SHINGLE_SIZE,
        "flagged_pairs": len(pairs),
        "clusters": len(clusters),
        "pairs": [
            {"skill_a": a, "skill_b": b, "similarity": round(s, 4)}
            for a, b, s in pairs
        ],
        "clusters": [
            {"members": members, "size": len(members)}
            for members in clusters
        ],
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report written to: {report_path}")


def print_summary(
    pairs: list[tuple[str, str, float]],
    clusters: list[list[str]],
    skill_count: int,
) -> None:
    """Print human-readable summary."""
    print("=" * 70)
    print("DUPLICATE DETECTION REPORT")
    print("=" * 70)
    print(f"Skills analysed:      {skill_count}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD:.0%}")
    print(f"Shingle size:         {SHINGLE_SIZE}-gram")
    print(f"Flagged pairs:        {len(pairs)}")
    print(f"Clusters:             {len(clusters)}")
    print("=" * 70)

    if not pairs:
        print("\nNo duplicates found above the threshold.")
        print()
        return

    # Sort pairs by similarity descending
    sorted_pairs = sorted(pairs, key=lambda p: p[2], reverse=True)

    print("\nFLAGGED PAIRS (sorted by similarity):")
    print("-" * 70)
    for a, b, sim in sorted_pairs:
        bar_len = int(sim * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"  {sim:5.1%}  [{bar}]  {a} <-> {b}")

    if clusters:
        print(f"\nCLUSTERS ({len(clusters)}):")
        print("-" * 70)
        for i, members in enumerate(clusters, 1):
            print(f"  Cluster {i}: {', '.join(members)}")

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

    # Load and shingle all skills
    skill_data: dict[str, set[str]] = {}
    skipped = 0
    for skill_dir in skills:
        text = load_skill_text(skill_dir)
        if text is None:
            skipped += 1
            continue
        shingles = compute_shingles(text)
        if shingles:
            skill_data[skill_dir.name] = shingles

    if skipped:
        print(f"Skipped {skipped} skill(s) without readable SKILL.md\n")

    # Compute pairwise similarity
    names = sorted(skill_data.keys())
    flagged_pairs: list[tuple[str, str, float]] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = jaccard_similarity(skill_data[names[i]], skill_data[names[j]])
            if sim >= SIMILARITY_THRESHOLD:
                flagged_pairs.append((names[i], names[j], sim))

    # Build clusters
    clusters = build_clusters(flagged_pairs)

    print_summary(flagged_pairs, clusters, len(skill_data))
    write_report(flagged_pairs, clusters, len(skill_data), reports_dir)

    return 1 if flagged_pairs else 0


if __name__ == "__main__":
    sys.exit(main())
