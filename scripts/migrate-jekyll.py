#!/usr/bin/env python3
"""
Jekyll Documentation Migration Script
Migrates markdown files to Jekyll-compatible frontmatter format.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

DOCS_ROOT = Path("/home/devuser/workspace/project/docs")

# Navigation order mappings
NAV_ORDER = {
    # Explanations
    "explanations/system-overview.md": 1,
    "explanations/architecture/README.md": 1,
    "explanations/architecture/hexagonal-cqrs.md": 2,
    "explanations/architecture/data-flow-complete.md": 3,
    "explanations/architecture/services-layer.md": 4,
    "explanations/architecture/services-architecture.md": 5,
    "explanations/architecture/adapter-patterns.md": 6,
    "explanations/architecture/integration-patterns.md": 7,
    "explanations/architecture/event-driven-architecture.md": 8,
    "explanations/architecture/database-architecture.md": 9,
    "explanations/architecture/semantic-physics-system.md": 10,
    "explanations/architecture/semantic-physics.md": 11,
    "explanations/architecture/semantic-forces-system.md": 12,
    "explanations/architecture/gpu-semantic-forces.md": 13,
    "explanations/architecture/stress-majorization.md": 14,
    "explanations/architecture/ontology-storage-architecture.md": 15,
    "explanations/architecture/ontology-reasoning-pipeline.md": 16,
    "explanations/architecture/ontology-physics-integration.md": 17,
    "explanations/architecture/ontology-analysis.md": 18,
    "explanations/architecture/pipeline-integration.md": 19,
    "explanations/architecture/pipeline-sequence-diagrams.md": 20,
    "explanations/architecture/reasoning-data-flow.md": 21,
    "explanations/architecture/reasoning-tests-summary.md": 22,
    "explanations/architecture/hierarchical-visualization.md": 23,
    "explanations/architecture/analytics-visualization.md": 24,
    "explanations/architecture/xr-immersive-system.md": 25,
    "explanations/architecture/multi-agent-system.md": 26,
    "explanations/architecture/ruvector-integration.md": 27,
    "explanations/architecture/github-sync-service-design.md": 28,
    "explanations/architecture/api-handlers-reference.md": 29,
    "explanations/architecture/cqrs-directive-template.md": 30,
    "explanations/architecture/quick-reference.md": 31,
    # Reference
    "reference/INDEX.md": 1,
    "reference/API_REFERENCE.md": 2,
    "reference/CONFIGURATION_REFERENCE.md": 3,
    "reference/DATABASE_SCHEMA_REFERENCE.md": 4,
    "reference/ERROR_REFERENCE.md": 5,
    "reference/PROTOCOL_REFERENCE.md": 6,
    "reference/README.md": 7,
    "reference/api-complete-reference.md": 8,
    "reference/code-quality-status.md": 9,
    "reference/error-codes.md": 10,
    "reference/implementation-status.md": 11,
    "reference/performance-benchmarks.md": 12,
    "reference/physics-implementation.md": 13,
    "reference/websocket-protocol.md": 14,
}

def get_parent_info(file_path: Path) -> Tuple[str, Optional[str]]:
    """Determine parent and grand_parent based on file location."""
    rel_path = file_path.relative_to(DOCS_ROOT)
    parts = rel_path.parts

    if parts[0] == "explanations":
        if len(parts) == 2:  # Direct child of explanations
            return "Explanations", None
        elif len(parts) == 3:  # Child of a subcategory
            subcategory = parts[1].replace("-", " ").title()
            return subcategory, "Explanations"
        elif len(parts) >= 4:  # Deeper nested
            parent = parts[-2].replace("-", " ").title()
            grand_parent = parts[1].replace("-", " ").title()
            return parent, grand_parent
    elif parts[0] == "reference":
        if len(parts) == 2:  # Direct child of reference
            return "Reference", None
        elif len(parts) >= 3:  # Child of a subcategory
            subcategory = parts[1].upper() if parts[1] in ["api"] else parts[1].replace("-", " ").title()
            return subcategory, "Reference"

    return "Documentation", None

def get_nav_order(file_path: Path) -> int:
    """Get navigation order for a file."""
    rel_path = str(file_path.relative_to(DOCS_ROOT))
    return NAV_ORDER.get(rel_path, 99)

def extract_title_from_content(content: str) -> Optional[str]:
    """Extract title from first H1 heading in content."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None

def parse_existing_frontmatter(content: str) -> Tuple[Dict, str]:
    """Parse existing YAML frontmatter and return dict + remaining content."""
    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    end_match = re.search(r'^---\s*$', content[3:], re.MULTILINE)
    if not end_match:
        return {}, content

    end_pos = end_match.start() + 3
    frontmatter_text = content[3:end_pos].strip()
    remaining_content = content[end_pos + 4:].lstrip()  # Skip closing ---

    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, remaining_content

def generate_jekyll_frontmatter(file_path: Path, existing_fm: Dict, content: str) -> str:
    """Generate Jekyll-compatible frontmatter."""
    parent, grand_parent = get_parent_info(file_path)
    nav_order = get_nav_order(file_path)

    # Get title from existing frontmatter or extract from content
    title = existing_fm.get("title", extract_title_from_content(content) or file_path.stem.replace("-", " ").title())

    # Build new frontmatter
    fm = {
        "layout": "default",
        "title": title,
        "parent": parent,
    }

    if grand_parent:
        fm["grand_parent"] = grand_parent

    fm["nav_order"] = nav_order

    # Generate YAML
    yaml_lines = ["---"]
    yaml_lines.append(f"layout: default")
    yaml_lines.append(f"title: \"{title}\"")
    yaml_lines.append(f"parent: {parent}")
    if grand_parent:
        yaml_lines.append(f"grand_parent: {grand_parent}")
    yaml_lines.append(f"nav_order: {nav_order}")
    yaml_lines.append("---")

    return "\n".join(yaml_lines)

def migrate_file(file_path: Path) -> bool:
    """Migrate a single file to Jekyll format."""
    try:
        content = file_path.read_text(encoding="utf-8")
        existing_fm, body = parse_existing_frontmatter(content)

        new_frontmatter = generate_jekyll_frontmatter(file_path, existing_fm, body)
        new_content = f"{new_frontmatter}\n\n{body}"

        file_path.write_text(new_content, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error migrating {file_path}: {e}")
        return False

def main():
    """Main migration function."""
    # Find all markdown files
    explanations_files = list(DOCS_ROOT.glob("explanations/**/*.md"))
    reference_files = list(DOCS_ROOT.glob("reference/**/*.md"))

    # Exclude index files we just created
    all_files = [f for f in explanations_files + reference_files
                 if f.name != "index.md"]

    print(f"Found {len(all_files)} files to migrate")

    migrated = 0
    failed = 0

    for file_path in sorted(all_files):
        rel_path = file_path.relative_to(DOCS_ROOT)
        if migrate_file(file_path):
            print(f"  [OK] {rel_path}")
            migrated += 1
        else:
            print(f"  [FAIL] {rel_path}")
            failed += 1

    print(f"\nMigration complete: {migrated} success, {failed} failed")

if __name__ == "__main__":
    main()
