#!/usr/bin/env python3
"""Add YAML frontmatter to documentation files.

This script adds standardized frontmatter to markdown files that don't have it.
"""

import re
import sys
from pathlib import Path
from datetime import date
from typing import Optional


FRONTMATTER_TEMPLATE = """---
title: "{title}"
category: "{category}"
status: "{status}"
last_updated: "{date}"
---

"""


def infer_category(file_path: Path, docs_root: Path) -> str:
    """Infer category from file path."""
    relative = file_path.relative_to(docs_root)
    parts = relative.parts

    if len(parts) > 1:
        top_dir = parts[0]
        category_map = {
            'getting-started': 'Getting Started',
            'guides': 'Guide',
            'concepts': 'Concept',
            'reference': 'Reference',
            'architecture': 'Architecture',
            'api': 'API Reference',
            'multi-agent-docker': 'Environment',
        }
        return category_map.get(top_dir, 'Documentation')

    return 'Documentation'


def infer_status(content: str) -> str:
    """Infer status from content."""
    if 'TODO:' in content or 'FIXME:' in content:
        return 'Draft'
    if 'DEPRECATED' in content or 'deprecated' in content.lower():
        return 'Deprecated'
    return 'Complete'


def extract_title(content: str, file_path: Path) -> str:
    """Extract title from first H1 heading or use filename."""
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)

    if title_match:
        return title_match.group(1).strip()

    # Fallback to filename
    return file_path.stem.replace('-', ' ').replace('_', ' ').title()


def add_frontmatter(
    file_path: Path,
    docs_root: Path,
    force: bool = False,
    dry_run: bool = False
) -> bool:
    """Add frontmatter to file if it doesn't have it."""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False

    # Check if already has frontmatter
    if content.startswith('---'):
        if not force:
            return False
        # Remove existing frontmatter if force=True
        parts = content.split('---', 2)
        if len(parts) >= 3:
            content = parts[2].lstrip()

    # Extract metadata
    title = extract_title(content, file_path)
    category = infer_category(file_path, docs_root)
    status = infer_status(content)

    # Generate frontmatter
    frontmatter = FRONTMATTER_TEMPLATE.format(
        title=title,
        category=category,
        status=status,
        date=date.today().isoformat()
    )

    new_content = frontmatter + content

    print(f"{'[DRY RUN] ' if dry_run else ''}Adding frontmatter to {file_path.relative_to(docs_root)}")
    print(f"  Title: {title}")
    print(f"  Category: {category}")
    print(f"  Status: {status}")

    if not dry_run:
        file_path.write_text(new_content, encoding='utf-8')

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Add YAML frontmatter to markdown files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--force', action='store_true', help='Replace existing frontmatter')
    parser.add_argument('--path', default='docs', help='Path to documentation root (default: docs)')
    parser.add_argument('files', nargs='*', help='Specific files to process (optional)')
    args = parser.parse_args()

    docs_root = Path(args.path)

    if not docs_root.exists():
        print(f"Error: Documentation path not found: {docs_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Adding frontmatter to markdown files in {docs_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"Force: {args.force}")
    print()

    # Determine files to process
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(docs_root.rglob('*.md'))

    modified_count = 0
    skipped_count = 0

    for md_file in files:
        if add_frontmatter(md_file, docs_root, args.force, args.dry_run):
            modified_count += 1
        else:
            skipped_count += 1

    print()
    print(f"{'=' * 60}")
    print(f"Total files processed: {len(files)}")
    print(f"Files {'that would be' if args.dry_run else ''} modified: {modified_count}")
    print(f"Files skipped (already have frontmatter): {skipped_count}")

    if args.dry_run and modified_count > 0:
        print()
        print("Run without --dry-run to apply changes")
