#!/usr/bin/env python3
"""
Remove the duplicated metaverse Current Landscape section from 355 files.
This is a critical copy-paste error affecting 20% of the corpus.
"""

import re
from pathlib import Path
import json
import sys

# The exact duplicated content signature
DUPLICATE_SIGNATURE = "metaverse platforms continue to evolve"


def has_duplicate_landscape(content: str) -> bool:
    """Check if file has the duplicated metaverse landscape section."""
    # Look for Current Landscape section with the metaverse signature
    match = re.search(
        r'##\s+Current Landscape.*?\n(.*?)(?=\n##\s+|\Z)',
        content,
        re.DOTALL | re.IGNORECASE
    )

    if match:
        section = match.group(1).lower()
        return DUPLICATE_SIGNATURE in section

    return False


def remove_duplicate_landscape(file_path: Path, dry_run: bool = True) -> bool:
    """Remove the duplicated Current Landscape section."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content

        if not has_duplicate_landscape(content):
            return False

        # Remove the entire Current Landscape section
        # Match ## Current Landscape through to next ## heading or end of file
        content = re.sub(
            r'##\s+Current Landscape\s*\(2025\)?\s*\n+.*?(?=\n##\s+[A-Z]|\Z)',
            '\n',
            content,
            count=1,
            flags=re.DOTALL
        )

        # Clean up excessive whitespace
        content = re.sub(r'\n{4,}', '\n\n\n', content)

        if content != original:
            if not dry_run:
                file_path.write_text(content, encoding='utf-8')
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Remove duplicate metaverse landscape sections')
    parser.add_argument('--apply', action='store_true', help='Apply fixes (default: dry run)')
    parser.add_argument('--report', help='Input report JSON with file list')

    args = parser.parse_args()

    # Load the report to get the list of files
    if args.report:
        report_path = Path(args.report)
        if not report_path.exists():
            print(f"Error: Report file not found: {report_path}", file=sys.stderr)
            sys.exit(1)

        report = json.loads(report_path.read_text())
        # Get files from the duplicate group
        files_to_fix = []
        for group_files in report.get('duplicate_groups', {}).values():
            files_to_fix.extend(group_files)

        print(f"Loaded {len(files_to_fix)} files from report")
    else:
        # Scan all files
        pages_dir = Path(__file__).parent.parent.parent / "mainKnowledgeGraph" / "pages"
        all_files = list(pages_dir.rglob("*.md"))

        files_to_fix = []
        for file_path in all_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                if has_duplicate_landscape(content):
                    files_to_fix.append(str(file_path.relative_to(pages_dir.parent.parent)))
            except Exception as e:
                print(f"Error scanning {file_path.name}: {e}", file=sys.stderr)

        print(f"Found {len(files_to_fix)} files with duplicate landscape")

    if not files_to_fix:
        print("No files to fix")
        return

    # Process files
    base_path = Path(__file__).parent.parent.parent
    fixed_count = 0

    for filepath in files_to_fix:
        file_path = base_path / filepath
        if file_path.exists():
            if remove_duplicate_landscape(file_path, dry_run=not args.apply):
                fixed_count += 1
                if args.apply:
                    print(f"✓ Fixed: {file_path.name}")

    if args.apply:
        print(f"\n✓ Removed duplicate landscape from {fixed_count} files")
    else:
        print(f"\nDry run: Would fix {fixed_count} files")
        print("Use --apply to make changes")


if __name__ == "__main__":
    main()
