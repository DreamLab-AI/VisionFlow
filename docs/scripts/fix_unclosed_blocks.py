#!/usr/bin/env python3
"""Fix unclosed code blocks in documentation.

This script scans all markdown files and automatically closes
any unclosed code blocks by adding a closing ```.
"""

import sys
from pathlib import Path
from typing import List, Tuple


def check_and_fix_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Check if file has unclosed code blocks and optionally fix them.

    Returns:
        (was_fixed, block_count): Whether file was modified and number of blocks
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False, 0

    block_count = content.count('```')

    if block_count % 2 != 0:
        # Unclosed block found
        print(f"{'[DRY RUN] ' if dry_run else ''}Fixing {file_path.relative_to(docs_root)}")

        if not dry_run:
            # Add closing block at end
            content = content.rstrip() + '\n```\n'
            file_path.write_text(content, encoding='utf-8')

        return True, block_count

    return False, block_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fix unclosed code blocks in markdown files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    parser.add_argument('--path', default='docs', help='Path to documentation root (default: docs)')
    args = parser.parse_args()

    docs_root = Path(args.path)

    if not docs_root.exists():
        print(f"Error: Documentation path not found: {docs_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {docs_root} for unclosed code blocks...")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FIX'}")
    print()

    fixed_files: List[Path] = []
    total_blocks = 0

    for md_file in sorted(docs_root.rglob('*.md')):
        was_fixed, block_count = check_and_fix_file(md_file, args.dry_run)
        total_blocks += block_count

        if was_fixed:
            fixed_files.append(md_file)

    print()
    print(f"{'=' * 60}")
    print(f"Total markdown files scanned: {len(list(docs_root.rglob('*.md')))}")
    print(f"Total code blocks found: {total_blocks}")
    print(f"Files {'that would be' if args.dry_run else ''} fixed: {len(fixed_files)}")

    if fixed_files:
        print()
        print("Fixed files:")
        for f in fixed_files:
            print(f"  - {f.relative_to(docs_root)}")

    if args.dry_run and fixed_files:
        print()
        print("Run without --dry-run to apply fixes")

    sys.exit(0 if not fixed_files else 1)
