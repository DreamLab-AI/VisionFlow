#!/usr/bin/env python3
"""
Fix invalid mermaid diagrams in documentation.

Fixes:
1. Replace <br> with <br/> in Note statements (XHTML compliance)
2. Fix unclosed brackets in erDiagram field definitions
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Base paths
PROJECT_ROOT = Path('/home/devuser/workspace/project')
DOCS_ROOT = PROJECT_ROOT / 'docs'
REPORT_FILE = PROJECT_ROOT / '.doc-alignment-reports' / 'mermaid-report-scoped.json'

def load_invalid_diagrams() -> List[Dict]:
    """Load invalid diagrams from report."""
    with open(REPORT_FILE) as f:
        data = json.load(f)
    return data['invalid_diagram_list']

def fix_br_tags(content: str) -> Tuple[str, int]:
    """Replace <br> with <br/> in Note statements.

    Returns:
        Tuple of (fixed_content, num_changes)
    """
    # First pass: Replace ALL <br> with <br/> globally (not just in Note statements)
    # This is simpler and more comprehensive
    original = content
    fixed = content.replace('<br>', '<br/>')

    # Count changes
    count = original.count('<br>') - fixed.count('<br>')

    return fixed, count

def fix_erdiagram_brackets(content: str, start_line: int, end_line: int) -> Tuple[str, int]:
    """Fix unclosed brackets in erDiagram field definitions.

    Common issues:
    - Missing closing braces for entity definitions
    - Field definitions extending beyond entity block

    Returns:
        Tuple of (fixed_content, num_changes)
    """
    lines = content.split('\n')
    changes = 0

    # Find the diagram section
    in_diagram = False
    in_entity = False
    entity_indent = 0

    for i in range(len(lines)):
        line = lines[i]

        # Check if we're in the target diagram range
        if i + 1 == start_line:
            in_diagram = True
        elif i + 1 == end_line:
            in_diagram = False

        if not in_diagram:
            continue

        # Check for entity definition start
        if re.match(r'^\s+\w+\s*\{', line):
            in_entity = True
            entity_indent = len(line) - len(line.lstrip())
            continue

        # Check for entity definition end
        if in_entity and re.match(r'^\s+\}', line):
            in_entity = False
            continue

        # If we're in an entity and encounter a line with lower indent, close entity
        if in_entity:
            current_indent = len(line) - len(line.lstrip())
            # If line is not indented more than entity start, we need to close entity
            if current_indent <= entity_indent and line.strip() and not line.strip().startswith('}'):
                # Insert closing brace before this line
                lines.insert(i, ' ' * (entity_indent + 4) + '}')
                in_entity = False
                changes += 1

    return '\n'.join(lines), changes

def fix_file(file_path: Path, diagrams: List[Dict]) -> Dict:
    """Fix all invalid diagrams in a file.

    Returns:
        Dictionary with fix statistics
    """
    stats = {
        'file': str(file_path),
        'br_tag_fixes': 0,
        'bracket_fixes': 0,
        'diagrams_fixed': len(diagrams)
    }

    # Try multiple path resolutions
    full_path = DOCS_ROOT / file_path
    if not full_path.exists():
        full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        # Try without docs prefix
        parts = Path(file_path).parts
        if parts[0] == 'docs':
            full_path = PROJECT_ROOT / Path(*parts[1:])
        else:
            full_path = DOCS_ROOT / file_path

    if not full_path.exists():
        print(f"Warning: File not found: {file_path}")
        print(f"  Tried: {full_path}")
        return stats

    with open(full_path, 'r') as f:
        content = f.read()

    original_content = content

    # Fix BR tags (applies to all Note syntax errors)
    content, br_fixes = fix_br_tags(content)
    stats['br_tag_fixes'] = br_fixes

    # Fix bracket issues in erDiagrams
    for diagram in diagrams:
        if 'Unclosed bracket' in diagram.get('error_message', ''):
            content, bracket_fixes = fix_erdiagram_brackets(
                content,
                diagram['start_line'],
                diagram['end_line']
            )
            stats['bracket_fixes'] += bracket_fixes

    # Write back if changes were made
    if content != original_content:
        with open(full_path, 'w') as f:
            f.write(content)
        try:
            rel_path = full_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = full_path
        print(f"✓ Fixed {rel_path}")
        print(f"  - BR tag fixes: {stats['br_tag_fixes']}")
        if stats['bracket_fixes']:
            print(f"  - Bracket fixes: {stats['bracket_fixes']}")

    return stats

def main():
    """Main fix routine."""
    print("Loading invalid diagrams report...")
    invalid_diagrams = load_invalid_diagrams()

    print(f"Found {len(invalid_diagrams)} invalid diagrams\n")

    # Group diagrams by file
    by_file: Dict[str, List[Dict]] = {}
    for diagram in invalid_diagrams:
        file_path = diagram['file']
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(diagram)

    print(f"Files to fix: {len(by_file)}\n")

    # Fix each file
    all_stats = []
    for file_path, diagrams in sorted(by_file.items()):
        print(f"\nProcessing {file_path} ({len(diagrams)} diagrams)...")
        stats = fix_file(Path(file_path), diagrams)
        all_stats.append(stats)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_br = sum(s['br_tag_fixes'] for s in all_stats)
    total_brackets = sum(s['bracket_fixes'] for s in all_stats)
    total_diagrams = sum(s['diagrams_fixed'] for s in all_stats)

    print(f"Files processed: {len(all_stats)}")
    print(f"Diagrams fixed: {total_diagrams}")
    print(f"BR tag fixes: {total_br}")
    print(f"Bracket fixes: {total_brackets}")

    # Save detailed stats
    stats_file = PROJECT_ROOT / 'docs' / 'MERMAID_FIXES_STATS.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'summary': {
                'files_processed': len(all_stats),
                'diagrams_fixed': total_diagrams,
                'br_tag_fixes': total_br,
                'bracket_fixes': total_brackets
            },
            'by_file': all_stats
        }, f, indent=2)

    print(f"\nDetailed stats saved to: {stats_file.relative_to(PROJECT_ROOT)}")

if __name__ == '__main__':
    main()
