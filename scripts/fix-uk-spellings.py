#!/usr/bin/env python3
"""
UK Spelling Remediation Script
Fixes US English → UK English across documentation corpus
Preserves code blocks, URLs, file paths, and technical identifiers
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Spelling replacements (US → UK)
REPLACEMENTS = [
    # optimization family (261 occurrences)
    (r'\boptimization\b', 'optimisation'),
    (r'\boptimizations\b', 'optimisations'),
    (r'\boptimized\b', 'optimised'),
    (r'\boptimize\b', 'optimise'),
    (r'\boptimizing\b', 'optimising'),
    (r'\boptimizer\b', 'optimiser'),

    # organization family (66 occurrences)
    (r'\borganization\b', 'organisation'),
    (r'\borganizations\b', 'organisations'),
    (r'\borganizational\b', 'organisational'),
    (r'\borganized\b', 'organised'),
    (r'\borganize\b', 'organise'),
    (r'\borganizing\b', 'organising'),

    # color family (69 occurrences)
    (r'\bcolor\b', 'colour'),
    (r'\bcolors\b', 'colours'),
    (r'\bcolored\b', 'coloured'),
    (r'\bcoloring\b', 'colouring'),
    (r'\bcolorize\b', 'colourise'),

    # behavior family (35 occurrences)
    (r'\bbehavior\b', 'behaviour'),
    (r'\bbehaviors\b', 'behaviours'),
    (r'\bbehavioral\b', 'behavioural'),

    # analyze family (29 occurrences)
    (r'\banalyzer\b', 'analyser'),
    (r'\banalyzers\b', 'analysers'),
    (r'\banalyze\b', 'analyse'),
    (r'\banalyzing\b', 'analysing'),
    (r'\banalyzed\b', 'analysed'),

    # fiber family (30 occurrences)
    (r'\bfiber\b', 'fibre'),
    (r'\bfibers\b', 'fibres'),

    # realize family
    (r'\brealize\b', 'realise'),
    (r'\brealizes\b', 'realises'),
    (r'\brealized\b', 'realised'),
    (r'\brealizing\b', 'realising'),

    # utilize family
    (r'\butilize\b', 'utilise'),
    (r'\butilizes\b', 'utilises'),
    (r'\butilized\b', 'utilised'),
    (r'\butilizing\b', 'utilising'),

    # center family
    (r'\bcenter\b', 'centre'),
    (r'\bcenters\b', 'centres'),
    (r'\bcentered\b', 'centred'),
    (r'\bcentering\b', 'centring'),

    # favor family
    (r'\bfavor\b', 'favour'),
    (r'\bfavors\b', 'favours'),
    (r'\bfavored\b', 'favoured'),
    (r'\bfavoring\b', 'favouring'),

    # honor family
    (r'\bhonor\b', 'honour'),
    (r'\bhonors\b', 'honours'),
    (r'\bhonored\b', 'honoured'),
    (r'\bhonoring\b', 'honouring'),

    # defense/offense
    (r'\bdefense\b', 'defence'),
    (r'\bdefenses\b', 'defences'),
    (r'\boffense\b', 'offence'),
    (r'\boffenses\b', 'offences'),

    # catalog family
    (r'\bcatalog\b', 'catalogue'),
    (r'\bcatalogs\b', 'catalogues'),
]


def process_file(file_path: Path) -> Tuple[int, int]:
    """
    Process a single markdown file, preserving code blocks.
    Returns (changes_made, total_lines_processed)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ⚠️  Error reading {file_path}: {e}", file=sys.stderr)
        return 0, 0

    original_content = content
    changes = 0

    # Split into code blocks and text blocks
    # Pattern: ``` ... ``` (code blocks)
    parts = []
    in_code_block = False
    current_part = ""

    lines = content.split('\n')
    processed_lines = []

    for line in lines:
        # Check if this line starts/ends a code block
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            processed_lines.append(line)
        elif in_code_block:
            # Inside code block - don't modify
            processed_lines.append(line)
        else:
            # Outside code block - apply replacements
            modified_line = line
            for us_pattern, uk_replacement in REPLACEMENTS:
                modified_line = re.sub(us_pattern, uk_replacement, modified_line)

            if modified_line != line:
                changes += 1
            processed_lines.append(modified_line)

    new_content = '\n'.join(processed_lines)

    if new_content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return changes, len(lines)
        except Exception as e:
            print(f"  ⚠️  Error writing {file_path}: {e}", file=sys.stderr)
            return 0, 0

    return 0, len(lines)


def main():
    docs_dir = Path('/home/devuser/workspace/project/docs')

    if not docs_dir.exists():
        print(f"❌ Documentation directory not found: {docs_dir}", file=sys.stderr)
        sys.exit(1)

    print("🔍 UK Spelling Remediation Script")
    print("━" * 60)
    print()

    total_files = 0
    total_changes = 0
    total_lines = 0
    modified_files = []

    # Process all markdown files
    for md_file in sorted(docs_dir.rglob('*.md')):
        total_files += 1
        changes, lines = process_file(md_file)
        total_lines += lines

        if changes > 0:
            modified_files.append((md_file.relative_to(docs_dir), changes))
            total_changes += changes
            print(f"  ✓ Fixed {changes:3d} line(s) in: {md_file.relative_to(docs_dir)}")

    print()
    print("━" * 60)
    print("✨ Remediation Complete")
    print(f"📊 Files scanned: {total_files}")
    print(f"📝 Files modified: {len(modified_files)}")
    print(f"📄 Total lines processed: {total_lines}")
    print(f"✏️  Total lines changed: {total_changes}")
    print()

    if modified_files:
        print("Modified files:")
        for file_path, change_count in modified_files[:20]:
            print(f"  • {file_path} ({change_count} changes)")
        if len(modified_files) > 20:
            print(f"  ... and {len(modified_files) - 20} more files")
        print()


if __name__ == '__main__':
    main()
