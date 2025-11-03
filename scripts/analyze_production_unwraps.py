#!/usr/bin/env python3
"""Analyze unwraps in production code (excluding tests)"""

import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

def is_test_code(filepath: Path, line_num: int) -> bool:
    """Check if a line is in test code"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Check if we're inside a #[cfg(test)] module
    in_test_module = False
    brace_depth = 0
    test_start_depth = None

    for i, line in enumerate(lines[:line_num], 1):
        # Track module nesting
        if 'mod tests' in line or '#[cfg(test)]' in line:
            in_test_module = True
            test_start_depth = brace_depth

        # Track braces
        brace_depth += line.count('{') - line.count('}')

        # Exit test module when braces close
        if in_test_module and test_start_depth is not None:
            if brace_depth <= test_start_depth:
                in_test_module = False
                test_start_depth = None

    return in_test_module

def analyze_unwraps():
    """Analyze all production unwraps"""
    unwraps_by_file = defaultdict(list)
    unwraps_by_pattern = defaultdict(int)

    for rust_file in SRC_DIR.rglob("*.rs"):
        with open(rust_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if '.unwrap()' in line and '// SAFETY' not in line:
                    if not is_test_code(rust_file, line_num):
                        rel_path = rust_file.relative_to(PROJECT_ROOT)
                        unwraps_by_file[str(rel_path)].append((line_num, line.strip()))

                        # Categorize pattern
                        if 'TempDir::new()' in line:
                            unwraps_by_pattern['TempDir'] += 1
                        elif '.get(' in line:
                            unwraps_by_pattern['Collection .get()'] += 1
                        elif '.parse()' in line:
                            unwraps_by_pattern['String .parse()'] += 1
                        elif '.to_str()' in line:
                            unwraps_by_pattern['Path .to_str()'] += 1
                        elif '.await' in line:
                            unwraps_by_pattern['Async Result'] += 1
                        elif 'Connection::open' in line:
                            unwraps_by_pattern['Database Connection'] += 1
                        else:
                            unwraps_by_pattern['Other'] += 1

    # Sort files by unwrap count
    sorted_files = sorted(
        unwraps_by_file.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print("=" * 70)
    print("PRODUCTION UNWRAPS ANALYSIS (Excluding Tests)")
    print("=" * 70)
    print()

    print("UNWRAP PATTERNS:")
    print("-" * 70)
    for pattern, count in sorted(unwraps_by_pattern.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern:30s}: {count:3d}")
    print()

    print(f"TOP 20 FILES WITH PRODUCTION UNWRAPS:")
    print("-" * 70)
    for filepath, unwraps in sorted_files[:20]:
        print(f"\n{filepath} ({len(unwraps)} unwraps):")
        for line_num, line in unwraps[:5]:  # Show first 5
            print(f"  Line {line_num:4d}: {line[:80]}")
        if len(unwraps) > 5:
            print(f"  ... and {len(unwraps) - 5} more")

    print()
    print("=" * 70)
    total = sum(len(u) for u in unwraps_by_file.values())
    print(f"TOTAL PRODUCTION UNWRAPS: {total}")
    print(f"FILES WITH UNWRAPS: {len(unwraps_by_file)}")
    print("=" * 70)

if __name__ == "__main__":
    analyze_unwraps()
