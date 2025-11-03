#!/usr/bin/env python3
"""
Automated unwrap migration script for VisionFlow project
Migrates unsafe .unwrap() calls to safe helper utilities
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Migration patterns
PATTERNS = {
    # SystemTime::now().duration_since(UNIX_EPOCH).unwrap()
    "duration_since": {
        "pattern": r'SystemTime::now\(\)\s*\.duration_since\(UNIX_EPOCH\)\s*\.unwrap\(\)',
        "replacement": 'SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0))',
        "import": None,  # Uses std::time::Duration
    },

    # .lock().unwrap() for Mutex
    "mutex_lock": {
        "pattern": r'\.lock\(\)\.unwrap\(\)',
        "replacement": '.lock().expect("Mutex poisoned")',
        "import": None,  # expect is built-in
    },

    # .read().unwrap() for RwLock
    "rwlock_read": {
        "pattern": r'\.read\(\)\.unwrap\(\)',
        "replacement": '.read().expect("RwLock poisoned")',
        "import": None,
    },

    # .write().unwrap() for RwLock
    "rwlock_write": {
        "pattern": r'\.write\(\)\.unwrap\(\)',
        "replacement": '.write().expect("RwLock poisoned")',
        "import": None,
    },
}

def analyze_file(filepath: Path) -> Dict[str, int]:
    """Analyze unwrap patterns in a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip test files
    if '#[cfg(test)]' in content or 'mod tests' in content:
        # Count but mark as test
        return {"test_file": 1}

    results = {}
    for name, config in PATTERNS.items():
        matches = re.findall(config["pattern"], content)
        if matches:
            results[name] = len(matches)

    return results

def migrate_file(filepath: Path, dry_run: bool = False) -> Tuple[int, List[str]]:
    """Migrate unwraps in a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Skip test files in actual migration
    if '#[cfg(test)]' in original_content or 'mod tests' in original_content:
        return 0, []

    content = original_content
    changes = []
    total_replacements = 0

    for name, config in PATTERNS.items():
        count = len(re.findall(config["pattern"], content))
        if count > 0:
            content = re.sub(config["pattern"], config["replacement"], content)
            total_replacements += count
            changes.append(f"  - {name}: {count} replacements")

            # Add import if needed
            if config.get("import") and config["import"] not in content:
                # Find first use statement and add after it
                use_pattern = r'(use\s+.*?;)'
                matches = list(re.finditer(use_pattern, content))
                if matches:
                    last_use = matches[-1]
                    insert_pos = last_use.end()
                    content = content[:insert_pos] + f"\n{config['import']}" + content[insert_pos:]

    if not dry_run and content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    return total_replacements, changes

def main():
    dry_run = '--dry-run' in sys.argv

    print("=" * 60)
    print("VisionFlow Unwrap Migration Tool")
    print("=" * 60)
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
    print()

    # Find all Rust files
    rust_files = list(SRC_DIR.rglob("*.rs"))
    print(f"Found {len(rust_files)} Rust files")
    print()

    # Analyze all files
    print("Analyzing files...")
    file_stats = {}
    for filepath in rust_files:
        stats = analyze_file(filepath)
        if stats and "test_file" not in stats:
            file_stats[filepath] = stats

    # Sort by total unwraps
    sorted_files = sorted(
        file_stats.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )

    # Show top files
    print("\nTop 20 files with unsafe unwraps:")
    print("-" * 60)
    for filepath, stats in sorted_files[:20]:
        rel_path = filepath.relative_to(PROJECT_ROOT)
        total = sum(stats.values())
        print(f"{total:3d} unwraps - {rel_path}")
        for pattern, count in stats.items():
            print(f"      {pattern}: {count}")
    print()

    # Migrate files
    print("Migrating files...")
    print("-" * 60)
    total_files_modified = 0
    total_replacements = 0

    for filepath, stats in sorted_files:
        replacements, changes = migrate_file(filepath, dry_run)
        if replacements > 0:
            total_files_modified += 1
            total_replacements += replacements
            rel_path = filepath.relative_to(PROJECT_ROOT)
            print(f"\n✓ {rel_path} ({replacements} replacements)")
            for change in changes:
                print(change)

    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Files analyzed: {len(rust_files)}")
    print(f"Files with unwraps: {len(sorted_files)}")
    print(f"Files modified: {total_files_modified}")
    print(f"Total replacements: {total_replacements}")
    print()

    if dry_run:
        print("This was a DRY RUN. Run without --dry-run to apply changes.")
    else:
        print("✅ Migration complete!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
