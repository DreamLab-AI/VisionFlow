#!/usr/bin/env python3
"""
HTTP Response Standardization Script
Task 1.4 - Phase 1: API Specialist Agent
Replaces direct HttpResponse construction with response macros
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Refactoring patterns
PATTERNS = [
    # Pattern 1: HttpResponse::Ok().json(...) → ok_json!(...)
    {
        'name': 'Ok responses',
        'pattern': r'HttpResponse::Ok\(\)\.json\(([^)]+)\)',
        'replacement': r'ok_json!(\1)',
        'macro': 'ok_json'
    },
    # Pattern 2: return HttpResponse::Ok().json(...) → return ok_json!(...)
    {
        'name': 'Ok responses with return',
        'pattern': r'return\s+HttpResponse::Ok\(\)\.json\(([^)]+)\)',
        'replacement': r'return ok_json!(\1)',
        'macro': 'ok_json'
    },
    # Pattern 3: HttpResponse::InternalServerError().json(...) → error_json!(...)
    {
        'name': 'Internal server errors',
        'pattern': r'HttpResponse::InternalServerError\(\)\.json\([^)]*\)',
        'replacement': lambda m: extract_error_message(m.group(0)),
        'macro': 'error_json'
    },
    # Pattern 4: HttpResponse::BadRequest().json(...) → bad_request!(...)
    {
        'name': 'Bad request responses',
        'pattern': r'HttpResponse::BadRequest\(\)\.json\([^)]*\)',
        'replacement': lambda m: extract_error_message(m.group(0), 'bad_request'),
        'macro': 'bad_request'
    },
    # Pattern 5: HttpResponse::NotFound().json(...) → not_found!(...)
    {
        'name': 'Not found responses',
        'pattern': r'HttpResponse::NotFound\(\)\.json\([^)]*\)',
        'replacement': lambda m: extract_error_message(m.group(0), 'not_found'),
        'macro': 'not_found'
    },
    # Pattern 6: HttpResponse::Created().json(...) → created_json!(...)
    {
        'name': 'Created responses',
        'pattern': r'HttpResponse::Created\(\)\.json\(([^)]+)\)',
        'replacement': r'created_json!(\1)',
        'macro': 'created_json'
    },
]

# Required imports
REQUIRED_IMPORTS = """
// Response macros - Task 1.4 HTTP Standardization
use crate::{ok_json, error_json, bad_request, not_found, created_json};
use crate::utils::handler_commons::HandlerResponse;
"""


def extract_error_message(match_str: str, macro_name: str = 'error_json') -> str:
    """Extract error message from HttpResponse::InternalServerError().json(...) patterns."""

    # Try to extract "error": "message" pattern
    error_match = re.search(r'"error":\s*"([^"]+)"', match_str)
    if error_match:
        return f'{macro_name}!("{error_match.group(1)}")'

    # Try to extract "message": value pattern
    msg_match = re.search(r'"message":\s*([^,}]+)', match_str)
    if msg_match:
        msg_value = msg_match.group(1).strip()
        # Remove quotes if it's a string literal
        if msg_value.startswith('"') and msg_value.endswith('"'):
            return f'{macro_name}!({msg_value})'
        else:
            return f'{macro_name}!({msg_value}.to_string())'

    # Fallback: just use generic error
    return f'{macro_name}!("Internal server error")'


def add_imports_if_needed(content: str) -> str:
    """Add necessary imports if not already present."""

    # Check if imports already exist
    if 'use crate::ok_json' in content or 'use crate::{ok_json' in content:
        return content

    # Find the last 'use' statement
    use_statements = list(re.finditer(r'^use\s+', content, re.MULTILINE))

    if use_statements:
        # Insert after the last use statement
        last_use = use_statements[-1]
        # Find the end of this use statement (semicolon)
        semicolon_pos = content.find(';', last_use.start())

        if semicolon_pos != -1:
            # Insert imports after the semicolon and newline
            insertion_point = semicolon_pos + 1
            return content[:insertion_point] + '\n' + REQUIRED_IMPORTS + content[insertion_point:]

    return content


def refactor_file(file_path: Path) -> Tuple[int, List[str]]:
    """Refactor a single file and return count of replacements and changes made."""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if no HttpResponse found
        if 'HttpResponse::' not in content:
            return 0, []

        original_content = content
        changes = []
        total_replacements = 0

        # Add imports
        content = add_imports_if_needed(content)

        # Apply each pattern
        for pattern_def in PATTERNS:
            pattern = pattern_def['pattern']
            replacement = pattern_def['replacement']

            matches = list(re.finditer(pattern, content))

            if matches:
                if callable(replacement):
                    # For complex replacements with functions
                    for match in reversed(matches):  # Reverse to maintain positions
                        new_text = replacement(match)
                        content = content[:match.start()] + new_text + content[match.end():]
                        total_replacements += 1
                        changes.append(f"  - {pattern_def['name']}: {match.group(0)[:50]}...")
                else:
                    # For simple regex replacements
                    content, count = re.subn(pattern, replacement, content)
                    total_replacements += count
                    if count > 0:
                        changes.append(f"  - {pattern_def['name']}: {count} occurrences")

        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        return total_replacements, changes

    except Exception as e:
        print(f"ERROR processing {file_path}: {e}", file=sys.stderr)
        return 0, [f"ERROR: {e}"]


def main():
    """Main refactoring function."""

    handlers_dir = Path("/home/devuser/workspace/project/src/handlers")

    if not handlers_dir.exists():
        print(f"ERROR: Handlers directory not found: {handlers_dir}")
        sys.exit(1)

    print("=" * 70)
    print("HTTP Response Standardization - Task 1.4")
    print("=" * 70)
    print(f"Target directory: {handlers_dir}\n")

    # Find all Rust files
    rust_files = list(handlers_dir.rglob("*.rs"))
    print(f"Found {len(rust_files)} Rust files\n")

    total_files_changed = 0
    total_replacements = 0
    detailed_changes: Dict[str, List[str]] = {}

    # Process each file
    for rust_file in sorted(rust_files):
        rel_path = rust_file.relative_to(handlers_dir)
        count, changes = refactor_file(rust_file)

        if count > 0:
            total_files_changed += 1
            total_replacements += count
            detailed_changes[str(rel_path)] = changes
            print(f"✓ {rel_path}: {count} replacements")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed:       {len(rust_files)}")
    print(f"Files changed:         {total_files_changed}")
    print(f"Total replacements:    {total_replacements}")

    # Verify remaining direct HttpResponse usage
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    remaining_count = 0
    for rust_file in rust_files:
        with open(rust_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Count non-standard HttpResponse usage (excluding imports)
        lines = content.split('\n')
        for line in lines:
            if 'HttpResponse::' in line and \
               'use actix' not in line and \
               'response_macros' not in line and \
               'handler_commons' not in line:
                remaining_count += 1

    print(f"Remaining direct HttpResponse usages: {remaining_count}")

    if remaining_count == 0:
        print("\n✓ SUCCESS: All HTTP responses standardized!")
    else:
        print(f"\n⚠ WARNING: {remaining_count} direct HttpResponse usages still remain")
        print("Manual review may be required for complex cases")

    # Detailed changes
    if detailed_changes and len(detailed_changes) <= 10:
        print("\n" + "=" * 70)
        print("DETAILED CHANGES")
        print("=" * 70)
        for file_name, changes in detailed_changes.items():
            print(f"\n{file_name}:")
            for change in changes:
                print(change)

    print("\nRefactoring complete!")
    return 0 if remaining_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
