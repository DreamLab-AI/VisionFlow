#!/usr/bin/env python3
"""
HTTP Response Standardization - Phase 2
Handles complex multiline cases and nested json! macros
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple


def extract_multiline_response(content: str, start_pos: int) -> Tuple[str, int]:
    """Extract a multiline HttpResponse construction."""

    # Find the matching closing parenthesis
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_pos, len(content)):
        char = content[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    return content[start_pos:i+1], i+1

    return "", start_pos


def refactor_complex_patterns(content: str) -> Tuple[str, int]:
    """Handle complex multiline response patterns."""

    replacements = 0

    # Pattern: return HttpResponse::Status().json(serde_json::json!({...}))
    # This handles multiline json! macros

    patterns_to_fix = [
        ('HttpResponse::InternalServerError()', 'error_json'),
        ('HttpResponse::BadRequest()', 'bad_request'),
        ('HttpResponse::NotFound()', 'not_found'),
        ('HttpResponse::Ok()', 'ok_json'),
    ]

    for http_pattern, macro_name in patterns_to_fix:
        # Find all occurrences
        pos = 0
        new_content = content
        offset = 0  # Track position offset due to replacements

        while True:
            pos = new_content.find(http_pattern + '.json(', pos + offset)
            if pos == -1:
                break

            # Extract the full response including multiline json
            start = pos
            json_start = new_content.find('.json(', pos) + 6

            # Extract the argument to .json(...)
            json_content, end_pos = extract_multiline_response(new_content, json_start)

            if not json_content:
                offset += 1
                continue

            # Remove the closing parenthesis from json_content
            json_content = json_content[:-1]

            # Check if it's a serde_json::json! macro
            if 'serde_json::json!' in json_content:
                # Extract the json macro content
                json_macro_match = re.search(r'serde_json::json!\s*\(\s*\{(.*?)\}\s*\)', json_content, re.DOTALL)

                if json_macro_match:
                    json_inner = json_macro_match.group(1).strip()

                    # Extract error or message field
                    error_match = re.search(r'"error":\s*"([^"]+)"', json_inner)
                    message_match = re.search(r'"message":\s*([^,}]+)', json_inner)

                    if error_match:
                        new_response = f'{macro_name}!("{error_match.group(1)}")'
                    elif message_match:
                        msg_value = message_match.group(1).strip()
                        if msg_value.startswith('"') and msg_value.endswith('"'):
                            new_response = f'{macro_name}!({msg_value})'
                        else:
                            # It's a variable or expression
                            new_response = f'{macro_name}!({{}})', msg_value
                    else:
                        # Fallback
                        new_response = f'{macro_name}!("Operation failed")'

                    # Check for 'return' statement
                    is_return = new_content[max(0, start-10):start].strip().endswith('return')

                    if is_return:
                        # Find the 'return' keyword
                        return_pos = new_content.rfind('return', max(0, start-10), start)
                        full_replacement = f'return {new_response}'
                        new_content = new_content[:return_pos] + full_replacement + new_content[end_pos:]
                        offset = len(full_replacement) - (end_pos - return_pos)
                    else:
                        new_content = new_content[:start] + new_response + new_content[end_pos:]
                        offset = len(new_response) - (end_pos - start)

                    replacements += 1
                    continue

            offset += 1

        content = new_content

    return content, replacements


def refactor_file_phase2(file_path: Path) -> Tuple[int, List[str]]:
    """Phase 2 refactoring for complex patterns."""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if no HttpResponse found
        if 'HttpResponse::' not in content or \
           ('ok_json' in content and 'error_json' in content):
            return 0, []

        original_content = content
        changes = []

        # Apply complex pattern refactoring
        content, count = refactor_complex_patterns(content)

        if count > 0:
            changes.append(f"  - Complex multiline responses: {count} occurrences")

        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        return count, changes

    except Exception as e:
        print(f"ERROR processing {file_path}: {e}", file=sys.stderr)
        return 0, [f"ERROR: {e}"]


def main():
    """Main function for phase 2 refactoring."""

    handlers_dir = Path("/home/devuser/workspace/project/src/handlers")

    print("=" * 70)
    print("HTTP Response Standardization - Phase 2")
    print("=" * 70)
    print("Handling complex multiline patterns\n")

    rust_files = list(handlers_dir.rglob("*.rs"))

    total_replacements = 0
    files_changed = 0

    for rust_file in sorted(rust_files):
        rel_path = rust_file.relative_to(handlers_dir)
        count, changes = refactor_file_phase2(rust_file)

        if count > 0:
            files_changed += 1
            total_replacements += count
            print(f"✓ {rel_path}: {count} replacements")

    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"Files changed:         {files_changed}")
    print(f"Total replacements:    {total_replacements}")

    # Final verification
    remaining = 0
    for rust_file in rust_files:
        with open(rust_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if 'HttpResponse::' in line and \
               'use actix' not in line and \
               'response_macros' not in line and \
               'handler_commons' not in line:
                remaining += 1

    print(f"\nRemaining direct HttpResponse usages: {remaining}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
