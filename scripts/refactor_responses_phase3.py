#!/usr/bin/env python3
"""
HTTP Response Standardization - Phase 3
Handles special status codes: TooManyRequests, ServiceUnavailable, PayloadTooLarge
"""

import re
from pathlib import Path


def refactor_special_status_codes(content: str) -> tuple[str, int]:
    """Refactor special HTTP status codes."""

    replacements = 0

    # Pattern 1: HttpResponse::TooManyRequests().json(json!({...}))
    pattern1 = r'HttpResponse::TooManyRequests\(\)\.json\(json!\(\{[^}]*"error":\s*"([^"]+)"[^}]*\}\)\)'
    content, count1 = re.subn(pattern1, r'too_many_requests!("\1")', content)
    replacements += count1

    # Pattern 2: HttpResponse::ServiceUnavailable().json(json!({...}))
    pattern2 = r'HttpResponse::ServiceUnavailable\(\)\.json\(json!\(\{[^}]*"error":\s*"([^"]+)"[^}]*\}\)\)'
    content, count2 = re.subn(pattern2, r'service_unavailable!("\1")', content)
    replacements += count2

    # Pattern 3: HttpResponse::PayloadTooLarge().json(json!({...}))
    pattern3 = r'HttpResponse::PayloadTooLarge\(\)\.json\(json!\(\{[^}]*"error":\s*"([^"]+)"[^}]*\}\)\)'
    content, count3 = re.subn(pattern3, r'payload_too_large!("\1")', content)
    replacements += count3

    return content, replacements


def update_imports(content: str) -> str:
    """Update imports to include new macros."""

    # Check if imports need updating
    if 'use crate::{ok_json' in content and 'too_many_requests' not in content:
        # Find the import line and update it
        pattern = r'use crate::\{([^}]+)\};'
        match = re.search(pattern, content)

        if match:
            current_imports = match.group(1)
            if 'too_many_requests' not in current_imports:
                new_imports = current_imports + ', too_many_requests, service_unavailable, payload_too_large'
                content = content.replace(match.group(0), f'use crate::{{{new_imports}}};')

    return content


def main():
    handlers_dir = Path("/home/devuser/workspace/project/src/handlers")

    print("=" * 70)
    print("HTTP Response Standardization - Phase 3")
    print("=" * 70)
    print("Handling special status codes: 429, 503, 413\n")

    total_replacements = 0
    files_changed = 0

    for rust_file in sorted(handlers_dir.rglob("*.rs")):
        with open(rust_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if no special status codes
        if not any(code in content for code in ['TooManyRequests', 'ServiceUnavailable', 'PayloadTooLarge']):
            continue

        original_content = content

        # Refactor special status codes
        content, count = refactor_special_status_codes(content)

        if count > 0:
            # Update imports
            content = update_imports(content)

            # Write file
            with open(rust_file, 'w', encoding='utf-8') as f:
                f.write(content)

            rel_path = rust_file.relative_to(handlers_dir)
            print(f"✓ {rel_path}: {count} replacements")

            total_replacements += count
            files_changed += 1

    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    print(f"Files changed:         {files_changed}")
    print(f"Total replacements:    {total_replacements}")

    # Final count
    remaining = 0
    for rust_file in handlers_dir.rglob("*.rs"):
        with open(rust_file, 'r') as f:
            for line in f:
                if 'HttpResponse::' in line and \
                   'use actix' not in line and \
                   'response_macros' not in line and \
                   'handler_commons' not in line:
                    remaining += 1

    print(f"\nFinal remaining direct HttpResponse usages: {remaining}")

    return 0


if __name__ == "__main__":
    exit(main())
