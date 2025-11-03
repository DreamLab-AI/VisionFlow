#!/usr/bin/env python3
"""
Final comprehensive migration using extended response macros
Handles complex error patterns with error + message fields
"""

import re
import sys
from pathlib import Path

class ComprehensiveMigrator:
    def __init__(self):
        self.stats = {
            'files_modified': 0,
            'responses_migrated': 0,
            'patterns': {}
        }

    def migrate_file(self, file_path: Path) -> int:
        """Migrate all patterns in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original = content
            migrations = 0

            # Pattern 1: HttpResponse::InternalServerError().json(serde_json::json!({ "error": "...", "message": ... }))
            pattern = r'HttpResponse::InternalServerError\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*,\s*"message"\s*:\s*([^}]+?)\s*\}\s*\)\)'

            def replace_internal_with_message(match):
                nonlocal migrations
                migrations += 1
                error_msg = match.group(1)
                message_expr = match.group(2).strip()
                return f'error_json!("{error_msg}", {message_expr})'

            content = re.sub(pattern, replace_internal_with_message, content, flags=re.DOTALL)

            # Pattern 2: HttpResponse::BadRequest().json(serde_json::json!({ "error": "...", "message": ... }))
            pattern = r'HttpResponse::BadRequest\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*,\s*"message"\s*:\s*([^}]+?)\s*\}\s*\)\)'

            def replace_badreq_with_message(match):
                nonlocal migrations
                migrations += 1
                error_msg = match.group(1)
                message_expr = match.group(2).strip()
                return f'bad_request!("{error_msg}", {message_expr})'

            content = re.sub(pattern, replace_badreq_with_message, content, flags=re.DOTALL)

            # Pattern 3: HttpResponse::NotFound().json(serde_json::json!({ "error": "...", "message": ... }))
            pattern = r'HttpResponse::NotFound\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*,\s*"message"\s*:\s*([^}]+?)\s*\}\s*\)\)'

            def replace_notfound_with_message(match):
                nonlocal migrations
                migrations += 1
                error_msg = match.group(1)
                message_expr = match.group(2).strip()
                return f'not_found!("{error_msg}", {message_expr})'

            content = re.sub(pattern, replace_notfound_with_message, content, flags=re.DOTALL)

            # Pattern 4: Simple errors without message field - InternalServerError
            pattern = r'HttpResponse::InternalServerError\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, r'error_json!("\1").unwrap()', content)
                migrations += matches

            # Pattern 5: Simple errors - BadRequest
            pattern = r'HttpResponse::BadRequest\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, r'bad_request!("\1").unwrap()', content)
                migrations += matches

            # Pattern 6: Simple errors - NotFound
            pattern = r'HttpResponse::NotFound\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, r'not_found!("\1").unwrap()', content)
                migrations += matches

            # Pattern 7: ServiceUnavailable
            pattern = r'HttpResponse::ServiceUnavailable\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, r'service_unavailable!("\1").unwrap()', content)
                migrations += matches

            # Pattern 8: TooManyRequests
            pattern = r'HttpResponse::TooManyRequests\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, r'too_many_requests!("\1").unwrap()', content)
                migrations += matches

            # Pattern 9: PayloadTooLarge
            pattern = r'HttpResponse::PayloadTooLarge\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, r'payload_too_large!("\1").unwrap()', content)
                migrations += matches

            if migrations > 0:
                # Ensure imports
                if not self.has_imports(content):
                    content = self.add_imports(content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats['files_modified'] += 1
                self.stats['responses_migrated'] += migrations

            return migrations

        except Exception as e:
            print(f"  ✗ Error: {str(e)}", file=sys.stderr)
            return 0

    def has_imports(self, content: str) -> bool:
        """Check if response macros are imported"""
        return 'use crate::' in content and any(
            macro in content for macro in ['error_json!', 'bad_request!', 'not_found!']
        )

    def add_imports(self, content: str) -> str:
        """Add macro imports"""
        if self.has_imports(content):
            return content

        # Find last use statement
        matches = list(re.finditer(r'(use\s+[^;]+;)', content))
        if matches:
            last_use = matches[-1]
            imports = '\nuse crate::{error_json, bad_request, not_found, service_unavailable, too_many_requests, payload_too_large};\n'
            content = content[:last_use.end()] + imports + content[last_use.end():]

        return content

    def migrate_directory(self, directory: Path):
        """Migrate all handler files"""
        rs_files = sorted(directory.rglob('*.rs'))

        print(f"\nComprehensive Migration with Extended Macros")
        print("=" * 60)

        for file_path in rs_files:
            if '/tests/' in str(file_path):
                continue

            try:
                with open(file_path, 'r') as f:
                    if 'HttpResponse::' in f.read():
                        print(f"\n{file_path.relative_to(directory.parent)}")
                        count = self.migrate_file(file_path)
                        if count > 0:
                            print(f"  ✓ Migrated {count} responses")
            except:
                continue

    def print_summary(self):
        """Print summary"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files modified: {self.stats['files_modified']}")
        print(f"Total responses migrated: {self.stats['responses_migrated']}")

def main():
    project_root = Path(__file__).parent.parent
    handlers_dir = project_root / 'src' / 'handlers'

    if not handlers_dir.exists():
        print("ERROR: Handlers directory not found", file=sys.stderr)
        sys.exit(1)

    migrator = ComprehensiveMigrator()
    migrator.migrate_directory(handlers_dir)
    migrator.print_summary()

if __name__ == '__main__':
    main()
