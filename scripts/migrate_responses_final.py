#!/usr/bin/env python3
"""
Final Comprehensive HTTP Response Migration
Handles all remaining patterns including multi-line JSON and complex error objects
"""

import re
import sys
from pathlib import Path

class FinalMigrator:
    def __init__(self):
        self.stats = {
            'files_modified': 0,
            'responses_migrated': 0,
            'manual_review': []
        }

    def migrate_multiline_errors(self, content: str) -> tuple[str, int]:
        """Migrate multi-line error responses"""
        migrations = 0

        # Pattern: HttpResponse::InternalServerError().json(serde_json::json!({...}))
        # Spanning multiple lines
        pattern = r'HttpResponse::InternalServerError\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*(?:"([^"]+)"|format!\("([^"]+)"(?:,\s*([^)]+))?\))\s*(?:,\s*"message"\s*:\s*[^}]+)?\s*\}\s*\)\)'

        def replace_internal_error(match):
            nonlocal migrations
            migrations += 1
            error_msg = match.group(1) or match.group(2)
            if match.group(3):  # has format args
                return f'error_json!("{error_msg}", {match.group(3)})'
            return f'error_json!("{error_msg}")'

        content = re.sub(pattern, replace_internal_error, content, flags=re.MULTILINE | re.DOTALL)

        # Pattern: HttpResponse::BadRequest().json(...)
        pattern = r'HttpResponse::BadRequest\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*(?:"([^"]+)"|format!\("([^"]+)"(?:,\s*([^)]+))?\))\s*\}\s*\)\)'

        def replace_bad_request(match):
            nonlocal migrations
            migrations += 1
            error_msg = match.group(1) or match.group(2)
            if match.group(3):
                return f'bad_request!("{error_msg}", {match.group(3)})'
            return f'bad_request!("{error_msg}")'

        content = re.sub(pattern, replace_bad_request, content, flags=re.MULTILINE | re.DOTALL)

        # Pattern: HttpResponse::NotFound().json(...)
        pattern = r'HttpResponse::NotFound\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*(?:"([^"]+)"|format!\("([^"]+)"(?:,\s*([^)]+))?\))\s*\}\s*\)\)'

        def replace_not_found(match):
            nonlocal migrations
            migrations += 1
            error_msg = match.group(1) or match.group(2)
            if match.group(3):
                return f'not_found!("{error_msg}", {match.group(3)})'
            return f'not_found!("{error_msg}")'

        content = re.sub(pattern, replace_not_found, content, flags=re.MULTILINE | re.DOTALL)

        # Pattern: HttpResponse::ServiceUnavailable().json(...)
        pattern = r'HttpResponse::ServiceUnavailable\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'

        def replace_service_unavail(match):
            nonlocal migrations
            migrations += 1
            return f'service_unavailable!("{match.group(1)}")'

        content = re.sub(pattern, replace_service_unavail, content, flags=re.MULTILINE | re.DOTALL)

        # Pattern: HttpResponse::TooManyRequests().json(...)
        pattern = r'HttpResponse::TooManyRequests\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'

        def replace_too_many(match):
            nonlocal migrations
            migrations += 1
            return f'too_many_requests!("{match.group(1)}")'

        content = re.sub(pattern, replace_too_many, content, flags=re.MULTILINE | re.DOTALL)

        # Pattern: HttpResponse::PayloadTooLarge().json(...)
        pattern = r'HttpResponse::PayloadTooLarge\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'

        def replace_payload_large(match):
            nonlocal migrations
            migrations += 1
            return f'payload_too_large!("{match.group(1)}")'

        content = re.sub(pattern, replace_payload_large, content, flags=re.MULTILINE | re.DOTALL)

        return content, migrations

    def ensure_imports(self, content: str, file_path: Path) -> str:
        """Ensure response macro imports are present"""
        macros = ['ok_json', 'error_json', 'bad_request', 'not_found',
                 'too_many_requests', 'service_unavailable', 'payload_too_large']

        has_imports = any(f'{macro}!' in content for macro in macros)

        if not has_imports:
            return content

        # Check if imports already exist
        for macro in macros:
            if f'use crate::{{{macro}' in content or f'use crate::{macro}' in content:
                return content

        # Add imports after the last use statement
        use_pattern = r'(use\s+[^;]+;)'
        matches = list(re.finditer(use_pattern, content))

        if matches:
            last_use = matches[-1]
            insert_pos = last_use.end()

            import_statement = '''
use crate::{
    ok_json, created_json, error_json, bad_request, not_found,
    unauthorized, forbidden, conflict, no_content, accepted,
    too_many_requests, service_unavailable, payload_too_large
};
'''
            content = content[:insert_pos] + import_statement + content[insert_pos:]
            print(f"  + Added macro imports")

        return content

    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip if no HttpResponse patterns
            if 'HttpResponse::' not in content or 'use actix_web' not in content:
                return False

            original_content = content
            content, migrations = self.migrate_multiline_errors(content)

            if migrations > 0:
                content = self.ensure_imports(content, file_path)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats['files_modified'] += 1
                self.stats['responses_migrated'] += migrations
                print(f"  ✓ Migrated {migrations} responses")
                return True

            return False

        except Exception as e:
            print(f"  ✗ Error: {str(e)}", file=sys.stderr)
            return False

    def migrate_directory(self, directory: Path):
        """Migrate all handler files"""
        rs_files = sorted(directory.rglob('*.rs'))

        print(f"\nFinal Migration Pass - {len(rs_files)} files")
        print("=" * 60)

        for file_path in rs_files:
            if '/tests/' in str(file_path):
                continue

            # Only process files with HttpResponse patterns
            try:
                with open(file_path, 'r') as f:
                    if 'HttpResponse::' in f.read():
                        print(f"\n{file_path.relative_to(directory.parent)}")
                        self.migrate_file(file_path)
            except:
                continue

    def print_summary(self):
        """Print migration summary"""
        print("\n" + "=" * 60)
        print("FINAL MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files modified: {self.stats['files_modified']}")
        print(f"Responses migrated: {self.stats['responses_migrated']}")

def main():
    project_root = Path(__file__).parent.parent
    handlers_dir = project_root / 'src' / 'handlers'

    if not handlers_dir.exists():
        print(f"ERROR: Handlers directory not found", file=sys.stderr)
        sys.exit(1)

    migrator = FinalMigrator()
    migrator.migrate_directory(handlers_dir)
    migrator.print_summary()

if __name__ == '__main__':
    main()
