#!/usr/bin/env python3
"""
Final working migration for complex multi-field error responses
Uses two-stage pattern matching for reliability
"""

import re
import sys
from pathlib import Path

class WorkingMigrator:
    def __init__(self):
        self.stats = {'files_modified': 0, 'responses_migrated': 0}

    def migrate_error_with_message(self, content: str, error_type: str, macro_name: str) -> tuple[str, int]:
        """Migrate error responses that have both error and message fields"""
        migrations = 0

        # Match the entire HttpResponse pattern
        pattern = rf'{error_type}\(\)\.json\((?:serde_)?json!\(\{{(.+?)\}}\)\)'

        def replacer(match):
            nonlocal migrations
            json_content = match.group(1)

            # Extract error message
            error_match = re.search(r'"error"\s*:\s*"([^"]+)"', json_content)
            # Extract message expression
            message_match = re.search(r'"message"\s*:\s*(.+?)(?=\s*$)', json_content, re.DOTALL)

            if error_match and message_match:
                error = error_match.group(1)
                message = message_match.group(1).strip()
                migrations += 1
                return f'{macro_name}("{error}", {message})'
            elif error_match:
                # Simple error without message
                error = error_match.group(1)
                migrations += 1
                return f'{macro_name}("{error}").unwrap()'

            # Return unchanged if pattern doesn't match expected structure
            return match.group(0)

        content = re.sub(pattern, replacer, content, flags=re.DOTALL)
        return content, migrations

    def migrate_file(self, file_path: Path) -> int:
        """Migrate a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original = content
            total_migrations = 0

            # Migrate InternalServerError
            content, count = self.migrate_error_with_message(
                content, 'HttpResponse::InternalServerError', 'error_json!')
            total_migrations += count

            # Migrate BadRequest
            content, count = self.migrate_error_with_message(
                content, 'HttpResponse::BadRequest', 'bad_request!')
            total_migrations += count

            # Migrate NotFound
            content, count = self.migrate_error_with_message(
                content, 'HttpResponse::NotFound', 'not_found!')
            total_migrations += count

            # Migrate ServiceUnavailable
            content, count = self.migrate_error_with_message(
                content, 'HttpResponse::ServiceUnavailable', 'service_unavailable!')
            total_migrations += count

            # Migrate TooManyRequests
            content, count = self.migrate_error_with_message(
                content, 'HttpResponse::TooManyRequests', 'too_many_requests!')
            total_migrations += count

            # Migrate PayloadTooLarge
            content, count = self.migrate_error_with_message(
                content, 'HttpResponse::PayloadTooLarge', 'payload_too_large!')
            total_migrations += count

            if total_migrations > 0:
                # Add imports if needed
                if not self.has_imports(content):
                    content = self.add_imports(content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats['files_modified'] += 1
                self.stats['responses_migrated'] += total_migrations

            return total_migrations

        except Exception as e:
            print(f"  ✗ Error: {str(e)}", file=sys.stderr)
            return 0

    def has_imports(self, content: str) -> bool:
        """Check if macros are imported"""
        macros = ['error_json!', 'bad_request!', 'not_found!']
        return any(macro in content for macro in macros)

    def add_imports(self, content: str) -> str:
        """Add macro imports after last use statement"""
        if self.has_imports(content):
            return content

        matches = list(re.finditer(r'(use\s+[^;]+;)', content))
        if matches:
            last_use = matches[-1]
            imports = '\nuse crate::{error_json, bad_request, not_found, service_unavailable, too_many_requests, payload_too_large};\n'
            content = content[:last_use.end()] + imports + content[last_use.end():]

        return content

    def migrate_directory(self, directory: Path):
        """Migrate all handler files"""
        rs_files = sorted(directory.rglob('*.rs'))

        print(f"\nFinal Working Migration - Complex Error Patterns")
        print("=" * 60)

        for file_path in rs_files:
            if '/tests/' in str(file_path):
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Check for error patterns
                    if 'HttpResponse::InternalServerError' in content or \
                       'HttpResponse::BadRequest' in content or \
                       'HttpResponse::NotFound' in content:
                        print(f"\n{file_path.relative_to(directory.parent)}")
                        count = self.migrate_file(file_path)
                        if count > 0:
                            print(f"  ✓ Migrated {count} responses")
            except Exception as e:
                continue

    def print_summary(self):
        """Print summary"""
        print("\n" + "=" * 60)
        print("FINAL WORKING MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files modified: {self.stats['files_modified']}")
        print(f"Total responses migrated: {self.stats['responses_migrated']}")

def main():
    project_root = Path(__file__).parent.parent
    handlers_dir = project_root / 'src' / 'handlers'

    if not handlers_dir.exists():
        print("ERROR: Handlers directory not found", file=sys.stderr)
        sys.exit(1)

    migrator = WorkingMigrator()
    migrator.migrate_directory(handlers_dir)
    migrator.print_summary()

if __name__ == '__main__':
    main()
