#!/usr/bin/env python3
"""
Comprehensive final migration - handles all remaining patterns
Processes both serde_json::json! and json! variants
"""

import re
import sys
from pathlib import Path

class FinalComprehensiveMigrator:
    def __init__(self):
        self.stats = {'files_modified': 0, 'responses_migrated': 0}

    def migrate_errors(self, content: str, http_response_type: str, macro_name: str) -> tuple[str, int]:
        """Migrate error responses for a specific HttpResponse type"""
        migrations = 0

        # Pattern 1: serde_json::json! variant
        pattern1 = rf'{http_response_type}\(\)\.json\(serde_json::json!\(\{{(.+?)\}}\)\)'

        def replacer1(match):
            nonlocal migrations
            json_content = match.group(1)

            # Extract error and message
            error_match = re.search(r'"error"\s*:\s*"([^"]+)"', json_content)
            message_match = re.search(r'"message"\s*:\s*(.+?)(?:\s*$)', json_content, re.DOTALL)

            if error_match and message_match:
                error = error_match.group(1)
                message = message_match.group(1).strip()
                migrations += 1
                return f'{macro_name}("{error}", {message})'
            elif error_match:
                error = error_match.group(1)
                migrations += 1
                return f'{macro_name}("{error}").unwrap()'

            return match.group(0)

        content = re.sub(pattern1, replacer1, content, flags=re.DOTALL)

        # Pattern 2: plain json! variant
        pattern2 = rf'{http_response_type}\(\)\.json\(json!\(\{{(.+?)\}}\)\)'

        def replacer2(match):
            nonlocal migrations
            json_content = match.group(1)

            error_match = re.search(r'"error"\s*:\s*"([^"]+)"', json_content)
            message_match = re.search(r'"message"\s*:\s*(.+?)(?:\s*$)', json_content, re.DOTALL)

            if error_match and message_match:
                error = error_match.group(1)
                message = message_match.group(1).strip()
                migrations += 1
                return f'{macro_name}("{error}", {message})'
            elif error_match:
                error = error_match.group(1)
                migrations += 1
                return f'{macro_name}("{error}").unwrap()'

            return match.group(0)

        content = re.sub(pattern2, replacer2, content, flags=re.DOTALL)

        return content, migrations

    def migrate_file(self, file_path: Path) -> int:
        """Migrate all error patterns in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            total = 0

            # Migrate all error types
            content, count = self.migrate_errors(content, 'HttpResponse::InternalServerError', 'error_json!')
            total += count

            content, count = self.migrate_errors(content, 'HttpResponse::BadRequest', 'bad_request!')
            total += count

            content, count = self.migrate_errors(content, 'HttpResponse::NotFound', 'not_found!')
            total += count

            content, count = self.migrate_errors(content, 'HttpResponse::ServiceUnavailable', 'service_unavailable!')
            total += count

            content, count = self.migrate_errors(content, 'HttpResponse::TooManyRequests', 'too_many_requests!')
            total += count

            content, count = self.migrate_errors(content, 'HttpResponse::PayloadTooLarge', 'payload_too_large!')
            total += count

            if total > 0:
                # Add imports
                if not self.has_imports(content):
                    content = self.add_imports(content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats['files_modified'] += 1
                self.stats['responses_migrated'] += total

            return total

        except Exception as e:
            print(f"  ✗ Error in {file_path.name}: {str(e)}", file=sys.stderr)
            return 0

    def has_imports(self, content: str) -> bool:
        return 'error_json!' in content or 'bad_request!' in content or 'not_found!' in content

    def add_imports(self, content: str) -> str:
        if self.has_imports(content):
            return content

        matches = list(re.finditer(r'(use\s+[^;]+;)', content))
        if matches:
            last_use = matches[-1]
            imports = '\nuse crate::{error_json, bad_request, not_found, service_unavailable, too_many_requests, payload_too_large};\n'
            content = content[:last_use.end()] + imports + content[last_use.end():]

        return content

    def migrate_directory(self, directory: Path):
        """Migrate all files"""
        rs_files = sorted(directory.rglob('*.rs'))

        print("\nFinal Comprehensive Migration - All Remaining Patterns")
        print("=" * 60)

        for file_path in rs_files:
            if '/tests/' in str(file_path):
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if 'HttpResponse::' in content and ('InternalServerError' in content or
                                                        'BadRequest' in content or
                                                        'NotFound' in content):
                        print(f"\n{file_path.relative_to(directory.parent)}")
                        count = self.migrate_file(file_path)
                        if count > 0:
                            print(f"  ✓ Migrated {count} responses")
            except:
                continue

    def print_summary(self):
        print("\n" + "=" * 60)
        print("FINAL COMPREHENSIVE MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files modified: {self.stats['files_modified']}")
        print(f"Total responses migrated: {self.stats['responses_migrated']}")

def main():
    project_root = Path(__file__).parent.parent
    handlers_dir = project_root / 'src' / 'handlers'

    if not handlers_dir.exists():
        print("ERROR: Handlers directory not found", file=sys.stderr)
        sys.exit(1)

    migrator = FinalComprehensiveMigrator()
    migrator.migrate_directory(handlers_dir)
    migrator.print_summary()

    # Print remaining count
    print("\nVerifying remaining patterns...")
    import subprocess
    result = subprocess.run(
        ['grep', '-r', 'HttpResponse::InternalServerError', 'src/handlers/', '--include=*.rs'],
        capture_output=True, text=True
    )
    remaining = len([l for l in result.stdout.split('\n') if l and 'use ' not in l])
    print(f"Remaining InternalServerError patterns: {remaining}")

if __name__ == '__main__':
    main()
