#!/usr/bin/env python3
"""
Migrate error responses with format!() expressions
These are the remaining stubborn patterns
"""

import re
import sys
from pathlib import Path

class FormatErrorMigrator:
    def __init__(self):
        self.stats = {'files_modified': 0, 'responses_migrated': 0}

    def migrate_format_errors(self, content: str, error_type: str, macro_name: str) -> tuple[str, int]:
        """Migrate errors with format!() expressions"""
        migrations = 0

        # Pattern: HttpResponse::Type().json(serde_json::json!({ "error": format!(...) }))
        pattern = rf'{error_type}\(\)\.json\(serde_json::json!\(\{{\s*"error"\s*:\s*format!\(([^)]+(?:\([^)]*\)[^)]*)*)\)\s*\}}\)\)'

        def replacer(match):
            nonlocal migrations
            format_content = match.group(1)
            migrations += 1
            # Use the format variant of the macro
            return f'{macro_name}({format_content})'

        content = re.sub(pattern, replacer, content, flags=re.DOTALL)

        # Also handle the json! variant (without serde_)
        pattern2 = rf'{error_type}\(\)\.json\(json!\(\{{\s*"error"\s*:\s*format!\(([^)]+(?:\([^)]*\)[^)]*)*)\)\s*\}}\)\)'
        content = re.sub(pattern2, replacer, content, flags=re.DOTALL)

        return content, migrations

    def migrate_file(self, file_path: Path) -> int:
        """Migrate all format! errors in file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            total = 0

            # Migrate all types
            content, count = self.migrate_format_errors(content, 'HttpResponse::InternalServerError', 'error_json!')
            total += count

            content, count = self.migrate_format_errors(content, 'HttpResponse::BadRequest', 'bad_request!')
            total += count

            content, count = self.migrate_format_errors(content, 'HttpResponse::NotFound', 'not_found!')
            total += count

            if total > 0:
                if not self.has_imports(content):
                    content = self.add_imports(content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats['files_modified'] += 1
                self.stats['responses_migrated'] += total

            return total

        except Exception as e:
            print(f"  ✗ Error: {str(e)}", file=sys.stderr)
            return 0

    def has_imports(self, content: str) -> bool:
        return 'error_json!' in content or 'bad_request!' in content

    def add_imports(self, content: str) -> str:
        if self.has_imports(content):
            return content

        matches = list(re.finditer(r'(use\s+[^;]+;)', content))
        if matches:
            last_use = matches[-1]
            imports = '\nuse crate::{error_json, bad_request, not_found};\n'
            content = content[:last_use.end()] + imports + content[last_use.end():]

        return content

    def migrate_directory(self, directory: Path):
        """Migrate all files"""
        rs_files = sorted(directory.rglob('*.rs'))

        print("\nMigrating format!() Error Patterns")
        print("=" * 60)

        for file_path in rs_files:
            if '/tests/' in str(file_path):
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if 'format!(' in content and 'HttpResponse::' in content:
                        print(f"\n{file_path.relative_to(directory.parent)}")
                        count = self.migrate_file(file_path)
                        if count > 0:
                            print(f"  ✓ Migrated {count} format!() responses")
            except:
                continue

    def print_summary(self):
        print("\n" + "=" * 60)
        print("FORMAT!() MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files modified: {self.stats['files_modified']}")
        print(f"format!() responses migrated: {self.stats['responses_migrated']}")

def main():
    project_root = Path(__file__).parent.parent
    handlers_dir = project_root / 'src' / 'handlers'

    migrator = FormatErrorMigrator()
    migrator.migrate_directory(handlers_dir)
    migrator.print_summary()

if __name__ == '__main__':
    main()
