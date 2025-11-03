#!/usr/bin/env python3
"""
HTTP Response Standardization Migration Script
Automates the migration of direct HttpResponse calls to standardized macros
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple

class ResponseMigrator:
    def __init__(self):
        self.stats = {
            'files_modified': 0,
            'responses_migrated': 0,
            'imports_added': 0,
            'errors': []
        }

        # Migration patterns (pattern, replacement, description)
        self.patterns: List[Tuple[str, str, str]] = [
            # Ok().json() patterns
            (r'HttpResponse::Ok\(\)\.json\(([^)]+)\)',
             r'ok_json!(\1)',
             'Ok().json()'),

            # Created().json() patterns
            (r'HttpResponse::Created\(\)\.json\(([^)]+)\)',
             r'created_json!(\1)',
             'Created().json()'),

            # BadRequest with error object
            (r'HttpResponse::BadRequest\(\)\.json\(serde_json::json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)',
             r'bad_request!("\1")',
             'BadRequest with error'),

            # InternalServerError with error object
            (r'HttpResponse::InternalServerError\(\)\.json\(serde_json::json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)',
             r'error_json!("\1")',
             'InternalServerError with error'),

            # NotFound with error object
            (r'HttpResponse::NotFound\(\)\.json\(serde_json::json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)',
             r'not_found!("\1")',
             'NotFound with error'),

            # Unauthorized with error object
            (r'HttpResponse::Unauthorized\(\)\.json\(serde_json::json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)',
             r'unauthorized!("\1")',
             'Unauthorized with error'),

            # Forbidden with error object
            (r'HttpResponse::Forbidden\(\)\.json\(serde_json::json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)',
             r'forbidden!("\1")',
             'Forbidden with error'),

            # Conflict with error object
            (r'HttpResponse::Conflict\(\)\.json\(serde_json::json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)',
             r'conflict!("\1")',
             'Conflict with error'),

            # NoContent
            (r'HttpResponse::NoContent\(\)\.finish\(\)',
             r'no_content!()',
             'NoContent'),

            # TooManyRequests
            (r'HttpResponse::TooManyRequests\(\)\.json\(serde_json::json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)',
             r'too_many_requests!("\1")',
             'TooManyRequests'),

            # ServiceUnavailable
            (r'HttpResponse::ServiceUnavailable\(\)\.json\(serde_json::json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)',
             r'service_unavailable!("\1")',
             'ServiceUnavailable'),

            # Accepted
            (r'HttpResponse::Accepted\(\)\.json\(([^)]+)\)',
             r'accepted!(\1)',
             'Accepted().json()'),
        ]

    def has_macro_import(self, content: str) -> bool:
        """Check if file already imports response macros"""
        return 'use crate::' in content and any(
            macro in content for macro in [
                'ok_json', 'error_json', 'bad_request', 'not_found',
                'created_json', 'unauthorized', 'forbidden', 'conflict'
            ]
        )

    def add_macro_imports(self, content: str) -> str:
        """Add response macro imports if not present"""
        if self.has_macro_import(content):
            return content

        # Find the last use statement
        use_pattern = r'(use\s+[^;]+;)'
        matches = list(re.finditer(use_pattern, content))

        if matches:
            last_use = matches[-1]
            insert_pos = last_use.end()

            import_statement = '\nuse crate::{ok_json, created_json, error_json, bad_request, not_found, unauthorized, forbidden, conflict, no_content, accepted};\n'

            content = content[:insert_pos] + import_statement + content[insert_pos:]
            self.stats['imports_added'] += 1

        return content

    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            content = original_content
            file_modified = False
            migrations_in_file = 0

            # Apply each pattern
            for pattern, replacement, desc in self.patterns:
                matches = len(re.findall(pattern, content))
                if matches > 0:
                    content = re.sub(pattern, replacement, content)
                    migrations_in_file += matches
                    file_modified = True
                    print(f"  - Migrated {matches} {desc} patterns")

            if file_modified:
                # Add imports
                content = self.add_macro_imports(content)

                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats['files_modified'] += 1
                self.stats['responses_migrated'] += migrations_in_file
                return True

            return False

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.stats['errors'].append(error_msg)
            print(f"ERROR: {error_msg}", file=sys.stderr)
            return False

    def migrate_directory(self, directory: Path):
        """Migrate all .rs files in directory"""
        rs_files = list(directory.rglob('*.rs'))

        print(f"\nFound {len(rs_files)} Rust files in {directory}")
        print("=" * 60)

        for file_path in rs_files:
            # Skip test files for now
            if '/tests/' in str(file_path):
                continue

            print(f"\nProcessing: {file_path.relative_to(directory.parent)}")
            self.migrate_file(file_path)

    def print_summary(self):
        """Print migration summary"""
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files modified: {self.stats['files_modified']}")
        print(f"HttpResponse calls migrated: {self.stats['responses_migrated']}")
        print(f"Import statements added: {self.stats['imports_added']}")

        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors']:
                print(f"  - {error}")
        else:
            print("\n✓ No errors encountered")

def main():
    """Main execution"""
    project_root = Path(__file__).parent.parent
    handlers_dir = project_root / 'src' / 'handlers'

    if not handlers_dir.exists():
        print(f"ERROR: Handlers directory not found: {handlers_dir}", file=sys.stderr)
        sys.exit(1)

    print("HTTP Response Standardization Migration")
    print("=" * 60)
    print(f"Target directory: {handlers_dir}")

    migrator = ResponseMigrator()
    migrator.migrate_directory(handlers_dir)
    migrator.print_summary()

    sys.exit(0 if not migrator.stats['errors'] else 1)

if __name__ == '__main__':
    main()
