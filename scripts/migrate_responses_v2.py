#!/usr/bin/env python3
"""
Enhanced HTTP Response Standardization Migration Script
Handles complex patterns including Ok() wrappers and multi-line formats
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Tuple

class EnhancedResponseMigrator:
    def __init__(self):
        self.stats = {
            'files_modified': 0,
            'responses_migrated': 0,
            'imports_added': 0,
            'errors': []
        }

    def has_macro_import(self, content: str) -> bool:
        """Check if file already imports response macros"""
        macro_names = ['ok_json', 'error_json', 'bad_request', 'not_found',
                      'created_json', 'unauthorized', 'forbidden', 'conflict',
                      'too_many_requests', 'service_unavailable', 'payload_too_large']
        return any(f'use crate::{{{macro}' in content or f'use crate::{macro}' in content
                   for macro in macro_names)

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

            import_statement = '''
use crate::{
    ok_json, created_json, error_json, bad_request, not_found,
    unauthorized, forbidden, conflict, no_content, accepted,
    too_many_requests, service_unavailable, payload_too_large
};
'''

            content = content[:insert_pos] + import_statement + content[insert_pos:]
            self.stats['imports_added'] += 1

        return content

    def migrate_content(self, content: str) -> Tuple[str, int]:
        """Migrate all HttpResponse patterns in content"""
        migrations = 0

        # Pattern 1: Ok(HttpResponse::InternalServerError().json(...))
        pattern = r'Ok\(HttpResponse::InternalServerError\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)\)'
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, r'error_json!("\1")', content)
            migrations += matches
            print(f"  - Migrated {matches} Ok(InternalServerError) patterns")

        # Pattern 2: Ok(HttpResponse::BadRequest().json(...))
        pattern = r'Ok\(HttpResponse::BadRequest\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)\)'
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, r'bad_request!("\1")', content)
            migrations += matches
            print(f"  - Migrated {matches} Ok(BadRequest) patterns")

        # Pattern 3: Ok(HttpResponse::NotFound().json(...))
        pattern = r'Ok\(HttpResponse::NotFound\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)\)'
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, r'not_found!("\1")', content)
            migrations += matches
            print(f"  - Migrated {matches} Ok(NotFound) patterns")

        # Pattern 4: Ok(HttpResponse::TooManyRequests().json(...))
        pattern = r'Ok\(HttpResponse::TooManyRequests\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)\)'
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, r'too_many_requests!("\1")', content)
            migrations += matches
            print(f"  - Migrated {matches} Ok(TooManyRequests) patterns")

        # Pattern 5: Ok(HttpResponse::PayloadTooLarge().json(...))
        pattern = r'Ok\(HttpResponse::PayloadTooLarge\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)\)'
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, r'payload_too_large!("\1")', content)
            migrations += matches
            print(f"  - Migrated {matches} Ok(PayloadTooLarge) patterns")

        # Pattern 6: Ok(HttpResponse::ServiceUnavailable().json(...))
        pattern = r'Ok\(HttpResponse::ServiceUnavailable\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)\)'
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, r'service_unavailable!("\1")', content)
            migrations += matches
            print(f"  - Migrated {matches} Ok(ServiceUnavailable) patterns")

        # Pattern 7: Direct HttpResponse::InternalServerError().json(...)
        pattern = r'HttpResponse::InternalServerError\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, r'error_json!("\1").unwrap()', content)
            migrations += matches
            print(f"  - Migrated {matches} direct InternalServerError patterns")

        # Pattern 8: Direct HttpResponse::NotFound().json(...)
        pattern = r'HttpResponse::NotFound\(\)\.json\((?:serde_)?json!\(\s*\{\s*"error"\s*:\s*"([^"]+)"\s*\}\s*\)\)'
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, r'not_found!("\1").unwrap()', content)
            migrations += matches
            print(f"  - Migrated {matches} direct NotFound patterns")

        # Pattern 9: HttpResponse::Ok() without .json() (streaming/SSE)
        pattern = r'HttpResponse::Ok\(\)(?!\.json)'
        matches = len(re.findall(pattern, content))
        if matches:
            # This needs manual review - keep as is for now
            print(f"  ⚠ Found {matches} HttpResponse::Ok() without .json() - needs manual review")

        return content, migrations

    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            content, migrations = self.migrate_content(original_content)

            if migrations > 0:
                # Add imports
                content = self.add_macro_imports(content)

                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.stats['files_modified'] += 1
                self.stats['responses_migrated'] += migrations
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

        print(f"\nProcessing {len(rs_files)} Rust files in {directory}")
        print("=" * 60)

        for file_path in rs_files:
            # Skip test files
            if '/tests/' in str(file_path):
                continue

            # Check if file has HttpResponse patterns
            with open(file_path, 'r') as f:
                content = f.read()
                if 'HttpResponse::' in content and 'use actix_web' in content:
                    print(f"\nProcessing: {file_path.relative_to(directory.parent)}")
                    self.migrate_file(file_path)

    def print_summary(self):
        """Print migration summary"""
        print("\n" + "=" * 60)
        print("ENHANCED MIGRATION SUMMARY")
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

    print("Enhanced HTTP Response Standardization Migration")
    print("=" * 60)
    print(f"Target directory: {handlers_dir}")

    migrator = EnhancedResponseMigrator()
    migrator.migrate_directory(handlers_dir)
    migrator.print_summary()

    sys.exit(0 if not migrator.stats['errors'] else 1)

if __name__ == '__main__':
    main()
