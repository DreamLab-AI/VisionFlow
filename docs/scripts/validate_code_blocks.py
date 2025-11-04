#!/usr/bin/env python3
"""Validate code blocks in documentation.

Extracts and validates code examples from markdown files.
"""

import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional


class CodeBlockValidator:
    """Validates code blocks extracted from markdown."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'rust': {'valid': 0, 'invalid': 0, 'errors': []},
            'typescript': {'valid': 0, 'invalid': 0, 'errors': []},
            'python': {'valid': 0, 'invalid': 0, 'errors': []},
            'sql': {'valid': 0, 'invalid': 0, 'errors': []},
        }

    def extract_code_blocks(self, content: str, language: str) -> List[Tuple[str, int]]:
        """Extract code blocks for a specific language."""
        pattern = rf'```{language}\n(.*?)```'
        matches = []

        for match in re.finditer(pattern, content, re.DOTALL):
            code = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            matches.append((code, line_num))

        return matches

    def validate_rust(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Rust code."""
        with tempfile.NamedTemporaryFile(suffix='.rs', mode='w', delete=False) as f:
            # Try as-is first
            f.write(code)
            f.flush()

            result = subprocess.run(
                ['rustc', '--crate-type', 'lib', f.name],
                capture_output=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, None

            # Try with main wrapper
            f.seek(0)
            f.truncate()
            f.write(f"fn main() {{\n{code}\n}}")
            f.flush()

            result = subprocess.run(
                ['rustc', f.name],
                capture_output=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, None

            return False, result.stderr.decode()[:500]

    def validate_typescript(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate TypeScript code."""
        with tempfile.NamedTemporaryFile(suffix='.ts', mode='w', delete=False) as f:
            f.write(code)
            f.flush()

            result = subprocess.run(
                ['npx', 'tsc', '--noEmit', '--skipLibCheck', f.name],
                capture_output=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, None

            return False, result.stderr.decode()[:500]

    def validate_python(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code (syntax only)."""
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def validate_file(self, file_path: Path, languages: List[str]):
        """Validate all code blocks in a file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            return

        print(f"\nValidating {file_path}")

        for lang in languages:
            if lang not in self.results:
                continue

            blocks = self.extract_code_blocks(content, lang)

            if not blocks:
                continue

            print(f"  Found {len(blocks)} {lang} block(s)")

            for code, line_num in blocks:
                validator = getattr(self, f'validate_{lang}', None)

                if not validator:
                    continue

                try:
                    is_valid, error = validator(code)

                    if is_valid:
                        self.results[lang]['valid'] += 1
                        if self.verbose:
                            print(f"    ✓ Line {line_num}: Valid")
                    else:
                        self.results[lang]['invalid'] += 1
                        print(f"    ✗ Line {line_num}: Invalid")
                        if self.verbose and error:
                            print(f"      Error: {error[:200]}")
                        self.results[lang]['errors'].append({
                            'file': str(file_path),
                            'line': line_num,
                            'error': error
                        })
                except Exception as e:
                    self.results[lang]['invalid'] += 1
                    print(f"    ✗ Line {line_num}: Validation error: {e}")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        total_valid = 0
        total_invalid = 0

        for lang, stats in self.results.items():
            valid = stats['valid']
            invalid = stats['invalid']
            total = valid + invalid

            if total == 0:
                continue

            total_valid += valid
            total_invalid += invalid

            pct = (valid / total * 100) if total > 0 else 0

            print(f"\n{lang.upper()}: {valid}/{total} valid ({pct:.1f}%)")

            if invalid > 0:
                print(f"  ✗ {invalid} invalid block(s)")

        print("\n" + "=" * 60)
        total = total_valid + total_invalid
        if total > 0:
            pct = (total_valid / total * 100)
            print(f"TOTAL: {total_valid}/{total} valid ({pct:.1f}%)")
        else:
            print("No code blocks found to validate")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate code blocks in markdown files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--path', default='docs', help='Path to documentation root')
    parser.add_argument('--languages', nargs='+', default=['rust', 'typescript', 'python'],
                       help='Languages to validate')
    args = parser.parse_args()

    docs_root = Path(args.path)

    if not docs_root.exists():
        print(f"Error: Documentation path not found: {docs_root}", file=sys.stderr)
        sys.exit(1)

    # Check for required tools
    tools = {
        'rust': 'rustc',
        'typescript': 'npx',
        'python': 'python3'
    }

    available_languages = []
    for lang in args.languages:
        tool = tools.get(lang)
        if tool and subprocess.run(['which', tool], capture_output=True).returncode == 0:
            available_languages.append(lang)
        else:
            print(f"Warning: {tool} not found, skipping {lang} validation")

    if not available_languages:
        print("Error: No validation tools available", file=sys.stderr)
        sys.exit(1)

    validator = CodeBlockValidator(verbose=args.verbose)

    print(f"Validating code blocks in {docs_root}")
    print(f"Languages: {', '.join(available_languages)}")

    for md_file in sorted(docs_root.rglob('*.md')):
        validator.validate_file(md_file, available_languages)

    validator.print_summary()
