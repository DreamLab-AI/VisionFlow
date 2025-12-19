#!/usr/bin/env python3
"""
Diagram Inspector for Documentation Alignment Hive Mind
Validates Mermaid diagrams and detects ASCII art for conversion
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Valid Mermaid diagram types
VALID_MERMAID_TYPES = {
    'graph', 'flowchart', 'sequenceDiagram', 'classDiagram',
    'stateDiagram', 'stateDiagram-v2', 'erDiagram', 'gantt',
    'pie', 'journey', 'gitGraph', 'C4Context', 'mindmap', 'timeline'
}

# ASCII box drawing characters
ASCII_PATTERNS = [
    r'[├─└│┌┐┬┴┼]',  # Basic box drawing
    r'[╔═╗║╚╝╠╣╦╩╬]',  # Double line box drawing
    r'[┏━┓┃┗┛┣┫┳┻╋]',  # Heavy box drawing
]

# Files that should have architecture diagrams
ARCHITECTURE_KEYWORDS = [
    'architecture', 'design', 'system', 'flow', 'pipeline',
    'data-flow', 'component', 'service', 'integration'
]

def find_markdown_files(docs_dir: str) -> List[Path]:
    """Find all markdown files in docs directory"""
    docs_path = Path(docs_dir)
    return list(docs_path.rglob("*.md"))

def extract_mermaid_blocks(file_path: Path) -> List[Dict]:
    """Extract all Mermaid code blocks from a file"""
    diagrams = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        return diagrams

    in_mermaid = False
    current_diagram = []
    start_line = 0

    for i, line in enumerate(lines, 1):
        if line.strip().startswith('```mermaid'):
            in_mermaid = True
            start_line = i
            current_diagram = []
        elif in_mermaid and line.strip().startswith('```'):
            in_mermaid = False
            diagram_content = '\n'.join(current_diagram)
            diagrams.append({
                'file': str(file_path.relative_to(file_path.parents[len(file_path.parts) - file_path.parts.index('docs') - 1])),
                'line': start_line,
                'content': diagram_content
            })
        elif in_mermaid:
            current_diagram.append(line)

    return diagrams

def validate_mermaid(diagram: Dict) -> Tuple[bool, List[str]]:
    """Validate Mermaid diagram syntax"""
    issues = []
    content = diagram['content'].strip()

    if not content:
        return False, ['Empty diagram']

    # Check first line for valid diagram type
    first_line = content.split('\n')[0].strip()

    # Extract diagram type
    diagram_type = None
    for valid_type in VALID_MERMAID_TYPES:
        if first_line.startswith(valid_type):
            diagram_type = valid_type
            break

    if not diagram_type:
        issues.append(f'Invalid or missing diagram type. First line: "{first_line}"')
        return False, issues

    # Check for common syntax errors
    if diagram_type in ['graph', 'flowchart']:
        if not re.search(r'(TB|TD|BT|RL|LR)', first_line):
            issues.append('Missing direction (TB/TD/BT/RL/LR) for graph/flowchart')

    # Check for unclosed brackets
    if content.count('[') != content.count(']'):
        issues.append('Unmatched square brackets')
    if content.count('(') != content.count(')'):
        issues.append('Unmatched parentheses')
    if content.count('{') != content.count('}'):
        issues.append('Unmatched curly braces')

    # Check for empty nodes
    if re.search(r'\[\s*\]|\(\s*\)|\{\s*\}', content):
        issues.append('Empty node labels detected')

    return len(issues) == 0, issues

def detect_ascii_diagrams(file_path: Path) -> List[Dict]:
    """Detect ASCII art diagrams in file"""
    ascii_diagrams = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        return ascii_diagrams

    for i, line in enumerate(lines, 1):
        for pattern in ASCII_PATTERNS:
            if re.search(pattern, line):
                # Get context (3 lines before and after)
                start = max(0, i - 4)
                end = min(len(lines), i + 3)
                context = ''.join(lines[start:end])

                ascii_diagrams.append({
                    'file': str(file_path.relative_to(file_path.parents[len(file_path.parts) - file_path.parts.index('docs') - 1])),
                    'line': i,
                    'content_preview': context.strip()[:200]
                })
                break  # Only report once per line

    return ascii_diagrams

def should_have_diagram(file_path: Path) -> bool:
    """Check if file should have a diagram based on name/path"""
    path_str = str(file_path).lower()
    return any(keyword in path_str for keyword in ARCHITECTURE_KEYWORDS)

def main():
    docs_dir = '/home/devuser/workspace/project/docs'
    output_json = '/home/devuser/workspace/project/docs/working/hive-diagram-validation.json'
    output_md = '/home/devuser/workspace/project/docs/working/hive-diagram-validation.md'

    print("Starting diagram validation...")

    # Find all markdown files
    md_files = find_markdown_files(docs_dir)
    print(f"Found {len(md_files)} markdown files")

    # Analyze diagrams
    mermaid_diagrams = []
    ascii_diagrams = []
    missing_diagrams = []
    diagram_types = {}

    for file_path in md_files:
        # Extract Mermaid diagrams
        diagrams = extract_mermaid_blocks(file_path)
        for diagram in diagrams:
            valid, issues = validate_mermaid(diagram)

            # Determine diagram type
            first_line = diagram['content'].strip().split('\n')[0]
            diagram_type = 'unknown'
            for valid_type in VALID_MERMAID_TYPES:
                if first_line.startswith(valid_type):
                    diagram_type = valid_type
                    break

            diagram_types[diagram_type] = diagram_types.get(diagram_type, 0) + 1

            mermaid_diagrams.append({
                'file': diagram['file'],
                'line': diagram['line'],
                'type': diagram_type,
                'valid': valid,
                'issues': issues
            })

        # Detect ASCII diagrams
        ascii = detect_ascii_diagrams(file_path)
        ascii_diagrams.extend(ascii)

        # Check if should have diagram but doesn't
        if should_have_diagram(file_path) and not diagrams:
            rel_path = str(file_path.relative_to(file_path.parents[len(file_path.parts) - file_path.parts.index('docs') - 1]))
            missing_diagrams.append(rel_path)

    # Calculate statistics
    total_mermaid = len(mermaid_diagrams)
    valid_mermaid = sum(1 for d in mermaid_diagrams if d['valid'])
    total_ascii = len(ascii_diagrams)
    git_compliant_percentage = (valid_mermaid / total_mermaid * 100) if total_mermaid > 0 else 0

    # Prepare output
    results = {
        'mermaid_diagrams': mermaid_diagrams,
        'ascii_diagrams': ascii_diagrams,
        'missing_diagrams': missing_diagrams,
        'diagram_type_distribution': diagram_types,
        'total_mermaid': total_mermaid,
        'valid_mermaid': valid_mermaid,
        'invalid_mermaid': total_mermaid - valid_mermaid,
        'total_ascii': total_ascii,
        'git_compliant_percentage': round(git_compliant_percentage, 2),
        'files_analyzed': len(md_files),
        'files_with_mermaid': len(set(d['file'] for d in mermaid_diagrams)),
        'files_with_ascii': len(set(d['file'] for d in ascii_diagrams))
    }

    # Write JSON output
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"JSON report written to {output_json}")

    # Generate Markdown summary
    md_content = f"""# Diagram Validation Report

## Summary Statistics

- **Total Markdown Files**: {results['files_analyzed']}
- **Files with Mermaid Diagrams**: {results['files_with_mermaid']}
- **Files with ASCII Diagrams**: {results['files_with_ascii']}
- **Total Mermaid Diagrams**: {results['total_mermaid']}
- **Valid Mermaid Diagrams**: {results['valid_mermaid']}
- **Invalid Mermaid Diagrams**: {results['invalid_mermaid']}
- **Git Compliance**: {results['git_compliant_percentage']}%
- **Total ASCII Diagrams**: {results['total_ascii']}

## Diagram Type Distribution

```
"""

    for dtype, count in sorted(diagram_types.items(), key=lambda x: x[1], reverse=True):
        md_content += f"{dtype}: {count}\n"

    md_content += """```

## Invalid Mermaid Diagrams

"""

    invalid_diagrams = [d for d in mermaid_diagrams if not d['valid']]
    if invalid_diagrams:
        for diagram in invalid_diagrams:
            md_content += f"\n### {diagram['file']}:{diagram['line']}\n"
            md_content += f"**Type**: {diagram['type']}\n\n"
            md_content += "**Issues**:\n"
            for issue in diagram['issues']:
                md_content += f"- {issue}\n"
    else:
        md_content += "No invalid Mermaid diagrams found.\n"

    md_content += f"""

## ASCII Diagrams Detected ({total_ascii} instances)

"""

    if ascii_diagrams:
        md_content += "The following files contain ASCII art that should be converted to Mermaid:\n\n"
        ascii_by_file = {}
        for ascii in ascii_diagrams:
            file = ascii['file']
            if file not in ascii_by_file:
                ascii_by_file[file] = []
            ascii_by_file[file].append(ascii['line'])

        for file, lines in sorted(ascii_by_file.items()):
            md_content += f"- **{file}**: {len(lines)} instances (lines: {', '.join(map(str, lines))})\n"
    else:
        md_content += "✅ No ASCII diagrams detected. All diagrams use Mermaid!\n"

    md_content += f"""

## Files Missing Diagrams ({len(missing_diagrams)} files)

Architecture-related files that should have diagrams:

"""

    if missing_diagrams:
        for file in sorted(missing_diagrams):
            md_content += f"- {file}\n"
    else:
        md_content += "All architecture files have diagrams.\n"

    md_content += """

## Recommendations

"""

    if total_ascii > 0:
        md_content += f"1. **Convert ASCII to Mermaid**: {total_ascii} ASCII diagrams detected across {results['files_with_ascii']} files\n"

    if results['invalid_mermaid'] > 0:
        md_content += f"2. **Fix Invalid Mermaid**: {results['invalid_mermaid']} diagrams have syntax errors\n"

    if len(missing_diagrams) > 0:
        md_content += f"3. **Add Missing Diagrams**: {len(missing_diagrams)} architecture files lack diagrams\n"

    if total_ascii == 0 and results['invalid_mermaid'] == 0 and len(missing_diagrams) == 0:
        md_content += "✅ All diagrams are valid Mermaid with proper syntax!\n"

    # Write Markdown output
    with open(output_md, 'w') as f:
        f.write(md_content)

    print(f"Markdown report written to {output_md}")
    print("\n=== Validation Complete ===")
    print(f"Mermaid Diagrams: {results['total_mermaid']} ({results['valid_mermaid']} valid)")
    print(f"ASCII Diagrams: {results['total_ascii']}")
    print(f"Git Compliance: {results['git_compliant_percentage']}%")

if __name__ == '__main__':
    main()
