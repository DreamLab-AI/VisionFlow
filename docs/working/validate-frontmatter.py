#!/usr/bin/env python3
"""YAML Front Matter Validator for Documentation"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Standardised tag vocabulary
STANDARD_TAGS = {
    'architecture', 'api', 'authentication', 'agents', 'actors', 'binary-protocol',
    'caching', 'client', 'configuration', 'cuda', 'database', 'deployment', 'docker',
    'error-handling', 'features', 'github', 'gpu', 'guides', 'hexagonal', 'integration',
    'json', 'logging', 'mcp', 'memory', 'monitoring', 'neo4j', 'ontology', 'performance',
    'physics', 'protocol', 'python', 'reference', 'rest-api', 'rust', 'schema', 'security',
    'semantic', 'server', 'settings', 'testing', 'troubleshooting', 'typescript',
    'visualization', 'websocket', 'xr'
}

VALID_CATEGORIES = {'tutorial', 'howto', 'reference', 'explanation'}
VALID_DIFFICULTY = {'beginner', 'intermediate', 'advanced'}

def extract_frontmatter(file_path: str) -> tuple[dict | None, str | None]:
    """Extract YAML front matter from markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.startswith('---'):
            return None, None

        # Find the closing ---
        parts = content.split('---', 2)
        if len(parts) < 3:
            return None, "Malformed: Missing closing ---"

        yaml_content = parts[1].strip()
        if not yaml_content:
            return None, "Empty front matter"

        try:
            metadata = yaml.safe_load(yaml_content)
            return metadata, None
        except yaml.YAMLError as e:
            return None, f"YAML parse error: {str(e)}"

    except Exception as e:
        return None, f"File read error: {str(e)}"

def validate_frontmatter(file_path: str, metadata: dict) -> List[str]:
    """Validate front matter fields."""
    issues = []

    # Required fields
    if 'title' not in metadata:
        issues.append("Missing required field: title")
    elif not isinstance(metadata['title'], str):
        issues.append("title must be a string")

    if 'description' not in metadata:
        issues.append("Missing required field: description")
    elif not isinstance(metadata['description'], str):
        issues.append("description must be a string")

    if 'category' not in metadata:
        issues.append("Missing required field: category")
    elif metadata['category'] not in VALID_CATEGORIES:
        issues.append(f"Invalid category: {metadata['category']} (must be one of {VALID_CATEGORIES})")

    if 'tags' not in metadata:
        issues.append("Missing required field: tags")
    elif not isinstance(metadata['tags'], list):
        issues.append("tags must be an array")
    elif len(metadata['tags']) < 3 or len(metadata['tags']) > 5:
        issues.append(f"tags should have 3-5 items (found {len(metadata['tags'])})")

    # Validate optional fields
    if 'difficulty-level' in metadata and metadata['difficulty-level'] not in VALID_DIFFICULTY:
        issues.append(f"Invalid difficulty-level: {metadata['difficulty-level']}")

    if 'related-docs' in metadata and not isinstance(metadata['related-docs'], list):
        issues.append("related-docs must be an array")

    if 'dependencies' in metadata and not isinstance(metadata['dependencies'], list):
        issues.append("dependencies must be an array")

    # Validate tags against vocabulary
    if 'tags' in metadata and isinstance(metadata['tags'], list):
        invalid_tags = [tag for tag in metadata['tags'] if tag not in STANDARD_TAGS]
        if invalid_tags:
            issues.append(f"Non-standard tags: {invalid_tags}")

    return issues

def validate_all_docs(docs_dir: str) -> Dict[str, Any]:
    """Validate all markdown files in docs directory."""
    results = {
        'total_files': 0,
        'files_with_frontmatter': 0,
        'files_missing_frontmatter': [],
        'missing_required_fields': [],
        'invalid_categories': [],
        'invalid_tags': [],
        'malformed_yaml': [],
        'all_issues': {}
    }

    docs_path = Path(docs_dir)

    for md_file in docs_path.rglob('*.md'):
        file_path = str(md_file)
        rel_path = str(md_file.relative_to(docs_path))
        results['total_files'] += 1

        metadata, error = extract_frontmatter(file_path)

        if error:
            results['malformed_yaml'].append({
                'file': rel_path,
                'error': error
            })
            continue

        if metadata is None:
            results['files_missing_frontmatter'].append(rel_path)
            continue

        results['files_with_frontmatter'] += 1

        # Validate metadata
        issues = validate_frontmatter(file_path, metadata)

        if issues:
            results['all_issues'][rel_path] = issues

            # Categorize issues
            for issue in issues:
                if 'Missing required field' in issue:
                    results['missing_required_fields'].append({
                        'file': rel_path,
                        'issue': issue
                    })
                elif 'Invalid category' in issue:
                    results['invalid_categories'].append({
                        'file': rel_path,
                        'category': metadata.get('category')
                    })
                elif 'Non-standard tags' in issue:
                    results['invalid_tags'].append({
                        'file': rel_path,
                        'tags': [tag for tag in metadata.get('tags', []) if tag not in STANDARD_TAGS]
                    })

    # Calculate compliance
    if results['total_files'] > 0:
        results['frontmatter_compliance_percentage'] = round(
            (results['files_with_frontmatter'] / results['total_files']) * 100, 2
        )
    else:
        results['frontmatter_compliance_percentage'] = 0.0

    return results

def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate markdown summary report."""
    report = f"""# YAML Front Matter Validation Report

**Generated**: {datetime.now().isoformat()}

## Summary Statistics

- **Total Files**: {results['total_files']}
- **Files With Front Matter**: {results['files_with_frontmatter']}
- **Files Missing Front Matter**: {len(results['files_missing_frontmatter'])}
- **Front Matter Compliance**: {results['frontmatter_compliance_percentage']}%

## Issues Breakdown

### Malformed YAML ({len(results['malformed_yaml'])} files)

"""

    if results['malformed_yaml']:
        for item in results['malformed_yaml'][:10]:
            report += f"- `{item['file']}`: {item['error']}\n"
        if len(results['malformed_yaml']) > 10:
            report += f"- ... and {len(results['malformed_yaml']) - 10} more\n"
    else:
        report += "None\n"

    report += f"\n### Missing Required Fields ({len(results['missing_required_fields'])} issues)\n\n"

    if results['missing_required_fields']:
        for item in results['missing_required_fields'][:10]:
            report += f"- `{item['file']}`: {item['issue']}\n"
        if len(results['missing_required_fields']) > 10:
            report += f"- ... and {len(results['missing_required_fields']) - 10} more\n"
    else:
        report += "None\n"

    report += f"\n### Invalid Categories ({len(results['invalid_categories'])} files)\n\n"

    if results['invalid_categories']:
        for item in results['invalid_categories'][:10]:
            report += f"- `{item['file']}`: {item['category']}\n"
        if len(results['invalid_categories']) > 10:
            report += f"- ... and {len(results['invalid_categories']) - 10} more\n"
    else:
        report += "None\n"

    report += f"\n### Non-Standard Tags ({len(results['invalid_tags'])} files)\n\n"

    if results['invalid_tags']:
        for item in results['invalid_tags'][:10]:
            report += f"- `{item['file']}`: {item['tags']}\n"
        if len(results['invalid_tags']) > 10:
            report += f"- ... and {len(results['invalid_tags']) - 10} more\n"
    else:
        report += "None\n"

    report += f"\n### Files Missing Front Matter ({len(results['files_missing_frontmatter'])} files)\n\n"

    if results['files_missing_frontmatter']:
        for file in results['files_missing_frontmatter'][:20]:
            report += f"- `{file}`\n"
        if len(results['files_missing_frontmatter']) > 20:
            report += f"- ... and {len(results['files_missing_frontmatter']) - 20} more\n"
    else:
        report += "None\n"

    report += f"""

## Standard Tag Vocabulary

{', '.join(sorted(STANDARD_TAGS))}

## Recommendations

"""

    if results['frontmatter_compliance_percentage'] < 99:
        report += f"- **Priority**: Add front matter to {len(results['files_missing_frontmatter'])} files\n"

    if results['missing_required_fields']:
        report += f"- Fix {len(results['missing_required_fields'])} required field issues\n"

    if results['invalid_tags']:
        report += f"- Standardize {len(results['invalid_tags'])} files with non-standard tags\n"

    if results['malformed_yaml']:
        report += f"- Repair {len(results['malformed_yaml'])} files with malformed YAML\n"

    if results['frontmatter_compliance_percentage'] >= 99:
        report += "- **Excellent**: Documentation metadata compliance is at target (99%+)\n"

    return report

def main():
    docs_dir = '/home/devuser/workspace/project/docs'

    print("Validating YAML front matter...")
    results = validate_all_docs(docs_dir)

    # Save JSON report
    json_path = '/home/devuser/workspace/project/docs/working/hive-frontmatter-validation.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON report saved: {json_path}")

    # Save Markdown report
    md_report = generate_markdown_report(results)
    md_path = '/home/devuser/workspace/project/docs/working/hive-frontmatter-validation.md'
    with open(md_path, 'w') as f:
        f.write(md_report)
    print(f"Markdown report saved: {md_path}")

    print(f"\nSummary:")
    print(f"  Total files: {results['total_files']}")
    print(f"  With front matter: {results['files_with_frontmatter']}")
    print(f"  Missing front matter: {len(results['files_missing_frontmatter'])}")
    print(f"  Compliance: {results['frontmatter_compliance_percentage']}%")
    print(f"  Issues: {len(results['all_issues'])} files with validation issues")

if __name__ == '__main__':
    main()
