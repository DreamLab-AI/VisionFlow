#!/usr/bin/env python3
"""
Generate backlinks file showing all inbound references to each document
"""

import json
from pathlib import Path
from collections import defaultdict

def generate_backlinks_report(docs_root: str, report_path: str):
    """Generate comprehensive backlinks report"""

    docs_path = Path(docs_root)
    backlinks = defaultdict(list)

    # Scan all markdown files for links
    for md_file in docs_path.rglob("*.md"):
        if 'working' in md_file.parts or 'node_modules' in md_file.parts:
            continue

        rel_path = str(md_file.relative_to(docs_path))

        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all markdown links
        import re
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

        for link_text, link_url in links:
            if link_url.startswith(('http://', 'https://', '#')):
                continue

            # Resolve relative link
            try:
                target = (md_file.parent / link_url).resolve()
                if target.exists() and target.suffix == '.md':
                    target_rel = str(target.relative_to(docs_path))
                    backlinks[target_rel].append({
                        'source': rel_path,
                        'text': link_text
                    })
            except:
                pass

    # Generate markdown report
    output = ["# Documentation Backlinks\n"]
    output.append(f"Generated for {len(backlinks)} documents with inbound links\n")
    output.append("---\n")

    for target, sources in sorted(backlinks.items()):
        output.append(f"\n## {target}\n")
        output.append(f"Referenced by {len(sources)} document(s):\n")

        for source_info in sources:
            output.append(f"- [{source_info['text']}]({source_info['source']})")

        output.append("")

    # Save report
    output_path = Path(report_path)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"Backlinks report generated: {output_path}")
    print(f"Documents with inbound links: {len(backlinks)}")
    print(f"Total backlinks: {sum(len(sources) for sources in backlinks.values())}")

if __name__ == "__main__":
    docs_root = "/home/devuser/workspace/project/docs"
    report_path = "/home/devuser/workspace/project/docs/working/BACKLINKS.md"

    generate_backlinks_report(docs_root, report_path)
