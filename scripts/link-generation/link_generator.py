#!/usr/bin/env python3
"""
Comprehensive Documentation Link Generator
Analyzes 306 documentation files and generates bidirectional links
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import yaml

class DocumentAnalyzer:
    def __init__(self, docs_root: str):
        self.docs_root = Path(docs_root)
        self.documents: Dict[str, Dict] = {}
        self.link_graph: Dict[str, Dict[str, float]] = defaultdict(dict)

    def scan_documents(self) -> int:
        """Scan all markdown files and extract metadata"""
        count = 0
        for md_file in self.docs_root.rglob("*.md"):
            if self._should_skip(md_file):
                continue

            rel_path = str(md_file.relative_to(self.docs_root))
            self.documents[rel_path] = self._analyze_document(md_file)
            count += 1

        print(f"Scanned {count} documents")
        return count

    def _should_skip(self, path: Path) -> bool:
        """Skip certain directories and files"""
        skip_dirs = {'node_modules', '.git', 'working'}
        return any(part in skip_dirs for part in path.parts)

    def _analyze_document(self, file_path: Path) -> Dict:
        """Extract front matter, content, and metadata from document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract front matter
        front_matter = {}
        if content.startswith('---'):
            match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if match:
                try:
                    front_matter = yaml.safe_load(match.group(1)) or {}
                except:
                    pass

        # Extract headers
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)

        # Extract existing links
        existing_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

        # Extract words for similarity
        words = set(re.findall(r'\b[a-z]{3,}\b', content.lower()))

        return {
            'path': file_path,
            'title': front_matter.get('title', headers[0] if headers else file_path.stem),
            'tags': set(front_matter.get('tags', [])),
            'category': front_matter.get('category', ''),
            'headers': headers,
            'existing_links': existing_links,
            'words': words,
            'word_count': len(content.split()),
            'content': content
        }

    def calculate_similarities(self):
        """Calculate similarity scores between all document pairs"""
        docs_list = list(self.documents.items())

        for i, (path1, doc1) in enumerate(docs_list):
            for path2, doc2 in docs_list[i+1:]:
                score = self._calculate_similarity(doc1, doc2)
                if score > 0.1:  # Threshold for relevance
                    self.link_graph[path1][path2] = score
                    self.link_graph[path2][path1] = score

        print(f"Calculated {sum(len(links) for links in self.link_graph.values()) // 2} relationships")

    def _calculate_similarity(self, doc1: Dict, doc2: Dict) -> float:
        """Calculate weighted similarity between two documents"""
        score = 0.0

        # Tag overlap (highest weight)
        if doc1['tags'] and doc2['tags']:
            tag_overlap = len(doc1['tags'] & doc2['tags']) / len(doc1['tags'] | doc2['tags'])
            score += tag_overlap * 0.4

        # Category match
        if doc1['category'] and doc1['category'] == doc2['category']:
            score += 0.3

        # Word similarity (Jaccard)
        if doc1['words'] and doc2['words']:
            word_overlap = len(doc1['words'] & doc2['words']) / len(doc1['words'] | doc2['words'])
            score += word_overlap * 0.2

        # Path proximity (same directory)
        path1_parts = Path(doc1['path']).parts
        path2_parts = Path(doc2['path']).parts
        if len(path1_parts) > 1 and len(path2_parts) > 1:
            if path1_parts[-2] == path2_parts[-2]:
                score += 0.1

        return score

    def detect_relationships(self) -> Dict[str, Dict[str, List[str]]]:
        """Detect specific relationship types between documents"""
        relationships = defaultdict(lambda: {
            'prerequisites': [],
            'related': [],
            'siblings': [],
            'children': [],
            'references': []
        })

        for path, doc in self.documents.items():
            # Find top related documents
            if path in self.link_graph:
                sorted_links = sorted(
                    self.link_graph[path].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                for related_path, score in sorted_links:
                    if score > 0.4:
                        relationships[path]['related'].append(related_path)

            # Detect hierarchical relationships
            path_obj = Path(path)
            parent_dir = path_obj.parent

            # Find siblings (same directory)
            for other_path in self.documents:
                if other_path == path:
                    continue
                other_path_obj = Path(other_path)

                if other_path_obj.parent == parent_dir:
                    relationships[path]['siblings'].append(other_path)

                # Children (subdirectory files)
                if str(other_path_obj.parent).startswith(str(path_obj.parent / path_obj.stem)):
                    relationships[path]['children'].append(other_path)

        return relationships

class LinkInjector:
    def __init__(self, analyzer: DocumentAnalyzer, relationships: Dict):
        self.analyzer = analyzer
        self.relationships = relationships
        self.injected_count = 0

    def inject_links(self, dry_run: bool = False) -> List[str]:
        """Inject navigation links into all documents"""
        modified_files = []

        for path, doc in self.analyzer.documents.items():
            new_content = self._generate_link_section(path, doc)

            if new_content and new_content != doc['content']:
                if not dry_run:
                    with open(doc['path'], 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    self.injected_count += 1
                modified_files.append(path)

        print(f"{'Would modify' if dry_run else 'Modified'} {len(modified_files)} files")
        return modified_files

    def _generate_link_section(self, path: str, doc: Dict) -> str:
        """Generate navigation section for a document"""
        content = doc['content']
        rels = self.relationships[path]

        # Build navigation section
        nav_lines = []

        # Prerequisites
        if rels['prerequisites']:
            nav_lines.append("## Prerequisites\n")
            for prereq in rels['prerequisites'][:3]:
                title = self.analyzer.documents[prereq]['title']
                nav_lines.append(f"- [{title}]({self._relative_link(path, prereq)})")
            nav_lines.append("")

        # Related Documentation
        if rels['related']:
            nav_lines.append("## Related Documentation\n")
            for related in rels['related'][:5]:
                title = self.analyzer.documents[related]['title']
                nav_lines.append(f"- [{title}]({self._relative_link(path, related)})")
            nav_lines.append("")

        # Child Pages
        if rels['children']:
            nav_lines.append("## Sub-Topics\n")
            for child in rels['children'][:5]:
                title = self.analyzer.documents[child]['title']
                nav_lines.append(f"- [{title}]({self._relative_link(path, child)})")
            nav_lines.append("")

        if not nav_lines:
            return content

        # Remove existing navigation section if present
        content = re.sub(
            r'\n## (?:Prerequisites|Related Documentation|Sub-Topics|See Also)\n.*?(?=\n##|\Z)',
            '',
            content,
            flags=re.DOTALL
        )

        # Insert before last heading or at end
        nav_section = "\n---\n\n" + "\n".join(nav_lines)

        # Find last ## heading
        last_heading = list(re.finditer(r'\n## ', content))
        if last_heading and not content[last_heading[-1].start():].strip().startswith('## See Also'):
            insert_pos = last_heading[-1].start()
            return content[:insert_pos] + nav_section + content[insert_pos:]
        else:
            return content.rstrip() + "\n" + nav_section

    def _relative_link(self, from_path: str, to_path: str) -> str:
        """Calculate relative markdown link between documents"""
        from_parts = Path(from_path).parts
        to_parts = Path(to_path).parts

        # Find common prefix
        common = 0
        for a, b in zip(from_parts[:-1], to_parts[:-1]):
            if a == b:
                common += 1
            else:
                break

        # Build relative path
        ups = len(from_parts) - common - 1
        rel_parts = ['..'] * ups + list(to_parts[common:])

        return '/'.join(rel_parts)

class LinkValidator:
    def __init__(self, analyzer: DocumentAnalyzer):
        self.analyzer = analyzer
        self.errors: List[Dict] = []

    def validate(self) -> Dict:
        """Validate all links in documentation"""
        broken_links = []
        orphaned_files = []
        stats = {
            'total_docs': len(self.analyzer.documents),
            'total_links': 0,
            'broken_links': 0,
            'orphaned_files': 0,
            'avg_outbound': 0,
            'avg_inbound': 0
        }

        inbound_counts = defaultdict(int)
        outbound_counts = defaultdict(int)

        for path, doc in self.analyzer.documents.items():
            for link_text, link_url in doc['existing_links']:
                stats['total_links'] += 1
                outbound_counts[path] += 1

                # Check if link is valid
                if link_url.startswith(('http://', 'https://', '#')):
                    continue

                # Resolve relative link
                try:
                    target = (doc['path'].parent / link_url).resolve()

                    if not target.exists():
                        broken_links.append({
                            'source': path,
                            'link': link_url,
                            'text': link_text
                        })
                        stats['broken_links'] += 1
                    elif target.is_relative_to(self.analyzer.docs_root):
                        target_rel = str(target.relative_to(self.analyzer.docs_root))
                        inbound_counts[target_rel] += 1
                except (ValueError, OSError):
                    # Link outside docs root or invalid
                    pass

        # Find orphaned files (no inbound links)
        for path in self.analyzer.documents:
            if inbound_counts[path] == 0 and outbound_counts[path] == 0:
                orphaned_files.append(path)

        stats['orphaned_files'] = len(orphaned_files)
        stats['avg_outbound'] = sum(outbound_counts.values()) / len(self.analyzer.documents)
        stats['avg_inbound'] = sum(inbound_counts.values()) / len(self.analyzer.documents)

        return {
            'stats': stats,
            'broken_links': broken_links,
            'orphaned_files': orphaned_files,
            'inbound_counts': dict(inbound_counts),
            'outbound_counts': dict(outbound_counts)
        }

def main():
    docs_root = "/home/devuser/workspace/project/docs"

    print("=== Documentation Link Generation System ===\n")

    # Phase 1: Scan and analyze
    print("Phase 1: Scanning documents...")
    analyzer = DocumentAnalyzer(docs_root)
    doc_count = analyzer.scan_documents()

    # Phase 2: Calculate similarities
    print("\nPhase 2: Calculating document similarities...")
    analyzer.calculate_similarities()

    # Phase 3: Detect relationships
    print("\nPhase 3: Detecting relationships...")
    relationships = analyzer.detect_relationships()

    # Phase 4: Inject links
    print("\nPhase 4: Injecting navigation links...")
    injector = LinkInjector(analyzer, relationships)
    modified = injector.inject_links(dry_run=False)

    # Phase 5: Validate
    print("\nPhase 5: Validating links...")
    validator = LinkValidator(analyzer)
    validation = validator.validate()

    # Generate reports
    print("\nGenerating reports...")

    report_data = {
        'total_documents': doc_count,
        'modified_documents': len(modified),
        'validation': validation,
        'sample_relationships': {
            path: rels for path, rels in list(relationships.items())[:5]
        }
    }

    # Save report
    report_path = Path(docs_root) / "working" / "link-injection-report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Documents scanned: {doc_count}")
    print(f"Documents modified: {len(modified)}")
    print(f"Total links: {validation['stats']['total_links']}")
    print(f"Broken links: {validation['stats']['broken_links']}")
    print(f"Orphaned files: {validation['stats']['orphaned_files']}")
    print(f"Avg outbound links: {validation['stats']['avg_outbound']:.1f}")
    print(f"Avg inbound links: {validation['stats']['avg_inbound']:.1f}")
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
