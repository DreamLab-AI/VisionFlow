#!/usr/bin/env python3
"""
Documentation Link Analysis Tool
Analyzes all markdown files in docs/ to build reference graphs and identify linking opportunities.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from urllib.parse import unquote

class DocumentationAnalyzer:
    def __init__(self, docs_root: str):
        self.docs_root = Path(docs_root)
        self.documents = {}
        self.link_graph = defaultdict(set)  # outgoing links
        self.backlink_graph = defaultdict(set)  # incoming links
        self.broken_links = []
        self.orphaned_docs = set()
        self.link_patterns = Counter()
        
    def analyze_all(self):
        """Run complete analysis"""
        print("🔍 Starting documentation analysis...")
        
        self.discover_documents()
        self.extract_all_links()
        self.build_reference_graph()
        self.identify_broken_links()
        self.identify_orphaned_docs()
        self.analyze_link_patterns()
        
        print("✅ Analysis complete!")
        return self.generate_report()
    
    def discover_documents(self):
        """Find all markdown documents"""
        print("📄 Discovering documents...")
        
        for md_file in self.docs_root.rglob("*.md"):
            rel_path = md_file.relative_to(self.docs_root)
            self.documents[str(rel_path)] = {
                'path': md_file,
                'relative_path': str(rel_path),
                'title': self.extract_title(md_file),
                'size': md_file.stat().st_size,
                'links': [],
                'headings': self.extract_headings(md_file)
            }
        
        print(f"Found {len(self.documents)} documents")
    
    def extract_title(self, file_path: Path) -> str:
        """Extract document title from first heading or filename"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('# '):
                        return line.strip()[2:]
                    elif line.strip():
                        break
        except:
            pass
        return file_path.stem
    
    def extract_headings(self, file_path: Path) -> List[str]:
        """Extract all headings from document"""
        headings = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        level = len(line) - len(line.lstrip('#'))
                        heading = line.lstrip('# ').strip()
                        headings.append({'level': level, 'text': heading})
        except:
            pass
        return headings
    
    def extract_all_links(self):
        """Extract all markdown links from documents"""
        print("🔗 Extracting links...")
        
        link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
        
        for doc_path, doc_info in self.documents.items():
            try:
                with open(doc_info['path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                links = []
                for match in link_pattern.finditer(content):
                    link_text = match.group(1)
                    link_url = match.group(2)
                    
                    # Skip external links (http/https)
                    if link_url.startswith(('http://', 'https://', 'mailto:')):
                        continue
                    
                    # Clean and normalize internal links
                    clean_url = self.normalize_link(link_url, doc_path)
                    
                    links.append({
                        'text': link_text,
                        'url': link_url,
                        'normalized': clean_url,
                        'line': content[:match.start()].count('\n') + 1
                    })
                
                doc_info['links'] = links
                
            except Exception as e:
                print(f"⚠️  Error processing {doc_path}: {e}")
    
    def normalize_link(self, link_url: str, source_doc: str) -> str:
        """Normalize link URL to consistent format"""
        # Remove fragment identifiers
        if '#' in link_url:
            link_url = link_url.split('#')[0]
        
        # Skip empty links
        if not link_url:
            return None
        
        # Convert relative paths to absolute paths from docs root
        if link_url.startswith('./'):
            link_url = link_url[2:]
        elif link_url.startswith('../'):
            # Handle relative paths
            source_dir = Path(source_doc).parent
            resolved = (source_dir / link_url).resolve()
            try:
                return str(resolved.relative_to(Path('.')))
            except ValueError:
                return link_url
        
        # Remove leading slash
        if link_url.startswith('/'):
            link_url = link_url[1:]
        
        return link_url
    
    def build_reference_graph(self):
        """Build bidirectional reference graph"""
        print("🕸️ Building reference graph...")
        
        for doc_path, doc_info in self.documents.items():
            for link in doc_info['links']:
                if link['normalized']:
                    target = link['normalized']
                    
                    # Add to outgoing links
                    self.link_graph[doc_path].add(target)
                    
                    # Add to incoming links (backlinks)
                    self.backlink_graph[target].add(doc_path)
    
    def identify_broken_links(self):
        """Find broken internal links"""
        print("🔍 Identifying broken links...")
        
        for doc_path, doc_info in self.documents.items():
            for link in doc_info['links']:
                if link['normalized']:
                    target = link['normalized']
                    
                    # Check if target exists
                    target_exists = (
                        target in self.documents or
                        any(target.endswith(doc) for doc in self.documents) or
                        (self.docs_root / target).exists()
                    )
                    
                    if not target_exists:
                        self.broken_links.append({
                            'source': doc_path,
                            'target': target,
                            'text': link['text'],
                            'line': link['line'],
                            'original_url': link['url']
                        })
    
    def identify_orphaned_docs(self):
        """Find documents with no incoming links"""
        print("🏝️ Identifying orphaned documents...")
        
        # Documents that have no incoming links (except index files)
        for doc_path in self.documents:
            if (doc_path not in self.backlink_graph and 
                not doc_path.endswith('index.md') and
                not doc_path.endswith('README.md') and
                doc_path != 'README.md'):
                self.orphaned_docs.add(doc_path)
    
    def analyze_link_patterns(self):
        """Analyze common linking patterns"""
        print("📊 Analyzing link patterns...")
        
        for doc_path, targets in self.link_graph.items():
            for target in targets:
                # Categorize link types
                if target.endswith('.md'):
                    self.link_patterns['markdown_files'] += 1
                elif target.startswith('api/'):
                    self.link_patterns['api_docs'] += 1
                elif target.startswith('architecture/'):
                    self.link_patterns['architecture'] += 1
                elif target.startswith('client/'):
                    self.link_patterns['client_docs'] += 1
                elif target.startswith('server/'):
                    self.link_patterns['server_docs'] += 1
                else:
                    self.link_patterns['other'] += 1
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        # Find hub documents (most outgoing links)
        hub_docs = sorted(
            [(doc, len(targets)) for doc, targets in self.link_graph.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        # Find most referenced documents (most incoming links)
        popular_docs = sorted(
            [(doc, len(sources)) for doc, sources in self.backlink_graph.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        return {
            'summary': {
                'total_documents': len(self.documents),
                'total_internal_links': sum(len(targets) for targets in self.link_graph.values()),
                'broken_links': len(self.broken_links),
                'orphaned_docs': len(self.orphaned_docs),
                'connected_docs': len(self.documents) - len(self.orphaned_docs)
            },
            'hub_documents': hub_docs,
            'popular_documents': popular_docs,
            'broken_links': self.broken_links[:20],  # Top 20 broken links
            'orphaned_documents': list(self.orphaned_docs),
            'link_patterns': dict(self.link_patterns),
            'documents': self.documents,
            'link_graph': {k: list(v) for k, v in self.link_graph.items()},
            'backlink_graph': {k: list(v) for k, v in self.backlink_graph.items()}
        }

def main():
    analyzer = DocumentationAnalyzer('/workspace/ext/docs')
    report = analyzer.analyze_all()
    
    # Save detailed report
    with open('/workspace/ext/docs/analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("📈 DOCUMENTATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"📄 Total Documents: {report['summary']['total_documents']}")
    print(f"🔗 Internal Links: {report['summary']['total_internal_links']}")
    print(f"❌ Broken Links: {report['summary']['broken_links']}")
    print(f"🏝️ Orphaned Docs: {report['summary']['orphaned_docs']}")
    print(f"🔗 Connected Docs: {report['summary']['connected_docs']}")
    
    print("\n📊 Top Hub Documents (most outgoing links):")
    for doc, count in report['hub_documents'][:5]:
        print(f"  {doc}: {count} links")
    
    print("\n⭐ Most Referenced Documents:")
    for doc, count in report['popular_documents'][:5]:
        print(f"  {doc}: {count} backlinks")
    
    if report['broken_links']:
        print("\n❌ Sample Broken Links:")
        for link in report['broken_links'][:5]:
            print(f"  {link['source']} -> {link['target']}")
    
    if report['orphaned_documents']:
        print(f"\n🏝️ Sample Orphaned Documents:")
        for doc in list(report['orphaned_documents'])[:5]:
            print(f"  {doc}")

if __name__ == "__main__":
    main()