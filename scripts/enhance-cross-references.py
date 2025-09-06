#!/usr/bin/env python3
"""
Documentation Cross-Reference Enhancement Tool
Adds comprehensive cross-references and backlinks throughout the documentation.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

@dataclass
class Document:
    path: Path
    relative_path: str
    title: str
    content: str
    links: List[Dict]
    headings: List[Dict]

class CrossReferenceEnhancer:
    def __init__(self, docs_root: str, analysis_file: str):
        self.docs_root = Path(docs_root)
        self.analysis_file = analysis_file
        self.load_analysis()
        self.documents = {}
        self.topic_mapping = self.build_topic_mapping()
        self.api_mapping = self.build_api_mapping()
        self.changes_made = []
        
    def load_analysis(self):
        """Load the analysis report"""
        with open(self.analysis_file, 'r') as f:
            self.analysis = json.load(f)
    
    def load_documents(self):
        """Load all documents with content"""
        print("📄 Loading documents...")
        for doc_path in self.analysis['documents']:
            full_path = self.docs_root / doc_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.documents[doc_path] = Document(
                    path=full_path,
                    relative_path=doc_path,
                    title=self.analysis['documents'][doc_path].get('title', ''),
                    content=content,
                    links=self.analysis['documents'][doc_path].get('links', []),
                    headings=self.analysis['documents'][doc_path].get('headings', [])
                )
    
    def build_topic_mapping(self) -> Dict[str, List[str]]:
        """Build mapping of topics to related documents"""
        topics = {
            'api': [],
            'websocket': [],
            'architecture': [],
            'client': [],
            'server': [],
            'deployment': [],
            'configuration': [],
            'agents': [],
            'mcp': [],
            'gpu': [],
            'cuda': [],
            'testing': [],
            'development': [],
            'security': [],
            'getting-started': [],
            'features': []
        }
        
        for doc_path in self.analysis['documents']:
            doc_lower = doc_path.lower()
            
            if 'api' in doc_lower:
                topics['api'].append(doc_path)
            if 'websocket' in doc_lower:
                topics['websocket'].append(doc_path)
            if 'architecture' in doc_lower:
                topics['architecture'].append(doc_path)
            if 'client' in doc_lower:
                topics['client'].append(doc_path)
            if 'server' in doc_lower:
                topics['server'].append(doc_path)
            if 'deployment' in doc_lower or 'docker' in doc_lower:
                topics['deployment'].append(doc_path)
            if 'config' in doc_lower or 'settings' in doc_lower:
                topics['configuration'].append(doc_path)
            if 'agent' in doc_lower:
                topics['agents'].append(doc_path)
            if 'mcp' in doc_lower:
                topics['mcp'].append(doc_path)
            if 'gpu' in doc_lower or 'cuda' in doc_lower:
                topics['gpu'].append(doc_path)
            if 'test' in doc_lower:
                topics['testing'].append(doc_path)
            if 'dev' in doc_lower or 'development' in doc_lower:
                topics['development'].append(doc_path)
            if 'security' in doc_lower or 'auth' in doc_lower:
                topics['security'].append(doc_path)
            if 'getting-started' in doc_lower or 'quickstart' in doc_lower:
                topics['getting-started'].append(doc_path)
            if 'features' in doc_lower:
                topics['features'].append(doc_path)
        
        return topics
    
    def build_api_mapping(self) -> Dict[str, List[str]]:
        """Build mapping of API endpoints to implementation docs"""
        api_docs = [doc for doc in self.analysis['documents'] if 'api/' in doc]
        server_docs = [doc for doc in self.analysis['documents'] if 'server/' in doc]
        
        mapping = {}
        for api_doc in api_docs:
            # Find related server documentation
            related_server = []
            api_name = Path(api_doc).stem
            
            for server_doc in server_docs:
                if api_name in server_doc or any(keyword in server_doc.lower() 
                                               for keyword in ['handler', 'service', 'actor']):
                    related_server.append(server_doc)
            
            mapping[api_doc] = related_server
        
        return mapping
    
    def add_breadcrumbs(self, doc_path: str, content: str) -> str:
        """Add breadcrumb navigation to documents"""
        path_parts = Path(doc_path).parts
        
        if len(path_parts) <= 1:
            return content
        
        breadcrumbs = []
        current_path = ""
        
        # Build breadcrumb chain
        for i, part in enumerate(path_parts[:-1]):  # Exclude filename
            if i == 0:
                current_path = part
                breadcrumbs.append(f"[{part.title()}](../index.md)")
            else:
                current_path = "/".join(path_parts[:i+1])
                # Calculate relative path back to section
                back_levels = len(path_parts) - i - 1
                back_path = "../" * back_levels + current_path + "/index.md"
                breadcrumbs.append(f"[{part.title().replace('-', ' ')}]({back_path})")
        
        breadcrumb_line = " > ".join(breadcrumbs)
        breadcrumb_section = f"*{breadcrumb_line}*\n\n"
        
        # Insert after first heading if exists
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('# '):
                lines.insert(i + 1, '')
                lines.insert(i + 2, breadcrumb_section.strip())
                break
        else:
            # Insert at beginning if no heading found
            lines.insert(0, breadcrumb_section.strip())
            lines.insert(1, '')
        
        return '\n'.join(lines)
    
    def add_related_topics(self, doc_path: str, content: str) -> str:
        """Add related topics section"""
        related = set()
        
        # Find related documents by topic
        doc_lower = doc_path.lower()
        for topic, docs in self.topic_mapping.items():
            if topic in doc_lower and doc_path in docs:
                # Add other documents from same topic
                for related_doc in docs:
                    if related_doc != doc_path:
                        relative_path = self.get_relative_path(doc_path, related_doc)
                        title = self.analysis['documents'][related_doc].get('title', Path(related_doc).stem)
                        related.add(f"- [{title}]({relative_path})")
        
        # Add API-related documents
        if doc_path in self.api_mapping:
            for server_doc in self.api_mapping[doc_path]:
                relative_path = self.get_relative_path(doc_path, server_doc)
                title = self.analysis['documents'][server_doc].get('title', Path(server_doc).stem)
                related.add(f"- [{title}]({relative_path}) - Implementation")
        
        # Find documents that link to this one (backlinks)
        if doc_path in self.analysis['backlink_graph']:
            for linking_doc in list(self.analysis['backlink_graph'][doc_path])[:3]:  # Limit to 3
                if linking_doc in self.analysis['documents']:
                    relative_path = self.get_relative_path(doc_path, linking_doc)
                    title = self.analysis['documents'][linking_doc].get('title', Path(linking_doc).stem)
                    related.add(f"- [{title}]({relative_path})")
        
        if not related:
            return content
        
        related_section = "\n\n## Related Topics\n\n" + "\n".join(sorted(related)) + "\n"
        
        # Add before any existing cross-references or at the end
        if "## See Also" in content or "## Related" in content:
            return content  # Already has related content
        
        return content + related_section
    
    def add_see_also_sections(self, doc_path: str, content: str) -> str:
        """Add See Also sections with relevant cross-references"""
        see_also = set()
        
        # For API docs, link to implementation
        if doc_path.startswith('api/'):
            # Find corresponding server documentation
            api_name = Path(doc_path).stem
            for server_doc in self.analysis['documents']:
                if (server_doc.startswith('server/') and 
                    (api_name in server_doc or 'handler' in server_doc)):
                    relative_path = self.get_relative_path(doc_path, server_doc)
                    title = self.analysis['documents'][server_doc].get('title', 'Server Implementation')
                    see_also.add(f"- [{title}]({relative_path}) - Server implementation")
        
        # For server docs, link to API documentation
        elif doc_path.startswith('server/'):
            server_name = Path(doc_path).stem
            for api_doc in self.analysis['documents']:
                if (api_doc.startswith('api/') and 
                    (server_name in api_doc or any(keyword in server_name 
                                                 for keyword in ['handler', 'actor']))):
                    relative_path = self.get_relative_path(doc_path, api_doc)
                    title = self.analysis['documents'][api_doc].get('title', 'API Documentation')
                    see_also.add(f"- [{title}]({relative_path}) - API specification")
        
        # For configuration docs, link to guides
        elif 'config' in doc_path.lower() or 'settings' in doc_path.lower():
            for guide_doc in self.analysis['documents']:
                if guide_doc.startswith('guides/') or guide_doc.startswith('getting-started/'):
                    relative_path = self.get_relative_path(doc_path, guide_doc)
                    title = self.analysis['documents'][guide_doc].get('title', 'Guide')
                    see_also.add(f"- [{title}]({relative_path})")
        
        # For architecture docs, link to implementation
        elif doc_path.startswith('architecture/'):
            for impl_doc in self.analysis['documents']:
                if (impl_doc.startswith('server/') or impl_doc.startswith('client/')):
                    relative_path = self.get_relative_path(doc_path, impl_doc)
                    title = self.analysis['documents'][impl_doc].get('title', 'Implementation')
                    see_also.add(f"- [{title}]({relative_path})")
                    if len(see_also) >= 3:  # Limit to avoid clutter
                        break
        
        if not see_also:
            return content
        
        see_also_section = "\n\n## See Also\n\n" + "\n".join(sorted(see_also)) + "\n"
        
        # Insert before any existing related topics or at the end
        if "## Related Topics" in content:
            content = content.replace("## Related Topics", see_also_section + "\n## Related Topics")
        else:
            content += see_also_section
        
        return content
    
    def get_relative_path(self, from_doc: str, to_doc: str) -> str:
        """Calculate relative path between two documents"""
        from_parts = Path(from_doc).parts[:-1]  # Exclude filename
        to_path = Path(to_doc)
        
        # Calculate how many levels to go up
        levels_up = len(from_parts)
        
        if levels_up == 0:
            return to_doc
        
        return "../" * levels_up + to_doc
    
    def add_topic_indices(self):
        """Create or enhance topic index files"""
        print("📚 Creating topic indices...")
        
        for topic, docs in self.topic_mapping.items():
            if not docs or len(docs) < 2:
                continue
            
            # Find or create index file for topic
            topic_dir = self.docs_root / topic
            if topic_dir.exists() and topic_dir.is_dir():
                index_file = topic_dir / "index.md"
                
                if index_file.exists():
                    with open(index_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    content = f"# {topic.title().replace('-', ' ')}\n\n"
                
                # Add document list if not already present
                if "## Documents" not in content and "## Pages" not in content:
                    docs_section = f"\n## Documents\n\n"
                    for doc in sorted(docs):
                        if doc.startswith(f"{topic}/") and doc != f"{topic}/index.md":
                            relative_path = Path(doc).name
                            title = self.analysis['documents'][doc].get('title', Path(doc).stem)
                            docs_section += f"- [{title}](./{relative_path})\n"
                    
                    content += docs_section
                    
                    with open(index_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.changes_made.append(f"Enhanced topic index: {topic}/index.md")
    
    def enhance_all_documents(self):
        """Apply all enhancements to documents"""
        print("🔧 Enhancing cross-references...")
        
        self.load_documents()
        enhanced_count = 0
        
        for doc_path, document in self.documents.items():
            original_content = document.content
            enhanced_content = original_content
            
            # Skip archive documents to avoid cluttering legacy content
            if 'archive/' in doc_path:
                continue
            
            # Skip if it's a very small document (likely just a stub)
            if len(enhanced_content.split('\n')) < 5:
                continue
            
            # Add breadcrumbs for nested documents
            if len(Path(doc_path).parts) > 1:
                enhanced_content = self.add_breadcrumbs(doc_path, enhanced_content)
            
            # Add related topics
            enhanced_content = self.add_related_topics(doc_path, enhanced_content)
            
            # Add see also sections
            enhanced_content = self.add_see_also_sections(doc_path, enhanced_content)
            
            # Save if changes were made
            if enhanced_content != original_content:
                with open(document.path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_content)
                
                self.changes_made.append(f"Enhanced cross-references: {doc_path}")
                enhanced_count += 1
        
        print(f"✅ Enhanced {enhanced_count} documents")
    
    def generate_link_report(self) -> str:
        """Generate comprehensive linking report"""
        report = []
        report.append("# Documentation Cross-Reference Enhancement Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        report.append(f"- **Total Documents**: {self.analysis['summary']['total_documents']}")
        report.append(f"- **Internal Links**: {self.analysis['summary']['total_internal_links']}")
        report.append(f"- **Broken Links**: {self.analysis['summary']['broken_links']}")
        report.append(f"- **Orphaned Documents**: {self.analysis['summary']['orphaned_docs']}")
        report.append(f"- **Connected Documents**: {self.analysis['summary']['connected_docs']}")
        report.append(f"- **Documents Enhanced**: {len(self.changes_made)}\n")
        
        # Top hub documents
        report.append("## Hub Documents (Most Outgoing Links)\n")
        for doc, count in self.analysis['hub_documents'][:10]:
            report.append(f"- **{doc}**: {count} outgoing links")
        report.append("")
        
        # Most referenced documents
        report.append("## Most Referenced Documents\n")
        for doc, count in self.analysis['popular_documents'][:10]:
            clean_doc = doc.replace('../', '')
            report.append(f"- **{clean_doc}**: {count} incoming links")
        report.append("")
        
        # Topic coverage
        report.append("## Topic Coverage\n")
        for topic, docs in self.topic_mapping.items():
            if docs:
                report.append(f"- **{topic.title()}**: {len(docs)} documents")
        report.append("")
        
        # Changes made
        report.append("## Changes Made\n")
        if self.changes_made:
            for change in self.changes_made:
                report.append(f"- {change}")
        else:
            report.append("- No changes made")
        report.append("")
        
        # Remaining issues
        report.append("## Remaining Issues\n")
        report.append(f"### Broken Links ({len(self.analysis['broken_links'])})\n")
        for link in self.analysis['broken_links'][:20]:
            report.append(f"- `{link['source']}` → `{link['target']}` (line {link['line']})")
        report.append("")
        
        report.append(f"### Orphaned Documents ({len(self.analysis['orphaned_documents'])})\n")
        non_archive_orphans = [doc for doc in self.analysis['orphaned_documents'] 
                             if not doc.startswith('archive/')]
        for doc in non_archive_orphans[:20]:
            report.append(f"- `{doc}`")
        
        return "\n".join(report)
    
    def run_enhancement(self):
        """Run complete cross-reference enhancement"""
        print("🚀 Starting cross-reference enhancement...")
        
        self.add_topic_indices()
        self.enhance_all_documents()
        
        # Generate report
        report_content = self.generate_link_report()
        report_file = self.docs_root / "CROSS_REFERENCE_REPORT.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📊 Generated report: {report_file}")
        print(f"✅ Enhancement complete! Made {len(self.changes_made)} changes.")
        
        return report_content

# Add missing import
from datetime import datetime

def main():
    enhancer = CrossReferenceEnhancer(
        '/workspace/ext/docs', 
        '/workspace/ext/docs/analysis_report.json'
    )
    
    report = enhancer.run_enhancement()
    print("\n" + "="*60)
    print("CROSS-REFERENCE ENHANCEMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()