#!/usr/bin/env python3
"""
Create comprehensive navigation indices for documentation
"""

import os
import json
from pathlib import Path
from collections import defaultdict

class NavigationIndexer:
    def __init__(self, docs_root: str, analysis_file: str):
        self.docs_root = Path(docs_root)
        self.analysis_file = analysis_file
        self.load_analysis()
    
    def load_analysis(self):
        """Load the analysis report"""
        with open(self.analysis_file, 'r') as f:
            self.analysis = json.load(f)
    
    def create_comprehensive_index(self):
        """Create comprehensive documentation index"""
        
        # Group documents by top-level category
        categories = defaultdict(list)
        
        for doc_path in self.analysis['documents']:
            if doc_path.startswith('archive/'):
                continue  # Skip archived content
                
            parts = Path(doc_path).parts
            if len(parts) > 1:
                category = parts[0]
                categories[category].append({
                    'path': doc_path,
                    'title': self.analysis['documents'][doc_path].get('title', Path(doc_path).stem),
                    'size': self.analysis['documents'][doc_path].get('size', 0)
                })
            else:
                categories['root'].append({
                    'path': doc_path,
                    'title': self.analysis['documents'][doc_path].get('title', Path(doc_path).stem),
                    'size': self.analysis['documents'][doc_path].get('size', 0)
                })
        
        # Create comprehensive index
        index_content = self.build_comprehensive_index_content(categories)
        
        with open(self.docs_root / 'DOCUMENTATION_INDEX.md', 'w') as f:
            f.write(index_content)
        
        return index_content
    
    def build_comprehensive_index_content(self, categories):
        """Build the comprehensive index content"""
        content = []
        content.append("# VisionFlow Documentation Index")
        content.append("")
        content.append("Complete index of all documentation organized by category and topic.")
        content.append("")
        
        # Add quick navigation
        content.append("## Quick Navigation")
        content.append("")
        for category in sorted(categories.keys()):
            if category != 'root':
                content.append(f"- [{category.title()}](#{category.replace('-', '')})")
        content.append("")
        
        # Add statistics
        total_docs = sum(len(docs) for docs in categories.values())
        content.append(f"**📊 Total Documents**: {total_docs}")
        content.append(f"**🔗 Internal Links**: {self.analysis['summary']['total_internal_links']}")
        content.append(f"**📈 Connected Documents**: {self.analysis['summary']['connected_docs']}")
        content.append("")
        
        # Add categories
        for category, docs in sorted(categories.items()):
            if category == 'root':
                content.append("## Root Documentation")
                content.append("")
                content.append("Core documentation files in the root directory.")
                content.append("")
            else:
                content.append(f"## {category.title().replace('-', ' ')}")
                content.append("")
                
                # Add category description
                descriptions = {
                    'api': 'REST and WebSocket API documentation, endpoints, and protocols.',
                    'architecture': 'System design documents, architectural decisions, and design patterns.',
                    'client': 'Frontend application documentation, components, and user interfaces.',
                    'server': 'Backend server documentation, actors, services, and handlers.',
                    'deployment': 'Deployment guides, Docker configurations, and production setup.',
                    'configuration': 'Configuration references, settings, and environment setup.',
                    'security': 'Security documentation, authentication, and access control.',
                    'getting-started': 'User guides, installation instructions, and quickstart tutorials.',
                    'guides': 'How-to guides, tutorials, and step-by-step instructions.',
                    'reference': 'API references, agent specifications, and lookup materials.',
                    'technical': 'Technical specifications, implementation details, and system docs.',
                    'development': 'Development guides, debugging, testing, and setup.',
                    'features': 'Feature documentation, capabilities, and functionality guides.',
                    'testing': 'Testing documentation, strategies, and quality assurance.',
                }
                
                if category in descriptions:
                    content.append(descriptions[category])
                content.append("")
            
            # Sort documents by title and add them
            sorted_docs = sorted(docs, key=lambda x: x['title'].lower())
            
            for doc in sorted_docs:
                title = doc['title']
                path = doc['path']
                size = doc['size']
                
                # Add size indicator
                if size > 50000:
                    size_indicator = " 📖"
                elif size > 20000:
                    size_indicator = " 📄"
                elif size > 5000:
                    size_indicator = " 📝"
                else:
                    size_indicator = " 📋"
                
                content.append(f"- [{title}](./{path}){size_indicator}")
            
            content.append("")
        
        # Add special sections
        content.append("## Most Referenced Documents")
        content.append("")
        content.append("Documents that are frequently linked to by others:")
        content.append("")
        
        for doc, count in self.analysis['popular_documents'][:10]:
            clean_doc = doc.replace('../', '')
            if clean_doc in self.analysis['documents']:
                title = self.analysis['documents'][clean_doc].get('title', Path(clean_doc).stem)
                content.append(f"- [{title}](./{clean_doc}) ({count} references)")
        
        content.append("")
        
        # Add hub documents
        content.append("## Hub Documents")
        content.append("")
        content.append("Documents with the most outgoing links:")
        content.append("")
        
        for doc, count in self.analysis['hub_documents'][:10]:
            if doc in self.analysis['documents']:
                title = self.analysis['documents'][doc].get('title', Path(doc).stem)
                content.append(f"- [{title}](./{doc}) ({count} outgoing links)")
        
        content.append("")
        
        # Add topic index
        content.append("## Topic Index")
        content.append("")
        content.append("Find documents by topic or keyword:")
        content.append("")
        
        # Build keyword index
        keywords = defaultdict(list)
        for doc_path, doc_info in self.analysis['documents'].items():
            if doc_path.startswith('archive/'):
                continue
                
            title = doc_info.get('title', '').lower()
            path_lower = doc_path.lower()
            
            # Extract keywords
            doc_keywords = set()
            
            # Common technical terms
            tech_terms = [
                'api', 'websocket', 'rest', 'graphql', 'mcp', 'agent', 'cuda', 'gpu',
                'architecture', 'deployment', 'docker', 'configuration', 'security',
                'authentication', 'testing', 'performance', 'optimization', 'monitoring',
                'logging', 'analytics', 'visualization', 'graph', 'node', 'edge',
                'client', 'server', 'backend', 'frontend', 'database', 'storage',
                'actor', 'service', 'handler', 'model', 'type', 'util'
            ]
            
            for term in tech_terms:
                if term in title or term in path_lower:
                    doc_keywords.add(term)
            
            # Add to keyword index
            for keyword in doc_keywords:
                keywords[keyword].append({
                    'path': doc_path,
                    'title': doc_info.get('title', Path(doc_path).stem)
                })
        
        # Display keyword index
        for keyword, docs in sorted(keywords.items()):
            if len(docs) >= 2:  # Only show keywords with multiple documents
                content.append(f"**{keyword.upper()}**")
                for doc in sorted(docs, key=lambda x: x['title'])[:5]:  # Limit to 5
                    content.append(f"  - [{doc['title']}](./{doc['path']})")
                content.append("")
        
        content.append("---")
        content.append("")
        content.append("*This index is automatically generated from the documentation structure.*")
        content.append("")
        
        return "\n".join(content)
    
    def create_site_map(self):
        """Create a visual sitemap of the documentation"""
        sitemap_content = []
        sitemap_content.append("# VisionFlow Documentation Sitemap")
        sitemap_content.append("")
        sitemap_content.append("Visual representation of the documentation structure.")
        sitemap_content.append("")
        sitemap_content.append("```")
        sitemap_content.append("docs/")
        
        # Build tree structure
        tree_structure = defaultdict(list)
        for doc_path in sorted(self.analysis['documents'].keys()):
            if doc_path.startswith('archive/'):
                continue
            
            parts = Path(doc_path).parts
            current_level = tree_structure
            
            for i, part in enumerate(parts[:-1]):
                if part not in current_level:
                    current_level[part] = defaultdict(list)
                current_level = current_level[part]
            
            # Add the file
            filename = parts[-1]
            current_level['_files'].append(filename)
        
        # Generate tree visualization
        def print_tree(structure, indent=""):
            for key, value in sorted(structure.items()):
                if key == '_files':
                    for filename in sorted(value):
                        sitemap_content.append(f"{indent}├── {filename}")
                else:
                    sitemap_content.append(f"{indent}├── {key}/")
                    if isinstance(value, dict):
                        print_tree(value, indent + "│   ")
        
        print_tree(tree_structure)
        sitemap_content.append("```")
        sitemap_content.append("")
        
        # Add statistics
        sitemap_content.append("## Documentation Statistics")
        sitemap_content.append("")
        sitemap_content.append(f"- **Total Files**: {len([d for d in self.analysis['documents'] if not d.startswith('archive/')])}")
        sitemap_content.append(f"- **Archive Files**: {len([d for d in self.analysis['documents'] if d.startswith('archive/')])}")
        sitemap_content.append(f"- **Directory Count**: {len(set(Path(d).parent for d in self.analysis['documents']))}")
        sitemap_content.append("")
        
        with open(self.docs_root / 'SITEMAP.md', 'w') as f:
            f.write("\n".join(sitemap_content))
        
        return "\n".join(sitemap_content)

def main():
    indexer = NavigationIndexer(
        '/workspace/ext/docs',
        '/workspace/ext/docs/analysis_report.json'
    )
    
    print("📚 Creating comprehensive documentation index...")
    index_content = indexer.create_comprehensive_index()
    
    print("🗺️ Creating documentation sitemap...")
    sitemap_content = indexer.create_site_map()
    
    print("✅ Navigation indices created successfully!")
    
    print("\nCreated files:")
    print("- /workspace/ext/docs/DOCUMENTATION_INDEX.md")
    print("- /workspace/ext/docs/SITEMAP.md")

if __name__ == "__main__":
    main()