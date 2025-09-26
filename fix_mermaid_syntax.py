#!/usr/bin/env python3
"""
Fix Mermaid diagram syntax issues in markdown files.
Specifically, wrap node labels containing <br/> in quotes.
"""

import re
import os
import sys

def fix_mermaid_nodes(content):
    """Fix node labels with <br/> that aren't quoted."""
    # Pattern to match node definitions like: NodeName[label<br/>label]
    # where the label doesn't start with a quote
    pattern = r'(\w+)\[([^"\[\]]+<br/>[^"\[\]]*)\]'
    
    def replace_node(match):
        node_name = match.group(1)
        label = match.group(2)
        # Wrap the label in quotes
        return f'{node_name}["{label}"]'
    
    # Process the content
    lines = content.split('\n')
    in_mermaid = False
    result_lines = []
    
    for line in lines:
        if '```mermaid' in line:
            in_mermaid = True
            result_lines.append(line)
        elif '```' in line and in_mermaid:
            in_mermaid = False
            result_lines.append(line)
        elif in_mermaid:
            # Only apply the fix inside mermaid blocks
            fixed_line = re.sub(pattern, replace_node, line)
            result_lines.append(fixed_line)
        else:
            result_lines.append(line)
    
    return '\n'.join(result_lines)

def process_file(filepath):
    """Process a single file to fix Mermaid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file contains mermaid blocks
        if '```mermaid' not in content:
            return False, "No Mermaid diagrams found"
        
        # Fix the content
        fixed_content = fix_mermaid_nodes(content)
        
        # Only write if changes were made
        if fixed_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True, "Fixed Mermaid syntax"
        else:
            return False, "No changes needed"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    # Files identified as needing fixes
    files_to_fix = [
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/getting-started/00-index.md",
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/hornedowl.md",
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/archive/owl.md",
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/guides/04-orchestrating-agents.md",
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/guides/01-deployment.md",
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/architecture/interface.md",
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/high-level.md",
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/concepts/01-system-overview.md",
        # Note: server-architecture.md and architecture/server.md don't need fixes
        # Note: client-architecture-current.md also has issues
        "/mnt/mldata/githubs/AR-AI-Knowledge-Graph/docs/client-architecture-current.md"
    ]
    
    print("Fixing Mermaid syntax in documentation files...")
    print("=" * 60)
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            success, message = process_file(filepath)
            status = "✓" if success else "✗"
            print(f"{status} {os.path.basename(filepath)}: {message}")
        else:
            print(f"✗ {os.path.basename(filepath)}: File not found")
    
    print("=" * 60)
    print("Fix complete!")

if __name__ == "__main__":
    main()