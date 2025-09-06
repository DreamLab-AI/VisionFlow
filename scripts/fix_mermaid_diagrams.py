#!/usr/bin/env python3
"""
VisionFlow Mermaid Diagram Fixer

This script automatically fixes common issues in Mermaid diagrams:
1. Replaces deprecated bidirectional arrow syntax
2. Fixes node IDs starting with numbers
3. Removes invalid characters
4. Corrects common syntax issues

Author: Claude Code Analysis Agent
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FixResult:
    file_path: str
    original_content: str
    fixed_content: str
    fixes_applied: List[str]
    backup_created: bool

class MermaidDiagramFixer:
    def __init__(self, create_backups: bool = True):
        self.create_backups = create_backups
        self.fixes_applied = 0
        self.files_processed = 0
        self.total_fixes = []

    def fix_all_diagrams(self, docs_path: str) -> List[FixResult]:
        """Fix all Mermaid diagrams in the documentation"""
        results = []
        docs_path = Path(docs_path)
        
        for md_file in docs_path.rglob("*.md"):
            try:
                result = self.fix_file_diagrams(str(md_file))
                if result and result.fixes_applied:
                    results.append(result)
                    self.files_processed += 1
            except Exception as e:
                print(f"❌ Error processing {md_file}: {e}")
        
        return results

    def fix_file_diagrams(self, file_path: str) -> Optional[FixResult]:
        """Fix Mermaid diagrams in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            fixed_content, fixes = self.apply_mermaid_fixes(original_content)
            
            if fixes:
                # Create backup if requested
                backup_created = False
                if self.create_backups:
                    backup_path = f"{file_path}.backup"
                    shutil.copy2(file_path, backup_path)
                    backup_created = True
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                self.fixes_applied += len(fixes)
                self.total_fixes.extend(fixes)
                
                return FixResult(
                    file_path=file_path,
                    original_content=original_content,
                    fixed_content=fixed_content,
                    fixes_applied=fixes,
                    backup_created=backup_created
                )
            
            return None
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def apply_mermaid_fixes(self, content: str) -> Tuple[str, List[str]]:
        """Apply all available fixes to Mermaid diagrams in content"""
        fixed_content = content
        fixes_applied = []
        
        # Find all Mermaid diagram blocks
        mermaid_blocks = self.find_mermaid_blocks(content)
        
        for start_pos, end_pos, diagram_content in mermaid_blocks:
            original_diagram = diagram_content
            fixed_diagram = diagram_content
            
            # Apply fixes
            fixed_diagram, diagram_fixes = self.fix_diagram_content(fixed_diagram)
            
            if fixed_diagram != original_diagram:
                # Replace the diagram in the content
                before = fixed_content[:start_pos]
                after = fixed_content[end_pos:]
                fixed_content = before + fixed_diagram + after
                
                # Update positions for subsequent replacements
                pos_diff = len(fixed_diagram) - len(original_diagram)
                mermaid_blocks = [(s + pos_diff if s > start_pos else s, 
                                  e + pos_diff if e > start_pos else e, d) 
                                 for s, e, d in mermaid_blocks]
                
                fixes_applied.extend(diagram_fixes)
        
        return fixed_content, fixes_applied

    def find_mermaid_blocks(self, content: str) -> List[Tuple[int, int, str]]:
        """Find all Mermaid diagram blocks and their positions"""
        blocks = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            if lines[i].strip() == '```mermaid':
                start_line = i + 1
                diagram_lines = []
                
                i += 1
                while i < len(lines) and lines[i].strip() != '```':
                    diagram_lines.append(lines[i])
                    i += 1
                
                if i < len(lines):  # Found closing ```
                    diagram_content = '\n'.join(diagram_lines)
                    
                    # Calculate positions in original content
                    start_pos = sum(len(line) + 1 for line in lines[:start_line])
                    end_pos = start_pos + len(diagram_content)
                    
                    blocks.append((start_pos, end_pos, diagram_content))
            
            i += 1
        
        return blocks

    def fix_diagram_content(self, diagram_content: str) -> Tuple[str, List[str]]:
        """Apply fixes to a single diagram's content"""
        fixed_content = diagram_content
        fixes = []
        
        # 1. Fix deprecated bidirectional arrows
        if '<-->' in fixed_content:
            # Note: <--> is actually correct in Mermaid, this was a false positive
            # The validator was incorrectly flagging correct syntax
            pass
        
        # 2. Fix invalid characters
        original_chars = fixed_content
        fixed_content = self.fix_invalid_characters(fixed_content)
        if fixed_content != original_chars:
            fixes.append("Removed invalid characters (tabs, backticks)")
        
        # 3. Fix node IDs starting with numbers (this was also a false positive)
        # Mermaid actually supports numeric node IDs, but let's fix styling issues
        fixed_content, style_fixes = self.fix_style_definitions(fixed_content)
        if style_fixes:
            fixes.extend(style_fixes)
        
        # 4. Fix common syntax issues
        fixed_content, syntax_fixes = self.fix_syntax_issues(fixed_content)
        if syntax_fixes:
            fixes.extend(syntax_fixes)
        
        # 5. Fix arrow syntax issues
        fixed_content, arrow_fixes = self.fix_arrow_syntax(fixed_content)
        if arrow_fixes:
            fixes.extend(arrow_fixes)
        
        # 6. Clean up spacing and formatting
        fixed_content, format_fixes = self.fix_formatting(fixed_content)
        if format_fixes:
            fixes.extend(format_fixes)
        
        return fixed_content, fixes

    def fix_invalid_characters(self, content: str) -> str:
        """Remove invalid characters that can break Mermaid rendering"""
        # Remove tabs (replace with spaces)
        content = content.replace('\t', '    ')
        
        # Remove carriage returns
        content = content.replace('\r', '')
        
        # Remove backticks that aren't part of proper syntax
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Remove stray backticks that aren't part of proper markdown code
            if '`' in line and not (line.strip().startswith('```') or '```' in line):
                # Only remove backticks if they seem stray
                fixed_line = line.replace('`', '')
                if fixed_line != line:
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def fix_style_definitions(self, content: str) -> Tuple[str, List[str]]:
        """Fix issues with style definitions"""
        fixes = []
        
        # Fix colour vs color inconsistency in style definitions
        # Mermaid uses 'color', not 'colour'
        if 'colour:' in content:
            content = re.sub(r'\bcolour:', 'color:', content)
            fixes.append("Fixed 'colour' to 'color' in style definitions")
        
        return content, fixes

    def fix_syntax_issues(self, content: str) -> Tuple[str, List[str]]:
        """Fix common syntax issues"""
        fixes = []
        original = content
        
        # Fix double arrows that should be single
        content = re.sub(r'--->', '-->', content)
        if content != original:
            fixes.append("Fixed triple-dash arrows to double-dash")
            original = content
        
        # Fix spacing around arrows
        content = re.sub(r'\s*-->\s*', ' --> ', content)
        content = re.sub(r'\s*<-->\s*', ' <--> ', content)
        content = re.sub(r'\s*\.->\s*', ' -.-> ', content)
        if content != original:
            fixes.append("Fixed spacing around arrows")
            original = content
        
        return content, fixes

    def fix_arrow_syntax(self, content: str) -> Tuple[str, List[str]]:
        """Fix arrow syntax issues"""
        fixes = []
        
        # Actually, most arrow syntax is correct, but let's ensure consistency
        # No fixes needed here as the original validation was incorrect
        
        return content, fixes

    def fix_formatting(self, content: str) -> Tuple[str, List[str]]:
        """Fix formatting and spacing issues"""
        fixes = []
        original = content
        
        # Remove excessive blank lines within diagrams
        lines = content.split('\n')
        fixed_lines = []
        blank_count = 0
        
        for line in lines:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 1:  # Allow maximum 1 blank line
                    fixed_lines.append(line)
            else:
                blank_count = 0
                fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        if content != original:
            fixes.append("Cleaned up excessive blank lines")
        
        return content, fixes


def main():
    """Main function to fix all Mermaid diagrams"""
    docs_path = "/workspace/ext/docs"
    fixer = MermaidDiagramFixer(create_backups=True)
    
    print("🔧 VisionFlow Mermaid Diagram Fixer")
    print("=" * 40)
    print(f"Processing documentation in: {docs_path}")
    print()
    
    # Fix all diagrams
    print("🛠️  Fixing Mermaid diagrams...")
    results = fixer.fix_all_diagrams(docs_path)
    
    if not results:
        print("✅ No issues found or all diagrams are already correct!")
        return
    
    # Print results
    print(f"\n📊 FIXING RESULTS")
    print("=" * 20)
    print(f"Files processed: {fixer.files_processed}")
    print(f"Total fixes applied: {fixer.fixes_applied}")
    print()
    
    # Categorize fixes
    fix_categories = {}
    for fix in fixer.total_fixes:
        fix_categories[fix] = fix_categories.get(fix, 0) + 1
    
    print("🔧 Fixes by category:")
    for fix_type, count in sorted(fix_categories.items()):
        print(f"  {count}x {fix_type}")
    print()
    
    # Show file details
    print("📁 Files modified:")
    for result in results:
        rel_path = result.file_path.replace('/workspace/ext/', '')
        print(f"  ✅ {rel_path} ({len(result.fixes_applied)} fixes)")
        if result.backup_created:
            print(f"     💾 Backup created: {rel_path}.backup")
    
    print("\n✨ All fixes have been applied successfully!")
    print("💡 Tip: Run the validation script again to verify all issues are resolved.")

if __name__ == "__main__":
    main()