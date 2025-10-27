# How to View Mermaid Diagrams

## Quick Start

The file `/home/devuser/workspace/project/docs/architecture/event-flow-diagrams.md` now contains **10 world-class Mermaid diagrams** that can be viewed in multiple ways.

---

## Viewing Options

### 1. GitHub (Recommended - Zero Setup)
Simply view the file on GitHub - Mermaid diagrams render automatically!

```
https://github.com/your-repo/docs/architecture/event-flow-diagrams.md
```

**Advantages**:
- ✅ No installation required
- ✅ Renders automatically
- ✅ Mobile-friendly
- ✅ Shareable links

---

### 2. VS Code (Best for Development)

#### Install Extension
```bash
code --install-extension bierner.markdown-mermaid
```

#### View Diagrams
1. Open `event-flow-diagrams.md`
2. Press `Ctrl+Shift+V` (Windows/Linux) or `Cmd+Shift+V` (Mac)
3. Diagrams render in preview pane

**Advantages**:
- ✅ Instant preview
- ✅ Side-by-side editing
- ✅ Zoom and pan
- ✅ Export to images

---

### 3. Mermaid Live Editor (Quick Testing)

Visit: https://mermaid.live

1. Copy any `mermaid` code block
2. Paste into editor
3. See live preview
4. Export to PNG/SVG

**Advantages**:
- ✅ No installation
- ✅ Export images
- ✅ Share diagrams
- ✅ Test syntax

---

### 4. MkDocs (Documentation Site)

#### Install MkDocs with Mermaid
```bash
pip install mkdocs mkdocs-material pymdown-extensions
```

#### Configure mkdocs.yml
```yaml
markdown_extensions:
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
```

#### Serve
```bash
mkdocs serve
```

**Advantages**:
- ✅ Professional documentation site
- ✅ Search functionality
- ✅ Dark mode support
- ✅ Mobile responsive

---

### 5. GitLab (Alternative Git Host)

GitLab has native Mermaid support - diagrams render automatically in markdown files.

**Advantages**:
- ✅ Native support
- ✅ CI/CD integration
- ✅ Wiki support

---

## Exporting Diagrams

### Export to PNG/SVG (Using Mermaid CLI)

#### Install Mermaid CLI
```bash
npm install -g @mermaid-js/mermaid-cli
```

#### Extract Individual Diagrams
Create separate `.mmd` files for each diagram, then:

```bash
# Export to PNG
mmdc -i diagram1.mmd -o diagram1.png -w 2048 -H 1536

# Export to SVG
mmdc -i diagram1.mmd -o diagram1.svg

# Export with theme
mmdc -i diagram1.mmd -o diagram1.png -t dark
```

#### Batch Export Script
```bash
#!/bin/bash
# Extract all mermaid blocks and export as images

# Create output directory
mkdir -p diagrams

# Export each diagram
for i in {1..10}; do
  mmdc -i "diagram${i}.mmd" -o "diagrams/diagram${i}.png" -w 2048 -H 1536
done
```

---

## Browser Extensions

### Chrome/Edge
- **Mermaid Diagrams** - Renders Mermaid in local HTML files
- Install: https://chrome.google.com/webstore

### Firefox
- **Markdown Viewer Webext** - Supports Mermaid rendering
- Install: https://addons.mozilla.org

---

## Integration with Documentation Tools

### Docusaurus
```javascript
// docusaurus.config.js
module.exports = {
  themes: ['@docusaurus/theme-mermaid'],
  markdown: {
    mermaid: true,
  },
};
```

### Hugo
```markdown
{{< mermaid >}}
sequenceDiagram
    participant A
    participant B
    A->>B: Hello
{{< /mermaid >}}
```

### Notion
- Use Mermaid Live Editor to export PNG
- Upload PNG to Notion

### Confluence
- Use Mermaid Live Editor to export PNG
- Embed PNG in Confluence page

---

## Diagram Overview

### Available Diagrams

1. **GitHub Sync Event Flow (BUG FIX)**
   - Diagram 1: Current Problem Flow (cache coherency bug)
   - Diagram 2: Fixed Event-Driven Flow (solution)

2. **Physics Simulation Event Flow**
   - Diagram 3: Physics Step Execution (60 FPS real-time)

3. **Node Creation Event Flow**
   - Diagram 4: User Creates New Node (CQRS pattern)

4. **WebSocket Connection Event Flow**
   - Diagram 5: Client Connects to WebSocket (real-time sync)

5. **Cache Invalidation Event Flow**
   - Diagram 6: When Cache Gets Invalidated (3-layer cache)
   - Diagram 7: Read Flow with Cache (cache-aside pattern)

6. **Semantic Analysis Event Flow**
   - Diagram 8: AI-Powered Semantic Analysis (GPU-accelerated)

7. **Error Handling Event Flow**
   - Diagram 9: When Commands Fail (validation + database errors)

8. **Event Store Replay (Event Sourcing)**
   - Diagram 10: Rebuilding State from Events (disaster recovery)

---

## Troubleshooting

### Diagram Not Rendering

1. **Check Syntax**: Copy to https://mermaid.live to validate
2. **Check Version**: Ensure Mermaid version supports features used
3. **Check Theme**: Some themes may not support all colors

### Performance Issues

- Large diagrams may take time to render
- Consider splitting into smaller diagrams
- Use `%%{init: {'theme':'base'}}%%` for faster rendering

### Export Quality

- For presentations, export at 2048x1536 or higher
- Use SVG for infinite scalability
- Use PNG with transparency for overlays

---

## Best Practices

### Viewing
- ✅ Use GitHub for quick sharing
- ✅ Use VS Code for development
- ✅ Use Mermaid Live for testing
- ✅ Use MkDocs for documentation sites

### Editing
- ✅ Keep diagrams under 20 participants
- ✅ Use colors for clarity
- ✅ Add notes for context
- ✅ Group related actions with `rect`

### Sharing
- ✅ Share GitHub links for live diagrams
- ✅ Export PNG for presentations
- ✅ Export SVG for design tools
- ✅ Embed in documentation sites

---

## Additional Resources

### Official Documentation
- Mermaid Docs: https://mermaid.js.org
- Sequence Diagrams: https://mermaid.js.org/syntax/sequenceDiagram.html
- Flowcharts: https://mermaid.js.org/syntax/flowchart.html

### Tutorials
- Interactive Tutorial: https://mermaid-js.github.io/mermaid-live-editor
- Video Tutorials: Search "Mermaid diagram tutorial" on YouTube

### Community
- GitHub Discussions: https://github.com/mermaid-js/mermaid/discussions
- Stack Overflow: Tag `mermaid`

---

**Enjoy exploring the event flow diagrams!** 🎉

For technical questions about the architecture, see:
- `/home/devuser/workspace/project/docs/architecture/MERMAID_CONVERSION_SUMMARY.md`
- `/home/devuser/workspace/project/docs/architecture/event-flow-diagrams.md`
