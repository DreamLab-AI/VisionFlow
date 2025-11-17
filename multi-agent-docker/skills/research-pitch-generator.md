---
name: research-pitch-generator
description: Generate research collaboration Beamer presentations using multi-agent orchestration with logseq corpus and VisionFlow context
version: 1.0.0
author: DreamLab
tags: [research, beamer, latex, presentations, multi-agent, visionflow, logseq]
requires:
  - latex (pdflatex, beamer)
  - imagemagick (convert/magick)
  - git
  - Multi-agent coordination (Task tool)
model: sonnet
---

# Research Pitch Generator

Generate intellectually rigorous, contextually adapted research collaboration presentations using multi-agent orchestration.

## Purpose

Create Beamer presentations pitching VisionFlow infrastructure for collaborative research with academics, combining:
- Target researcher's profile and publications
- VisionFlow ontology alignment (from logseq corpus)
- Regional/institutional context
- Power-conscious, non-tech-solutionist framing

## When to Use

- User provides researcher profile (LinkedIn, website, publications)
- Request to "build a pitch" or "create presentation" for collaboration
- Need contextually appropriate research proposals
- Require production-ready PDF with previews

## Core Workflow

### Phase 1: Context Preparation

**1. Sync Logseq Corpus**
```bash
cd /home/devuser/workspace/logseq
git pull origin main
```

**2. Fetch VisionFlow Context**
Use WebFetch to get current VisionFlow architecture from:
- https://deepwiki.com/DreamLab-AI/VisionFlow

Extract key capabilities:
- GPU-accelerated 3D knowledge graphs (100k+ nodes at 60 FPS)
- Neo4j + OWL ontologies (2000+ concepts)
- d3-force-3d immersive visualization
- Nostr protocol (self-sovereign identity)
- Bitcoin/Lightning (micropayments, platform cooperatives)

### Phase 2: Multi-Agent Research Orchestration

Deploy 4 specialized agents **in parallel** using Claude Code's Task tool:

**Agent 1: Profile Researcher**
```
Prompt: "Research [TARGET NAME/URL] comprehensively. Create profile covering:
- Core research domains and methodologies
- Key publications with citation counts
- Institutional affiliations and roles
- Regional/policy context if applicable
- 5 potential collaboration opportunities aligned with their work

Save to: /home/devuser/workspace/logseq/docs/[slug]-research-profile.md"
```

**Agent 2: Ontology Aligner**
```
Prompt: "Analyze logseq corpus at /home/devuser/workspace/logseq/pages/
and journals/ to identify 10-15 ontology concepts with high alignment to
[TARGET]'s research domains.

For each concept, extract:
- Concept name and definition
- Cross-domain links (bridge concepts)
- Relevance score to target's work
- Potential research applications

Save to: /home/devuser/workspace/logseq/docs/[slug]-ontology-alignments.md"
```

**Agent 3: Structure Planner**
```
Prompt: "Design 15-20 slide presentation structure for research collaboration
with [TARGET].

Requirements:
- Opening: Their finding + Central infrastructure question
- 3 research provocations (each 4-6 slides)
- Synthesis: Mutual benefit + Next steps
- Backup slides: Technical details, cross-cutting themes

Determine tone split based on target's domain:
- Critical/theoretical researchers: 70% critical, 20% technical, 10% possibility
- Applied/policy researchers: 60% their research, 30% infrastructure, 10% VisionFlow
- Technical researchers: 50% research, 30% technical, 20% implementation

Save to: /home/devuser/workspace/logseq/docs/[slug]-presentation-structure.md"
```

**Agent 4: Theme Designer**
```
Prompt: "Design Beamer framework template using AmurMaple theme with color
variant appropriate for [TARGET]'s research domain:

- Blue (knowledge systems, information science)
- Green (cooperative economics, sustainability, policy)
- Orange (technical/engineering, innovation)
- Purple (social science, critical theory)

Include:
- TikZ diagram templates for: network visualization, comparison tables,
  geographic maps, ecosystem diagrams
- Alert/example/block styles for research hypotheses
- Progressive reveal patterns with \\pause
- Bibliography integration if needed

Save to: /home/devuser/workspace/logseq/docs/[slug]-presentation-framework.tex"
```

### Phase 3: LaTeX Generation & Compilation

**Agent 5: Beamer Coder**
```
Prompt: "Generate complete, production-ready Beamer presentation LaTeX file.

Source materials:
1. [slug]-research-profile.md
2. [slug]-ontology-alignments.md
3. [slug]-presentation-structure.md
4. [slug]-presentation-framework.tex
5. VisionFlow capabilities from WebFetch

Title format: '[Research Theme] for [Target Domain]: A Collaboration Proposal'
Author: Dr. John O'Hare (DreamLab / Manchester)

Requirements:
- Cite target's specific publications/frameworks
- Regional specificity (if applicable)
- 3 concrete research opportunities with:
  * What: Research question and methodology
  * Why: Intellectual contribution to their field
  * How: Infrastructure capabilities enabling novel research
  * Pilot: Specific deliverables and timeline
- Power-conscious framing throughout:
  * Who benefits? Who decides? Who's excluded?
  * Federated vs centralized architecture
  * Global South inclusion without extraction
- Anti-tech-solutionism:
  * Explicit acknowledgment of structural limits
  * Technology as research infrastructure, not solution
  * Political commitment required for change
- Low-pressure collaborative invitation (not vendor pitch)

Save to: /home/devuser/workspace/logseq/docs/[slug]-collaboration-presentation.tex"
```

**Compilation Steps:**
```bash
cd /home/devuser/workspace/logseq/docs

# Check for missing LaTeX packages
pdflatex -interaction=nonstopmode [slug]-collaboration-presentation.tex 2>&1 | grep "not found"

# Install if needed
# sudo pacman -S --noconfirm texlive-latexextra texlive-pictures

# Compile (2 passes for TOC/navigation)
pdflatex -interaction=nonstopmode [slug]-collaboration-presentation.tex
pdflatex -interaction=nonstopmode [slug]-collaboration-presentation.tex

# Verify output
ls -lh [slug]-collaboration-presentation.pdf
```

**Fix Common LaTeX Errors:**
- `\end{frame>}` → `\end{frame}` (closing brace typo)
- `\end{alertblock>}` → `\end{alertblock}` (closing brace typo)
- Unicode characters (↔, ≠, →) - non-fatal warnings, ignore
- Missing packages - install texlive-latexextra

### Phase 4: Deliverables Creation

**Preview Images:**
```bash
# Use magick (not deprecated convert)
magick -density 150 '[slug]-collaboration-presentation.pdf[0-4]' \
       -quality 90 [slug]-preview-%02d.png
```

**Documentation Summary:**
Create `[slug]-presentation-summary.md` with:
- Presentation details (pages, file size, theme)
- Content overview (3 provocations detailed)
- Design principles implemented
- Visual elements (TikZ diagrams, color scheme)
- Compilation details
- Files generated list
- Cover email framing suggestions
- Key positioning messages

### Phase 5: Repository Commit

```bash
git add docs/[slug]-research-profile.md \
        docs/[slug]-ontology-alignments.md \
        docs/[slug]-presentation-structure.md \
        docs/[slug]-presentation-framework.tex \
        docs/[slug]-collaboration-presentation.tex \
        docs/[slug]-collaboration-presentation.pdf \
        docs/[slug]-preview-*.png \
        docs/[slug]-presentation-summary.md

git commit -m "Add [TARGET NAME] research collaboration presentation

[Brief description of target's research focus]

Key deliverables:
- Comprehensive research profile
- Ontology alignments ([N] high-alignment concepts)
- [N]-slide presentation with AmurMaple [color] theme
- Three research provocations: [list]

Design approach:
- [Tone split description]
- [Regional/contextual specificity]
- [Key framing principles]

Generated using multi-agent coordination.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
```

## Success Factors (Apply to Every Presentation)

### 1. Intellectual Integrity
- **NO tech-solutionism** ("AI will fix everything")
- **NO crypto-libertarianism** (Lightning = infrastructure, not ideology)
- **YES structural analysis** (power dynamics, who benefits)
- **YES explicit limits** (technology alone changes nothing)

### 2. Contextual Adaptation
- **Regional specificity**: Name institutions, councils, regional initiatives
- **Domain calibration**: Adjust tone for critical vs applied vs technical audiences
- **Citation accuracy**: Reference their specific frameworks/publications
- **Pilot proposals**: Concrete deliverables, not vague possibilities

### 3. Collaborative Framing
- **Mutual benefit**: What they bring + What we bring
- **Research-first**: Questions, not solutions
- **Low-pressure**: Initial call → Demo → Workshop → Grant (staged engagement)
- **Honest invitation**: Acknowledge risks, unknowns, structural barriers

### 4. Visual & Structural Quality
- **TikZ diagrams**: Networks, ecosystems, comparisons (not stock images)
- **Progressive reveals**: Build arguments incrementally with `\pause`
- **Alert blocks**: Highlight research hypotheses and power questions
- **Backup slides**: Technical details, cross-cutting themes, anticipated questions

### 5. Power-Conscious Infrastructure
- **Federated vs centralized**: Data sovereignty, cooperative governance
- **Protocol vs platform**: Open standards vs walled gardens
- **Global South inclusion**: Lightning micropayments, not extractive dynamics
- **Worker ownership**: Platform cooperatives, algorithmic transparency

## Tone Calibration Guide

| Target Domain | Critical | Technical | Possibility | Notes |
|---------------|----------|-----------|-------------|-------|
| Critical Theory / Social Science | 70% | 20% | 10% | Foreground power dynamics, structural analysis |
| Applied Policy / Governance | 60% | 30% | 10% | Balance their research + infrastructure perspective |
| Knowledge Management / Info Sci | 60% | 30% | 10% | Orientation dynamics, systematic methods |
| Technical / Engineering | 50% | 30% | 20% | Implementation details, performance, scalability |
| Design / HCI | 60% | 20% | 20% | User experience, participatory methods |

## AmurMaple Theme Color Selection

```latex
% Blue - Knowledge systems, information science
\definecolor{amurblue}{RGB}{0,74,128}

% Green - Cooperative economics, sustainability, policy
\definecolor{amurgreen}{RGB}{34,139,34}

% Orange - Technical/engineering, innovation
\definecolor{amurorange}{RGB}{230,126,34}

% Purple - Social science, critical theory
\definecolor{amurpurple}{RGB}{142,68,173}

% Accent colors for power dynamics alerts
\definecolor{amuraccent}{RGB}{220,20,60}  % Crimson
\definecolor{amurdark}{RGB}{25,25,25}     % Near-black
```

## Three Provocation Structure Template

Each presentation should have 3 research opportunities. Use this template:

### Provocation [N]: [Compelling Title]

**Slides Structure:**
1. **Opening** (1 slide): Their finding/framework + Infrastructure gap
2. **Research Question** (1 slide): Novel intellectual contribution
3. **Methodology** (2-3 slides): How VisionFlow enables new research
4. **Pilot Proposal** (1-2 slides): Concrete deliverables, timeline, outputs

**Content Requirements:**
- **What**: Research question advancing their field
- **Why**: Intellectual contribution (not just "making it easier")
- **How**: Specific VisionFlow capabilities required:
  * 3D immersive visualization for [X]
  * Knowledge graph federation for [Y]
  * Real-time collaboration for [Z]
  * Nostr/Lightning for [W]
- **Pilot**: Specific institution/project/dataset, timeline, research outputs

**Framing Checklist:**
- [ ] Cites their specific work
- [ ] Advances their research agenda (not ours)
- [ ] Regional/institutional context included
- [ ] Power dynamics addressed (who benefits?)
- [ ] Federated architecture emphasized
- [ ] Concrete deliverables specified
- [ ] Research outputs identified (papers, tools, methods)

## Common Pitfalls to Avoid

1. **Generic proposals**: "Improve collaboration" → Specific: "Map Manchester AI vendor ecosystem for participatory audit"
2. **Tech-solutionism**: "AI will democratize governance" → "Infrastructure that could enable democratic governance if combined with political commitment"
3. **Overselling**: "Revolutionary platform" → "Research infrastructure with open questions"
4. **Missing citations**: Generic "research shows" → "Your DPAF framework (Whittle & Mills, 2023)"
5. **Vague pilots**: "Test in a city" → "Manchester City Council vendor audit pilot, Q2 2026"
6. **Ignoring power**: "Everyone benefits" → "Who controls data? Who sets priorities? Who's excluded?"

## Example Agent Coordination Message

```
I need to create a research collaboration presentation for [TARGET NAME/URL].

Please execute the following agents in parallel:

1. Profile Researcher: Analyze [TARGET]'s research, create comprehensive profile
2. Ontology Aligner: Map logseq corpus concepts to their research domains
3. Structure Planner: Design 18-slide presentation with 3 provocations
4. Theme Designer: Create AmurMaple [COLOR] Beamer framework

Then sequentially:
5. Beamer Coder: Generate complete LaTeX from all research materials
6. Compile PDF and create preview images
7. Generate documentation summary
8. Commit all deliverables to repository

Target domain: [e.g., "AI policy and dark patterns"]
Regional context: [e.g., "Manchester/Salford, UK"]
Tone: [e.g., "70% critical solidarity, 20% technical, 10% possibility"]
```

## Quick Reference: File Naming Convention

- `[slug]-research-profile.md` - Comprehensive background analysis
- `[slug]-ontology-alignments.md` - Logseq concept mapping
- `[slug]-presentation-structure.md` - Slide outline and planning
- `[slug]-presentation-framework.tex` - Beamer theme template
- `[slug]-collaboration-presentation.tex` - Complete LaTeX source
- `[slug]-collaboration-presentation.pdf` - Final presentation
- `[slug]-preview-00.png` through `[slug]-preview-04.png` - First 5 slides
- `[slug]-presentation-summary.md` - Documentation and framing guide

Where `[slug]` is lowercase-hyphenated version of target name (e.g., `whittle`, `ul-durar`)

## Verification Checklist

Before committing, verify:

- [ ] PDF compiles cleanly (warnings OK, errors not)
- [ ] Preview images show first 5 slides correctly
- [ ] Citations are accurate (check publications)
- [ ] Regional context is specific (name institutions)
- [ ] Power dynamics addressed throughout
- [ ] Three provocations each have concrete pilots
- [ ] Tone matches target domain calibration
- [ ] No tech-solutionism or overselling
- [ ] Summary document includes cover email framing
- [ ] All source files committed to git

## Advanced: Handling Special Cases

### Target with No Public Research Profile
If researcher has minimal online presence:
1. Use institutional affiliation to infer domain
2. Research their department's focus areas
3. Create broader domain-aligned provocations
4. Emphasize exploratory collaboration over specific projects

### Multi-Institutional Targets
For research groups or centers:
1. Profile 2-3 key researchers
2. Identify common themes across members
3. Propose infrastructure serving multiple research agendas
4. Frame as "platform for diverse projects" not single-purpose tool

### Non-Academic Targets (Policy, Industry)
Adjust framing:
- Less academic citations, more deliverables
- Shorter timeline for pilots (months not years)
- Emphasize operational impact over publications
- Include ROI/impact metrics if appropriate (with caveats)

## Integration with Other Skills

This skill works well with:
- **`latex-documents`**: For detailed LaTeX/Beamer syntax
- **`logseq-ontology`**: For ontology manipulation and querying
- **`web-summary`**: For fetching target's web content
- **`perplexity-research`**: For real-time research updates
- **`github-workflow-automation`**: For automated PR creation

## Example Usage

```bash
# User provides target
User: "Create a pitch for Dr. Jane Smith at UCL who works on participatory AI governance"

# You respond
I'll generate a research collaboration presentation for Dr. Jane Smith using
multi-agent orchestration. This will take 5-10 minutes.

[Execute 5-phase workflow]

# Result
Created comprehensive pitch deck:
- 72-page PDF with AmurMaple green theme (policy focus)
- 3 provocations: Participatory AI Auditing Toolkit, London AI Commons,
  Community Governance Protocols
- Tone: 70% critical solidarity, 20% technical, 10% possibility
- UCL-specific pilot proposals with concrete deliverables
- All files committed to repository
```

## Maintenance Notes

**Update when:**
- VisionFlow architecture changes (re-fetch from deepwiki)
- Logseq ontology structure evolves (adjust concept queries)
- LaTeX packages update (test compilation)
- AmurMaple theme gets new variants (add color options)

**Version History:**
- v1.0.0 (2025-11-17): Initial skill based on successful Ul-Durar and Whittle presentations
