# Research Pitch Generator - Example Usage

This document shows how to use the research-pitch-generator skill to create collaboration presentations.

## Quick Start

### 1. Basic Invocation

```
Create a research collaboration presentation for [TARGET NAME] at [INSTITUTION].

Profile: [LinkedIn URL or website]
Domain: [Their research area]
Context: [Regional or institutional specifics]
```

### 2. Example: Policy Researcher

```
Create a research collaboration presentation for Dr. Sarah Johnson at King's College London.

Profile: https://www.kcl.ac.uk/people/sarah-johnson
Domain: Digital rights and AI regulation in UK public services
Context: Working with Westminster City Council on algorithmic transparency
```

Expected output:
- Green AmurMaple theme (policy focus)
- 70% critical solidarity, 20% technical, 10% possibility
- Three provocations focused on: Public Service AI Auditing, Westminster Digital Twin, Algorithmic Accountability Infrastructure
- UK public services regional context throughout

### 3. Example: Technical Researcher

```
Create a research collaboration presentation for Prof. Michael Chen.

Profile: https://research.example.edu/chen
Domain: Knowledge graph construction and semantic web technologies
Context: Leading EU-funded project on federated data systems
```

Expected output:
- Blue AmurMaple theme (knowledge systems)
- 50% research, 30% technical, 20% implementation
- Three provocations focused on: Federated Knowledge Graph Protocols, Cross-Border Data Sovereignty, Semantic Interoperability Standards
- EU research infrastructure context

### 4. Example: Critical Researcher

```
Create a research collaboration presentation for Dr. Aisha Patel at Manchester Metropolitan.

Profile: https://www.linkedin.com/in/aisha-patel-critical-ai/
Domain: Postcolonial computing, algorithmic bias, Global South technology justice
Context: Community organizing work with Manchester migrant support networks
```

Expected output:
- Green or purple AmurMaple theme (critical theory)
- 70% critical solidarity, 20% technical, 10% possibility
- Three provocations focused on: Decolonial AI Mapping, Community Data Sovereignty, South-South Solidarity Networks
- Explicit anti-extractive framing, Manchester migrant community context

## Workflow Overview

When you invoke this skill, the following happens:

### Phase 1: Context Sync (30 seconds)
- Pulls latest logseq corpus from git
- Fetches VisionFlow architecture from deepwiki
- Establishes current ontology concepts

### Phase 2: Multi-Agent Research (5-7 minutes)
- **Agent 1**: Creates comprehensive research profile (publications, frameworks, roles)
- **Agent 2**: Maps 10-15 ontology concepts to target's domains
- **Agent 3**: Designs 18-slide structure with 3 provocations
- **Agent 4**: Creates themed Beamer framework template

All agents run in parallel for efficiency.

### Phase 3: Generation (3-5 minutes)
- **Agent 5**: Generates complete LaTeX (typically 1000-1200 lines)
- Compiles PDF (2 passes for navigation)
- Fixes any LaTeX syntax errors automatically
- Generates preview images of first 5 slides

### Phase 4: Documentation (1 minute)
- Creates comprehensive summary document
- Includes cover email framing suggestions
- Lists all deliverables and file sizes

### Phase 5: Repository Commit (30 seconds)
- Commits all 8 files to git
- Pushes to remote repository
- Provides commit hash for reference

**Total time**: 10-15 minutes for complete, production-ready presentation

## Advanced Usage

### Custom Tone Calibration

```
Create a research collaboration presentation for Dr. Alex Morgan.

Profile: https://example.com/morgan
Domain: Mixed methods research on AI in healthcare
Tone: 60% research methods, 25% technical infrastructure, 15% clinical possibilities

Focus on: How immersive visualization enables novel qualitative-quantitative integration
```

### Multi-Institutional Target

```
Create a research collaboration presentation for the Urban AI Lab consortium.

Key members:
- Prof. Lisa Wang (UCL) - Smart cities and urban data
- Dr. James O'Brien (Bristol) - Participatory design
- Dr. Fatima Hassan (Leeds) - Environmental justice

Theme: Infrastructure serving diverse urban research agendas
Context: Northern cities consortium (Manchester, Leeds, Newcastle, Bristol)
```

### Non-Academic Target

```
Create a research collaboration presentation for Manchester City Council Digital Strategy Team.

Contact: Tom Richardson (Head of Digital)
Domain: Local government digital transformation
Timeline: Shorter pilots (3-6 months, not multi-year)
Emphasis: Operational deliverables over publications
Context: Post-COVID digital service redesign, budget constraints
```

## Customization Options

### Theme Color Override

```
Use AmurMaple orange theme (innovation/technical focus)
```

### Provocation Count

```
Create 4 provocations instead of 3
(for targets with very broad research portfolio)
```

### Length Adjustment

```
Create compact 12-slide version for initial outreach
(Can expand to full 18-slide version if interest confirmed)
```

## Output Files Explained

After completion, you'll have 8 files in `/home/devuser/workspace/logseq/docs/`:

1. **`[slug]-research-profile.md`** (300-400 lines)
   - Comprehensive background research
   - Publications with citations
   - 5 collaboration opportunities identified

2. **`[slug]-ontology-alignments.md`** (500-600 lines)
   - 10-15 concepts from logseq corpus
   - Cross-domain links and relevance scores
   - Research application suggestions

3. **`[slug]-presentation-structure.md`** (700-900 lines)
   - Complete slide outline
   - Tone calibration rationale
   - Provocation details with pilots

4. **`[slug]-presentation-framework.tex`** (800-1000 lines)
   - Beamer theme template
   - TikZ diagram templates
   - Color scheme definitions

5. **`[slug]-collaboration-presentation.tex`** (1000-1200 lines)
   - Complete presentation source code
   - All content integrated
   - Ready for compilation

6. **`[slug]-collaboration-presentation.pdf`** (400-600KB, 60-80 pages)
   - Production-ready presentation
   - Includes all slides, navigation, transitions

7. **`[slug]-preview-00.png` through `[slug]-preview-04.png`** (10-50KB each)
   - First 5 slides as PNG images
   - For quick review without opening PDF

8. **`[slug]-presentation-summary.md`** (150-200 lines)
   - Documentation of content and design
   - Cover email framing suggestions
   - Key positioning messages

## Troubleshooting

### "Can't find target's research profile"
- Provide alternative sources (institutional bio, Google Scholar, ResearchGate)
- If minimal online presence, specify research domain broadly
- Skill will create domain-aligned provocations even without detailed profile

### "Logseq corpus seems outdated"
- Skill automatically pulls latest from git
- If still outdated, manually pull: `cd /home/devuser/workspace/logseq && git pull`

### "LaTeX compilation errors"
- Skill fixes common errors automatically (typos, missing packages)
- If persistent errors, check skill output for specific error messages
- Most Unicode warnings are non-fatal (presentation still compiles)

### "Tone seems off for target"
- Specify desired split explicitly: "70% critical, 20% technical, 10% possibility"
- Provide example of their writing style if available
- Request revision after seeing preview slides

## Quality Checklist

Before sending presentation to target, verify:

- [ ] **Citations accurate**: Check target's publication titles, years, co-authors
- [ ] **Regional context**: Specific institutions/projects named (not generic)
- [ ] **Power dynamics**: Who benefits? Who decides? addressed throughout
- [ ] **Pilots concrete**: Specific deliverables, timelines, institutions
- [ ] **No tech-solutionism**: Limits acknowledged, political commitment required
- [ ] **Tone appropriate**: Matches target's domain and communication style
- [ ] **Visuals clean**: TikZ diagrams render correctly, colors accessible
- [ ] **Preview images**: First 5 slides show title, opening, first provocation

## Integration with Workflow

### After Generation

1. **Review preview images** (slides 0-4) to verify tone and framing
2. **Customize if needed**: Minor edits to LaTeX, recompile
3. **Draft cover email** using suggestions from summary document
4. **Send PDF** with brief introduction (not full summary)

### If Target Responds Positively

1. **Schedule demo call**: Show VisionFlow 3D capabilities live
2. **Co-design workshop**: Refine provocations with their input
3. **Grant proposal**: Collaborate on ESRC/Research England/regional funding
4. **Pilot project**: Start with one provocation, 3-6 month timeline

### If Target Declines or No Response

1. **Document learnings**: What worked/didn't in summary
2. **Refine skill**: Update tone calibration or provocation templates
3. **Try related researchers**: Use skill for their collaborators/colleagues

## Version History

- **v1.0.0** (2025-11-17): Initial skill with 5-phase workflow, multi-agent orchestration
- Future: Add support for video presentation generation, shorter pitch decks

## Related Skills

- **latex-documents**: For detailed LaTeX/Beamer editing
- **web-summary**: For fetching target's web content
- **perplexity-research**: For real-time research on target
- **logseq-ontology**: For querying ontology concepts
