# Fashion Markdown Content Enrichment Plan

## Problem Statement

The 147 fashion markdown files in `jjohare/logseq/fashionMarkdown/` contain ONLY
structured OntologyBlock metadata — no human-readable body text. The knowledge
graph has 147 nodes and **0 functional edges** because:

1. **No body text** — files end after the OntologyBlock `#### Relationships` section
2. **Title ≠ filename** — wikilinks use titles (`[[Bag]]`) but node IDs hash filenames
   (`accessory--bag`), so `hash("Bag") ≠ hash("accessory--bag")` — edges land on
   orphan stubs instead of real page nodes
3. **Some files are empty** — at least `garment--jacket.md`, `style--streetwear.md`,
   `occasion--formal.md`, `commerce--brand.md`, `context--season.md`,
   `footwear--boot.md` are 0-line stubs (wikilink targets that were never populated)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 0: Parser Fix (Rust)                                 │
│  Build title→filename lookup so [[Bag]] resolves to         │
│  hash("accessory--bag") instead of hash("Bag")              │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  Phase 1: Taxonomy Index (one-shot)                         │
│  Build canonical page registry: filename → title → namespace│
│  + existing wikilink targets from OntologyBlocks            │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  Phase 2: Research Agents (parallel, batched)               │
│  Per-file: generate 150-300 word body text in Logseq format │
│  with 5-15 wikilinks targeting REAL pages from the registry │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  Phase 3: Cross-link Validation                             │
│  Verify every [[wikilink]] targets an existing page         │
│  Ensure bidirectional coverage (if A→B, ideally B→A)        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  Phase 4: Push to GitHub                                    │
│  Batch commit updated files via gh API                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Parser Title Resolution Fix

### Problem

`KnowledgeGraphParser.page_name_to_id()` hashes the raw filename:
```
"accessory--bag" → hash("accessory--bag") → ID 483291
```
But wikilinks say `[[Bag]]`:
```
"Bag" → hash("Bag") → ID 729104   (DIFFERENT — orphan stub)
```

### Fix: Two-pass title→ID resolution

In `knowledge_graph_parser.rs` (or at the sync-service level where all files
are available):

1. **First pass**: Parse every file, extract the `# Title` heading line, build a
   `HashMap<String, u32>` mapping canonical titles to page IDs:
   ```
   "Bag"              → hash("accessory--bag")
   "Cotton"           → hash("material--cotton")
   "Body Shape"       → hash("body--body-shape")
   "Garment or Clothing" → hash("garment--garment-or-clothing")
   ```

2. **Second pass** (or post-processing): For every edge whose `target` ID doesn't
   match any page node ID, look up the wikilink text in the title map. If found,
   rewrite `edge.target` to the resolved page ID. Discard the orphan stub node.

3. **Also normalise**: `alt-terms::` wikilinks (e.g., `[[Tote]]`, `[[Carry-All]]`)
   won't resolve to any page. These remain as stub nodes (expected — they represent
   synonym concepts, not full pages).

### Files to edit
- `src/services/parsers/knowledge_graph_parser.rs` — add title extraction + lookup
- `src/services/github_sync_service.rs` — build title map across batch, resolve post-parse
- `src/services/local_file_sync_service.rs` — same resolution logic

---

## Phase 1: Taxonomy Index

### 12 Namespaces (from filename prefixes)

| Namespace | Count | Description |
|-----------|-------|-------------|
| accessory | 16 | Bags, belts, jewellery, scarves, hats |
| body | 5 | Body shapes, sizes, gender, characteristics |
| bridge | 8 | Cross-ontology alignment nodes |
| care | 14 | Washing, bleaching, drying, certifications |
| commerce | 12 | Business entities, pricing, delivery, payments |
| context | 6 | Sustainability, ethical sourcing, UK high street |
| footwear | 9 | Boots, heels, sneakers, sandals |
| garment | 18 | Core clothing items (dresses, shirts, trousers) |
| material | 22 | Fabrics and textiles (cotton, silk, leather, synthetics) |
| occasion | 10 | Events and contexts (wedding, funeral, interview) |
| style | 5 | Dress codes, textures, seasons, weather |
| underwear | 12 | Lingerie, bras, socks, shapewear |

### Canonical Page Registry Format

Build a JSON registry for agents to consume:
```json
{
  "accessory--bag": {
    "title": "Bag",
    "namespace": "accessory",
    "term_id": "FASH-0151",
    "relationships": ["Accessory", "Handbag", "Shoulder Bag", "Leather"],
    "has_body": false,
    "wikilink_target": "Bag"
  },
  ...
}
```

This registry serves as the agents' wikilink vocabulary — they can ONLY create
`[[Target]]` wikilinks where `Target` matches a `wikilink_target` value in the
registry. This guarantees every link resolves to a real page node.

### Build Steps

1. Fetch all 147 files from GitHub API
2. For each: extract `# Title`, `term-id::`, `relates-to::`, `is-subclass-of::`,
   existing body presence (has_body)
3. Output `/tmp/fashion-registry.json`
4. Identify empty stub files that need full generation vs files needing body append

---

## Phase 2: Research Agent Content Generation

### Agent Architecture

```
                    ┌──────────────────┐
                    │  Coordinator     │
                    │  (1 agent)       │
                    │  Distributes     │
                    │  batches, merges │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
     │ Writer #1   │  │ Writer #2  │  │ Writer #N   │
     │ ~25 files   │  │ ~25 files  │  │ ~25 files   │
     │ per batch   │  │ per batch  │  │ per batch   │
     └─────────────┘  └────────────┘  └─────────────┘
```

- **6 writer agents** processing ~25 files each (147 / 6)
- Each agent receives: the file's current content + the full page registry
- Each agent produces: updated markdown with body text appended

### Content Template

For a file like `accessory--bag.md`, the agent appends AFTER the OntologyBlock:

```markdown
-
- ## Overview
	- A bag is one of the most versatile pieces in the [[Accessory]] category, serving both functional and aesthetic purposes across every [[Occasion]] from casual outings to formal [[Business Meeting]]s. Available in materials ranging from [[Leather]] and [[Suede]] to [[Recycled Polyester]] and [[Nylon]], bags bridge the gap between [[Clothing Style]] and practical utility.
	-
- ## Types and Variations
	- The [[Handbag]] remains the most iconic form, typically carried by hand or on the forearm, whilst the [[Shoulder Bag]] offers hands-free convenience with a longer strap. Structured variants in [[Leather]] or [[Nubuck]] suit [[Interview]]s and [[Dinner Party]] settings, whereas canvas [[Tote]]s align with [[Sustainability]] values and everyday use.
	-
- ## Styling and Context
	- In UK high-street fashion, bags are styled to complement the [[Outfit]] rather than dominate it. A well-chosen bag in a coordinating [[Clothing Texture]] can elevate a simple [[Dress]] or [[Blazer]] combination. Seasonal shifts (see [[Season]]) influence material choice: lighter [[Cotton]] and canvas for summer, heavier [[Leather]] and [[Wool]] blends for autumn and winter.
	-
- ## Care
	- Maintenance depends on material: [[Leather]] bags benefit from regular conditioning, whilst [[Synthetic Clothing Material]] options tolerate [[Washing Care]] more readily. See [[Care Instruction]] for material-specific guidance.
```

### Content Rules for Agents

1. **Logseq format**: Every line is a bullet (`- `) with tab indentation for nesting
2. **Line endings**: `\r\n` (critical for Logseq parsing)
3. **UK English**: colour, centre, favourite, specialise, etc.
4. **Wikilinks**: 5-15 per file, ONLY targeting pages in the registry
5. **No duplicate links**: Each `[[Target]]` appears at most twice per file
6. **Cross-namespace linking**: Prioritise links to OTHER namespaces (not just siblings).
   A `garment--dress.md` should link to materials, occasions, styles, accessories — not
   just other garments
7. **Word count**: 150-300 words of body text (3-5 sections of 1-2 paragraphs each)
8. **Sections**: Overview, Types/Variations, Styling/Context, Care/Maintenance (adapt per
   namespace — care items get Application sections, materials get Properties sections)
9. **Preserve OntologyBlock**: Append body AFTER the existing content, never modify it
10. **Empty stubs**: Generate the full file: `public:: true`, `# Title`, OntologyBlock
    (from related pages' relationship data), then body text
11. **No fabricated citations**: Do not invent `[[Author YEAR]]` references — only use
    wikilinks to other fashion pages
12. **definition:: line as seed**: Use the existing `definition::` property text as the
    seed/basis for the Overview paragraph — expand, don't contradict

### Per-Namespace Section Templates

| Namespace | Sections |
|-----------|----------|
| accessory | Overview, Types, Styling, Care |
| body | Overview, Classification, Styling Implications, Measurement |
| bridge | Overview, Alignment Purpose, Mapped Concepts |
| care | Overview, Application Method, Applicable Materials, Certifications |
| commerce | Overview, Industry Context, Consumer Impact |
| context | Overview, Relevance to Fashion, UK Perspective, Trends |
| footwear | Overview, Types, Styling, Materials, Care |
| garment | Overview, Fit and Silhouette, Styling, Materials, Occasions |
| material | Overview, Properties, Common Uses, Care, Sustainability |
| occasion | Overview, Dress Code, Key Pieces, Seasonal Considerations |
| style | Overview, Characteristics, Application, Cultural Context |
| underwear | Overview, Types, Fit, Materials, Care |

---

## Phase 3: Cross-link Validation

After all agents complete:

1. **Parse every generated file** with the wikilink regex `\[\[([^\]|]+)(?:\|[^\]]+)?\]\]`
2. **Check each target** against the page registry — flag any `[[Target]]` where Target
   is not in the registry
3. **Build adjacency matrix** — count edges per node:
   - Target: minimum 3 inbound links per page (popularity floor)
   - Maximum: no page should have >30 inbound links (avoid hub dominance)
4. **Bidirectional coverage**: If page A links to page B, check whether B links back to A.
   Flag one-way links for manual review (not all need reciprocation, but high-value
   relationships should be bidirectional)
5. **Orphan detection**: After resolution, any page with 0 edges is a failure — every page
   must participate in at least 1 edge
6. **Edge count target**: 147 pages × 8 avg outbound links = ~1,176 edges (vs current 0)

### Validation Script

```bash
# After content generation, run locally:
for f in /tmp/enriched/*.md; do
  grep -oP '\[\[([^\]|]+)' "$f" | sed 's/\[\[//' | sort -u
done | sort | uniq -c | sort -rn > /tmp/link-frequency.txt
```

---

## Phase 4: Push to GitHub

### Strategy

Use `gh api` to update files via the GitHub Contents API in batches:

```bash
# For each file:
gh api repos/jjohare/logseq/contents/fashionMarkdown/{filename} \
  --method PUT \
  --field message="feat: add body text with wikilinks to {filename}" \
  --field content="$(base64 -w0 < enriched/{filename})" \
  --field sha="{current_sha}"
```

### Batch Plan

- Batch size: 10 files per commit (avoid rate limits)
- 15 batches total (147 / 10)
- Each batch: single commit with descriptive message
- Alternative: create a branch, push all files, open a PR for review

### Recommended: PR-based approach

1. Create branch `feat/fashion-body-text`
2. Push all 147 updated files
3. Open PR with summary of changes + edge count metrics
4. Review sample files before merge
5. After merge, VisionFlow server reloads on next sync cycle

---

## Execution Timeline

| Phase | Agents | Parallelism | Estimated Work |
|-------|--------|-------------|----------------|
| 0 | 1 coder | Sequential | Parser fix: ~3 files |
| 1 | 1 researcher | Sequential | Registry build: 147 API calls |
| 2 | 6 writers | Parallel | Content generation: ~25 files each |
| 3 | 1 validator | Sequential | Cross-link audit |
| 4 | 1 deployer | Sequential | GitHub push |

---

## Success Criteria

1. All 147 files have human-readable body text (150-300 words)
2. Every file has 5-15 outbound wikilinks to real pages
3. Parser resolves `[[Title]]` → correct page node ID
4. Graph loads with 147 nodes and 500+ edges (target: ~1,000)
5. GPU physics simulation produces a connected, spread-out layout
6. Labels display readable fashion content at typical viewing distances

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Wikilink targets typos | Validate against registry; reject unknown targets |
| Over-linking to popular pages | Cap inbound links at 30; distribute across namespace |
| Body text contradicts definition | Seed from `definition::` property |
| Empty stubs missing OntologyBlock | Generate from related pages' relationship data |
| Parser title resolution collisions | Use exact case-sensitive title match first |
| Rate limit on GitHub API | Batch 10 files per commit; 3s delay between batches |
| Content quality variance | Coordinator reviews random 10% sample before push |
