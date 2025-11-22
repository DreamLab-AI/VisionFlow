# Canonical Content Format Specification

**Version:** 1.0.0
**Date:** 2025-11-21
**Status:** Authoritative Specification
**Author:** Content Architect - Swarm Content Standardization
**Purpose:** Definitive standard for non-OntologyBlock content across all knowledge graph pages

---

## Executive Summary

This document defines THE canonical format for page content that appears AFTER the OntologyBlock in the Logseq knowledge graph. It complements the canonical OntologyBlock schema by standardizing the rich narrative content, ensuring consistency, quality, and usability across all 1,709 markdown files.

**Key Design Decisions:**
- **UK English spelling** throughout all content (organise, colour, behaviour, etc.)
- **Logseq markdown** with wiki-style links `[[Term]]` for semantic connections
- **Hyphen-defined blocks** (bullets with `-`) for all content structure
- **Four-layer architecture**: Technical précis → Rich narrative → Regional context → Metadata
- **Domain-specific variations** while maintaining core structure consistency
- **Quality-driven approach** with measurable standards for completeness

---

## Design Principles

### 1. UK English Standard

All content MUST use UK English spelling conventions:

**Common US → UK Conversions:**
- organize → **organise**
- color → **colour**
- behavior → **behaviour**
- center → **centre**
- defense → **defence**
- realize → **realise**
- analyze → **analyse**
- optimize → **optimise**
- catalog → **catalogue**
- traveled → **travelled**
- labeled → **labelled**
- modeling → **modelling**
- favor → **favour**
- honor → **honour**
- neighbor → **neighbour**

**Technical Terms:**
- digitize → **digitise**
- standardize → **standardise**
- synchronize → **synchronise**
- parameterize → **parameterise**

**Exceptions:**
- Proper nouns and brand names maintain original spelling (e.g., "Google Colab")
- Code snippets and technical identifiers remain unchanged
- Direct quotations preserve original spelling

### 2. Logseq Markdown Conventions

**Wiki-Style Links:**
```markdown
- [[Concept Name]] - Link to concept pages
- [[Technology|Display Text]] - Link with custom display text
- #tag - Tag references
```

**Hyphen-Defined Blocks:**
```markdown
- Section header
  - Subsection content
    - Nested detail
  - Another subsection
```

**Embedded Media:**
```markdown
- ![Image description](../assets/image.png)
- {{embed [[Block Reference]]}}
- {{video https://youtube.com/...}}
```

**Code Blocks:**
````markdown
- ```python
  # Always specify language
  def example():
      return "formatted code"
  ```
````

### 3. Content Architecture

Content follows a progressive disclosure pattern:

**Layer 1: Technical Précis** (200-300 words)
- Concise technical overview
- Key characteristics and features
- Immediate context and relevance
- 3-5 wiki links to related concepts

**Layer 2: Rich Narrative** (500-1000 words)
- Comprehensive explanation
- Academic and theoretical foundations
- Current state of technology/concept
- Practical implementations and applications
- 10-15 wiki links throughout

**Layer 3: Contextual Information**
- UK-specific context (institutions, implementations)
- Research directions and key literature
- Future trends and emerging developments
- Case studies and real-world examples

**Layer 4: Termination Metadata**
- References (formatted citations)
- Metadata (review status, dates, quality indicators)

### 4. Semantic Linking Strategy

**When to Create Wiki Links:**

✅ **DO link:**
- Technical terms defined elsewhere in corpus
- Domain concepts (AI, Blockchain, Robotics, etc.)
- Proper nouns (technologies, standards, frameworks)
- Related concepts mentioned in explanations
- Parent and sibling concepts
- Prerequisite knowledge
- Enabled capabilities or applications

❌ **DON'T link:**
- Common words (computer, data, system in generic sense)
- Terms already linked in same paragraph
- Generic concepts not in knowledge graph
- Action verbs unless they represent specific methodologies
- Adjectives and adverbs

**Linking Frequency:**
- Technical Overview: 3-5 links (critical concepts only)
- Detailed Explanation: 10-15 links (comprehensive coverage)
- Overall density: ~1-2 links per 100 words

---

## Canonical Structure

### Complete Page Template

```markdown
- ### OntologyBlock
  [Already standardized - see canonical-ontology-block.md]

# {Preferred Term}

## Technical Overview
- **Definition**: [2-3 sentence precise technical definition with [[key]] [[concept]] links]
- **Key Characteristics**:
  - [Characteristic 1 with [[wiki link]] where appropriate]
  - [Characteristic 2 with technical detail]
  - [Characteristic 3 with context]
  - [Characteristic 4 if applicable]
- **Primary Applications**: [Where/how it's used in practice]
- **Related Concepts**: [[Concept1]], [[Concept2]], [[Concept3]]

## Detailed Explanation
- Comprehensive overview
  - [Opening paragraph: What it is, why it matters, fundamental principles]
  - [Second paragraph: How it works, key mechanisms, underlying technologies]
  - [Third paragraph: Evolution and development, historical context]

- Technical architecture
  - [Core components and their relationships]
  - [System design and structure]
  - [Key technologies and dependencies]

- Capabilities and features
  - [Primary capabilities with examples]
  - [Advanced features and functionalities]
  - [Distinguishing characteristics]

- Implementation considerations
  - [Practical deployment aspects]
  - [Integration requirements]
  - [Performance and scalability factors]

## Academic Context
- Theoretical foundations
  - [Academic grounding and theoretical frameworks]
  - [Fundamental principles from computer science, engineering, etc.]
  - [Mathematical or logical foundations where applicable]

- Key researchers and institutions
  - [Leading researchers in the field]
  - [Major research institutions (emphasise UK where applicable)]
  - [Significant academic contributions]

- Seminal papers and publications
  - [Foundational research papers]
  - [Key academic publications with brief descriptions]
  - [Standards documents and specifications]

- Current research directions (2025)
  - [Active areas of investigation]
  - [Emerging research themes]
  - [Open problems and challenges]

## Current Landscape (2025)
- Industry adoption and implementations
  - [Current state of commercial adoption]
  - [Major companies and organisations using the technology]
  - [Market trends and deployment patterns]

- Technical capabilities and limitations
  - [What it can do well]
  - [Current limitations and constraints]
  - [Performance characteristics]

- Standards and frameworks
  - [Industry standards and specifications]
  - [Regulatory frameworks]
  - [Best practices and guidelines]

- Ecosystem and tools
  - [Supporting technologies and tools]
  - [Developer ecosystem]
  - [Available resources and platforms]

## UK Context
- British contributions and implementations
  - [UK innovations and developments]
  - [British researchers and pioneers]
  - [UK-specific implementations]

- Major UK institutions and organisations
  - **Universities**: [Leading UK universities with research programmes]
  - **Research Labs**: [UK-based research facilities and centres]
  - **Companies**: [British companies implementing or developing the technology]

- Regional innovation hubs
  - **London**: [Capital-based activity and organisations]
  - **Edinburgh**: [Scottish contributions]
  - **Manchester**: [North West developments]
  - **Newcastle**: [North East initiatives]
  - **Leeds/Sheffield**: [Yorkshire contributions]
  - **Cambridge/Oxford**: [Academic concentrations]

- Regional case studies
  - [Specific examples of UK implementations]
  - [Success stories and lessons learnt]
  - [Challenges and solutions in UK context]

## Practical Implementation
- Technology stack and tools
  - [Required technologies and frameworks]
  - [Recommended tools and platforms]
  - [Development environments and SDKs]

- Best practices and patterns
  - [Industry-proven approaches]
  - [Design patterns and architectures]
  - [Optimisation strategies]

- Common challenges and solutions
  - [Typical problems encountered]
  - [Proven solutions and workarounds]
  - [Pitfalls to avoid]

- Case studies and examples
  - [Real-world implementations]
  - [Lessons learnt from deployments]
  - [Quantified outcomes where available]

## Research & Literature
- Key academic papers and sources
  - [Author, Year. "Title". Journal/Conference. DOI/URL]
  - [Formatted citations with brief annotations]
  - [Prioritise seminal and recent (2020+) works]

- Ongoing research directions
  - [Current research themes]
  - [Emerging areas of investigation]
  - [Future research priorities]

- Academic conferences and venues
  - [Major conferences in the field]
  - [Key academic journals]
  - [Research communities and networks]

## Future Directions
- Emerging trends and developments
  - [Technology trends for 2025-2030]
  - [Anticipated advances and breakthroughs]
  - [Industry direction and evolution]

- Anticipated challenges
  - [Technical challenges on the horizon]
  - [Ethical and social considerations]
  - [Regulatory and policy issues]

- Research priorities
  - [Key areas requiring investigation]
  - [Grand challenges in the field]
  - [Long-term research goals]

- Predicted impact
  - [Expected societal impact]
  - [Economic implications]
  - [Transformative potential]

## References
1. [Author, A., & Author, B. (Year). Title of work. Publisher/Journal. DOI/URL]
2. [Author, C. (Year). Title. Conference/Journal. DOI/URL]
3. [Organisation. (Year). Standard/Specification Title. URL]
4. [Author, D. et al. (Year). Title. Journal. DOI]
5. [Minimum 5 references, ideally 8-12 for comprehensive coverage]

## Metadata
- **Last Updated**: YYYY-MM-DD
- **Review Status**: [Initial Draft | Comprehensive Editorial Review | Expert Reviewed]
- **Content Quality**: [High | Medium | Requires Enhancement]
- **Completeness**: [100% | 80% | 60% | Stub]
- **Verification**: [Academic sources verified | Requires verification]
- **Regional Context**: [UK/North England where applicable | Global | Needs UK context]
- **Curator**: [Team or individual responsible]
- **Version**: [M.m.p]
```

---

## Section Requirements

### Tier 1: Mandatory Sections

These sections MUST be present in every page:

#### Technical Overview
**Required elements:**
- ✅ Definition (2-3 sentences, 50-150 words)
- ✅ Key Characteristics (3-5 bullet points)
- ✅ At least 3 wiki links to related concepts
- ✅ Clear statement of primary applications

**Quality standards:**
- Definition is precise and unambiguous
- Characteristics are specific and informative
- Links connect to existing pages or intentional forward references
- Applications are concrete and relevant

#### Detailed Explanation
**Required elements:**
- ✅ Minimum 500 words of substantive content
- ✅ 3-5 subsections providing comprehensive coverage
- ✅ At least 10 wiki links throughout
- ✅ Technical depth appropriate to concept complexity
- ✅ Examples or illustrations of key points

**Quality standards:**
- Content is technically accurate
- Explanations are clear and accessible
- Progressive detail from general to specific
- Examples clarify abstract concepts
- UK English spelling throughout

### Tier 2: Strongly Recommended Sections

These sections should be included for all substantive concepts:

#### Academic Context
**Recommended for:** Technical concepts, methodologies, theoretical frameworks

**Required elements:**
- ✅ Theoretical foundations (academic grounding)
- ✅ Key researchers or institutions (emphasise UK)
- ✅ At least 2 seminal papers or publications
- ✅ Current research directions (2025 state)

**Quality standards:**
- Academic sources are authoritative
- UK contributions highlighted where applicable
- Research directions are current (2020+)
- Citations are properly formatted

#### Current Landscape (2025)
**Recommended for:** Technologies, platforms, methodologies in active use

**Required elements:**
- ✅ Industry adoption status and trends
- ✅ Technical capabilities and limitations
- ✅ Standards and frameworks
- ✅ Current state as of 2025

**Quality standards:**
- Information is current (2024-2025)
- Claims are verifiable or clearly marked as estimates
- Both strengths and limitations discussed
- Standards cited are active and relevant

#### UK Context
**Recommended for:** All concepts (universally valuable)

**Required elements:**
- ✅ At least 2 UK institutions or organisations mentioned
- ✅ British contributions or innovations noted
- ✅ Regional information (North England where applicable)
- ✅ UK-specific implementations or case studies

**Quality standards:**
- UK institutions are accurately represented
- Regional distribution considered (not only London)
- Contributions are verifiable
- Case studies are concrete and informative

#### Practical Implementation
**Recommended for:** Technologies, frameworks, methodologies

**Required elements:**
- ✅ Technology stack or tools
- ✅ Best practices
- ✅ Common challenges and solutions
- ✅ At least one case study or example

**Quality standards:**
- Advice is actionable and practical
- Tools are current and relevant
- Challenges are realistic
- Case studies are detailed enough to be useful

### Tier 3: Optional Sections

Include based on concept type and available information:

#### Research & Literature
**Optional but valuable for:** Academic concepts, emerging technologies

**Suggested elements:**
- Key academic papers (beyond those in Academic Context)
- Ongoing research directions
- Academic conferences and venues
- Research communities

#### Future Directions
**Optional but valuable for:** Emerging technologies, evolving concepts

**Suggested elements:**
- Emerging trends (2025-2030 horizon)
- Anticipated challenges
- Research priorities
- Predicted impact

---

## Content Quality Standards

### Technical Overview Quality

**High Quality (Target):**
- Definition: 75-150 words, technically precise
- Key Characteristics: 4-5 specific, informative points
- Wiki Links: 5-7 links to directly related concepts
- Applications: 2-3 concrete examples
- UK English: Perfect compliance

**Medium Quality (Acceptable):**
- Definition: 50-75 words, generally clear
- Key Characteristics: 3-4 points with some detail
- Wiki Links: 3-5 links
- Applications: 1-2 examples
- UK English: Minor inconsistencies

**Requires Enhancement (Needs Work):**
- Definition: <50 words or vague
- Key Characteristics: <3 points or generic
- Wiki Links: <3 links
- Applications: Missing or too generic
- UK English: Significant US spelling

### Detailed Explanation Quality

**High Quality (Target):**
- Length: 750-1000 words
- Structure: 4-5 well-organised subsections
- Wiki Links: 15-20 links throughout
- Technical Depth: Appropriate to concept complexity
- Examples: 3-5 concrete examples or illustrations
- UK English: Perfect compliance
- Clarity: Accessible to target audience while maintaining rigor

**Medium Quality (Acceptable):**
- Length: 500-750 words
- Structure: 3-4 subsections
- Wiki Links: 10-15 links
- Technical Depth: Generally appropriate
- Examples: 2-3 examples
- UK English: Minor inconsistencies
- Clarity: Generally clear

**Requires Enhancement (Needs Work):**
- Length: <500 words
- Structure: <3 subsections or unorganised
- Wiki Links: <10 links
- Technical Depth: Too shallow or inappropriately complex
- Examples: <2 examples or too abstract
- UK English: Significant US spelling
- Clarity: Confusing or inaccessible

### UK Context Quality

**High Quality (Target):**
- UK Institutions: 4+ organisations or universities mentioned
- Regional Distribution: At least 2 UK regions represented
- Contributions: Specific innovations or developments cited
- Case Studies: 2+ detailed UK-based examples
- Depth: Substantive information, not just name-dropping

**Medium Quality (Acceptable):**
- UK Institutions: 2-3 organisations mentioned
- Regional Distribution: At least 1 region beyond London
- Contributions: General contributions noted
- Case Studies: 1 UK example
- Depth: Basic information provided

**Requires Enhancement (Needs Work):**
- UK Institutions: <2 organisations or all non-UK
- Regional Distribution: London-only or missing
- Contributions: None identified or too vague
- Case Studies: No UK examples
- Depth: Token mention only

---

## Wiki Linking Guidelines

### Strategic Linking

**Primary Links (Always Link):**
- Core concepts fundamental to understanding
- Direct prerequisites
- Parent concepts in taxonomy
- Key technologies or methodologies
- Proper nouns (when in knowledge graph)

**Secondary Links (Usually Link):**
- Related concepts in same domain
- Alternative approaches or competitors
- Applications or use cases
- Supporting technologies

**Tertiary Links (Selectively Link):**
- Tangential concepts
- Historical references
- Future possibilities
- Cross-domain connections

### Linking Best Practices

1. **First Mention Rule**: Link a term at first mention in each major section
2. **Density Balance**: Aim for 1-2 links per 100 words
3. **Anchor Text**: Use natural, descriptive anchor text
4. **Forward References**: Link to pages you intend to create
5. **Avoid Over-linking**: Don't link every occurrence
6. **Context Matters**: Link when the connection adds value

### Examples

**Good Linking:**
```markdown
- [[Machine Learning]] is a subset of [[Artificial Intelligence]] that
  enables systems to learn from data. Common approaches include
  [[Supervised Learning]], [[Unsupervised Learning]], and
  [[Reinforcement Learning]].
```

**Over-linking (Avoid):**
```markdown
- [[Machine Learning]] is a [[subset]] of [[Artificial Intelligence]]
  that enables [[systems]] to [[learn]] from [[data]]. [[Common]]
  [[approaches]] include [[Supervised Learning]], [[Unsupervised Learning]],
  and [[Reinforcement Learning]].
```

**Under-linking (Avoid):**
```markdown
- Machine learning is a subset of artificial intelligence that enables
  systems to learn from data. Common approaches include supervised learning,
  unsupervised learning, and reinforcement learning.
```

---

## Code Block Standards

### Language Specification

**Always specify language for syntax highlighting:**

````markdown
- ```python
  # Python code example
  def hello_world():
      print("Hello, World!")
  ```

- ```javascript
  // JavaScript code example
  function helloWorld() {
      console.log("Hello, World!");
  }
  ```

- ```bash
  # Shell commands
  npm install package-name
  ```
````

### Common Languages

- `python` - Python code
- `javascript` - JavaScript
- `typescript` - TypeScript
- `bash` / `shell` - Shell commands
- `sql` - SQL queries
- `json` - JSON data
- `yaml` - YAML configuration
- `markdown` - Markdown examples
- `clojure` - OWL Functional Syntax (by convention)
- `rust` - Rust code
- `go` - Go code
- `java` - Java code
- `cpp` - C++ code

### Code Block Best Practices

1. **Include Comments**: Explain non-obvious code
2. **Keep It Concise**: Focus on relevant portions
3. **Proper Indentation**: Use consistent formatting
4. **Complete Examples**: Code should be runnable (when applicable)
5. **Error Handling**: Include for production examples

---

## Domain-Specific Variations

### Artificial Intelligence Domain

**Emphasis:**
- Mathematical and algorithmic foundations
- Training and inference characteristics
- Model architecture details
- Performance metrics and benchmarks
- Ethical considerations (bias, fairness, transparency)

**Additional Sections (Optional):**
- Model Architecture
- Training Methodology
- Performance Characteristics
- Ethical Considerations

### Blockchain Domain

**Emphasis:**
- Cryptographic foundations
- Consensus mechanisms
- Security properties
- Decentralisation characteristics
- Economic incentive structures

**Additional Sections (Optional):**
- Cryptographic Foundations
- Consensus Mechanism
- Security Properties
- Economic Model

### Robotics Domain

**Emphasis:**
- Physical components and hardware
- Control systems and algorithms
- Sensor and actuator details
- Real-world deployment considerations
- Safety and reliability

**Additional Sections (Optional):**
- Hardware Components
- Control Architecture
- Sensor Systems
- Safety Considerations

### Metaverse Domain

**Emphasis:**
- Immersive experience characteristics
- Virtual world technologies
- User interaction paradigms
- Content creation and distribution
- Social and economic aspects

**Additional Sections (Optional):**
- Virtual World Architecture
- User Experience Design
- Content Creation Tools
- Economic Systems

### Telecollaboration Domain

**Emphasis:**
- Communication protocols and technologies
- Collaboration frameworks
- Synchronous vs asynchronous mechanisms
- User experience and accessibility
- Integration with existing tools

**Additional Sections (Optional):**
- Communication Architecture
- Collaboration Features
- Integration Capabilities
- Accessibility Considerations

### Disruptive Technologies Domain

**Emphasis:**
- Innovation characteristics
- Market disruption patterns
- Adoption curves and barriers
- Transformative potential
- Regulatory and social implications

**Additional Sections (Optional):**
- Disruption Analysis
- Adoption Patterns
- Market Impact
- Regulatory Landscape

---

## Reference Formatting

### Citation Styles

**Academic Papers:**
```markdown
- Author, A., & Author, B. (Year). Title of paper. Journal Name, Volume(Issue), page-page. DOI
```

**Conference Papers:**
```markdown
- Author, A., Author, B., & Author, C. (Year). Title. In Conference Name (pp. page-page). Publisher. DOI/URL
```

**Books:**
```markdown
- Author, A. (Year). Book Title (Edition ed.). Publisher. ISBN
```

**Web Resources:**
```markdown
- Organisation. (Year). Document Title. Retrieved from URL
```

**Standards:**
```markdown
- Standards Body. (Year). Standard ID: Standard Title. URL
```

### Examples

```markdown
## References

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444. https://doi.org/10.1038/nature14539
2. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008). Curran Associates.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. ISBN: 978-0262035613
4. OpenAI. (2023). GPT-4 Technical Report. Retrieved from https://arxiv.org/abs/2303.08774
5. ISO/IEC. (2022). ISO/IEC 23894:2022 - Artificial Intelligence. https://www.iso.org/standard/77304.html
```

---

## Metadata Standards

### Required Metadata Fields

```markdown
## Metadata
- **Last Updated**: YYYY-MM-DD
- **Review Status**: [Status value]
- **Content Quality**: [Quality rating]
- **Completeness**: [Percentage or description]
```

### Optional Metadata Fields

```markdown
- **Verification**: [Verification status]
- **Regional Context**: [Geographic relevance]
- **Curator**: [Responsible party]
- **Version**: [Semantic version]
- **Contributors**: [Additional contributors]
- **Expert Reviewed By**: [Expert reviewer if applicable]
```

### Metadata Value Standards

**Review Status Values:**
- `Initial Draft` - First version, needs review
- `Comprehensive Editorial Review` - Reviewed for structure and quality
- `Expert Reviewed` - Reviewed by domain expert
- `Peer Reviewed` - Multiple experts reviewed
- `Requires Update` - Content needs refresh

**Content Quality Values:**
- `High` - Meets all quality standards
- `Medium` - Acceptable, minor improvements needed
- `Requires Enhancement` - Significant improvements needed
- `Stub` - Minimal content, major expansion needed

**Completeness Values:**
- `100%` - All sections complete and comprehensive
- `80%` - Core sections complete, some enhancement possible
- `60%` - Essential sections present, others missing
- `40%` - Basic information only
- `Stub` - Minimal content

---

## Quality Assurance Checklist

### Pre-Publication Checklist

**Content Standards:**
- [ ] UK English spelling throughout (no US spellings)
- [ ] Hyphen-defined blocks used consistently
- [ ] Wiki links use `[[Page Name]]` format
- [ ] Code blocks specify language
- [ ] All URLs are properly formatted or annotated

**Structure Standards:**
- [ ] OntologyBlock present at top of page
- [ ] Technical Overview section present and complete
- [ ] Detailed Explanation section present (≥500 words)
- [ ] At least 3 Tier 2 sections included
- [ ] References section present with ≥5 citations
- [ ] Metadata section present and complete

**Quality Standards:**
- [ ] Definition is precise and comprehensive (2-3 sentences)
- [ ] At least 10 wiki links throughout content
- [ ] UK Context includes ≥2 institutions
- [ ] Regional distribution considered (not London-only)
- [ ] Technical depth appropriate to concept
- [ ] Examples and illustrations provided
- [ ] References are authoritative and verifiable

**Verification Standards:**
- [ ] Academic sources verified for accuracy
- [ ] UK institutions and organisations verified
- [ ] Technical claims are supportable
- [ ] Last Updated date is current
- [ ] Review Status accurately reflects state
- [ ] No placeholder text (e.g., "TODO", "TBD")

---

## Migration and Update Procedures

### Updating Existing Content

**Step 1: Assessment**
1. Read existing content completely
2. Identify missing mandatory sections
3. Assess quality of existing sections
4. Note US spellings to convert
5. Identify opportunities for wiki links

**Step 2: Enhancement**
1. Convert US English to UK English
2. Add missing mandatory sections
3. Enhance thin sections to meet quality standards
4. Add wiki links (target: 10+ total)
5. Improve Technical Overview if needed

**Step 3: Enrichment**
1. Add UK Context if missing or thin
2. Enhance Academic Context
3. Add Practical Implementation if applicable
4. Expand References (target: 8-12 citations)
5. Update metadata

**Step 4: Verification**
1. Run quality assurance checklist
2. Verify all wiki links
3. Verify UK institutions
4. Check formatting consistency
5. Update Last Updated date

### Creating New Content

**Step 1: Planning**
1. Choose appropriate domain template
2. Research concept thoroughly
3. Identify related concepts for linking
4. Locate authoritative references
5. Identify UK context and institutions

**Step 2: Writing**
1. Create OntologyBlock first (see canonical-ontology-block.md)
2. Write Technical Overview (use template)
3. Write Detailed Explanation (500-1000 words)
4. Add mandatory Tier 2 sections
5. Add optional sections as applicable

**Step 3: Enhancement**
1. Add wiki links (target: 10-15)
2. Verify UK English spelling
3. Add code examples if relevant
4. Include diagrams or media if helpful
5. Compile References section

**Step 4: Finalisation**
1. Complete Metadata section
2. Run quality assurance checklist
3. Set appropriate Review Status
4. Mark Completeness level
5. Submit for review if applicable

---

## Tools and Automation

### Validation Tools

**Spelling Checker:**
- Tool: Custom dictionary with UK English + technical terms
- Usage: Automated scan for US spellings
- Action: Highlight for manual review and correction

**Link Checker:**
- Tool: Logseq query or script
- Usage: Identify broken wiki links
- Action: Create forward reference pages or fix links

**Quality Scorer:**
- Tool: Custom script analysing word count, link density, section presence
- Usage: Generate quality score for each page
- Action: Prioritise low-scoring pages for enhancement

**UK Context Validator:**
- Tool: Script checking for UK institution mentions
- Usage: Identify pages lacking UK context
- Action: Flag for UK context addition

### Recommended Workflows

**Daily Writing:**
1. Use domain-specific template
2. Write in UK English from start
3. Add wiki links as you write
4. Complete all mandatory sections
5. Run quality checklist before saving

**Bulk Enhancement:**
1. Generate quality report for all pages
2. Prioritise by importance and deficiency
3. Batch process US→UK spelling conversions
4. Add UK Context systematically
5. Enhance wiki linking density
6. Update metadata

**Periodic Review:**
1. Review pages quarterly
2. Update Current Landscape (2025) annually
3. Verify Research & Literature currency
4. Check for broken wiki links
5. Update Last Updated dates

---

## Examples and Templates

### See Domain-Specific Templates

Detailed templates with examples for each domain:

- `/docs/content-standardization/templates/content-template-ai.md`
- `/docs/content-standardization/templates/content-template-blockchain.md`
- `/docs/content-standardization/templates/content-template-robotics.md`
- `/docs/content-standardization/templates/content-template-metaverse.md`
- `/docs/content-standardization/templates/content-template-telecollaboration.md`
- `/docs/content-standardization/templates/content-template-disruptive-tech.md`

---

## References

1. W3C. (2012). OWL 2 Web Ontology Language Document Overview. https://www.w3.org/TR/owl2-overview/
2. Logseq Documentation. (2025). Markdown Format and Block Properties. https://docs.logseq.com/
3. British Standards Institution. (2023). BS ISO 24765:2023 - Systems and Software Engineering Vocabulary. https://www.bsigroup.com/
4. Oxford English Dictionary. (2025). British vs American English Spelling. Oxford University Press.
5. UK Government Digital Service. (2025). GOV.UK Content Design Style Guide. https://www.gov.uk/guidance/style-guide

---

**Document Control:**
- **Version**: 1.0.0
- **Status**: Authoritative
- **Approved By**: Content Architect
- **Review Date**: 2025-11-21
- **Next Review**: 2025-12-21
- **Changelog**: Initial canonical specification
