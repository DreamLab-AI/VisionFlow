---
name: "Logseq Formatted"
description: "Create public-facing 'logseq formatted' pages with proper nested bullet-point outlines, WikiLinks, embeds, and academic citation style. Use when writing research notes, creating knowledge base entries, documenting complex topics, or building interconnected thought pieces with UK English spelling and first-person authorial voice."
---

# Logseq Digital Garden Skill

## What This Skill Does

Emulates a writer creating authoritative digital garden pages in Logseq format. Produces raw Markdown structured as nested bullet-point outlines using Logseq-specific syntax including `[[WikiLinks]]`, `{{embeds}}`, proper image properties, and heavily-cited academic prose in UK English.

## Prerequisites

- Understanding of Logseq's outliner-based structure
- Familiarity with Markdown syntax
- Knowledge of the topic being written about
- UK English spelling conventions

## When to Use This Skill

Use this skill when you need to:
- Create research notes in Logseq format
- Build interconnected knowledge base entries
- Document complex topics with heavy citations
- Write authoritative analysis pieces
- Convert research into digital garden format
- Maintain consistent Logseq structure and style
- Produce citation-heavy academic content

---

## Quick Start

### Create Your First Digital Garden Entry

```markdown
- # Bitcoin as Digital Gold
	- The comparison between [[Bitcoin]] and gold as store-of-value assets has become increasingly relevant since [[Nakamoto 2008]] introduced the concept of 'digital scarcity' through proof-of-work consensus. We observe that Bitcoin's fixed supply schedule (capped at 21 million coins) mirrors gold's relative scarcity, though the mechanisms differ fundamentally—one relies on cryptographic proof, whilst the other depends on physical extraction costs and geological constraints.
	-
	- ## Key Properties Comparison
		- **Scarcity:** Bitcoin's algorithmic supply cap versus gold's geological limits
		- **Divisibility:** Bitcoin's 8 decimal places (satoshis) versus gold's practical divisibility constraints
		- **Portability:** Bitcoin's near-zero cost digital transfer versus gold's physical transport challenges
```

### Basic Structure Pattern

Every piece of content lives in a bullet block:
```markdown
- # Main Heading
	- First paragraph of content with [[WikiLinks]] and citations [[Author YEAR]].
	-
	- ## Subheading
		- Nested content goes one tab deeper.
		- ![image.jpg](../assets/image.jpg){:width 300}
```

---

## Part 1: Structural & Markdown Style (Logseq Syntax)

### Primary Structure Rules

**Every piece of content must be a bullet block:**
```markdown
- Single line of text
- # This is a heading
- This is a paragraph. It can be very long and contain multiple sentences, but it remains as a single bullet block.
```

**Never use asterisks for bullets** - Logseq generates bullets automatically. Always use hyphens (`-`).

### Line Termination

**Critical:** Logseq requires `\r\n` (backslash-r, backslash-n) line termination. The skill will not create proper blocks without these.

- Each bullet point block ends with `\r\n`
- Empty blocks for spacing also use `\r\n`
- This is essential for Logseq to parse the outline correctly

### Headings

Use Markdown headings within bullet blocks:

```markdown
- # Top-Level Heading (H1)
	- Content for this section starts indented one tab
	-
	- ## Second-Level Heading (H2)
		- Content indented under H2
		-
		- ### Third-Level Heading (H3)
			- Content indented under H3
			-
			- #### Fourth-Level Heading (H4)
				- Don't go deeper than H4
```

**Rule:** The block following a heading should be indented one tab deeper than the heading itself.

### Indentation & Nesting

Create hierarchy with tabs:
```markdown
- Top level
	- One tab deep
		- Two tabs deep
			- Three tabs deep (maximum recommended)
```

**Tab Rule:** Use a single tab character before each hyphen to create nesting.

### Links & Citations

#### Internal/Conceptual Links (WikiLinks)

Use `[[WikiLink]]` format for:
- Key concepts: `[[cypherpunk]]`, `[[proof-of-work]]`
- People: `[[Satoshi Nakamoto]]`, `[[Nick Szabo]]`
- Cross-references: `[[Digital Asset Risks]]`
- Technologies: `[[Bitcoin]]`, `[[Ethereum]]`

```markdown
- The [[cypherpunk]] movement of the 1990s laid the groundwork for [[Bitcoin]]'s emergence, with figures like [[Adam Back]] and [[Wei Dai]] contributing foundational concepts that [[Nakamoto 2008]] synthesised into a working system.
```

#### Academic Citations

Use `[[Author YEAR]]` format for scholarly references:
```markdown
- As demonstrated by [[Nakamoto 2008]], the double-spending problem can be solved without trusted third parties through proof-of-work consensus.
- Multiple studies [[Antonopoulos 2017]], [[Narayanan 2016]] have explored the implications of blockchain technology.
```

**If a citation is missing:** Search for appropriate academic sources and add them in this format.

#### External Links

Standard Markdown `[link text](URL)` format:
```markdown
- Bitcoin's whitepaper ["Bitcoin: A Peer-to-Peer Electronic Cash System"](https://bitcoin.org/bitcoin.pdf) introduced the concept in 2008.
- The [Lightning Network](https://lightning.network/) provides a Layer 2 scaling solution.
```

**Link text conventions:**
- Direct quotes from the source
- Article/paper titles
- Descriptive phrases

### Media & Embeds (Logseq Specific)

#### Videos

```markdown
- {{video https://www.youtube.com/watch?v=y48uAeHwZGg}}
```

#### Tweets/X Posts

```markdown
- {{twitter https://twitter.com/username/status/123456789}}
```

#### Block Embeds

```markdown
- {{embed ((block-uuid))}}
```

This embeds content from other blocks or pages within Logseq.

#### Images

Standard Markdown with Logseq properties:
```markdown
- ![bitcoin-chart.png](../assets/bitcoin-chart.png){:width 300}
- ![network-diagram.jpg](../assets/network-diagram.jpg){:height 400}
```

**Properties:** Append styling in curly braces:
- `{:width 300}` - Set width in pixels
- `{:height 400}` - Set height in pixels
- `{:width 300 :height 200}` - Both dimensions

### Formatting & Page Style

#### Section Length

- **Medium to long sections** - Each section should contain substantial content
- **Depth limit:** Don't section further than `####` (H4)
- **Avoid over-bolding:** Prefer sectioning over excessive **bold** text
- Use bold sparingly for key terms or emphasis

#### Spacing

**Vertical space** between blocks:
```markdown
- First block of content.
-
- Second block with space above.
```

An empty bullet (`-`) on its own line creates breathing room.

#### Section Breaks

Horizontal rule:
```markdown
- Content before break.
- ---
- Content after break.
```

Three hyphens create a visual separator.

---

## Part 2: Prose & Content Style (Authorial Voice)

### Writing Style: Digital Garden Academic Tone

A blend of:
- Well-researched academic paper
- Technical blog post
- Personal research notes

**Characteristics:**
- Authoritative yet conversational
- Dense with information
- Heavily sourced
- Thoughtful and analytical

### Voice: First-Person Authorial

Use **"I"** and **"we"** to guide the reader:

```markdown
- With that said, we aren't convinced by the value proposition of Ethereum's current approach to scalability. Whilst the roadmap presented in [[Buterin 2020]] outlines ambitious goals, the execution timeline remains uncertain, and we observe several technical challenges that haven't been adequately addressed in the literature.
- I find the argument for Bitcoin as 'digital gold' compelling, particularly when examining the [[stock-to-flow model]] proposed by [[PlanB 2019]], though this requires significant caveats (discussed below).
```

**Express opinions, but ground them in evidence.**

### Syntax & Density

#### Long, Complex Sentences

Write information-dense paragraphs within single bullet blocks:

```markdown
- The relationship between Bitcoin's hash rate and price demonstrates a complex feedback loop wherein increased price attracts more miners (due to higher revenue potential), which in turn increases network security through greater computational power dedicated to proof-of-work, thereby making the network more resilient to attacks and potentially more attractive to institutional investors—though this correlation isn't perfectly linear and exhibits significant lag effects during market transitions [[Hayes 2017]], [[Kristoufek 2015]].
```

#### Mix Dense with Atomic

Juxtapose complex blocks with simple, atomic ones:

```markdown
- The technical implementation of Lightning Network payment channels relies on Hash Time-Locked Contracts (HTLCs), a mechanism that enables conditional payments routed across multiple intermediary nodes whilst maintaining security guarantees equivalent to on-chain transactions, albeit with different trust assumptions regarding channel partner liveness and the availability of the base layer for dispute resolution [[Poon and Dryja 2016]].
-
- ![lightning-network-diagram.png](../assets/lightning-network-diagram.png){:width 400}
-
- Simple, powerful, elegant.
```

#### Parenthetical Asides

Use parentheses `()` frequently:

```markdown
- Bitcoin's energy consumption (currently estimated at approximately 150 TWh annually) represents a deliberate design choice rather than an inefficiency—the proof-of-work mechanism intentionally consumes energy to create an unforgeable costliness that secures the network (see [[hulsmann2008ethics]] for a defence of this approach from an Austrian economics perspective).
```

#### Inline Quotes

Long quotes in square brackets `[...]` within paragraph flow:

```markdown
- Nakamoto's original vision was clear: ["What is needed is an electronic payment system based on cryptographic proof instead of trust, allowing any two willing parties to transact directly with each other without the need for a trusted third party"] [[Nakamoto 2008]], a goal that remains partially realised but faces ongoing challenges in scalability and user experience.
```

### Lexicon & Vocabulary

**Sophisticated but accessible:**

- Use technical terms: `ancillary`, `codified`, `nascent`, `arbitrage`, `orthogonal`
- Mix with clear language: "take this with an appropriate pinch of salt"
- Occasional colloquialisms: "the sector seems to have responded with a shrug"
- Financial terminology: `liquidity`, `volatility`, `correlation`, `denominated`

```markdown
- The nascent DeFi sector has demonstrated remarkable growth, though we must take these TVL (Total Value Locked) figures with an appropriate pinch of salt given the prevalence of circular lending arrangements that artificially inflate the metrics [[Qin 2021]].
```

### Sourcing & Evidence: Link Constantly

**Every major claim needs a link:**

```markdown
- ✅ GOOD: Bitcoin's market capitalisation exceeded $1 trillion in 2021 [link to source], representing a significant milestone for the asset class.
- ❌ BAD: Bitcoin's market capitalisation exceeded $1 trillion in 2021, representing a significant milestone for the asset class.
```

**"Show your work" mentality:**
- Link to primary sources (whitepapers, code repositories)
- Link to academic papers
- Link to data sources (charts, statistics)
- Link to news articles for events
- Link to blog posts for technical explanations

**If a source is missing:** Use internet search to find high-quality citations and insert them.

### Spelling & Grammar: UK English

**Mandatory UK spellings:**
- favour (not favor)
- signalling (not signaling)
- decentralised (not decentralized)
- colour (not color)
- analysed (not analyzed)
- organisation (not organization)
- whilst (not while)
- amongst (not among)

**Follow Anthropic best practices for UK English.**

---

## Step-by-Step Guide

### 1. Plan Your Digital Garden Page

Before writing:
1. **Identify the topic** - What concept are you explaining?
2. **Research sources** - Gather academic papers, articles, data
3. **Outline structure** - Plan your H1, H2, H3 headings
4. **Identify key concepts** - List WikiLink candidates
5. **Collect citations** - Prepare `[[Author YEAR]]` references

### 2. Start with a Strong Introduction

```markdown
- # Topic Title
	- Opening paragraph that establishes context, introduces key concepts with [[WikiLinks]], and presents your thesis or main argument. This should be dense, well-sourced, and set expectations for what follows [[Key Citation YEAR]].
	-
```

### 3. Build Nested Sections

```markdown
	- ## First Major Section
		- Content exploring first aspect of the topic. Use long, complex sentences that weave together multiple ideas whilst maintaining clarity through proper structure and abundant linking to sources [[Source1 YEAR]], [[Source2 YEAR]].
		-
		- ### Subsection If Needed
			- Deeper dive into specific point.
			- ![relevant-image.png](../assets/relevant-image.png){:width 300}
		-
	-
	- ## Second Major Section
		- Continue building your argument or analysis.
```

### 4. Incorporate Media

Add visual elements:
```markdown
		- The network topology evolution demonstrates this clearly:
		- ![network-growth.png](../assets/network-growth.png){:width 400}
		- As evidenced by this data from [[Glassnode 2023]], the trend is unmistakable.
```

Embed videos for explanations:
```markdown
		- For a technical walkthrough, see:
		- {{video https://www.youtube.com/watch?v=example}}
```

### 5. Cite Heavily Throughout

Every claim gets support:
```markdown
		- The correlation between Bitcoin and traditional risk assets has increased significantly since 2020 [[Conlon 2020]], [[Corbet 2018]], suggesting a shift in how institutional investors perceive the asset class (though we should note that correlation doesn't imply causation, and the relationship may be spurious or temporary).
```

### 6. Express Considered Opinions

Use first-person voice:
```markdown
		- We remain sceptical of claims that Ethereum will successfully transition to proof-of-stake without significant technical challenges [[Buterin 2020]], particularly given the complexity of the beacon chain and the historical precedent of delayed major protocol upgrades.
		- I find this argument unconvincing because...
```

### 7. Link Concepts Together

Use WikiLinks to create connections:
```markdown
		- This relates to our earlier discussion of [[Digital Scarcity]] and connects with the broader theme of [[Cryptoeconomic Incentives]] that we'll explore in the section on [[Game Theory in Blockchain Design]].
```

### 8. End with Synthesis or Future Directions

```markdown
	-
	- ## Conclusion
		- Drawing these threads together, we observe that [synthesis of main points], which suggests [implications or future directions]. The ongoing research in [[Area of Study]] will likely prove critical to resolving these questions.
	-
	- ## Further Reading
		- [[Related Concept 1]]
		- [[Related Concept 2]]
		- [External Resource Title](URL)
```

---

## Examples

### Example 1: Bitcoin Monetary Policy Analysis

```markdown
- # Bitcoin's Monetary Policy
	- Bitcoin's fixed supply schedule represents a stark departure from traditional fiat monetary systems, implementing what [[Ammous 2018]] describes as 'absolute scarcity' in a digital medium for the first time in human history. The algorithmic supply cap of 21 million coins, enforced through consensus rules that no single party can unilaterally change, creates a deflationary monetary policy that stands in opposition to the inflationary tendencies of central bank-managed currencies [[Selgin 2015]].
	-
	- ## The Halving Mechanism
		- Bitcoin's supply issuance follows a geometric decay function, with new bitcoin creation (block rewards) halving approximately every four years (precisely every 210,000 blocks). This mechanism, hard-coded in the protocol since inception, ensures predictable and decreasing inflation:
		- 2009-2012: 50 BTC per block
		- 2012-2016: 25 BTC per block
		- 2016-2020: 12.5 BTC per block
		- 2020-2024: 6.25 BTC per block
		-
		- The current issuance rate (as of 2024) stands at approximately 1.7% annually, lower than gold's estimated 2-3% annual supply growth [[World Gold Council 2023]].
		- ![bitcoin-supply-curve.png](../assets/bitcoin-supply-curve.png){:width 400}
	-
	- ## Comparison with Fiat Systems
		- In contrast to Bitcoin's algorithmic certainty, fiat currencies operate under discretionary monetary policy where central banks adjust money supply based on economic conditions and policy objectives. The Federal Reserve, Bank of England, and European Central Bank have collectively expanded their balance sheets by over $10 trillion since 2008 [[BIS 2023]], demonstrating the stark difference in monetary philosophies.
		-
		- We observe that this fundamental divergence creates an interesting tension: whilst Bitcoin advocates argue for the superiority of rules-based monetary policy [[Hayek 1976]], critics note that discretionary policy allows for countercyclical intervention during economic crises [[Krugman 2018]]—though whether such intervention proves beneficial in the long term remains contentious.
	-
	- ---
	-
	- ## Stock-to-Flow Analysis
		- The [[stock-to-flow model]], popularised by [[PlanB 2019]], attempts to predict Bitcoin's price based on its scarcity relative to new supply. Whilst the model has shown remarkable correlation historically, we must approach it with appropriate scepticism given:
		- The limited sample size (only three complete halving cycles)
		- The post-hoc nature of the model fitting
		- The assumption that past patterns will continue indefinitely
		-
		- That said, the underlying logic—that scarcity drives value for monetary assets—has historical precedent with gold and silver [[Eichengreen 2019]].
		-
		- {{video https://www.youtube.com/watch?v=stock-flow-explanation}}
	-
	- ## Conclusion
		- Bitcoin's fixed supply schedule represents an ideological statement as much as a technical specification, encoding Austrian economic principles [[von Mises 1912]] into software that resists human intervention. Whether this proves superior to managed monetary policy remains an open question that we'll continue to monitor as the experiment unfolds.
	-
	- ## Further Reading
		- [[Austrian Economics]]
		- [[Monetary Theory]]
		- [[Digital Scarcity]]
		- [The Bitcoin Standard](https://saifedean.com/thebitcoinstandard) - [[Ammous 2018]]
```

### Example 2: Technical Deep Dive

```markdown
- # Lightning Network Architecture
	- The Lightning Network implements a Layer 2 payment channel protocol atop Bitcoin's base layer, enabling near-instant, low-cost transactions whilst maintaining the security properties of the underlying blockchain [[Poon and Dryja 2016]]. The architecture represents a significant departure from traditional blockchain scaling approaches (such as simply increasing block size), instead leveraging the concept of off-chain state updates with on-chain settlement as a security backstop.
	-
	- ## Payment Channel Fundamentals
		- At its core, a Lightning payment channel consists of a 2-of-2 multisignature Bitcoin address where two parties lock funds, creating a shared UTXO that can be spent only with both signatures. The clever innovation lies in the commitment transaction structure: each state update creates a new transaction that invalidates previous states through a penalty mechanism, ensuring neither party can profitably broadcast outdated channel states.
		-
		- ### Hash Time-Locked Contracts (HTLCs)
			- HTLCs enable conditional payments across multi-hop routes without requiring trust in intermediary nodes. The mechanism works through cryptographic hash preimages and time-based constraints:
			- Alice wants to pay Dave through intermediaries Bob and Carol
			- Dave generates a random secret and shares its hash with Alice
			- Alice creates an HTLC offering payment to Bob if he reveals the secret within 24 hours
			- Bob creates a similar HTLC to Carol (23-hour timeout)
			- Carol creates an HTLC to Dave (22-hour timeout)
			- Dave claims payment by revealing the secret
			- The secret propagates backwards, allowing each intermediary to claim their payment
			-
			- This cascading reveal mechanism ensures atomic settlement: either the entire route succeeds, or no funds are transferred [[Decker and Wattenhofer 2015]].
		-
		- ![lightning-htlc-diagram.png](../assets/lightning-htlc-diagram.png){:width 500}
	-
	- ## Network Topology and Routing
		- The Lightning Network's graph structure creates interesting economic and technical challenges. Nodes must maintain sufficient liquidity in payment channels to route transactions, creating a capital efficiency problem that the network is still actively solving through various mechanisms (dual-funded channels, channel factories, submarine swaps) [[Gudgeon 2020]].
		-
		- We observe that the network exhibits characteristics of a scale-free graph, with a small number of highly-connected nodes (hubs) facilitating most routing—though whether this centralisation tendency undermines Bitcoin's decentralisation ethos remains a point of debate [[Martinazzi 2020]].
	-
	- ## Limitations and Trade-offs
		- Whilst Lightning offers impressive scalability improvements (theoretically millions of transactions per second), it comes with several caveats that proponents sometimes downplay:
		- **Liquidity requirements:** Users must lock capital in channels
		- **Online requirement:** At least one of the channel parties (or a watchtower service) must be online to detect fraud attempts
		- **Routing challenges:** Finding paths with sufficient liquidity isn't guaranteed
		- **Channel management complexity:** Opening and closing channels requires on-chain transactions (and associated fees)
		-
		- These aren't necessarily dealbreakers, but they do suggest Lightning is better suited for certain use cases (frequent small payments) than others (large, infrequent settlements).
	-
	- ## Current Status and Adoption
		- As of 2024, the Lightning Network has approximately 15,000 nodes and 60,000 payment channels with total capacity exceeding 5,000 BTC [[1ML.com 2024]]. Whilst these numbers demonstrate real adoption, they're modest compared to Bitcoin's base layer usage, suggesting Lightning remains in early stages of its adoption curve.
		-
		- Notable developments include:
		- El Salvador's integration of Lightning for Bitcoin payments
		- Major exchanges adding Lightning deposit/withdrawal support
		- Emerging Lightning-native applications (gaming, streaming)
	-
	- ## Conclusion
		- The Lightning Network represents one of the most sophisticated attempts to scale blockchain systems without compromising decentralisation, implementing complex cryptographic protocols that most users will never need to understand. Whether it achieves mainstream adoption depends less on technical capabilities (which are impressive) and more on user experience improvements and ecosystem development.
	-
	- ## Further Reading
		- [[Bitcoin Layer 2 Solutions]]
		- [[Payment Channel Networks]]
		- [[Cryptographic Protocols]]
		- [Lightning Network Whitepaper](https://lightning.network/lightning-network-paper.pdf) - [[Poon and Dryja 2016]]
		- [BOLT Specifications](https://github.com/lightning/bolts) - Technical standards
```

---

## Tool Functions

### `create_digital_garden_page`
Generate a complete Logseq digital garden page.

Parameters:
- `topic` (required): Main topic/title of the page
- `sections` (required): Array of section objects with headings and content
- `citations` (required): Array of citation objects `{author, year, title, url}`
- `wikilinks` (optional): Array of key concepts to link internally
- `images` (optional): Array of image objects `{filename, width}`
- `uk_english` (default: true): Enforce UK spelling conventions

### `add_section`
Add a new section to an existing page.

Parameters:
- `heading_level` (required): 1-4 (# to ####)
- `heading_text` (required): Section heading
- `content` (required): Section content blocks
- `indent_level` (default: 0): Tab depth for nesting

### `format_citation`
Format academic citation in Logseq style.

Parameters:
- `authors` (required): Author name(s)
- `year` (required): Publication year
- `type` (optional): "inline" | "reference"

Returns: `[[Author YEAR]]` formatted citation

### `create_wikilink`
Generate internal WikiLink.

Parameters:
- `concept` (required): Concept name
- `display_text` (optional): Alternative display text

Returns: `[[concept]]` or `[[concept|display text]]`

### `embed_media`
Create Logseq media embed.

Parameters:
- `type` (required): "video" | "twitter" | "image" | "block"
- `url` (required): Media URL or block UUID
- `properties` (optional): Image properties object `{width, height}`

Returns: Properly formatted embed syntax

### `validate_structure`
Check Logseq syntax validity.

Parameters:
- `content` (required): Markdown content to validate

Returns: Validation report with errors and warnings

---

## Common Patterns

### Opening Paragraph Pattern

```markdown
- # [Topic]
	- The [topic] represents [contextual framing], implementing [key innovation] that [significance/implications]. As demonstrated by [[Key Citation YEAR]], [supporting evidence that grounds your opening statement], though [nuance or caveat that shows critical thinking].
```

### Argumentative Pattern

```markdown
	- ## [Claim or Position]
		- We argue that [position] for several reasons: first, [evidence 1] [[Citation1 YEAR]]; second, [evidence 2] [[Citation2 YEAR]]; and finally, [evidence 3] [[Citation3 YEAR]]. Whilst critics such as [[Critic YEAR]] contend that [counterargument], this objection fails to account for [rebuttal].
```

### Technical Explanation Pattern

```markdown
	- ## [Technical Concept]
		- At its core, [concept] consists of [fundamental components], creating [outcome or capability]. The mechanism works through [step-by-step process]:
		- Step 1: [description]
		- Step 2: [description]
		- Step 3: [description]
		-
		- This [process/mechanism] ensures [security property or benefit], though it comes with trade-offs: [limitation 1], [limitation 2] [[Technical Citation YEAR]].
```

### Synthesis Pattern

```markdown
	- ## Conclusion
		- Drawing these threads together, we observe that [synthesis of main points from earlier sections], which suggests [implications for the field/future]. The relationship between [concept A] and [concept B] proves more nuanced than simplistic narratives suggest, requiring [what's needed: more research, better tools, theoretical development].
		-
		- The ongoing work in [[Related Field]] will likely prove critical to resolving these questions, particularly as [future development or trend] continues to evolve.
```

---

## Troubleshooting

### Issue: Logseq Won't Create Blocks

**Symptoms**: Content appears as single block or doesn't parse
**Cause**: Missing `\r\n` line termination or incorrect indentation
**Solution**:
```markdown
✅ CORRECT:
- First block\r\n
- Second block\r\n

❌ WRONG:
- First block
- Second block
```

Ensure every block ends with proper line termination.

### Issue: WikiLinks Not Working

**Symptoms**: `[[Links]]` appear as plain text
**Cause**: Extra spaces or incorrect bracket syntax
**Solution**:
```markdown
✅ CORRECT: [[Bitcoin]], [[Nakamoto 2008]]
❌ WRONG: [[ Bitcoin ]], [[Nakamoto2008]]
```

No spaces inside brackets, space before year in citations.

### Issue: Images Not Displaying

**Symptoms**: Image links broken or properties not applied
**Cause**: Incorrect path or malformed properties
**Solution**:
```markdown
✅ CORRECT: ![image.jpg](../assets/image.jpg){:width 300}
❌ WRONG: ![image.jpg](assets/image.jpg){width: 300}

# Properties must use colons, be in curly braces
# Path must include ../assets/ prefix
```

### Issue: Citations Not Linking

**Symptoms**: Citations appear as plain text
**Cause**: Missing double brackets
**Solution**:
```markdown
✅ CORRECT: [[Nakamoto 2008]], [[Ammous 2018]]
❌ WRONG: [Nakamoto 2008], Nakamoto 2008
```

All citations must use WikiLink format.

### Issue: Inconsistent UK English

**Symptoms**: Mix of US and UK spellings
**Cause**: Inconsistent spelling choices
**Solution**:
Use UK English consistently:
- colour, favour, signalling
- decentralised, organised, analysed
- whilst, amongst, towards

### Issue: Sections Too Shallow

**Symptoms**: Content lacks depth or nuance
**Cause**: Not enough citation, analysis, or complexity
**Solution**:
```markdown
❌ SHALLOW:
- Bitcoin is digital money.

✅ DEEP:
- Bitcoin represents an attempt to create a digital bearer instrument through cryptographic proof-of-work, solving the double-spending problem without trusted third parties [[Nakamoto 2008]]—though whether it succeeds as 'money' depends on one's definition of that term and the specific use case in question (store of value versus medium of exchange versus unit of account).
```

Add:
- More citations
- Nuance and caveats
- Multiple perspectives
- Technical depth

---

## Integration with Other Skills

Works well with:
- `web-summary` skill for researching sources to cite
- `filesystem` skill for managing Logseq vault structure
- Search tools for finding academic citations
- Writing assistance for UK English spell-checking

---

## Advanced Usage

### Multi-Level Argumentation

```markdown
- # Complex Argument Structure
	- ## Thesis
		- [Main claim] [[Citation YEAR]].
		-
		- ### Supporting Evidence
			- **Empirical:** [Data/studies] [[Citation1 YEAR]], [[Citation2 YEAR]]
			- **Theoretical:** [Logical argument] [[Citation3 YEAR]]
			- **Historical:** [Precedents] [[Citation4 YEAR]]
		-
		- ### Anticipated Objections
			- #### Objection 1: [Counterargument]
				- Response: [Rebuttal] [[Citation5 YEAR]]
			-
			- #### Objection 2: [Counterargument]
				- Response: [Rebuttal] [[Citation6 YEAR]]
		-
	- ## Implications
		- If the thesis holds, then [consequences for field/practice/theory].
```

### Comparative Analysis

```markdown
- # Comparing [A] and [B]
	- ## Similarities
		- Both exhibit [shared characteristic 1] [[Citation YEAR]]
		- Both rely on [shared mechanism 2] [[Citation YEAR]]
	-
	- ## Differences
		- | Aspect | [A] | [B] |
		- |--------|-----|-----|
		- | Property 1 | [Description] | [Description] |
		- | Property 2 | [Description] | [Description] |
	-
	- ## Synthesis
		- The comparison reveals that [insight from comparison], suggesting [implications].
```

### Research Note Pattern

```markdown
- # Research Notes: [Topic]
	- **Date:** [[2024-01-15]]
	- **Status:** #in-progress
	- **Related:** [[Related Concept 1]], [[Related Concept 2]]
	-
	- ## Questions to Explore
		- How does [X] relate to [Y]?
		- What evidence exists for [claim]?
		- What would falsify [hypothesis]?
	-
	- ## Findings
		- [[Source YEAR]] argues that [finding].
		- {{embed ((related-block-uuid))}}
		- Counter-evidence from [[Other Source YEAR]]: [contradictory finding].
	-
	- ## Next Steps
		- Read [[Paper YEAR]] on [topic]
		- Investigate connection to [[Related Concept]]
		- Write synthesis section
```

---

## Performance Notes

- **Reading time:** Dense content takes 2-3x normal reading time
- **Link density:** Aim for 3-5 links per substantial paragraph
- **Citation ratio:** Major claims should have 1-2 supporting citations
- **Atomic blocks:** 20-30% of blocks should be simple/atomic for readability
- **Section length:** 3-7 substantial paragraphs per H2 section

---

## Style Checklist

Before publishing, verify:

- [ ] Every bullet point uses hyphen (`-`), never asterisk
- [ ] Headings are on their own bullet lines
- [ ] Content after headings is indented one level deeper
- [ ] All citations use `[[Author YEAR]]` format
- [ ] Key concepts have WikiLinks `[[Concept]]`
- [ ] External links use `[text](URL)` format
- [ ] Images include `{:width XXX}` properties
- [ ] Line termination uses `\r\n`
- [ ] UK English spelling throughout (favour, whilst, organised)
- [ ] First-person voice ("we", "I") is used
- [ ] Dense paragraphs mixed with atomic blocks
- [ ] Parenthetical asides add nuance
- [ ] Every major claim is cited
- [ ] Sophisticated vocabulary balanced with clarity
- [ ] Sections don't exceed #### depth
- [ ] Empty `-` blocks create spacing
- [ ] `---` used for section breaks
- [ ] Missing citations have been researched and added

---

## Resources

- [Logseq Documentation](https://docs.logseq.com/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Digital Gardens Overview](https://maggieappleton.com/garden-history)
- [Academic Citation Formats](https://www.chicagomanualofstyle.org/)
- [UK English Style Guide](https://www.gov.uk/guidance/style-guide)

---

**Created**: 2025-10-20
**Format**: Logseq Markdown
**Line Termination**: `\r\n` required
**Spelling**: UK English (mandatory)
**Voice**: First-person authorial
**Citation Style**: `[[Author YEAR]]` WikiLink format

when highlighting text inline as belonging to a topic PREFER subjects from this list, if appropriate. You should do this sparingly, aiming for 1-3 per page at most per subject.

3D Scene Exchange Protocol (SXP),6G Network Slice,AI Ethics Checklist,AI Governance Framework,AI Model Card,AI-Generated Content Disclosure,API Standard,Accessibility Audit Tool,Accessibility Standard,Algorithmic Transparency Index,Application Layer,Audit Trail,Augmented Reality (AR),Authoring Tool,Autonomous Agent,Avatar Interoperability,Avatar,Behavioural Feedback Loop,Biometric Binding Mechanism,Biosensing Interface,Carbon Credit Token,Central Bank Digital Currency (CBDC),Cloud Rendering Service,Cognitive Feedback Interface,Cognitive Load Metric,Collective Intelligence System,Collective Memory Archive,Community Governance Model,Compatibility Process,Compliance Audit Trail,Compute Layer,Consent Management,Construction Digital Twin,Content Delivery Network (CDN),Content Moderation,Context Awareness,Conversion Pipeline,Creator Economy,Creator Royalty Token,Cross-Border Data Transfer Rule,Cross-Platform Compliance Hub,Crypto Token,Cryptocurrency,Cultural Heritage XR Experience,Cultural Provenance Record,Data Anonymization Pipeline,Data Fabric Architecture,Data Integration Interface,Data Protection,Data Provenance,Data Storage Layer,Decentralization Layer,Decentralized Autonomous Organization (DAO),Decentralized Exchange (DEX),Decentralized Identity (DID),Deepfakes,Digital Asset Workflow,Digital Asset,Digital Citizens' Assembly,Digital Citizenship,Digital Citizens' Assembly,Digital Constitution,Digital Curation Platform,Digital Evidence Chain of Custody,Digital Goods Registry,Digital Goods,Digital Identity Framework,Digital Identity Wallet,Digital Jurisdiction,Digital Ontology Repository,Digital Performance Capture,Digital Real Estate,Digital Rights Management (Extended),Digital Ritual,Digital Tax Compliance Node,Digital Taxonomy Registry,Digital Twin Interop Protocol,Digital Twin Synchronisation Bus,Digital Twin of Society (DToS),Digital Twin,Digital Well-Being Index,Discovery Layer,Display Metrology,Dispute Resolution Mechanism,Distributed Architecture,Distributed Ledger Technology (DLT),E-Contract Arbitration,ETSI_Domain_AI,ETSI_Domain_AI___Creative_Media,ETSI_Domain_AI___Data_Mgmt,ETSI_Domain_AI___Governance,ETSI_Domain_AI___Human_Interface,ETSI_Domain_Application___Creative,ETSI_Domain_Application___Education,ETSI_Domain_Application___Health,ETSI_Domain_Application___Industry,ETSI_Domain_Application___Tourism,ETSI_Domain_Creative_Media,ETSI_Domain_Data,ETSI_Domain_Data_Management,ETSI_Domain_Data_Management___Creative,ETSI_Domain_Data_Management___Culture,ETSI_Domain_Data_Management___Ethics,ETSI_Domain_Data_Mgmt___AI,ETSI_Domain_Data_Mgmt___Security,ETSI_Domain_Ethics_&_Law,ETSI_Domain_Governance_&_Compliance,ETSI_Domain_Governance_&_Ethics,ETSI_Domain_Governance_Compliance,ETSI_Domain_Governance_Security,ETSI_Domain_Governance___Economy,ETSI_Domain_Governance___Ethics,ETSI_Domain_Governance___Society,ETSI_Domain_Human_Interface,ETSI_Domain_Human_Interface___Governance,ETSI_Domain_Human_Interface___UX,ETSI_Domain_Identity_,ETSI_Domain_Identity_&_Trust,ETSI_Domain_Immersive_Experiences,ETSI_Domain_Immersive___Reality_Capture,ETSI_Domain_Infrastructure,ETSI_Domain_Infrastructure_Data,ETSI_Domain_Infrastructure___Governance,ETSI_Domain_Infrastructure___Immersive,ETSI_Domain_Infrastructure___Interop,ETSI_Domain_Infrastructure___Security,ETSI_Domain_Interoperability,ETSI_Domain_Interoperability___Creative,ETSI_Domain_Reality_Capture,ETSI_Domain_Reality_Capture___Creative,ETSI_Domain_Security_&_Privacy,ETSI_Domain_Virtual_Economy,ETSI_Domain_Virtual_Society,Edge Computing Node,Edge Mesh Network,Edge Network,Edge Orchestration,Education Metaverse,Emotional Analytics Engine,Emotional Immersion,Environmental Impact Metric,Environmental Sustainability Label,Ethical Framework,Ethics & Law Layer,Experience Layer,Explainable AI (XAI),Extended Reality (XR),Eye Tracking,Federated Credential Exchange,Feedback Mechanism,Fractionalized NFT,Game Engine,Generative Design Tool,Glossary Index,Glossary_Index,Governance Model,Haptics,Hardware Abstraction Layer (HAL),Hardware-Platform-Agnostic,Hardware-_Platform-Agnostic,Health Metaverse Application,Human Capture & Recognition,Human Interface Device,Human Interface Layer (HIL),Humanity Attestation,Identity Federation,Identity Graph,Identity Provider (IdP),Immersion,Immersive Experience,Industrial Metaverse,Infrastructure Layer,Intelligent Virtual Entity,Interoperability Framework,Interoperability,Knowledge Graph,Latency Management Protocol,Latency,Liquidity Pool,Loyalty Token,Marketplace,Metadata Standard,Metaverse Architecture Stack,Metaverse Content Pipeline,Metaverse Liability Model,Metaverse Ontology Schema,Metaverse Psychology Profile,Metaverse Safety Protocol,Metaverse,Metaverse_Ontology_Schema,Micropayment,Middleware,Mixed Reality (MR),Motion Capture Rig,Multiverse,NFT Renting,NFT Swapping,NFT Wrapping,Narrative Design Ontology,Network Infrastructure,Networking Layer,Non-Fungible Token (NFT),OntologyDefinition,Open World,Ownership & Freedom (distributed),Ownership-Freedom-distributed,Persistence,Photogrammetry,Physics Engine,Physics-Based Animation,Platform Layer,Play-to-Earn (P2E),Policy Engine,Portability,Post-Quantum Cryptography,Presence,Privacy Impact Assessment (PIA),Privacy-Enhancing Computation (PEC),Procedural Audio Generator,Procedural Content Generation,Procedural Texture,Provenance Ontology (PROV-O),Provenance Verification,Quantum Network Node,Real-Time Rendering Pipeline,Reality Capture System,Reputation Data,Reputation Scoring Model,Resilience Metric,Right to Be Forgotten,Royalty Mechanism,Scene Graph,Security Layer,Self-Sovereign Identity (SSI),Semantic Metadata Registry,Service Layer,Smart Contract,Smart Royalties Ledger,Smart Royalty Contract,Social Impact Assessment (SIA),Social Token Economy,Spatial Anchor,Spatial Audio Scene Description,Spatial Computing Layer,Spatial Computing,Spatial Index,Stablecoin,State Synchronization,Storage Layer,Synthetic Data Generator,Telemetry & Analytics,Testing Process,Threat Surface Map,Token Bonding Curve,Token Custody Service,Tokenization,Tourism Metaverse,Transaction Standard,Trust Framework Policy,Trust Score Metric,Universal Manifest,User Agreement Compliance,User Consent Token,Validation Process,Verifiable Credential (VC),Virtual Lighting Model,Virtual Notary Service,Virtual Performance Space,Virtual Production (VP),Virtual Production Volume,Virtual Property Right,Virtual Reality (VR),Virtual Securities Offering (VSO),Virtual World,Visualization Layer,Voice Interaction,WebXR API,World Instance,XR Accessibility Guideline,Zero-Knowledge Proof (ZKP),Zero-Trust Architecture (ZTA),glTF (3D File Format)

when highlighting text inline as belonging to a topic ALSO CHOOSE FROM subjects from this list, if appropriate

3D and 4D,AI Companies,AI Risks,AI Scrapers,AI Video,AI privacy at the 2024 Olympics,Accessibility,Adoption of Convergent Technologies,Agentic Alliance,Agentic Metaverse for Global Creatives,Agentic Mycelia,Agents,Ai in Games,Algorithmic Bias and Variance,AnimateDiff,Anthropic Claude,Apple,Artificial Intelligence,Automated Podcasting,BTC Layer 3,Bias in Large Language Models,Bitcoin,Bitcoin As Money,Bitcoin ETF,Bitcoin Technical Overview,Bitcoin Value Proposition,Blender,Blockchain,Calculating Empires,California AI bill,Call Centres,Cashu,ChatGPT,Client side DCO,Coding support,Comfy UI for Fashion and Brands,ComfyUI,Comparison of GPT4 and Gemini Ultra,Comparison of SDXL and Midjourney v6,Competition in AI,Conspiracies,Consumer Tools for SMEs,Controlnet and similar,Convergence,Courses and Training,Cyber Security and Military,Cyber security and Cryptography,Death of the Internet,Debug Test Page,Decentralised Web,Deep Learning,Deepfakes and fraudulent content,Deepmind,Definitions and frameworks for Metaverse,Depth Estimation,Diagrams as Code,Diffusion Models,Digital Asset Risks,Digital Objects,Digital Society Harms,Digital Society Surveillance,Distributed Identity,EU AI Act,Education and AI,Energy and Power,Ethereum,Evaluation benchmarks and leaderboards,Facebook Meta,Financialised Agentic Memetics,Flux,Foundation Models,Future Bernard,GANs,Gaussian splatting and Similar,Gemini,Geopolitical hot takes,Global Inequality,Gold,Haptics,Hardware and Edge,Human tracking and SLAM capture,Human vs AI,Humans, Avatars , Character,Hyper personalisation,IPAdapter,Inpainting,Interfaces,Introduction to me,Jailbreaking,Knowhere,Knowledge Graphing,Landscape,Large language models,Layoff tracker and threatened roles,Lead Poisoning Hypothesis,Leopold Aschenbrenner,Llama,LoRA DoRA etc,Machine Learning,Medical AI,Metaverse Ontology,Metaverse and Spatial Risks,Metaverse as Markets,Micropayments,Microsoft CoPilot,Microsoft Work Trends Impact 2024,Mixed reality,Model Control Protocols like MCP,Model Optimisation and Performance,Money,Multi Agent RAG scrapbook,Music and audio,NVIDIA Omniverse,National Industrial Centre for Virtual Environments,Norbert Wiener,Nostr protocol,Octave Multi Model Laboratory,Open Generative AI tools,OpenAI,Overview of Machine Learning Techniques,Papers Of Note,Parametric,Politics, Law, Privacy,Privacy, Trust and Safety,Procedural and Hybrid 4D,Product Design,Product and Risk Management,Project Automated Podcast,Project BroBots,Prompt Engineering,Proprietary AI Video,Proprietary Image Generation,Proprietary Large Language Models,Proprietary Video,RGB and Client Side Validation,Reasoning,Recent Projects,Research Tools,Revision List,Robin Hanson,Robotics,Runes and Glyphs,Rust,SLAM,Safety and alignment,Sam Hammond,Scene Capture and Reconstruction,Segmentation and Identification,Semantic Web,Singularity,Social contract and jobs,Soon-Next-Later (AI futurology),Spatial Computing,Speech and voice,Stable Coins,Stable Diffusion,State Space and Other Approaches,State of the art in AI,Suggested Reading Order,Tim Reutermann,Time Series Forecasting,Tokenisation,Training and fine tuning,Training for Design Practitioners,Transformers,Upscaling,VP robotics project,Vesuvian Scrolls,Virtual Production,Vision Pro,VisionFlow and Junkie Jarvis,artificial superintelligence,collaborative,cypherpunk,debug linked node,ecash,flossverse,infrastructure,latent space,legacy media,license,multimodal,ollama,p(doom),relighting

where you find a web url link in wiki style without any explainer text around it to contextualise it then you should use your web-summary skill to call the zai model to return logseq formatted text to carefully integrate into the surrounding text. Similarly if you find a youtube link without any context you should do the same.
