### **Master Instruction for the Documentation Swarm**

**Mission:** Your collective mission is to transform the existing corpus of development notes, reports, and documentation into a formalized, consistent, and authoritative knowledge base for the "AR-AI Knowledge Graph" (VisionFlow) project. You will analyze all provided documents, synthesize the information, eliminate redundancy, formalize the content, and restructure it according to the target architecture defined below. The final output should be a clean, professional, and easily navigable documentation set that preserves all critical technical details.

**Guiding Framework:** The target structure is based on the [Diátaxis framework](https://diataxis.fr/), which organizes documentation into four distinct modes: **Tutorials** (Getting Started), **How-To Guides** (Guides), **Reference**, and **Explanation** (Concepts).

---

### **Target Documentation Architecture**

All agents must work towards refactoring the file structure to match this clean, logical hierarchy.

```
docs/
├── README.md                # NEW: High-level project entry point, brief overview, and navigation.
├── index.md                 # DEPRECATED: Content to be merged into README.md and 00-INDEX.md.
├── 00-INDEX.md              # REFACTORED: The master table of contents and navigation hub.
│
├── getting-started/         # (Tutorials) Onboarding for new users/developers.
│   ├── 01-installation.md
│   ├── 02-first-graph.md
│   └── 03-deploying-agents.md
│
├── guides/                  # (How-To) Practical step-by-step guides for specific tasks.
│   ├── development-workflow.md
│   ├── deployment-production.md
│   ├── orchestrating-agents.md
│   ├── extending-the-system.md
│   ├── testing-and-verification.md
│   ├── troubleshooting.md
│   └── xr-setup.md
│
├── concepts/                # (Explanation) Understanding the "why" behind the system.
│   ├── system-architecture.md
│   ├── agentic-workers.md
│   ├── gpu-compute.md
│   ├── networking-protocols.md
│   ├── security-model.md
│   ├── ontology-and-validation.md
│   └── decisions/             # Architectural Decision Records (ADRs).
│       ├── adr-001-unified-api-client.md
│       └── ...
│
├── reference/               # (Reference) Technical descriptions and API specs.
│   ├── glossary.md
│   ├── configuration.md
│   ├── api/
│   │   ├── index.md
│   │   ├── rest-api.md
│   │   ├── websocket-protocol.md  # CANONICAL: Single source of truth for WebSocket.
│   │   ├── binary-protocol.md
│   │   ├── mcp-protocol.md
│   │   └── openapi-spec.yml
│   └── agents/                # Master directory for all agent definitions.
│       ├── index.md
│       ├── conventions.md
│       ├── core/
│       ├── consensus/
│       ├── github/
│       └── ... (consolidated agent definitions)
│
└── _archive/                  # For outdated but historically relevant files.
    ├── code-examples-2025-10/
    ├── reports/
    └── legacy-implementations/
```

---

### **Agent Roles & Specializations**

Your swarm is composed of the following specialists:

1.  **Chief Documentation Architect (Coordinator):** Oversees the entire process, assigns tasks to specialized agents, resolves structural conflicts, and ensures the final output adheres to the target architecture.
2.  **Content Analyst Agents:** Read and parse all existing documents to extract metadata, topics, key details, code snippets, metrics, and relationships between documents.
3.  **Synthesis & Consolidation Agents:** Identify and merge duplicate or overlapping content into a single, canonical source of truth.
4.  **Technical Writer Agents:** Rewrite informal notes and reports into formal, clear, and consistent documentation. They enforce the style guide.
5.  **Structural Engineer Agents:** Rename, move, and delete files to implement the target file structure. They are responsible for the physical organization of the documentation.
6.  **Link & Graph Integrity Agents:** Update all internal links, cross-references, and Mermaid diagrams to reflect the new structure and ensure navigational integrity.

---

### **Multi-Phase Execution Plan**

Execute the mission in the following phases:

#### **Phase 1: Analysis & Metadata Extraction**

*   **Assigned to:** Content Analyst Agents
*   **Task:**
    1.  Read every file in the provided `<file_tree>`.
    2.  For each file, create a metadata record containing:
        *   **Topics Covered:** (e.g., "WebSocket", "Binary Protocol", "Client Architecture", "Agent Spawning").
        *   **Document Type:** (e.g., "Architecture Deep Dive", "Status Report", "Development Note", "API Reference", "ADR").
        *   **Key Entities:** List all major components or concepts mentioned (e.g., `GraphServiceActor`, `MCP Session Bridge`, `Binary Protocol V2`).
        *   **Status Flags:** Extract any explicit status markers like `⭐ NEW`, `🔄 UPDATED`, `✅ CURRENT`, `⚠️ DEPRECATED`.
        *   **Cross-References:** List all explicit links to other documents.
        *   **Code & Diagrams:** Index all code blocks and Mermaid diagrams.
    3.  Pay special attention to `00-INDEX.md` and the root `README.md`, as they contain high-level structural information and recent updates. The "Recent Updates (October 2025)" section in `00-INDEX.md` is a primary source of details to be formalized.

#### **Phase 2: Synthesis & Consolidation (Creating the Single Source of Truth)**

*   **Assigned to:** Synthesis & Consolidation Agents
*   **Task:**
    1.  Using the metadata from Phase 1, identify all documents covering the same topic.
    2.  **Merge Duplicates:** Create a single, canonical document for each topic.
        *   **Example 1:** Merge `architecture/components/websocket-protocol.md` and `reference/api/websocket-protocol.md`. The content from both should be combined into a single, comprehensive file at the new canonical location: `reference/api/websocket-protocol.md`.
        *   **Example 2:** Consolidate content from `architecture/overview.md`, `architecture/system-overview.md`, and `concepts/system-architecture.md` into a single, authoritative `concepts/system-architecture.md`.
    3.  **Integrate Notes:** Extract informal notes and updates and merge them into the relevant canonical documents.
        *   **Example:** Take the "Recent Updates (October 2025)" section from `00-INDEX.md`.
            *   The "Binary Protocol V2 Upgrade" details must be integrated into the canonical `reference/api/binary-protocol.md`.
            *   The "Dual-Graph Broadcasting Fix" details must be integrated into `reference/api/websocket-protocol.md`.
            *   The "Agent Management Implementation" details should be integrated into `concepts/agentic-workers.md` and `architecture/hybrid_docker_mcp_architecture.md`.
    4.  **Consolidate Entry Points:** Merge the useful, non-redundant parts of the root `README.md` and `00-INDEX.md`. The new `docs/README.md` should be a welcoming project overview, and `docs/00-INDEX.md` should become the master table of contents.
    5.  **Resolve Overlaps:** Analyze overlapping directories like `deployment/` and `development/` and merge their content into unified guides (e.g., `guides/deployment-production.md`).

#### **Phase 3: Restructuring & Formalization**

*   **Assigned to:** Structural Engineer Agents & Technical Writer Agents
*   **Task (Structural Engineers):**
    1.  Implement the **Target Documentation Architecture**.
    2.  Rename files to be descriptive and consistent (e.g., `hybrid_docker_mcp_architecture.md` -> `hybrid-docker-mcp-architecture.md`).
    3.  Move the newly consolidated files from Phase 2 into their correct locations.
    4.  Archive obsolete content. Move the contents of `code-examples/archived-2025-10/` into `_archive/code-examples-2025-10/`. Move reports from `reports/` into `_archive/reports/`.
*   **Task (Technical Writers):**
    1.  Rewrite all consolidated content into a formal, consistent tone.
    2.  Convert bullet points and informal notes into structured prose with clear headings.
    3.  Ensure all documents have a consistent header, introduction, and conclusion.
    4.  Standardize terminology based on `reference/glossary.md`.
    5.  Apply the project's style guide (e.g., **UK English spelling**: "colour", "optimisation").
    6.  Ensure code blocks are correctly formatted and diagrams are legible and titled.

#### **Phase 4: Integration & Verification**

*   **Assigned to:** Link & Graph Integrity Agents
*   **Task:**
    1.  **Update All Internal Links:** Traverse every markdown file and update all relative links (`[link](./path)`) to point to the new file locations.
    2.  **Regenerate Master Index:** Rebuild `00-INDEX.md` to accurately reflect the new, clean structure.
    3.  **Update Navigation Hub:** Update `README.md` with correct links to the main sections.
    4.  **Verify Diagrams:** Review all Mermaid diagrams (especially in `00-INDEX.md` and architecture docs) and update them to reflect the new document relationships and structure.
    5.  **Final Pass:** Perform a full-corpus scan to find and fix any dead links or incorrect cross-references.

---

### **General Principles for All Agents**

*   **Preserve Detail:** The goal is to formalize, not summarize. Do not discard technical specifications, metrics, or code examples.
*   **Prioritize Authoritative Sources:** Documents based on "direct code examination" (like `client.md`, `server.md`) are more authoritative than older, high-level design documents. Use them to resolve conflicts.
*   **Formalize Metadata:** Convert informal tags like `⭐ NEW` into formal document status headers (e.g., `Status: New (October 2025)`).
*   **Archive, Don't Delete (Initially):** Move any file you are unsure about into the `_archive/` directory with a descriptive path. Deletion can be a final, manual step.
*   **UK English spelling:** Apply UK English spelling conventions throughout (e.g., "colour", "optimisation").
*   **Mermaid Diagrams** Make extensive use of Mermaid diagrams to visualize complex relationships and workflows. Ensure all diagrams are updated to reflect the new structure and are included in relevant documents. Brackets should be avoided for github compatibility.
*   **Iterative Refinement:** After each phase, conduct a review to ensure quality and consistency before proceeding to the next phase.
