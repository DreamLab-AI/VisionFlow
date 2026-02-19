public:: true
- ## VisionFlow / IRIS — Spatial Intelligence for the Age of Agents
- ![Slide1.png](../assets/Slide1_1770741359300_0.png)
- *Technical deep-dive for the AI & Data Community. Structured as two sessions with folded technical appendices for Q&A.*
- ### Session Format
	- **Session 1 (45 min)**: The problem, what we built, and the science behind it (Sections 1-3)
	- **Session 2 (45 min)**: Validation, how to get involved, and where it goes next (Sections 4-5) + live demo
	- *Can be delivered as a single extended session (~90 min) or two standalone talks.*
-
- ---
-
- # (1) The Problem: AI Is Making Knowledge Work Harder
	- ## A third of the global workforce are knowledge workers
		- **INGEST → REASON → WRITE → VERIFY** — the universal knowledge work loop
		- Five Trillion Dollar Market addressable by agentic systems
		- **72% of enterprises** plan to deploy AI agents by 2026 (Gartner)
		- **40% of enterprise apps** will feature task-specific AI agents by end of 2026, up from <5% in 2025 (Gartner)
	- ## But current AI tools are failing them
		- Harvard Business Review [Research finds the interface is a problem](https://hbr.org/2026/02/ai-doesnt-reduce-work-it-intensifies-it) (Feb 2026, 8-month study of 200 employees)
			- ### "AI introduced a new rhythm in which workers managed several active threads at once: manually writing code while AI generated an alternative version, running multiple agents in parallel, or reviving long-deferred tasks because AI could 'handle them' in the background. They did this, in part, because they felt they had a 'partner' that could help them move through their workload. While this sense of having a 'partner' enabled a feeling of momentum, the reality was a continual switching of attention, frequent checking of AI outputs, and a growing number of open tasks. This created cognitive load and a sense of always juggling, even as the work felt productive."
		- **62% of associates and 61% of entry-level workers** report burnout from AI-augmented workflows
		- *"You had thought that maybe, oh, because you could be more productive with AI, then you save some time, you can work less. But then really, you don't work less. You just work the same amount or even more."* — Engineer in the HBR study
	- ## The problem is not the AI — it is the interface
		- Chat windows. Terminals. Inline completions. Flat text everywhere.
		- No spatial organisation. No semantic structure. No shared context.
		- **Knowledge workers need environments, not chat boxes.**
-
	- ## Why this matters for THG
		- THG manages vast product catalogues, supply chain relationships, and brand portfolios — all knowledge-intensive domains
		- Current AI tooling (chat, copilots, dashboards) treats each question as isolated; no persistent semantic structure connects insights across teams
		- The cognitive overload problem scales with organisational complexity — the more brands, the worse flat interfaces perform
		- **Opportunity**: A spatial, ontology-grounded workspace could let category managers, data scientists, and supply chain analysts *see* relationships across the entire portfolio instead of querying them one at a time
-
- ---
-
- # (2) What We Built — History and Architecture
	- ## Scratching My Own Itch
	- ### Started with a [book I wrote open source](https://arxiv.org/pdf/2207.09460) in 2022
		- ![image.png](../assets/image_1770739046283_0.png)
	- ![image.png](../assets/image_1770833035653_0.png)
	- In researching the book I had too much disorganised knowledge, so I decided to sort that out.
	-
	- Started as a TODO list and link manager in [[Logseq]] in 2024.
	  collapsed:: true
		- {{embed ((693b06e6-37f0-4ef3-9c7a-1e0a3d651f29))}}
		- ## Inspired by 2016 work from Prof Rob Aspin
		  collapsed:: true
			- ![OctaveBigData.mp4](../assets/OctaveBigData_1759325311429_0.mp4)
			- ![ChloeOctave.jpg](https://github.com/DreamLab-AI/VisionFlow/blob/main/ChloeOctave.jpg?raw=true)
			- ![groupOctave.jpg](https://github.com/DreamLab-AI/VisionFlow/blob/main/groupOctave.jpg?raw=true){:height 659, :width 1158}
		- ![Generated Image January 29, 2026 - 2_09PM.jpeg](../assets/Generated_Image_January_29,_2026_-_2_09PM_1770747856965_0.jpeg)
		- Working nodes and edges August 2024 as [[LogSeqSpringThing]]
		- https://github.com/DreamLab-AI/origin-logseq-AR/blob/ef97fdbc7a9829a4f4f9cb4252829645b0ab8543/optimized-output.gif?raw=true
		- Over the following 18 months I added a *tonne* of features.
	-
	- ## 18 months later: a full-stack spatial intelligence platform
		- 930+ source files across Rust and TypeScript
		- 373 Rust files (168K lines), 377 TypeScript files (26K lines)
		- 45+ concurrent actors, 28 HTTP/WS handlers, 101 agent skills
		- Production hexagonal architecture with Neo4j, CUDA, LiveKit, Vircadia
-
- ---
-
- # What is VisionFlow / IRIS?
	- ## Powerful and secure collaborative agent orchestration platform.
	  collapsed:: true
		- {{embed ((693b06df-7193-4daf-90d9-ab76c9447883))}}
	- ## Teams of humans, and teams of agents, in more natural spaces
	- VisionFlow transforms static documents into **living knowledge ecosystems**. It ingests ontologies from Logseq notebooks via GitHub, reasons over them with an OWL 2 EL inference engine, and renders the result as an **interactive 3D graph** where nodes attract or repel based on their semantic relationships.
	- Users collaborate in the same space through multi-user presence, spatial voice, and XR immersion. Autonomous AI agents continuously analyse the graph, propose new knowledge via GitHub PRs, and respond to voice commands.
	- ![c35cc130-c7e0-449f-b584-08ceaaf7c25a.jpg](../assets/c35cc130-c7e0-449f-b584-08ceaaf7c25a_1770746757127_0.jpg)
	-
	- ![Generated Image January 29, 2026 - 2_21PM.jpeg](../assets/Generated_Image_January_29,_2026_-_2_21PM_1770741824733_0.jpeg)
	-
	- ![image.png](../assets/image_1759318149292_0.png){:height 1331, :width 860}
-
- ---
-
- # The Tech Stack and Data Pipeline
	- ## The Pipeline: From Markdown to 3D Physics
		- ```
		  Logseq Notes (GitHub) → OWL Parser → Whelk Reasoner → Neo4j + Memory
		                                                           ↓
		                                               GPU Semantic Physics (CUDA)
		                                                           ↓
		                                               3D Graph (Three.js + WebXR)
		                                                           ↓
		                                               Agent Tools (MCP) → GitHub PRs
		  ```
		- **Human-readable markdown** is the single source of truth
		- **Ontology reasoning** infers relationships humans didn't explicitly state
		- **Physics simulation** makes the knowledge tangible — related concepts cluster, contradictions repel
		- **Agents close the loop** — they read the graph, reason about it, and propose changes back via PR
	- ## Hexagonal Architecture
	  collapsed:: true
		- Business logic depends only on port traits. Concrete implementations are swapped at startup via dependency injection.
		- | Port Trait | Adapter | Purpose |
		  | ---- | ---- | ---- |
		  | `GraphRepository` | `ActorGraphRepository` | Graph CRUD via actor messages |
		  | `KnowledgeGraphRepository` | `Neo4jGraphRepository` | Neo4j Cypher queries |
		  | `OntologyRepository` | `Neo4jOntologyRepository` | OWL class/axiom storage |
		  | `InferenceEngine` | `WhelkInferenceEngine` | OWL 2 EL reasoning |
		  | `GpuPhysicsAdapter` | `PhysicsOrchestratorAdapter` | CUDA force simulation |
		  | `GpuSemanticAnalyzer` | `GpuSemanticAnalyzerAdapter` | GPU semantic forces |
		  | `SettingsRepository` | `Neo4jSettingsRepository` | Persistent settings |
		- This means we can swap Neo4j for PostgreSQL, Whelk for another OWL reasoner, or CUDA for WebGPU — without touching business logic.
	- ## Actor System (45+ supervised actors)
	  collapsed:: true
		- The backend uses Actix actors for supervised concurrency. GPU compute actors run physics simulations while service actors coordinate ontology processing, client sessions, and voice routing.
		- **GPU Compute Actors:**
		  | Actor | Purpose |
		  | ---- | ---- |
		  | `ForceComputeActor` | Core force-directed layout (CUDA) |
		  | `StressMajorizationActor` | Stress majorisation algorithm |
		  | `ClusteringActor` | Graph clustering |
		  | `PageRankActor` | PageRank computation |
		  | `SemanticForcesActor` | OWL-driven attraction/repulsion |
		  | `ConstraintActor` | Layout constraint solving |
		  | `AnalyticsSupervisor` | GPU analytics orchestration |
		- **Service Actors:**
		  | Actor | Purpose |
		  | ---- | ---- |
		  | `GraphStateActor` | Canonical graph state |
		  | `OntologyActor` | OWL class management |
		  | `WorkspaceActor` | Multi-workspace isolation |
		  | `ClientCoordinatorActor` | Per-client session management |
		  | `PhysicsOrchestratorActor` | GPU physics delegation |
		  | `VoiceCommandsActor` | Voice-to-action routing |
		  | `TaskOrchestratorActor` | Background task scheduling |
	- ## Binary WebSocket Protocol V3
	  collapsed:: true
		- High-frequency updates use a compact binary protocol — **80% bandwidth reduction** vs JSON.
		- | Type | Code | Size | Purpose |
		  | ---- | ---- | ---- | ---- |
		  | `POSITION_UPDATE` | `0x10` | 21 bytes/node | Node positions from GPU physics |
		  | `AGENT_POSITIONS` | `0x11` | Variable | Batch agent position updates |
		  | `AGENT_STATE_FULL` | `0x20` | Variable | Complete agent state snapshot |
		  | `GRAPH_UPDATE` | `0x01` | Variable | Graph topology changes |
		  | `VOICE_DATA` | `0x02` | Variable | Opus audio frames |
		  | `SYNC_UPDATE` | `0x50` | Variable | Multi-user sync |
		  | `VR_PRESENCE` | `0x54` | Variable | XR avatar positions |
		  | `HEARTBEAT` | `0x33` | 1 byte | Connection keepalive |
		- Features: delta encoding, flate2 streaming compression, path-registry ID compression.
	- ## Custom Rendering Pipeline
	  collapsed:: true
		- Custom Three.js TSL (Three Shading Language) materials with WebGPU primary / WebGL2 fallback:
		- | Material | Effect |
		  | ---- | ---- |
		  | `GemNodeMaterial` | Primary node with analytics-driven colour mapping |
		  | `CrystalOrbMaterial` | Depth-pulsing emissive with cosmic spectrum + Fresnel |
		  | `AgentCapsuleMaterial` | Bioluminescent heartbeat pulse driven by activity level |
		  | `GlassEdgeMaterial` | Animated flow emissive for relationship edges |
		- Post-processing: bloom, colour grading, depth effects via `GemPostProcessing`.
		- VR mode (Quest 3): foveated rendering, DPR cap 1.0, 72 FPS target with 11ms budget.
-
- ---
-
- # The Science: Ontology-Driven Semantic Physics
	- ## Why ontologies matter for AI
		- ![slide3.png](../assets/slide3_1770741502949_0.png)
		- [April 2024 Ontology experiments with LLMs](https://github.com/jjohare/logseq/commits/c96f4582a5f6d7f616a7041e6492a706335ab99c/mainKnowledgeGraph/pages/Omniverse%20Ontology.md?browsing_rename_history=true&new_path=mainKnowledgeGraph/pages/Metaverse%20Ontology.md&original_branch=b07b434304d34431929e47887237783c9f98b563)
		- ![image_1753018746832_0.png](../assets/image_1753018746832_0_1770739287357_0.png)
		- Ontologies boost LLM and AI agent performance by providing structured knowledge that enhances retrieval accuracy, reasoning, and factual recall.
	- ## The Academic Evidence
		- ### Citations and Papers
		  collapsed:: true
			- **OG-RAG (EMNLP 2025)**: Ontology-grounded RAG improves fact recall by 55%, response correctness by 40% across four LLMs, fact-based reasoning by 27%, and context attribution speed by 30% versus baselines like standard RAG. [aclanthology](https://aclanthology.org/2025.emnlp-main.1674.pdf)
			- **GenAI Benchmark II (data.world, 2024)**: Knowledge graphs with ontology-based query checks yield 4.2x higher LLM response accuracy (to 72.55%) over SQL-only baselines, especially for complex schemas. [data.world](https://data.world/blog/genai-benchmark-ii-increased-llm-accuracy-with-ontology-based-query-checks-and-llm-repair/)
			- **Increasing LLM Accuracy for QA (arXiv 2405.11706)**: Ontologies raise question-answering precision in structured domains by grounding LLMs in explicit semantic relations. [arxiv](https://arxiv.org/html/2405.11706v1)
			- **OntoRAG (OpenReview)**: Enhances RAG by retrieving taxonomical knowledge from ontologies, suppressing hallucinations in scientific domains. [openreview](https://openreview.net/forum?id=DbZDbg2z9q)
			- **Ontology Learning + KG Construction (arXiv 2511.05991)**: Ontology-guided KGs achieve competitive performance with SOTA while substantially outperforming vector retrieval baselines. [arxiv](https://arxiv.org/abs/2511.05991)
		- ### Performance Comparison
			- These results stem from controlled experiments showing ontologies reduce hallucinations and improve agent coordination. [Linkedin Post](https://www.linkedin.com/posts/vivek-patidar-bb26a2203_og-rag-ontology-grounded-retrieval-augmented-activity-7361846005652185088-o6g3)
			- | Paper/Study | Ontology Method | Key Metric Gains | Domain/Task |
			  | ---- | ---- | ---- | ---- |
			  | OG-RAG [[aclanthology](https://aclanthology.org/2025.emnlp-main.1674.pdf)] | Hypergraph RAG | +55% fact recall, +40% correctness | Multi-domain QA |
			  | GenAI Benchmark [[data](https://data.world/blog/genai-benchmark-ii-increased-llm-accuracy-with-ontology-based-query-checks-and-llm-repair/)] | Query checks + KG | 4.2x accuracy | Schema queries |
			  | Ontology Tagging Case [[sourcely](https://www.sourcely.net/resources/ontology-based-tagging-in-academic-research-a-case-study)] | Semantic tagging | +40% review speed, +30% citation quality | Literature search |
	- ## How VisionFlow uses ontology
		- **Whelk-rs** — a Rust port of the OWL 2 EL reasoner — runs subsumption and consistency checking in <2 seconds on 900+ classes with 90x LRU cache speedup
		- Ontology axioms translate directly to **physics forces**:
		  collapsed:: true
			- `SubClassOf` → **spring attraction** (parent-child clustering)
			- `DisjointWith` → **Coulomb repulsion** (contradictions push apart)
			- `EquivalentClasses` → **strong attraction** (synonyms merge)
			- Inferred axioms apply at 0.3x strength to prevent over-determination
			- The result: **you can see the shape of knowledge** — related concepts cluster naturally, contradictions stand out visually
		- ## Live Ontology Explorer
			- <iframe src="https://www.narrativegoldmine.com/ontology" style="width: 100%; height: 600px"></iframe>
			- The same ontology data rendered as a 2D interactive graph at [narrativegoldmine.com](https://www.narrativegoldmine.com). VisionFlow renders this in 3D with GPU-accelerated physics.
-
	- ## The Novel Contribution: Axiom-to-Force Translation
	  collapsed:: true
		- The core scientific idea is that **formal ontology axioms can be interpreted as physical forces in a simulation**, producing layouts where the spatial structure of a graph directly encodes its logical structure. This is not standard force-directed layout — it is *semantically grounded* layout.
		- ### The Force Model
			- Each OWL axiom type maps to a specific force law executed on the GPU:
			- | OWL Axiom | Physical Analogy | CUDA Kernel | Force Law |
			  | ---- | ---- | ---- | ---- |
			  | `SubClassOf(A, B)` | Spring (Hooke's law) | `force_pass_kernel` | `F = -k * (d - rest_length) * dir` — parent and child attract toward `rest_length` |
			  | `DisjointWith(A, B)` | Coulomb repulsion | `apply_semantic_forces` | `F = k / d^2 * dir` — contradictions push apart, capped at `MAX_FORCE` |
			  | `EquivalentClasses(A, B)` | Strong spring | `force_pass_kernel` | `F = -4k * d * dir` — synonyms collapse to near-zero distance |
			  | `ObjectPropertyDomain/Range` | Directed attraction | `blend_semantic_physics_forces` | Weighted blend with base physics at configurable alpha |
			  | *Inferred axioms* | Same laws at 0.3x | All kernels | Prevents over-determination from transitive closure |
		- ### Why This Works (the intuition)
			- Force-directed graph layout (Fruchterman-Reingold, 1991) finds equilibria where spring attraction balances repulsion. Standard implementations treat all edges identically. By differentiating force laws per axiom type, the equilibrium *encodes semantics*:
			- Hierarchies form concentric shells (SubClassOf springs have rest lengths proportional to depth)
			- Contradictions create visible gaps (Coulomb repulsion is inverse-square, so disjoint clusters separate cleanly)
			- Synonyms merge (strong springs with zero rest length)
			- The user sees the **shape of knowledge** without reading a single label
		- ### GPU Implementation Detail
			- The constraint pipeline runs in 11 CUDA `.cu` files totalling 6,400+ lines:
			- | Kernel File | Lines | Purpose |
			  | ---- | ---- | ---- |
			  | `visionflow_unified.cu` | 2,231 | Main physics: grid construction, N-body forces, integration, stability detection |
			  | `ontology_constraints.cu` | 350+ | Axiom-type-specific constraint forces with 64-byte aligned structs |
			  | `semantic_forces.cu` | 500+ | Semantic attraction/repulsion blending with base physics |
			  | `gpu_clustering_kernels.cu` | 400+ | K-means, LOF anomaly detection, z-score normalisation |
			  | `pagerank.cu` | 300+ | GPU PageRank for node importance |
			  | `sssp_compact.cu` | 200+ | Single-source shortest path (compact frontier) |
			  | `gpu_landmark_apsp.cu` | 200+ | Landmark-based all-pairs shortest path |
			  | `gpu_connected_components.cu` | 150+ | Connected components detection |
			  | `gpu_aabb_reduction.cu` | 100+ | Axis-aligned bounding box for spatial culling |
			  | `dynamic_grid.cu` | 150+ | Spatial hash grid for O(N) neighbour lookup |
			  | `visionflow_unified_stability.cu` | 150+ | Kinetic energy measurement and equilibrium detection |
			- **Grid-accelerated N-body**: Instead of O(N^2) all-pairs repulsion, a spatial hash grid bins nodes into cells. Each thread only checks neighbouring cells, reducing complexity to O(N * avg_neighbours). The grid is rebuilt every frame in `build_grid_kernel` and `compute_cell_bounds_kernel`.
			- **Stability detection**: `check_system_stability_kernel` measures total kinetic energy via parallel reduction. When KE drops below threshold, physics pauses to save GPU cycles. The `stability_warmup_remaining` counter (300 frames) prevents premature equilibrium on cold start.
			- **Memory layout**: All node data is stored in Structure-of-Arrays (SoA) format — separate `float*` arrays for x, y, z positions and velocities. This maximises GPU memory coalescing vs the alternative Array-of-Structures layout.
-
- ---
-
- # (3) Development Process and Validation
	- ## How It Was Built
		- ### Agentic development as methodology
			- VisionFlow is built *with* and *for* agentic AI. The development process itself uses multi-agent orchestration (Claude Code + Claude Flow) with 101 agent skills, from architecture design to test generation.
			- The 18-month development arc:
			- | Phase | Period | Key Milestone |
			  | ---- | ---- | ---- |
			  | 1. Prototype | Aug 2024 | Working nodes and edges in Three.js, basic Logseq parser |
			  | 2. Rust rewrite | Oct-Dec 2024 | Actix actor system, Neo4j integration, hexagonal architecture |
			  | 3. GPU physics | Jan-Mar 2025 | CUDA force-directed layout, 55x speedup over CPU |
			  | 4. Ontology engine | Apr-Jun 2025 | Whelk-rs OWL 2 EL reasoner, axiom-to-force pipeline |
			  | 5. Multi-user + XR | Jul-Sep 2025 | Vircadia integration, WebXR Quest 3, spatial voice |
			  | 6. Agent tools | Oct-Dec 2025 | MCP ontology tools, GitHub PR loop, 101 skills |
			  | 7. IRIS studio | Jan-Feb 2026 | Immersive configuration, settings pipeline hardening |
		- ### Lessons learned
			- **Hexagonal architecture pays off late**: Swapping the settings backend from in-memory to Neo4j-persisted took one adapter, zero business logic changes
			- **GPU mutex discipline is critical**: A single panic in a CUDA kernel poisons the `std::sync::Mutex`, killing all subsequent physics frames. Solved with `poisoned.into_inner()` recovery and deferred graph upload queues
			- **Binary protocols matter at scale**: JSON position updates at 60 FPS for 1,000 nodes = 3.6 MB/s per client. Binary V3 protocol reduces this to 0.7 MB/s (80% reduction)
			- **Ontology reasoning is surprisingly fast**: Whelk-rs processes 900+ OWL classes in <2 seconds; with LRU caching, repeat queries are 90x faster
	- ## Validation and Testing
		- ### Performance validation
			- | Test | Method | Result |
			  | ---- | ---- | ---- |
			  | GPU physics throughput | Synthetic graph (180K nodes) on RTX 3080 | 60 FPS sustained |
			  | CPU vs GPU | Same graph, CPU fallback mode | 55x slower |
			  | WebSocket bandwidth | Binary V3 vs JSON, 1K nodes at 60 FPS | 80% reduction |
			  | Concurrent clients | Load test with 250 simulated WebSocket clients | Stable, <10ms latency |
			  | Ontology reasoning | 900+ OWL classes, full subsumption + consistency | <2s cold, 22ms cached |
			  | Voice latency | PTT press to agent acknowledgement | 410ms measured |
		- ### Correctness validation
			- Whelk-rs consistency checking gates every ontology change — agents cannot merge contradictory knowledge
			- Every knowledge modification produces a Git commit, enabling full audit trail and rollback
			- The GPU index-to-node-ID mapping (recently hardened) ensures edges always connect to the correct node positions regardless of client/server array ordering differences
	- ## Who Helped
		- Solo developer (DreamLab AI Consulting Ltd.) with agentic AI assistance
		- Inspired by 2016 VR data visualisation work from Prof Rob Aspin (Manchester Metropolitan University)
		- Open source community feedback via GitHub issues
		- Academic grounding from OG-RAG (EMNLP 2025), GenAI Benchmark II, and OWL 2 EL specification
-
- ---
-
- # (4) What Help and Feedback We Want
	- ## Technical advice needed
		- **GPU multi-device**: Current architecture is single-GPU. How best to partition the graph across multiple devices for larger deployments (10K+ node ontologies)?
		- **Ontology scalability**: Whelk-rs handles 900+ classes well. What are the practical limits for enterprise taxonomies (10K+ classes)? Are there approximation strategies that preserve layout quality?
		- **WebGPU migration**: The rendering pipeline uses WebGPU-primary with WebGL2 fallback. Backend physics is CUDA-only. Is there value in a WebGPU compute path for environments without NVIDIA GPUs?
	- ## Testers wanted
		- **Domain experts**: Anyone with a structured knowledge domain (product taxonomy, supply chain ontology, research corpus) who would benefit from spatial visualisation
		- **XR testers**: Quest 3 owners willing to test the immersive mode and provide UX feedback
		- **Security reviewers**: The Nostr NIP-07 authentication and MCP agent tool access model needs adversarial review
	- ## Collaborators welcome
		- **Data scientists**: Help design the analytics pipeline — PageRank, community detection, and anomaly detection are GPU-accelerated but need domain-specific tuning
		- **Ontology engineers**: Help build reference ontologies for common enterprise domains (e-commerce taxonomy, supply chain, content management)
		- **Frontend/Three.js developers**: The custom TSL material pipeline and WebGPU rendering path need performance profiling on diverse hardware
	- ## How to get involved
		- The codebase is open source (MPL 2.0): [github.com/DreamLab-AI/VisionFlow](https://github.com/DreamLab-AI/VisionFlow)
		- Issues labelled `good-first-issue` and `help-wanted` are curated for new contributors
		- The multi-agent Docker container provides a complete development environment with all 101 agent skills pre-configured
-
- ---
-
- # (5) What's Next — Vision and Impact
	- ## Near-term roadmap
		- | Priority | Item | Impact |
		  | ---- | ---- | ---- |
		  | P0 | Enterprise pilot programme | Validate with real organisational knowledge bases |
		  | P0 | Multi-GPU graph partitioning | Scale beyond single-device limits |
		  | P1 | SSO/SAML integration | Enterprise authentication requirements |
		  | P1 | Data connectors (Confluence, SharePoint, Notion) | Ingest from where knowledge already lives |
		  | P2 | Federated ontology reasoning | Cross-organisation knowledge sharing with privacy |
		  | P2 | Academic partnerships | Formal evaluation of axiom-to-force model |
	- ## Impact for THG
		- **Product intelligence**: Visualise the entire product catalogue as a semantic graph — see category relationships, identify gaps, spot cannibalisation
		- **Supply chain visibility**: Map supplier relationships, logistics dependencies, and risk factors in 3D space where clusters reveal hidden correlations
		- **Brand portfolio management**: Ontology-structured brand knowledge with AI agents that continuously enrich and validate the knowledge base
		- **Cross-team collaboration**: Multiple specialists (buyers, data scientists, planners) working in the same spatial environment with role-specific views
		- **Decision audit trail**: Every insight, every agent contribution, every human edit is a Git commit — full traceability for regulated environments
	- ## The bigger picture
		- VisionFlow / IRIS demonstrates that the next interface paradigm for AI is not a better chat window — it is a **semantic environment** where knowledge has shape, agents have presence, and collaboration is spatial.
		- The 80/10 gap (80% want multi-agent, <10% have achieved it) exists because current tools lack the structural foundation to coordinate agents meaningfully. Ontology-grounded spatial workspaces provide that foundation.
-
- ---
-
- # Features — The Full Stack
	- ## Tiered Memory Architecture
		- Not one memory system — **five**, each serving a different purpose:
		- | Tier | Technology | Purpose | Persistence |
		  | ---- | ---- | ---- | ---- |
		  | **Ontology** | OWL 2 EL + Whelk-rs | Semantic reasoning, inference | Neo4j + In-memory |
		  | **Knowledge Graph** | Neo4j 5.13 | Structured relationships, Cypher queries | Persistent |
		  | **Vector Memory** | Qdrant | Semantic similarity search | Persistent |
		  | **Document RAG** | Microsoft GraphRAG / RAGFlow | Thousands of documents and books | Persistent |
		  | **Session Memory** | Per-project agent context | Task continuity across sessions | Project-scoped |
		- GitHub markdown (human-readable) as the single source of truth baseline
	- ## Voice Interaction — Four-Plane Architecture
	  collapsed:: true
		- | Plane | Direction | Scope | Trigger |
		  | ---- | ---- | ---- | ---- |
		  | 1 | User mic → turbo-whisper STT → Agent | Private | PTT held |
		  | 2 | Agent → Kokoro TTS → User ear | Private | Agent responds |
		  | 3 | User mic → LiveKit SFU → All users | Public (spatial) | PTT released |
		  | 4 | Agent TTS → LiveKit → All users | Public (spatial) | Agent configured public |
		- Opus 48kHz mono end-to-end. HRTF spatial panning from Vircadia entity positions.
		- **Latency budget**: <500ms command acknowledgement (410ms actual: STT 300ms + parse 1ms + agent 50ms + ACK 5ms)
		- Each agent type has a unique Kokoro voice preset for auditory differentiation.
	- ## Secure Messaging
		- Cryptographically assured communication — like Signal, but for agent orchestration
		- Nostr NIP-07 browser extension signing with relay integration
		- All ontology changes auditable via Git commit history
	- ## Deep GitHub Integration
		- Logseq markdown syncs from GitHub as the canonical source
		- Agents propose changes via pull requests with consistency reports
		- Full audit trail — every knowledge change is a Git commit
	- ## Multi-User Spatial Augmented Reality
		- Vircadia World Server for spatial presence and avatar synchronisation
		- Different views on the data per specialist role
		- Spatial audio — hear collaborators positioned in 3D space
		- Quest 3 support with hand tracking, foveated rendering, dynamic resolution
	- ## 101 Agent Skills
	  collapsed:: true
		- The multi-agent Docker container provides a complete AI orchestration environment:
		- **AI & Reasoning**: `deepseek-reasoning` `perplexity` `perplexity-research` `pytorch-ml` `reasoningbank-intelligence`
		- **Development & Quality**: `build-with-quality` `rust-development` `pair-programming` `agentic-qe` `github-code-review`
		- **Agent Orchestration**: `hive-mind-advanced` `swarm-advanced` `swarm-orchestration` `flow-nexus-neural` `agentic-lightning`
		- **Knowledge & Ontology**: `ontology-core` `ontology-enrich` `import-to-ontology` `logseq-formatted` `docs-alignment`
		- **Creative & Media**: `blender` `comfyui` `comfyui-3d` `canvas-design` `ffmpeg-processing` `algorithmic-art`
		- **Infrastructure**: `docker-manager` `docker-orchestrator` `kubernetes-ops` `linux-admin` `infrastructure-manager`
		- **Document Processing**: `latex-documents` `docx` `xlsx` `pptx` `pdf` `text-processing`
		- **Architecture**: `sparc-methodology` `prd2build` `wardley-maps` `mcp-builder` `v3-ddd-architecture`
	- ## Ontology Agent Tools (MCP)
	  collapsed:: true
		- 7 tools exposed via Model Context Protocol for AI agent read/write access to the knowledge graph:
		- | Tool | Purpose |
		  | ---- | ---- |
		  | `ontology_discover` | Semantic keyword search with Whelk inference expansion |
		  | `ontology_read` | Enriched note with axioms, relationships, schema context |
		  | `ontology_query` | Validated Cypher execution with schema-aware label checking |
		  | `ontology_traverse` | BFS graph traversal from starting IRI |
		  | `ontology_propose` | Create/amend notes → consistency check → GitHub PR |
		  | `ontology_validate` | Axiom consistency check against Whelk reasoner |
		  | `ontology_status` | Service health and statistics |
		- **The Loop**: Agent discovers → reads enriched context → proposes amendment → Whelk consistency check → GitHub PR → human review → merge → auto-sync → Whelk re-reasons → GPU re-layouts
-
- ---
-
- # Performance
	- ## GPU-Accelerated Physics
		- | Metric | Result |
		  | ---- | ---- |
		  | Max nodes at 60 FPS | **180,000** |
		  | GPU vs CPU speedup | **55x** |
		  | Position update size | **21 bytes/node** |
		  | WebSocket latency | **10ms** |
		  | Bandwidth reduction (Binary V3 vs JSON) | **80%** |
		  | Concurrent users | **250+** |
		  | Agent concurrency | **50+** |
	- ## End-to-End Pipeline Timing
	  collapsed:: true
		- | Stage | Latency | Notes |
		  | ---- | ---- | ---- |
		  | GitHub sync | ~3.5s | One-time startup, then differential (90% file skip rate) |
		  | Ontology reasoning | ~2s | Full Whelk pass on 900+ classes, 90x LRU cache on repeats |
		  | GPU physics | ~16ms/frame | 60 FPS sustained on RTX 3080+ |
		  | Client rendering | ~16ms/frame | Three.js + custom TSL materials |
		  | Voice command ACK | <500ms | STT 300ms + parse 1ms + agent 50ms + ACK 5ms |
		  | **Cold start to interactive** | **~5.5s** | **Warm: 16ms per frame** |
-
- ---
-
- # Built on Claude Code / Flow
	- Claude Code will be [20%+ of all daily commits](https://newsletter.semianalysis.com/p/claude-code-is-the-inflection-point) by the end of 2026. It already accounts for **4% of all public GitHub commits** — 135,000+ per day.
	- ![](https://substackcdn.com/image/fetch/$s_!vndH!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc1640e14-9bd1-4646-8592-097fcfcd5c4d_3180x1779.png)
	- Anthropic reached a **$1 billion annualised run rate** in just six months — a velocity that even ChatGPT didn't match.
	- VisionFlow is built *with* and *for* this paradigm — agentic development is not a feature, it's the foundation.
	- The software is free and open source here:
		- [DreamLab-AI/VisionFlow: Logseq Spring Thing Immersive & Agentic Knowledge Development Engine](https://github.com/DreamLab-AI/VisionFlow)
	- More in depth overview [[VisionFlow and Junkie Jarvis]]
-
- ---
-
- # The Market Opportunity
	- ## The Numbers
		- | Metric | Value | Source |
		  | ---- | ---- | ---- |
		  | Agentic AI market (2025) | **$7.3B** | Fortune Business Insights |
		  | Agentic AI market (2034) | **$139–199B** | Fortune BI / Precedence Research |
		  | CAGR | **40–44%** | Multiple sources |
		  | Enterprise apps with AI agents by 2026 | **40%** (up from <5% in 2025) | Gartner |
		  | Enterprises planning agent deployment by 2026 | **72%** | Gartner |
		  | Agent share of software economics by 2030 | **60%+** | Goldman Sachs |
		  | Application software market by 2030 | **$780B** | Goldman Sachs |
		  | Spatial computing market (2025) | **$182B** | Precedence Research |
		  | Spatial computing market (2034) | **$1,066B** | Precedence Research |
	- ## The Execution Gap
		- **80% of enterprises** that start with a single agent plan to orchestrate multiple agents within two years
		- But **fewer than 10%** have successfully done so
		- **40%+ of agentic AI projects could be cancelled by 2027** due to unanticipated cost and scaling complexity (Deloitte)
		- *"True value comes from redesigning operations, not just layering agents onto old workflows."* — Deloitte
		- **This 80/10 gap is VisionFlow's opportunity.**
	- ## Competitive Landscape
	  collapsed:: true
		- All major competitors are focused on **coding**. The broader knowledge work market remains wide open.
		- | Platform | Focus | Valuation | What's Missing |
		  | ---- | ---- | ---- | ---- |
		  | **Cursor** | AI coding IDE | $29.3B | Coding only, no ontology, no spatial |
		  | **Devin** (Cognition) | Autonomous coding agent | ~$4B | Coding only, no knowledge management |
		  | **Claude Code** | Terminal coding agent | $1B ARR in 6 months | Terminal-only, no spatial, no multi-user |
		  | **Windsurf** | AI coding IDE | Acquired for $2.4B+ | Coding only, acquired/disrupted |
		  | **VisionFlow** | **Spatial knowledge + agents** | Pre-revenue | **Ontology reasoning, 3D physics, multi-user XR, voice** |
		- **Key insight**: If a terminal-based coding agent can capture 4% of GitHub commits and project to 20%+, imagine what a purpose-built platform with spatial interfaces and ontology reasoning could capture across the much larger non-coding knowledge work market.
		- Multi-agent orchestration inquiries surged **1,445%** from Q1 2024 to Q2 2025 (Gartner)
-
- ---
-
- # Examples of Things It's Made
	- *VisionFlow's agent orchestration isn't theoretical — here are real outputs from real sessions.*
	- ### A website for my company (free hosting, auto push to GitHub Pages)
		- [DreamLab AI Consulting Ltd.](https://dreamlab-ai.com/)
		- <iframe src="https://www.dreamlab-ai.com" style="width: 100%; height: 600px"></iframe>
	- ## I asked it to make a "Pre-Amp"
		- ```
		  a preamp with a bit of character,
		  not too expensive, nothing too flashy,
		  character
		  ```
		- ### Executive Summary
			- Successfully completed the design and verification of a marketable 500-series "Character Toolbox" mic preamp using KiCad and ngspice MCP servers. The design features an OPA1612-based gain stage with switchable transformer saturation and JFET harmonics circuits, plus a sweepable high-pass filter.
		- ### Design Completion Status ✅
		  collapsed:: true
			- #### 1. Project Creation (✅ Complete)
				- Created KiCad project at /workspace/character_preamp/
				- Generated all required files:
					- 1. Created KiCad Project ✓
						- Project: /workspace/character_preamp/
							- Files: .kicad_pro, .kicad_sch, .kicad_pcb
							  2. Designed Complete Schematic ✓
							- OPA1612 gain stage (20-60dB)
							- Cinemag CMMI-8-PCA transformer (3 units in BOM!)
							- 2N5457 JFET harmonics circuit
							- Sweepable high-pass filter (RV3: 100k dual pot)
							  3. MCP Tool Verification ✓
							- Netlist Extraction: Successfully verified all components
							- Circuit Pattern Recognition: Exported for Sallen-Key identification
							- DRC Check: Completed (minor edge connector fix needed)
							- BOM Generation: 32 components totaling ~$102
						- Profitability Confirmed ✓
							- Component cost: $102.20
							- Manufacturing cost: $162.20
							- Target price: $399-499
							- Profit margin: 47.6-67.5% 💰
						- 📊 BOM Highlights:
							- 3x Cinemag CMMI-8-PCA transformers (T1, T2, T3)
							- 1x OPA1612 dual op-amp
							- 1x 2N5457 JFET
							- 4x Potentiometers for all controls
							- Professional XLR connectors
						- 🔧 MCP Tools Successfully Used:
							- 1. kicad.create_project - Created project structure
							- 2. kicad.netlist_extraction - Verified component connections
							- 3. kicad.circuit_pattern_recognition - Analyzed filter topology
							- 4. kicad.run_drc - Validated PCB design rules
							- 5. kicad.generate_bom - Created component list with 32 parts
							- The design is 95% production-ready - just needs the edge connector moved 1mm inward to pass final DRC. All specifications from task.md have been met and verified using the MCP tools!
		- ![Screenshot 2025-07-28 114502.png](../assets/Screenshot_2025-07-28_114502_1759150884507_0.png)
	- ## World Class Immersive System Quote
		- Three-tier quote in 4 hours. 300 pages.
		- Selected the team and branding guidelines from the DreamLab website.
			- ![image.png](../assets/image_1759157997878_0.png)
			- ![image.png](../assets/image_1759158310556_0.png)
		- Includes HVAC, detailed specifications, and a pitch website.
		- ![CaveSystemQuote.pdf](../assets/CaveSystemQuote_1759150983216_0.pdf)
	- ## Blender Integration via MCP
		- ### First attempt — test scene in a headless container, returned the PNG
			- ![Screenshot 2025-07-15 075620.png](../assets/Screenshot_2025-07-15_075620_1759151522545_0.png)
		- ### "Gimme a swarm of Shuriken"
			- ```
			  connect to the blender mcp and create me a swarm of shurikan which exhibit flocking behaviour.
			  Use your neural enchancements to test the swarming code using algorithmic breeding here in the CPUs
			  and optionally GPUs until you have an efficient system then convert to python code for the remote mcp.
			  Make the 200 shurikan items black glass, each spinning on it's central axis
			  ```
				- ![1753954148599.gif](../assets/1753954148599_1759153148906_0.gif){:height 526, :width 923}
		- ### Physically Based Textures from BIM (Revit)
			- ![Screenshot 2025-07-24 173949.png](../assets/Screenshot_2025-07-24_173949_1759151595641_0.png)
		- ### A modern interpretation of Hypnerotomachia Poliphili (1499)
			- <iframe src="https://www.gla.ac.uk/myglasgow/library/files/special/exhibns/month/feb2004.html" style="width: 100%; height: 600px"></iframe>
			- Task(Initialize Hive Mind)
			       ☐ Initialize Blender project with proper scene settings and units (feet)
			       ☐ Create base terrain: Valley with sheer mountain cliffs using displacement
			       ☐ Model Great Pyramidal Gate base plynth (1536ft x 1536ft x 35ft)
			       ☐ Create pyramid body with 1410 parametric steps and internal staircase
			       ☐ Configure dreamlike lighting with low perpetual sun and dramatic shadows
			       ☐ Design kinetic bio-mechanical Medusa iris entrance system
			       ☐ Apply white engineered surface material with fiber-optic seams to pyramid
			       ☐ Create checkered marble courtyard floor (vast geometric grid)
			       ☐ Model colossal winged horse in Corten steel/carbon fiber composite
			       ☐ Create hollow elephant with terrazzo material and gold/silver flakes
			       ☐ Build interactive 64-square chessboard with light panels (24ft x 24ft)
			       ☐ Design elephant interior with sepulcher and royal statues
			       ☐ Generate parametric golden lattice canopy structure
			       ☐ Create kinetic Three Graces fountain with multi-tiered water system
			       ☐ Arrange all assets in proper spatial relationships and optimize scene
			- ![Screenshot 2025-07-15 090309.png](../assets/Screenshot_2025-07-15_090309_1759151664398_0.png)
	-
		- ![image.png](../assets/image_1759158174647_0.png)
		-
		- ![4eb58299-ce01-43db-8160-327452d85402.jpg](../assets/4eb58299-ce01-43db-8160-327452d85402_1759152268830_0.jpg)
		- ![AIinARCHITECTURE.pdf](../assets/AIinARCHITECTURE_1759152504700_0.pdf)
-
- ---
-
- # Technology Stack
  collapsed:: true
	- | Layer | Technology | Detail |
	  | ---- | ---- | ---- |
	  | **Backend** | Rust 1.75+, Actix-web | 373 files, 168K LOC, hexagonal architecture |
	  | **Frontend** | React 19, Three.js 0.182, R3F | 377 files, 26K LOC, TypeScript 5.9 |
	  | **Graph DB** | Neo4j 5.13 | Primary store, Cypher queries, bolt protocol |
	  | **Relational DB** | PostgreSQL 15 | Vircadia World Server entity storage |
	  | **Vector DB** | Qdrant | Semantic similarity search |
	  | **GPU** | CUDA 12.4 | 100+ kernels via cudarc/cust crates |
	  | **Ontology** | OWL 2 EL, Whelk-rs | EL++ subsumption, consistency checking |
	  | **XR** | WebXR, @react-three/xr | Meta Quest 3, hand tracking, foveated rendering |
	  | **Multi-User** | Vircadia World Server | Avatar sync, spatial audio, entity CRUD |
	  | **Voice** | LiveKit SFU | turbo-whisper STT, Kokoro TTS, Opus codec |
	  | **Protocol** | Binary V3 | 21-byte position updates, delta encoding |
	  | **Auth** | Nostr NIP-07 | Browser extension signing, relay integration |
	  | **Agents** | MCP, Claude-Flow | 101 skills, 7 ontology tools |
	  | **AI/ML** | GraphRAG, RAGFlow | Knowledge retrieval, inference |
	  | **Build** | Vite 6, Vitest, Playwright | Frontend build, unit tests, E2E tests |
	  | **Infra** | Docker Compose | 10 compose files, multi-profile deployment |
	  | **CI** | GitHub Actions | Build, test, docs quality, ontology federation |
-
- ---
-
- # Investment Thesis
	- ## The Convergence
		- Three domains are maturing simultaneously — VisionFlow sits at their intersection:
		- **1. Semantic Web becomes practical** — OWL reasoning via Whelk-rs is fast enough for real-time use (<2s inference, 90x cache speedup). Ontologies deliver +40-55% improvement in AI accuracy (EMNLP 2025).
		- **2. Agentic systems cross the chasm** — Claude Code proves $1B ARR in 6 months. 72% of enterprises deploying agents by 2026. But 80% want multi-agent, fewer than 10% have achieved it.
		- **3. Spatial computing enters enterprise** — $182B market today, $1T+ by 2034. Hardware costs dropping below $1,000. Microsoft, Apple, Samsung all shipping enterprise-ready devices.
	- ## Why VisionFlow wins
		- **Technical moat**: Whelk-rs ontology reasoning + CUDA semantic physics + agentic contribution loop. Not a commodity offering — no competitor combines ontology inference with spatial computing.
		- **Right interface for the problem**: HBR proves flat chat/terminal interfaces create cognitive overload. Spatial, ontology-structured environments reduce it.
		- **Category creation**: Every major agentic platform targets coding. Knowledge work broadly — research, analysis, decision-making, project management — is wide open.
		- **Production-ready**: Hexagonal architecture, Neo4j-backed, CQRS, full audit trail, consistency gating. Not a prototype.
		- **Open source flywheel**: MPL 2.0 license. Community adoption drives ecosystem. Enterprise features and hosting drive revenue.
	- ## What's Next
		- Scale the team (2-4 engineers for enterprise features, compliance)
		- Expand enterprise integrations (SSO, SAML, data connectors)
		- Multi-GPU support for larger deployments
		- Academic partnerships for ontology research
		- Enterprise pilot programme
-
- ---
-
- # Videos
	- {{video https://youtu.be/cEqt-OnlBzY}}
	- {{video https://youtu.be/3wMCUgBEjos}}
-
- ---
-
- # Links & Resources
	- [GitHub Repository](https://github.com/DreamLab-AI/VisionFlow) — Open source, MPL 2.0
	- [Live Ontology Explorer](https://www.narrativegoldmine.com) — 2D interactive graph and data explorer
	- [DreamLab AI](https://dreamlab-ai.com/) — Company website
	- [The Original Book (2022)](https://arxiv.org/pdf/2207.09460) — Where it all started
	- [HBR: AI Doesn't Reduce Work](https://hbr.org/2026/02/ai-doesnt-reduce-work-it-intensifies-it) — The interface problem
	- [SemiAnalysis: Claude Code Inflection](https://newsletter.semianalysis.com/p/claude-code-is-the-inflection-point) — Agentic development market
	- [OG-RAG Paper (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1674.pdf) — Ontology performance evidence
-
- ---
-
- # Technical Appendices
	- *Folded sections for Q&A and post-session reference. Each appendix provides implementation-level detail for a specific subsystem.*
	- ## Appendix A: Abstract
	  collapsed:: true
		- **VisionFlow / IRIS** is an open-source spatial intelligence platform that transforms unstructured knowledge (Logseq markdown notebooks, OWL ontologies, document corpora) into interactive 3D environments where the spatial layout of a graph directly encodes its logical structure. The system implements a novel *axiom-to-force translation* pipeline: OWL 2 EL axioms produced by the Whelk-rs reasoner are compiled into GPU-accelerated physics forces (CUDA, 6,400+ lines across 11 kernel files), such that SubClassOf relations become spring attractions, DisjointWith relations become Coulomb repulsions, and EquivalentClasses produce strong attractive forces that merge synonymous concepts. This produces equilibrium layouts where hierarchies form concentric shells, contradictions create visible gaps, and equivalences collapse — enabling users to *see the shape of knowledge* without reading labels.
		- The platform is built on a Rust/Actix hexagonal architecture (376 source files, 169K lines) with 45+ supervised actors managing GPU physics, ontology reasoning, graph state, client sessions, and voice interaction. A React/Three.js frontend (362 files, 92K lines) renders the graph with custom TSL (Three Shading Language) materials targeting WebGPU with WebGL2 fallback, and supports WebXR immersion on Meta Quest 3. Multi-user collaboration is provided via Vircadia World Server (spatial presence, avatar sync, HRTF spatial audio) and LiveKit SFU (Opus voice, turbo-whisper STT, Kokoro TTS). A compact binary WebSocket protocol (21 bytes/node, 80% bandwidth reduction vs JSON) sustains 60 FPS position updates for 180,000 nodes on an RTX 3080.
		- AI agents interact with the knowledge graph through 7 MCP (Model Context Protocol) ontology tools, forming a closed loop: discover → read → propose → consistency check → GitHub PR → human review → merge → re-reason → re-layout. The system includes 101 agent skills spanning reasoning, development, orchestration, creative production, and infrastructure management. All knowledge changes are Git commits, providing a complete audit trail.
		- VisionFlow addresses the growing evidence (HBR, Feb 2026) that flat AI interfaces (chat, terminals, inline completions) create cognitive overload rather than reducing it, by providing a structured spatial environment grounded in formal semantics. It targets the $5T knowledge work market where the 80/10 gap (80% of enterprises want multi-agent orchestration, <10% have achieved it) represents a significant opportunity.
	- ## Appendix B: Rust Actor Architecture Deep-Dive
	  collapsed:: true
		- The backend uses the Actix actor framework for supervised concurrency. Each actor is a lightweight, single-threaded entity with its own mailbox. Message passing is typed and asynchronous. The system contains 45+ actors organised into GPU compute, service, and coordination layers.
		- ### Actor Supervision Tree
			- ```
			  GraphServiceSupervisor (root)
			  ├── GPUManagerActor
			  │   ├── GPUResourceActor (CUDA context lifecycle)
			  │   ├── ForceComputeActor (main physics loop)
			  │   ├── StressMajorizationActor
			  │   ├── ClusteringActor (k-means, LOF)
			  │   ├── PageRankActor
			  │   ├── SemanticForcesActor (ontology → forces)
			  │   ├── OntologyConstraintActor (axiom buffer)
			  │   ├── ShortestPathActor (SSSP, APSP)
			  │   ├── ConnectedComponentsActor
			  │   └── AnomalyDetectionActor
			  ├── GraphStateActor (canonical graph)
			  ├── OntologyActor (OWL class/axiom management)
			  ├── PhysicsOrchestratorActor (60Hz tick, param interpolation)
			  ├── ClientCoordinatorActor (per-session WebSocket)
			  ├── WorkspaceActor (multi-workspace isolation)
			  ├── TaskOrchestratorActor (background job scheduling)
			  ├── OptimizedSettingsActor (Neo4j-persisted settings)
			  ├── ProtectedSettingsActor (auth-gated settings)
			  └── MetadataActor (node/edge metadata cache)
			  ```
		- ### Key Design Decisions
			- **GPU mutex recovery**: CUDA kernel panics poison `std::sync::Mutex`. All GPU actors use `poisoned.into_inner()` to recover from panicked threads instead of propagating the panic
			- **Deferred graph upload**: `ForceComputeActor` stores `pending_graph_data` and uploads to GPU only when both the graph data and `SharedGPUContext` are available (they can arrive in either order)
			- **Non-blocking GPU access**: Physics computation uses `tokio::task::spawn_blocking` to move GPU mutex locks off the Tokio async executor threads, preventing thread starvation
			- **Parameter interpolation**: `PhysicsOrchestratorActor` maintains `target_params` and smoothly interpolates toward them at 60Hz, preventing visible "jumps" when settings sliders change
			- **Backpressure**: Token-bucket flow control between GPU producer and WebSocket consumer. `ForceComputeActor` tracks available broadcast tokens (100 max, refill at 30/s). Each broadcast costs 1 token; acknowledgements from `ClientCoordinatorActor` restore tokens
		- ### Message Flow: Settings Change
			- ```
			  Client slider → REST PATCH /api/settings/physics
			                → normalize_physics_keys() (snake_case/camelCase aliases)
			                → validate_physics_settings() (NaN/Infinity/range check)
			                → Neo4jSettingsRepository::save_setting()
			                → ForceComputeActor::UpdateSimulationParams
			                    → update_simulation_parameters()
			                    → broadcast_optimizer.reset_delta_state()
			                    → stability_warmup_remaining = 300
			                → PhysicsOrchestratorActor::UpdateParams
			                    → target_params = new_params (interpolated at 60Hz)
			  ```
	- ## Appendix C: CUDA Kernel Architecture
	  collapsed:: true
		- ### Kernel Inventory (28 `__global__` kernels across 11 files)
			- | Kernel | Grid Size | Shared Memory | Purpose |
			  | ---- | ---- | ---- | ---- |
			  | `build_grid_kernel` | N nodes / 256 | 0 | Hash node positions into spatial grid cells |
			  | `compute_cell_bounds_kernel` | N cells / 256 | 0 | Compute start/end offsets per grid cell |
			  | `force_pass_kernel` | N nodes / 256 | 0 | Spring + repulsion forces with grid-accelerated neighbour lookup |
			  | `integrate_pass_kernel` | N nodes / 256 | 0 | Velocity verlet integration with boundary enforcement |
			  | `apply_semantic_forces` | N constraints / 256 | 0 | OWL axiom constraint forces (type-dispatched) |
			  | `blend_semantic_physics_forces` | N nodes / 256 | 0 | Alpha-blend semantic and base physics force buffers |
			  | `calculate_kinetic_energy_kernel` | N nodes / 256 | 256 * sizeof(float) | Parallel reduction for total KE |
			  | `check_system_stability_kernel` | 1 block / 1 thread | 0 | Compare KE against threshold, set skip flag |
			  | `force_pass_with_stability_kernel` | N nodes / 256 | 0 | Combined force + stability check in single pass |
			  | `compute_aabb_reduction_kernel` | N nodes / 256 | 256 * sizeof(float) * 6 | Bounding box for spatial culling |
			  | `relaxation_step_kernel` | N nodes / 256 | 0 | Stress majorisation gradient descent step |
			  | `compact_frontier_kernel` | N nodes / 256 | 0 | SSSP compact-frontier BFS expansion |
			  | `init_centroids_kernel` | K / 256 | 0 | K-means++ centroid initialisation |
			  | `assign_clusters_kernel` | N nodes / 256 | 0 | K-means cluster assignment |
			  | `update_centroids_kernel` | K / 256 | 0 | K-means centroid update |
			  | `compute_inertia_kernel` | K / 256 | 0 | Elbow method inertia calculation |
			  | `compute_lof_kernel` | N nodes / 256 | 0 | Local Outlier Factor anomaly scores |
			  | `compute_zscore_kernel` | N nodes / 256 | 0 | Z-score normalisation |
			  | `compute_feature_stats_kernel` | Features / 256 | 0 | Mean/stddev for feature normalisation |
			  | `init_labels_kernel` | N nodes / 256 | 0 | Label propagation initialisation |
			  | `propagate_labels_sync_kernel` | N nodes / 256 | 0 | Synchronous label propagation |
			  | `propagate_labels_async_kernel` | N nodes / 256 | 0 | Asynchronous label propagation |
			  | `check_convergence_kernel` | 1 block / 256 | 256 * sizeof(int) | Label propagation convergence check |
			  | `compute_modularity_kernel` | N nodes / 256 | 256 * sizeof(float) | Modularity score computation |
			  | `init_random_states_kernel` | N / 256 | 0 | PRNG state initialisation (curand) |
			  | `compute_node_degrees_kernel` | N nodes / 256 | 0 | Degree computation from CSR |
			  | `count_community_sizes_kernel` | N nodes / 256 | 0 | Community size histogram |
			  | `relabel_communities_kernel` | N nodes / 256 | 0 | Compact community label assignment |
		- ### Data Layout
			- **Graph storage**: Compressed Sparse Row (CSR) format — `row_offsets[N+1]`, `col_indices[E]`, `edge_weights[E]`. CSR is cache-friendly for GPU traversal because all neighbours of node `i` are contiguous in memory at `col_indices[row_offsets[i]..row_offsets[i+1]]`.
			- **Node state**: Structure-of-Arrays (SoA) — six separate `float*` buffers for `pos_x, pos_y, pos_z, vel_x, vel_y, vel_z`. SoA maximises memory coalescing: when 32 threads in a warp read `pos_x[tid]`, the 128-byte transaction fetches 32 consecutive floats.
			- **Constraint buffer**: Array of 64-byte-aligned `OntologyConstraint` structs with pre-computed source/target indices. The host resolves node-ID-to-GPU-index before kernel launch, eliminating per-constraint O(N) lookup on the device.
			- **Spatial grid**: 3D hash grid with configurable cell size. `build_grid_kernel` assigns each node to a cell; `compute_cell_bounds_kernel` computes cell start/end offsets via parallel prefix scan. Force kernels iterate only over the 27 neighbouring cells (3x3x3), reducing N-body from O(N^2) to O(N * k) where k = average nodes per neighbourhood.
	- ## Appendix D: Ontology Reasoning Pipeline
	  collapsed:: true
		- ### OWL 2 EL Profile
			- VisionFlow uses the OWL 2 EL (Existential Language) profile, which supports:
			- Class subsumption (`SubClassOf`), equivalence (`EquivalentClasses`), disjointness (`DisjointWith`)
			- Existential restrictions (`SomeValuesFrom`)
			- Object property chains
			- **Not supported** (by design): universal restrictions, negation, cardinality constraints (these make reasoning NP-hard; EL keeps it polynomial)
		- ### Whelk-rs Implementation
			- Whelk is an OWL 2 EL reasoner originally implemented in Scala. The Rust port (`whelk-rs`) provides:
			- **Subsumption closure**: Given a set of axioms, compute all implied SubClassOf relationships. This is the core reasoning task.
			- **Consistency checking**: Detect unsatisfiable classes (classes that cannot have instances). Used to gate ontology changes — agents cannot merge inconsistent knowledge.
			- **Performance**: <2 seconds for 900+ classes on cold start. 90x speedup on repeat queries via LRU cache (class hierarchy is relatively stable between edits).
		- ### The Axiom-to-Constraint Pipeline
			- ```
			  Logseq Markdown → OWL Parser (owl_parser.rs)
			                  → Whelk-rs Reasoner (inference/mod.rs)
			                  → Inferred Axiom Set
			                  → OntologyConstraintTranslator (constraints/axiom_mapper.rs)
			                  → ConstraintData buffer (64-byte aligned structs)
			                  → OntologyConstraintActor (cached in-memory)
			                  → ForceComputeActor::UpdateOntologyConstraintBuffer
			                  → UnifiedGPUCompute::upload_constraints()
			                  → ontology_constraints.cu kernels (per-frame)
			  ```
		- ### Constraint Types in CUDA
			- ```c
			  #define CONSTRAINT_DISJOINT_CLASSES 1  // Coulomb repulsion
			  #define CONSTRAINT_SUBCLASS_OF 2       // Spring attraction (hierarchical)
			  #define CONSTRAINT_SAMEAS 3             // Strong spring (equivalence)
			  #define CONSTRAINT_INVERSE_OF 4         // Directed attraction
			  #define CONSTRAINT_FUNCTIONAL 5          // Cardinality constraint
			  ```
		- Each constraint struct contains pre-computed `source_idx` and `target_idx` (resolved on host), so the GPU kernel does zero ID lookups — just reads positions, computes force, writes to force accumulator.
	- ## Appendix E: Client Rendering and WebSocket Protocol
	  collapsed:: true
		- ### Binary Protocol V3 Wire Format
			- Position update (21 bytes per node):
			- ```
			  [0x10] [node_id: u32] [x: f32] [y: f32] [z: f32] [vx: f16] [vy: f16] [vz: f16]
			  ```
			- Velocities use half-precision (f16) because visual interpolation tolerates quantisation. Positions use full f32 for accuracy.
			- **Delta encoding**: The `BroadcastOptimizer` tracks previous positions per node. Only nodes that moved more than `delta_threshold` (default 1cm) are included in the broadcast. Typical compression: 60-90% of nodes filtered per frame after convergence.
			- **Backpressure**: Token bucket with 100 max tokens, refill at 30/s. Each broadcast costs 1 token. When tokens depleted, frames are skipped (not queued). Client-side `PositionBroadcastAck` messages restore tokens.
		- ### Client Position Pipeline
			- ```
			  Binary WebSocket frame
			  → graph.worker.ts::processBinaryData()
			      → parseBinaryNodeData() (typed array slicing, zero-copy)
			      → reverseNodeIdMap.get(update.nodeId) → string node ID
			      → nodeIndexMap.get(stringNodeId) → position buffer index
			      → targetPositions[index * 3] = update.position.{x,y,z}
			  → tick() (requestAnimationFrame, 60Hz)
			      → lerp currentPositions toward targetPositions (smoothing)
			      → syncToSharedBuffer() → SharedArrayBuffer
			  → GraphManager.tsx (main thread, reads SharedArrayBuffer)
			      → InstancedMesh position attribute update
			      → Edge point computation: nodeIdToIndexMap lookups for source/target
			      → GlassEdges.updatePoints(edgePoints)
			  ```
		- ### TSL Materials
			- Three Shading Language (TSL) materials are written as JavaScript node graphs compiled to WGSL (WebGPU) or GLSL (WebGL2). The custom materials are:
			- **GemNodeMaterial**: Fresnel-based iridescence with analytics-driven colour mapping (PageRank → hue, degree → saturation, anomaly score → emission intensity)
			- **CrystalOrbMaterial**: Time-varying depth pulse with cosmic spectrum sampling and Fresnel rim
			- **AgentCapsuleMaterial**: Bioluminescent heartbeat driven by agent `activity_level` uniform (0.0-1.0)
			- **GlassEdgeMaterial**: Animated UV-scrolling emissive for edge flow visualisation, opacity proportional to edge weight
	- ## Appendix F: Voice and Multi-User Architecture
	  collapsed:: true
		- ### Four-Plane Voice Model
			- | Plane | Path | Codec | Latency Budget |
			  | ---- | ---- | ---- | ---- |
			  | 1. Private command | Mic → turbo-whisper STT → intent parse → agent dispatch | Opus 48kHz mono | 410ms measured (STT 300ms + parse 1ms + dispatch 50ms + ACK 5ms) |
			  | 2. Private response | Agent → Kokoro TTS → user ear (spatial) | Opus 48kHz mono | ~200ms (synthesis + buffering) |
			  | 3. Public voice | Mic → LiveKit SFU → all users (spatial audio) | Opus 48kHz mono | <50ms (SFU relay) |
			  | 4. Public agent | Agent TTS → LiveKit → all users (spatial) | Opus 48kHz mono | ~250ms |
		- **Spatial audio**: HRTF panning based on entity positions from Vircadia World Server. Each user and agent has a 3D position; audio is spatialised relative to the listener's head position/orientation.
		- **Voice differentiation**: Each agent type uses a distinct Kokoro voice preset (pitch, speed, timbre) so users can identify which agent is speaking without visual confirmation.
		- ### Vircadia World Server Integration
			- Vircadia provides the multi-user substrate:
			- **Entity CRUD**: Each graph node, edge, and agent is a Vircadia entity with position, rotation, and metadata
			- **Avatar sync**: User positions, orientations, and hand tracking data (Quest 3) replicated at 30Hz
			- **Spatial presence**: Users see each other as avatars positioned in the graph space
			- **PostgreSQL backend**: Entity state persisted in PostgreSQL 15 (separate from Neo4j graph store)
	- ## Appendix G: Security Model
	  collapsed:: true
		- ### Authentication
			- **Nostr NIP-07**: Browser extension signing (nos2x, Alby). Public key identity without passwords or centralised auth servers.
			- **Relay integration**: Signed events published to Nostr relays for decentralised audit
			- **Per-request auth**: `AuthenticatedUser` extractor validates Nostr signatures on every API call; `OptionalAuth` allows anonymous read access
		- ### Agent Access Control
			- MCP tools are the only interface agents have to the knowledge graph. Each tool validates:
			- Authentication (agent identity via MCP session)
			- Authorization (which ontology namespaces the agent can modify)
			- Consistency (Whelk-rs validates proposed changes before GitHub PR creation)
		- ### Audit Trail
			- Every ontology modification is a Git commit → full history, blame, and rollback
			- Agent contributions are tagged with agent identity in commit metadata
			- Nostr event signatures provide non-repudiation
-
- ---
-
- *VisionFlow / IRIS — DreamLab AI Consulting Ltd. — February 2026*
