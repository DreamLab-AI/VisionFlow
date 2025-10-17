# 🌌 VisionFlow

[![License](https://img.shields.io/badge/License-Mozilla%202.0-blue.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/Performance-60FPS%20@%20100k%20nodes-red.svg)](docs/)
[![Agents](https://img.shields.io/badge/AI%20Agents-50%2B%20Concurrent-orange.svg)](docs/)
[![CUDA](https://img.shields.io/badge/CUDA-40%20Kernels-green.svg)](docs/)

### **Immersive Multi-User Multi-Agent Knowledge Graphing**

**VisionFlow deploys self-sovereign AI agent teams that continuously research, analyse, and surface insights from your entire data corpus—visualised for collaborative teams in a stunning, real-time 3D interface.**

<div align="center">
  <table>
    <tr>
      <td><img src="./visionflow.gif" alt="VisionFlow Visualisation" style="width:100%; border-radius:10px;"></td>
      <td><img src="./jarvisSept.gif" alt="Runtime Screenshot" style="width:100%; border-radius:10px;"></td>
    </tr>
  </table>
</div>

---

## 🚀 Why VisionFlow?

Unlike passive AI tools that wait for your prompts, VisionFlow's autonomous agent teams work continuously in the background, discovering patterns and connections in your private knowledge base that you didn't know existed.

**VisionFlow vs Traditional AI Tools:**

| VisionFlow | Traditional AI Chat |
| :--- | :--- |
| ✅ **Continuous**, real-time agent research | ❌ Reactive, query-based responses |
| ✅ Discovers patterns in **your private knowledge corpus** | ❌ Limited to conversation context |
| ✅ **Interactive 3D visualisation** you explore with your team | ❌ Static text-based output |
| ✅ **Human-in-the-loop** collaboration with Git version control | ❌ No audit trail or oversight |
| ✅ **Self-sovereign** and enterprise-secure | ❌ Hosted on third-party infrastructure |
| ✅ **Voice-first** spatial interaction | ❌ Text-only interface |

---

## ✨ Key Features

*   **🧠 Continuous AI Analysis**: Deploy teams of specialist AI agents (Researcher, Analyst, Coder) that work 24/7 in the background, using advanced GraphRAG to uncover deep semantic connections within your private data.

*   **🤝 Real-Time Collaborative 3D Space**: Invite your team into a shared virtual environment. Watch agents work, explore the knowledge graph together, and maintain independent specialist views while staying perfectly in sync.

*   **🎙️ Voice-First Interaction**: Converse naturally with your AI agents. Guide research, ask questions, and receive insights through seamless, real-time voice-to-voice communication with spatial audio.

*   **🔐 Enterprise-Grade & Self-Sovereign**: Your data remains yours. Built on a thin-client, secure-server architecture with Git-based version control for all knowledge updates, ensuring a complete audit trail and human-in-the-loop oversight.

*   **🔌 Seamless Data Integration**: Connect to your existing knowledge sources with our powerful Markdown-based data management system, built on [Logseq](https://logseq.com/). Enjoy block-based organisation, bidirectional linking, and local-first privacy.

*   **🦉 Ontology-Driven Validation**: Ensure logical consistency with OWL 2 semantic validation powered by horned-owl 1.2.0. Automatically infer new relationships through transitive, symmetric, and inverse property reasoning. Validate cardinality, domain/range, and disjoint class constraints in real-time with <2s latency. Visualize ontological constraints as physics forces for intuitive graph layouts. Supports OWL Functional Syntax and OWL/XML formats with SQLite persistence.

*   **⚡ GPU-Accelerated Performance**: 40 production CUDA kernels deliver 100x CPU speedup for physics simulation, clustering, and pathfinding—enabling 60 FPS rendering at 100k+ nodes with sub-10ms latency.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/VisionFlow.git
cd VisionFlow

# 2. Configure your environment
cp .env.example .env
# Edit .env to add your data sources and API keys

# 3. Deploy with Docker
docker-compose up -d

# 4. Access your AI research universe
# Open http://localhost:3001 in your browser
```

**That's it!** Your AI agent teams will begin analysing your data immediately.

**[📚 Full Documentation](docs/index.md)** | **[🎯 Getting Started Guide](docs/getting-started/02-quick-start.md)** | **[🔧 Installation Details](docs/getting-started/01-installation.md)**

---

## 🛠️ Technology Stack

VisionFlow combines cutting-edge technologies for unmatched performance and scalability:

| Layer | Technology | Highlights |
| :--- | :--- | :--- |
| **Frontend** | React + Three.js (React Three Fiber) | 60 FPS @ 100k+ nodes, WebGL 3D rendering |
| **Backend** | Rust + Actix | Supervised actor system, 1,000+ req/min |
| **GPU Acceleration** | CUDA (40 Kernels) | Physics, clustering, pathfinding—100x speedup |
| **AI Orchestration** | MCP Protocol + Claude | 50+ concurrent specialist agents |
| **Semantic Layer** | OWL 2 + Horned-OWL 1.2.0 | Ontology validation, reasoning, inference |
| **Networking** | Binary WebSocket (34-byte protocol) | <10ms latency, 95% bandwidth reduction |
| **Data Layer** | SQLite + Git + Markdown | Ontology persistence, version control, graph data |

**Advanced AI Architecture:**
- **Microsoft GraphRAG** for hierarchical knowledge structures
- **Leiden Clustering** for community detection
- **Shortest Path Analysis** enabling multi-hop reasoning
- **OWL 2 Reasoning** for semantic validation and inference
  - Constraint validation (cardinality, domain/range, disjoint classes)
  - Inference generation (transitive, symmetric, inverse properties)
  - Real-time WebSocket validation updates
  - SQLite persistence with <5ms query latency

---

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, improving documentation, or proposing new features, your help makes VisionFlow better.

**[📖 Contributing Guide](docs/contributing.md)** | **[🏗️ Architecture Overview](docs/architecture/index.md)** | **[🔍 API Documentation](docs/api/index.md)**

---

## 🔮 Roadmap

- ✅ **Current**: Real-time multi-user collaboration, voice-to-voice AI, 50+ concurrent agents, GPU acceleration, ontology validation, mobile companion app (Logseq), ML tools
- 🔄 **Coming Soon**: AR/VR (Quest 3) interface, multi-language voice support, email integration, SPARQL query interface
- 🎯 **Future Vision**: Scaled collaborative VR, predictive intelligence, autonomous workflows, federated ontologies, community plugin marketplace

---

## 🌟 Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/VisionFlow/issues)
- **Documentation**: [Full Documentation Hub](docs/index.md)

#### 🙏 Acknowledgements

Inspired by the innovative work of **Prof. Rob Aspin** and powered by the tools and concepts from **Anthropic**, **OpenAI**, and the incredible open-source community.

---

## 📄 Licence

This project is licensed under the Mozilla Public Licence 2.0. See the [LICENSE](LICENSE) file for details.

---

**Ready to transform how your team discovers knowledge?**

**[Get Started Now](docs/getting-started/02-quick-start.md)** | **[View Full Documentation](docs/index.md)**
