# Project OWL: A Proposal for a Hybrid Semantic Graph System

## 1. Executive Summary

This document proposes **Project OWL (Ontology-Webbed Links)**, a feature set to elevate the AR-AI-Knowledge-Graph platform from a high-performance visualization tool to a system for verifiable knowledge representation and logical inference. Our current system excels at discovering emergent patterns through statistical analysis and force-directed layouts. However, it lacks a mechanism for ensuring logical consistency and cannot deduce new information from existing facts.

Project OWL addresses this by integrating a formal, lightweight validation engine into our existing Rust backend. This hybrid approach will augment our flexible property graph with the rigor of rule-based validation, enabling three core capabilities in its first version:

1.  **Lightweight Validation:** Users can define a set of rules and validate their graph against them, instantly identifying and visualizing inconsistencies in their data model.
2.  **Clear API for Integration:** A simple, asynchronous API allows the frontend to trigger validation and poll for results without blocking the UI.
3.  **Actionable Visual Feedback:** The frontend will be enhanced to clearly visualize validation errors, providing users with immediate, intuitive feedback on data quality.

This initial version focuses on establishing the architecture and workflow for validation, de-risking the approach and delivering immediate value. Future phases will build upon this foundation to incorporate more advanced OWL reasoning and inference capabilities.

For detailed specifications, see:
- [Lightweight Rules (Version 1)](./rules_v1.md)
- [API Specification (Version 1)](./api.md)
- [Precise Implementation Plan (Version 1)](./implementation_plan.md)

## 2. The Hybrid Workflow: Lightweight Rules v1

The interplay between the two systems will follow a clear, asynchronous workflow. This version uses a lightweight, in-process rules engine instead of a full OWL reasoner to ensure performance and architectural simplicity for v1.

```mermaid
flowchart TD
    subgraph Frontend
        A[User clicks 'Validate Graph'] --> B{POST /api/analytics/validate};
        B --> C[Poll GET /api/analytics/validate/status/{taskId}];
        C --> D{Display ValidationReport};
    end

    subgraph Backend
        B --> E[GraphServiceActor];
        E --> F[Clone GraphData];
        F --> G[OntologyActor: ValidateGraph];
        G --> H[Run Lightweight Rules];
        H --> I[Return ValidationReport];
        I --> E;
        E --> C;
    end

    style Frontend fill:#e6f3ff,stroke:#0066cc
    style Backend fill:#d6f5d6,stroke:#006400
```

**Step-by-Step Breakdown:**

1.  **Trigger:** The user initiates validation from the frontend.
2.  **API Request:** The client sends a `POST` request to `/api/analytics/validate`. The backend returns a `taskId`.
3.  **Task Management:** The `GraphServiceActor` receives the request, creates a task ID, clones the current `GraphData`, and sends a `ValidateGraph` message to the `OntologyActor`.
4.  **Validation:** The `OntologyActor` (acting as our lightweight rules engine) executes a predefined set of rules against the graph data.
5.  **Polling & Response:** The frontend polls the status endpoint. Once validation is complete, the `GraphServiceActor` returns the `ValidationReport` containing any inconsistencies.
6.  **Visualization:** The frontend uses the report to highlight inconsistent nodes/edges.

## 3. Roadmap & Risks

*   **Version 1 (This Proposal):** Focuses on establishing the end-to-end validation workflow with a lightweight, in-process rules engine. This de-risks the architecture and provides immediate data quality feedback to users.
*   **Version 2 (Future):**
    *   Introduce a feature-flagged integration with `horned-owl` and `whelk-rs`.
    *   Implement a more sophisticated mapping from `GraphData` to OWL axioms.
    *   Begin populating the `inferred_triples` array with results from the reasoner.
    *   Address the licensing implications of statically linking an LGPLv3 library.

**Risk:** The primary risk for future versions is the LGPLv3 license of `horned-owl`. A mitigation strategy could involve using it as an optional, dynamically loaded plugin, which would need further investigation. This risk is completely avoided in Version 1.