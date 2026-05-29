# VisionFlow Unity XR Module

Unity **6000.4** project for the Meta Quest XR client: 3D knowledge graph, Nostr governance, and Solid Pod provenance.

## Location

Open `VisionFlow/` in Unity Hub (this folder).

## Prerequisites

- Unity **6000.4.3f1** (or compatible 6000.4.x)
- XR Interaction Toolkit **3.4** + OpenXR + Meta OpenXR (see `Packages/manifest.json`)
- Optional local backends: VisionClaw graph WS, Agentbox Solid Pod (see repo `scripts/dev-backends/`)

## First-time scene setup

1. Open `Assets/VisionFlow/Scenes/VisionFlowXR.unity` (or any scene).
2. Menu: **GameObject → VisionFlow → Bootstrap (Manager + Graph + UI)**.
3. Add XR rig: **GameObject → VisionFlow → Add XR Origin (XR Rig) From Samples** (requires XRI Starter Assets sample import).
4. **GameObject → VisionFlow → Wire Near-Far Interactors**.

## Play / device

- Start VisionClaw (`ws://localhost:8765/v1/graph`) and Solid Pod (`http://localhost:8484`) for full stack, or use `scripts/dev-backends/` stubs.
- Build **Android** IL2CPP ARM64 for Quest 3.

## Docs

See [`docs/VisionFlow_XR_TDD.md`](../docs/VisionFlow_XR_TDD.md) for milestones M1–M6 and architecture.
