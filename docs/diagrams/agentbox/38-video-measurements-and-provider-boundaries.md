---
id: AB-38
title: Video measurements and provider qualification boundaries
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2150, ADR-2163]
sources:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/skills/video-production/runtime/src/windows.rs
  - ../project/agentbox/skills/video-production/runtime/src/render_cache.rs
  - ../project/agentbox/skills/video-production/runtime/src/media.rs
  - ../project/agentbox/skills/video-production/runtime/src/stitch.rs
  - ../project/agentbox/skills/video-production/runtime/src/seam_qa.rs
  - ../project/agentbox/skills/video-production/runtime/src/acceptance.rs
  - ../project/agentbox/skills/comfyui/scripts/comfy_client.py
  - ../project/agentbox/skills/comfyui/mcp-server/server.js
  - ../project/agentbox/comfyui-sidecar/nodes/agentbox_video/__init__.py
  - ../project/agentbox/comfyui-sidecar/nodes/agentbox_video/identity.py
  - ../project/agentbox/comfyui-sidecar/nodes/agentbox_video/staging.py
  - ../project/agentbox/comfyui-sidecar/verify-spede.sh
verified_commit: {agentbox: 9dfc27b16e82e9786535a55e790710493dd51fdf}
---

## For developers

The [media audit](../../estate-review/2026-11-01-media-measurements-and-cancellation.md) records 305 passing existing tests, two skips, four audit tests and three controlled cancellation cases. Live model, image-build and GPU qualification remain unrun.

## For the business

Measurements help an editor locate defects and repeat a bounded edit. They do not accept a film or prove that a model workflow runs on the installed GPU.

## AB-38.1 Window lineage constrains selective reruns

```mermaid
flowchart TD
    REQUEST["Profile, geometry and model hashes<br/>windows.rs:352"] --> PLAN["Current schema: window-plan/2<br/>windows.rs:42"]
    PLAN --> KEYS["Plan hash and chained window keys<br/>windows.rs:439,442,443"]
    KEYS --> VERIFY["Verify plan and available model bytes<br/>windows.rs:460,487"]
    VERIFY --> DIFF["Compare verified plans<br/>windows.rs:536"]
    DIFF --> BIND["Bind parent plan hash and final key<br/>render_cache.rs:122,123,128"]
    BIND --> REUSE["Validate timing, coverage and allowed changes<br/>render_cache.rs:196,206,225,227"]
```

**What it shows.** Cache reuse binds the actual parent plan, not only equal frame counts. **Why it is this way.** A seed reroll changes that window and its successors; geometry or model-verification changes deliberately invalidate the broader plan.

## AB-38.2 File validation and measurements precede editorial review

```mermaid
flowchart TD
    INPUT["Local-file decode and frame allocation cap<br/>media.rs:24,28,33"] --> JOIN["Trim and blend assembly<br/>stitch.rs:153,402"]
    JOIN --> GATE["Check frames, rate, canvas, colour and audio<br/>stitch.rs:286,290,293,308"]
    GATE --> BAD["Discard rejected output<br/>stitch.rs:314"]
    GATE --> SEAM["Resolve colour and measure seams<br/>seam_qa.rs:703,711"]
    SEAM --> BOUNDS["GAP: extreme seam index overflows in debug<br/>seam_qa.rs:753"]
    SEAM --> STAMP["Provisional; no acceptance backing<br/>acceptance.rs:72,73,74,75"]
    STAMP --> EDITOR["REQUIRED editorial review of the production"]
```

**What it shows.** A supported file can pass technical checks and still need editorial judgement. **Why it is this way.** Thresholds are uncalibrated measurements; stale calibration claims are refused, and untagged YUV needs an explicit supported colour declaration.

## AB-38.3 ID-bearing interrupt still crosses a server race

```mermaid
flowchart TD
    RECEIPT["Read owned prompt and queue snapshot<br/>comfy_client.py:306,315"] --> PY["POST interrupt with prompt_id<br/>comfy_client.py:325"]
    JS["MCP running-job cancellation<br/>server.js:149"] --> SNAP["Pinned upstream checks running snapshot"]
    PY --> SNAP
    SNAP --> GAP["CONTROLLED hand-off: owned A finishes; B starts"]
    GAP --> SIGNAL["Upstream global interrupt can reach B"]
    PY --> ACK["Request acknowledgement is not terminal history<br/>comfy_client.py:309,366"]
    SIGNAL -.-> FIX["REQUIRED use atomic per-job cancellation route<br/>and verify terminal history"]
```

**What it shows.** An ID in the request does not make the selected server operation atomic. **Why it is this way.** The [pinned upstream handler](https://github.com/Comfy-Org/ComfyUI/blob/faa7e494f3a0c9e45e1c3db8d20ee48fd5fe1b52/server.py#L1258) checks a snapshot before signalling; the audit reproduces the hand-off with controlled effects, without a live queue.

## AB-38.4 Registered nodes and measurements leave qualification work

```mermaid
flowchart TD
    IMPORT["Import each module; warn and skip failures<br/>__init__.py:17,20,21,22"] --> MAP["Merge available node mappings<br/>__init__.py:23,24"]
    MAP --> CHECK["Post-build required-node list<br/>verify-spede.sh:33,34,35"]
    CHECK --> OMIT["GAP: load-image node omitted from check<br/>staging.py:206"]
    IDENTITY["Identity scores explicitly editorial<br/>identity.py:368"] --> REVIEW["REQUIRED real-face and workflow qualification"]
    DRIFT["GAP: drift report omits two measurement fields<br/>identity.py:424,425"] --> REVIEW
    CHECK --> GPU["GPU smoke is optional<br/>verify-spede.sh:54"]
    GPU --> REVIEW
```

**What it shows.** CPU tests and registration checks establish limited properties; neither executes the complete model workflow. **Why it is this way.** The staged image, real-face tests and live relay need separate evidence, and the registration list must cover every required node.
