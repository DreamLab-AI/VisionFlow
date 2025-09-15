Below is a comprehensive debugging guide for the startup process and client-server synchronization issues in your VisionFlow application, with a focus on the knowledge graph (handled by GraphServiceActor and related components) and settings (handled by AppFullSettings and the REST API). I'll break it down step-by-step, based on the codebase structure. The problems likely stem from initialization timing, missing dependencies (e.g., metadata loading), authentication/validation failures, or WebSocket protocol mismatches. I'll include specific checks, commands, and fixes.

1. Server Startup Debugging
The server startup involves loading settings, metadata, building the graph, and initializing GPU/physics. If the knowledge graph isn't built, clients won't receive position data. If settings aren't loaded, UI/features will be broken.

Step 1.1: Verify Server Logs on Startup
What to Check:

Run the server with RUST_LOG=debug cargo run to enable verbose logging.
Look for errors in:
Metadata loading (FileService::load_or_create_metadata).
Graph building (GraphServiceActor::build_from_metadata).
GPU initialization (UnifiedGPUCompute::new or ptx::load_ptx_sync).
Settings loading (AppFullSettings::new).
Key log lines to search for:
"Loaded {} metadata entries" (should be >0 if metadata exists).
"Graph built successfully with {} nodes" (should show node count >0).
"PTX loaded successfully" (for GPU).
"SimParams successfully updated" (for physics).
Any "Failed to" or "Error" messages around these.
Common Startup Errors & Fixes:

Metadata Loading Failure:

Error: "Failed to load metadata" or "No metadata entries".
Cause: Empty or corrupted data/metadata.json or file permissions.
Fix:
Run cargo run --bin generate_metadata to regenerate.
Check permissions: chmod 644 data/metadata.json.
Verify file: ls -la data/metadata.json (should exist and be non-empty).
Graph Building Failure:

Error: "Failed to build graph from metadata" or "No nodes created".
Cause: Metadata empty, or build_from_metadata fails (e.g., invalid data).
Fix:
Ensure metadata has valid entries (at least 1-2 nodes).
Check actor logs for "semantic_analyzer" errors.
Test standalone: In src/bin/test_graph.rs, add a test to build from sample metadata.
GPU/PTX Loading Failure:

Error: "PTX validation failed" or "Failed to create CUDA device".
Cause: Missing PTX file, CUDA mismatch, or GPU not available.
Fix:
Run cargo run --bin generate_types to rebuild PTX.
Check env: echo $CUDA_ARCH (should be 75, 80, 86, etc.).
Verify GPU: nvidia-smi (should show device 0 available).
Docker: Ensure --gpus all flag and NVIDIA container toolkit installed.
Settings Loading Failure:

Error: "Failed to load AppFullSettings" or "Settings validation failed".
Cause: Invalid data/settings.yaml or missing required fields.
Fix:
Regenerate: cargo run --bin generate_settings.
Validate: cat data/settings.yaml (should have valid YAML, no syntax errors).