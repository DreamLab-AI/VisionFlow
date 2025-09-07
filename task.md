Project Brief: GPU Analytics Engine Maturation — Execution Plan v2
Objective:
Transition all analytics to a fully GPU-accelerated engine (CUDA), eliminating CPU fallbacks and mock implementations once GPU parity is proven.

Phase 0: Architectural Decision and Build Pipeline Hardening
Decision: Select Path A — extend the active UnifiedGPUCompute engine.
Rationale:
- Lowest delivery risk; builds on the working physics pipeline in [src/utils/unified_gpu_compute.rs](src/utils/unified_gpu_compute.rs).
- Incremental kernel additions for analytics without a wholesale data model migration.
- Option to add adapters later to the richer data model in [src/gpu/visual_analytics.rs](src/gpu/visual_analytics.rs) if required.
Re-evaluation trigger: Consider Path B only if a planned feature demonstrably needs VisualAnalyticsGPU semantics that cannot be mapped to SoA buffers with acceptable performance.
Record of decision: Path A is the baseline for this plan; Path B tasks are parked as optional deltas.

Build/Runtime PTX pipeline (Phase 0 blocker)
Actions:
- Verify env export of VISIONFLOW_PTX_PATH in [build.rs](build.rs:111) and keep the build script linking consistent in [build.rs](build.rs:1).
- Align runtime PTX loading via [Module::from_ptx()](src/utils/unified_gpu_compute.rs:119).
- Unify and guard the on-the-fly fallback [compile_ptx_fallback()](src/actors/graph_actor.rs:2178) so it only triggers when the env path is absent or invalid.
- Add diagnostics in [src/utils/gpu_diagnostics.rs](src/utils/gpu_diagnostics.rs) to surface precise PTX errors.
Validation gates:
- Cold start should not surface “device kernel image is invalid”.
- PTX files are found through VISIONFLOW_PTX_PATH or the fallback compiles successfully on first run.
- Per-kernel launch succeeds at least once under CI smoke tests.

Phase 1: Stabilize and Re-enable Existing Features
1.1 Stress Majorization
Scope:
- Re-enable scheduling in [execute_stress_majorization_step()](src/actors/graph_actor.rs:500) by lowering [AdvancedParams::stress_step_interval_frames](src/models/constraints.rs:161) from u32::MAX to a safe cadence (e.g., 600 frames).
- Keep the CPU implementation as the authority in [StressMajorizationSolver](src/physics/stress_majorization.rs:86). GPU porting is optional and deferred.
Safety controls:
- Clamp delta positions per iteration; cap displacement per node (e.g., <= 5% of layout extent).
- Reject NaN/Inf; if detected, revert to last stable snapshot and exponentially back-off step size.
- Bound the layout domain (AABB) during optimization to avoid “position explosions”.
Observability:
- Emit iteration residuals and max displacement to performance metrics.
Validation gate:
- Over 5 runs on representative graphs, no divergence; improves stress (Kamada-Kawai style) by >= 10% vs baseline, frame-time overhead < 10 ms at cadence.

1.2 Semantic Constraints end-to-end
Scope:
- Ensure constraint data is GPU-transferable: update struct in [ConstraintData](src/models/constraints.rs:211) to use GPU-safe marker derives (Pod/Zeroable) equivalent to those used in SimParams.
- Implement host-side upload via [UnifiedGPUCompute::set_constraints()](src/utils/unified_gpu_compute.rs:336).
- Add constraint force accumulation in [force_pass_kernel()](src/utils/visionflow_unified.cu:178), balancing with repulsion/springs.
- Re-enable handler without clearing in [UpdateConstraints](src/actors/gpu_compute_actor.rs:695).
Safety controls:
- Scale constraint forces relative to local degree/edge weights; hard-cap force per node.
- Progressive activation: ramp from 0% to 100% over N frames to avoid “bouncing”.
Validation gate:
- With constraints on, no sustained oscillation; average kinetic energy returns to baseline within 2 seconds; constraint violations decrease monotonically in first 200 frames.

1.3 Finalize SSSP integration
Scope:
- Confirm kernel [relaxation_step_kernel()](src/utils/visionflow_unified.cu:301) and host path [UnifiedGPUCompute::run_sssp()](src/utils/unified_gpu_compute.rs:520).
- Gate spring adjustment with [FeatureFlags::ENABLE_SSSP_SPRING_ADJUST](src/models/simulation_params.rs:92) and expose API toggle in [src/handlers/api_handler/analytics/mod.rs](src/handlers/api_handler/analytics/mod.rs:1154).
Validation gate:
- SSSP distances validated on small graphs against CPU Dijkstra (tolerance 1e-5); enabling spring adjust improves edge length variance by >= 10% without destabilizing layout.

1.4 Spatial hashing robustness
Scope:
- Replace fixed allocations keyed to max grid cells near [src/utils/unified_gpu_compute.rs](src/utils/unified_gpu_compute.rs:156) with dynamic sizing based on node count and scene extent.
- Auto-tune grid cell size to target 4–16 neighbors per cell; validate [build_grid_kernel()](src/utils/visionflow_unified.cu:108) and [compute_cell_bounds_kernel()](src/utils/visionflow_unified.cu:134).
Validation gate:
- Hashing efficiency (non-empty cells / total) stays within 0.2–0.6 across workloads; repulsion pass time variance < 20% under node count doubling.

1.5 Buffer resizing strategy
Scope:
- Implement [UnifiedGPUCompute::resize_buffers()](src/utils/unified_gpu_compute.rs:323) with growth factor (e.g., 1.5x) and no data loss; prefer in-place reallocation with temporary staging if needed.
- Update resize flow in [update_graph_data_internal()](src/actors/gpu_compute_actor.rs:330) to carry over positions/velocities seamlessly.
Validation gate:
- Resizing during live simulation produces no panics, no NaNs, and preserves positions within 1e-6 relative error.

Phase 2: Port Mocked Analytics to GPU
2.1 GPU-Accelerated Clustering (K-means MVP)
Algorithm plan:
- Initialization: k-means++ or random seeding on GPU; maintain centroids in device memory.
- Assignment kernel: compute distance of each node to k centroids; write cluster_id per node.
- Update kernel: parallel reduce to sum positions per cluster and counts; update centroids.
- Convergence: stop when centroid delta < epsilon or max_iters reached; expose deterministic seed for reproducibility.
Implementation:
- Add device buffers for centroids, cluster assignments, and temporary reductions in [src/utils/unified_gpu_compute.rs](src/utils/unified_gpu_compute.rs).
- Implement kernels in [src/utils/visionflow_unified.cu](src/utils/visionflow_unified.cu).
- Wire through message [PerformGPUClustering](src/actors/messages.rs:734) via the handler path in [src/actors/gpu_compute_actor.rs](src/actors/gpu_compute_actor.rs:953).
- Remove CPU fallback paths in [src/handlers/api_handler/analytics/clustering.rs](src/handlers/api_handler/analytics/clustering.rs).
Validation gate:
- On labeled benchmark graphs, ARI/NMI within 2% of CPU reference at equal k; runtime 10–50x faster for 100k nodes; stable across three seeds.

2.2 Community Detection (Phase 2b)
Plan:
- Implement label propagation first (fully parallel friendly); consider Louvain later for modularity maximization.
- Share CSR and temporary buffers; expose results through the same API as K-means for UI reuse.

2.3 GPU Anomaly Detection MVP
Algorithm options:
- Local Outlier Factor using spatial grid/graph neighbors; or z-score on degree/centrality/velocity residuals.
Implementation:
- Add device buffers for anomaly scores in [src/utils/unified_gpu_compute.rs](src/utils/unified_gpu_compute.rs).
- Implement scoring kernels in [src/utils/visionflow_unified.cu](src/utils/visionflow_unified.cu); schedule periodically.
- Introduce an actor message to trigger detection and return top-N; replace the simulated loop in [src/handlers/api_handler/analytics/anomaly.rs](src/handlers/api_handler/analytics/anomaly.rs:18).
Validation gate:
- On synthetic injections, AUC >= 0.85 for detectable anomalies; latency < 100 ms for 100k nodes.

Phase 3: Advanced Integration and Observability
3.1 Real “AI Insights”
- Replace mocked strings by synthesizing insights from clustering and anomaly outputs in the analytics module in [src/handlers/api_handler/analytics/mod.rs](src/handlers/api_handler/analytics/mod.rs).
- Examples: “Detected 5 clusters; largest cohesion 0.85”, “Node X anomaly score 0.95; isolated but central”.
3.2 Telemetry and auto-balance
- Expose GPU-side metrics (kinetic energy, hashing efficiency, active constraints) and feed into [GraphServiceActor::update_node_positions()](src/actors/graph_actor.rs:832).
- Add per-kernel timings and memory stats; surface via [get_performance_stats](src/handlers/api_handler/analytics/mod.rs:520) and extend [get_gpu_metrics](src/handlers/api_handler/analytics/mod.rs:642).
Validation gate:
- Metrics endpoints live; kernel timing overhead < 2%; auto-balance improves frame time jitter by >= 10%.

Phase 4: Deprecation and Cleanup
Scope:
- Delete simulated CPU functions and fallbacks in [src/handlers/api_handler/analytics/clustering.rs](src/handlers/api_handler/analytics/clustering.rs) and [src/handlers/api_handler/analytics/anomaly.rs](src/handlers/api_handler/analytics/anomaly.rs).
- If Path A remains, deprecate [src/gpu/visual_analytics.rs](src/gpu/visual_analytics.rs) to avoid confusion.
- Remove obsolete FIXME/TODOs related to GPU parity; update docs.

Success Metrics (Program-level)
- No CPU fallback paths remain; GPU is mandatory.
- Steady-state frame time meets budget on target hardware; no “device kernel image is invalid” in normal runs.
- Clustering accuracy: ARI/NMI within 2–5% of CPU references on small graphs.
- Anomaly detection: AUC >= 0.85 on synthetic tests; top-N anomalies stable across runs (deterministic seed).
- Startup: PTX pipeline stable; cold start within acceptable latency (< 3 s).
- Memory: No leaks; VRAM usage scales linearly with nodes/edges within projections.

Risks and Mitigations
- PTX/driver mismatches: lock NVCC version; add explicit diagnostics in [src/utils/gpu_diagnostics.rs](src/utils/gpu_diagnostics.rs); CI GPU runners execute a smoke kernel.
- Numerical instability: clamps, bounded AABB, adaptive step sizes; fall back to last stable snapshot.
- OOM under large graphs: capacity growth and back-pressure; optional batch processing for analytics; expose clear 429/503 API responses.
- Kernel nondeterminism: fixed seeds; avoid racey atomics in reductions (use segmented reductions via Thrust where possible).
- Actor concurrency: single-threaded GPU command queue per device; serialize kernel launches; back-pressure via mailbox limits.

Rollout and Feature Flags
- Introduce analytics feature flags and expose toggles via API in [src/handlers/api_handler/analytics/mod.rs](src/handlers/api_handler/analytics/mod.rs:1154).
- Default off until validation gates pass; then flip to on in staged environments.
- Telemetry-first rollout; alert on regressions; documented rollback (disable flags without process restarts).

Documentation and CI
- Update this plan and developer CUDA setup/troubleshooting in docs (gpu-analytics.md, cuda-parameters.md).
- Ensure [build.rs](build.rs:1) runs in CI GPU jobs; provide informative failures when GPU is unavailable.
- Keep and extend safety tests in [tests/gpu_safety_tests.rs](tests/gpu_safety_tests.rs).

Deliverables per Phase (DoD)
Phase 0:
- PTX pipeline green; diagnostics actionable; decision recorded (Path A).
Phase 1:
- Stress majorization enabled and stable; constraints applied on-GPU; SSSP feature-gated and validated; spatial hash and resizing robust.
Phase 2:
- K-means GPU clustering live; community detection prototype; anomaly detection MVP live; API returns real results.
Phase 3:
- Insights generated from real analytics; telemetry endpoints expose timings and GPU metrics; auto-balance uses GPU metrics.
Phase 4:
- CPU fallbacks removed; deprecated modules cleaned; docs updated.