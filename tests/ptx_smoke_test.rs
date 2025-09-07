// PTX cold-start smoke tests
// Run only when RUN_GPU_SMOKE=1 to avoid failing on non-GPU CI.
// Validates that PTX loads, Module::from_ptx succeeds, and required kernels exist.
// Also sanity-checks UnifiedGPUCompute::new for a tiny graph.

#![allow(unused_imports)]

use cust::device::Device;
use cust::context::Context;
use cust::module::Module;

fn should_run() -> bool {
    std::env::var("RUN_GPU_SMOKE").ok().as_deref() == Some("1")
}

fn try_create_context() -> Option<Context> {
    match Device::get_device(0) {
        Ok(device) => match Context::new(device) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("[PTX-SMOKE] Failed to create CUDA context: {e}");
                None
            }
        },
        Err(e) => {
            eprintln!("[PTX-SMOKE] No CUDA device(0): {e}");
            None
        }
    }
}

#[test]
fn ptx_module_loads_and_kernels_present() {
    if !should_run() {
        eprintln!("[PTX-SMOKE] Skipping (set RUN_GPU_SMOKE=1 to enable)");
        return;
    }

    // 1) Load PTX (build-time path or runtime nvcc fallback)
    let ptx = match crate::utils::ptx::load_ptx_sync() {
        Ok(s) => {
            println!("[PTX-SMOKE] PTX loaded ({} bytes), arch=sm_{}", s.len(), crate::utils::ptx::effective_cuda_arch());
            s
        }
        Err(e) => {
            panic!("[PTX-SMOKE] Failed to load PTX: {e}");
        }
    };

    // 2) Create CUDA context
    let _ctx = match try_create_context() {
        Some(ctx) => ctx,
        None => {
            panic!("[PTX-SMOKE] Cannot create CUDA context - ensure NVIDIA drivers and GPU availability.");
        }
    };

    // 3) Create module from PTX
    let module = match Module::from_ptx(&ptx, &[]) {
        Ok(m) => {
            println!("[PTX-SMOKE] PTX module created successfully");
            m
        }
        Err(e) => {
            // Surface diagnosis like the runtime does
            let diag = crate::utils::gpu_diagnostics::diagnose_ptx_error(&format!("Module::from_ptx error: {e}"));
            panic!("[PTX-SMOKE] Module::from_ptx failed: {e}\n{diag}");
        }
    };

    // 4) Validate expected kernels exist
    let expected = [
        "build_grid_kernel",
        "compute_cell_bounds_kernel",
        "force_pass_kernel",
        "integrate_pass_kernel",
        "relaxation_step_kernel",
    ];

    for k in expected {
        assert!(
            module.get_function(k).is_ok(),
            "[PTX-SMOKE] Missing expected kernel symbol: {k}"
        );
    }

    println!("[PTX-SMOKE] ✅ All expected kernels present");
}

#[test]
fn unified_gpu_compute_new_smoke() {
    if !should_run() {
        eprintln!("[PTX-SMOKE] Skipping (set RUN_GPU_SMOKE=1 to enable)");
        return;
    }

    // Try building a minimal UnifiedGPUCompute context to catch wiring issues end-to-end.
    let ptx = match crate::utils::ptx::load_ptx_sync() {
        Ok(s) => s,
        Err(e) => {
            panic!("[PTX-SMOKE] Failed to load PTX: {e}");
        }
    };

    // Tiny graph: 16 nodes, 0 edges
    match crate::utils::unified_gpu_compute::UnifiedGPUCompute::new(16, 0, &ptx) {
        Ok(_gpu) => {
            println!("[PTX-SMOKE] ✅ UnifiedGPUCompute::new succeeded for tiny graph");
            // Drop to release resources
        }
        Err(e) => {
            panic!("[PTX-SMOKE] UnifiedGPUCompute::new failed: {e}");
        }
    };
}