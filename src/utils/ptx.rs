// ptx.rs - unified PTX loading and runtime compilation utilities
// This module centralizes PTX acquisition for CUDA kernel modules.
// Strategy:
// 1) Prefer build-time PTX pointed to by environment variables (set by build.rs).
// 2) If unavailable, corrupted, or in Docker (DOCKER_ENV set), compile on-the-fly via nvcc -ptx.
// 3) Support multiple PTX modules for different kernel sets.

use log::{info, warn};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_CUDA_ARCH: &str = "75";
pub const CUDA_ARCH_ENV: &str = "CUDA_ARCH";
pub const DOCKER_ENV_VAR: &str = "DOCKER_ENV";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PTXModule {
    VisionflowUnified,
    GpuClusteringKernels,
    DynamicGrid,
    GpuAabbReduction,
    GpuLandmarkApsp,
    SsspCompact,
    VisionflowUnifiedStability,
    OntologyConstraints,
    Pagerank,
    GpuConnectedComponents,
}

impl PTXModule {
    pub fn source_file(&self) -> &'static str {
        match self {
            PTXModule::VisionflowUnified => "visionflow_unified.cu",
            PTXModule::GpuClusteringKernels => "gpu_clustering_kernels.cu",
            PTXModule::DynamicGrid => "dynamic_grid.cu",
            PTXModule::GpuAabbReduction => "gpu_aabb_reduction.cu",
            PTXModule::GpuLandmarkApsp => "gpu_landmark_apsp.cu",
            PTXModule::SsspCompact => "sssp_compact.cu",
            PTXModule::VisionflowUnifiedStability => "visionflow_unified_stability.cu",
            PTXModule::OntologyConstraints => "ontology_constraints.cu",
            PTXModule::Pagerank => "pagerank.cu",
            PTXModule::GpuConnectedComponents => "gpu_connected_components.cu",
        }
    }

    pub fn env_var(&self) -> &'static str {
        match self {
            PTXModule::VisionflowUnified => "VISIONFLOW_UNIFIED_PTX_PATH",
            PTXModule::GpuClusteringKernels => "GPU_CLUSTERING_KERNELS_PTX_PATH",
            PTXModule::DynamicGrid => "DYNAMIC_GRID_PTX_PATH",
            PTXModule::GpuAabbReduction => "GPU_AABB_REDUCTION_PTX_PATH",
            PTXModule::GpuLandmarkApsp => "GPU_LANDMARK_APSP_PTX_PATH",
            PTXModule::SsspCompact => "SSSP_COMPACT_PTX_PATH",
            PTXModule::VisionflowUnifiedStability => "VISIONFLOW_UNIFIED_STABILITY_PTX_PATH",
            PTXModule::OntologyConstraints => "ONTOLOGY_CONSTRAINTS_PTX_PATH",
            PTXModule::Pagerank => "PAGERANK_PTX_PATH",
            PTXModule::GpuConnectedComponents => "GPU_CONNECTED_COMPONENTS_PTX_PATH",
        }
    }

    pub fn all_modules() -> Vec<PTXModule> {
        vec![
            PTXModule::VisionflowUnified,
            PTXModule::GpuClusteringKernels,
            PTXModule::DynamicGrid,
            PTXModule::GpuAabbReduction,
            PTXModule::GpuLandmarkApsp,
            PTXModule::SsspCompact,
            PTXModule::VisionflowUnifiedStability,
            PTXModule::OntologyConstraints,
            PTXModule::Pagerank,
            PTXModule::GpuConnectedComponents,
        ]
    }
}

// Build-time exported paths from build.rs (if present)
pub static COMPILED_PTX_PATH: Option<&'static str> = option_env!("VISIONFLOW_UNIFIED_PTX_PATH");

pub fn get_compiled_ptx_path(module: PTXModule) -> Option<PathBuf> {
    std::env::var(module.env_var()).ok().map(PathBuf::from)
}

pub fn get_compiled_ptx_path_legacy() -> Option<PathBuf> {
    COMPILED_PTX_PATH.map(PathBuf::from)
}

pub fn effective_cuda_arch() -> String {
    std::env::var(CUDA_ARCH_ENV).unwrap_or_else(|_| DEFAULT_CUDA_ARCH.to_string())
}

/// Maximum PTX ISA version the CUDA 13.0 driver (version 13000) can JIT-compile.
/// CUDA toolkit 13.1 emits `.version 9.1` PTX, but driver 13.0 only supports up to 9.0.
/// Downgrading the header is safe when the code uses no ISA-9.1-specific instructions
/// (verified empirically: our kernels use only basic ops available since ISA 7.x).
const MAX_DRIVER_PTX_ISA: &str = "9.0";

/// Downgrades the PTX ISA version header when it exceeds what the installed driver
/// can handle.  This avoids `CUDA_ERROR_INVALID_PTX` (222) on systems where the
/// toolkit is newer than the driver.
pub fn downgrade_ptx_isa_if_needed(ptx: String) -> String {
    // Fast path: already compatible
    if !ptx.contains(".version 9.1") {
        return ptx;
    }
    let fixed = ptx.replacen(".version 9.1", &format!(".version {}", MAX_DRIVER_PTX_ISA), 1);
    info!(
        "PTX ISA downgrade: .version 9.1 -> .version {} (driver compatibility)",
        MAX_DRIVER_PTX_ISA
    );
    fixed
}

/// Validates PTX assembly code structure
pub fn validate_ptx(ptx: &str) -> Result<(), String> {
    if !ptx.contains(".version") {
        return Err("PTX validation failed: missing .version directive".into());
    }
    if !ptx.contains(".target") {
        return Err("PTX validation failed: missing .target directive".into());
    }
    if !ptx.contains(".entry ") {
        return Err("PTX validation failed: missing kernel entry points (.entry directive)".into());
    }
    Ok(())
}

pub fn load_ptx_module_sync(module: PTXModule) -> Result<String, String> {
    info!("load_ptx_module_sync: Loading PTX for {:?}", module);

    let raw = load_ptx_module_sync_raw(module)?;
    Ok(downgrade_ptx_isa_if_needed(raw))
}

fn load_ptx_module_sync_raw(module: PTXModule) -> Result<String, String> {
    if std::env::var(DOCKER_ENV_VAR).is_ok() {
        info!("Docker environment detected, checking for pre-compiled PTX first");

        if let Ok(content) = load_precompiled_ptx(module) {
            return Ok(content);
        }

        info!("Pre-compiled PTX not found, using runtime compilation");
        return compile_ptx_fallback_sync_module(module);
    }

    if let Some(path) = get_compiled_ptx_path(module) {
        match fs::read_to_string(&path) {
            Ok(content) => {
                if let Err(e) = validate_ptx(&content) {
                    warn!(
                        "Build-time PTX at {} failed validation: {}. Trying alternatives.",
                        path.display(),
                        e
                    );
                } else {
                    info!("Loaded build-time PTX from {}", path.display());
                    return Ok(content);
                }
            }
            Err(read_err) => {
                warn!(
                    "Failed to read build-time PTX at {}: {}. Trying alternatives.",
                    path.display(),
                    read_err
                );
            }
        }
    }

    if let Ok(content) = load_precompiled_ptx(module) {
        return Ok(content);
    }

    warn!(
        "No pre-compiled PTX found for {:?}. Falling back to runtime compile.",
        module
    );
    compile_ptx_fallback_sync_module(module)
}

fn load_precompiled_ptx(module: PTXModule) -> Result<String, String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let ptx_file = module.source_file().replace(".cu", ".ptx");

    let ptx_paths = vec![
        PathBuf::from(manifest_dir)
            .join("src/utils/ptx")
            .join(&ptx_file),
        PathBuf::from("/app/src/utils/ptx").join(&ptx_file),
        PathBuf::from("./src/utils/ptx").join(&ptx_file),
    ];

    for path in ptx_paths {
        if let Ok(content) = fs::read_to_string(&path) {
            if validate_ptx(&content).is_ok() {
                info!("Loaded pre-compiled PTX from {}", path.display());
                return Ok(content);
            }
        }
    }

    Err(format!("Pre-compiled PTX not found for {:?}", module))
}

pub fn load_ptx_sync() -> Result<String, String> {
    load_ptx_module_sync(PTXModule::VisionflowUnified)
}

pub fn load_all_ptx_modules_sync() -> Result<HashMap<PTXModule, String>, String> {
    let mut modules = HashMap::new();
    let mut failed: Vec<(PTXModule, String)> = Vec::new();

    for module in PTXModule::all_modules() {
        match load_ptx_module_sync(module) {
            Ok(content) => {
                info!(
                    "Successfully loaded PTX for {:?}, size: {} bytes",
                    module,
                    content.len()
                );
                modules.insert(module, content);
            }
            Err(e) => {
                warn!(
                    "Failed to load PTX module {:?}: {}. Feature will be unavailable.",
                    module, e
                );
                failed.push((module, e));
            }
        }
    }

    if modules.is_empty() {
        return Err(format!(
            "No PTX modules loaded successfully. Failures: {:?}",
            failed
        ));
    }

    info!(
        "Loaded {}/{} PTX modules",
        modules.len(),
        modules.len() + failed.len()
    );

    Ok(modules)
}

pub async fn load_ptx() -> Result<String, String> {
    
    
    load_ptx_sync()
}

pub fn compile_ptx_fallback_sync_module(module: PTXModule) -> Result<String, String> {
    info!(
        "compile_ptx_fallback_sync_module: Starting runtime PTX compilation for {:?}",
        module
    );
    let arch = effective_cuda_arch();
    info!("Using CUDA architecture: sm_{}", arch);

    
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let cu_path = Path::new(manifest_dir)
        .join("src")
        .join("utils")
        .join(module.source_file());

    if !cu_path.exists() {
        return Err(format!(
            "CUDA source not found at {}. Ensure the path is correct.",
            cu_path.display()
        ));
    }

    // Use unique temp filename to avoid race conditions
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let ptx_file = format!(
        "ptx_{}_{}.ptx",
        module.source_file().replace(".cu", ""),
        unique
    );
    let out_path = std::env::temp_dir().join(&ptx_file);

    let nvcc = "nvcc";
    let arch_flag = format!("-arch=sm_{}", arch);

    let output = Command::new(nvcc)
        .args(["-ptx", "-std=c++17"])
        .arg(arch_flag)
        .arg(&cu_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .map_err(|e| format!("Failed to spawn nvcc: {}", e))?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "nvcc failed for {:?} (code {:?}). Command: nvcc -ptx -std=c++17 -arch=sm_{} {} -o {}\nstdout:\n{}\nstderr:\n{}",
            module,
            output.status.code(),
            arch,
            cu_path.display(),
            out_path.display(),
            stdout,
            stderr
        ));
    }

    let ptx_content = fs::read_to_string(&out_path).map_err(|e| {
        format!(
            "Failed to read generated PTX at {}: {}",
            out_path.display(),
            e
        )
    })?;

    validate_ptx(&ptx_content)?;
    info!(
        "Successfully compiled PTX for {:?}, size: {} bytes",
        module,
        ptx_content.len()
    );
    Ok(ptx_content)
}

pub fn compile_ptx_fallback_sync() -> Result<String, String> {
    compile_ptx_fallback_sync_module(PTXModule::VisionflowUnified)
}
