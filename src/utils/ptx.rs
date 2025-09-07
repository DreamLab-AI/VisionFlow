// ptx.rs - unified PTX loading and runtime compilation utilities
// This module centralizes PTX acquisition for CUDA kernel modules.
// Strategy:
// 1) Prefer build-time PTX pointed to by VISIONFLOW_PTX_PATH (set by build.rs).
// 2) If unavailable, corrupted, or in Docker (DOCKER_ENV set), compile on-the-fly via nvcc -ptx.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_CUDA_ARCH: &str = "75";
pub const CUDA_ARCH_ENV: &str = "CUDA_ARCH";
pub const DOCKER_ENV_VAR: &str = "DOCKER_ENV";

// Build-time exported path from build.rs (if present)
pub static COMPILED_PTX_PATH: Option<&'static str> = option_env!("VISIONFLOW_PTX_PATH");

/// Get the PTX path exported by build.rs, if available.
pub fn get_compiled_ptx_path() -> Option<PathBuf> {
    COMPILED_PTX_PATH.map(PathBuf::from)
}

/// Resolve the effective CUDA arch used for fallback compilation.
pub fn effective_cuda_arch() -> String {
    std::env::var(CUDA_ARCH_ENV).unwrap_or_else(|_| DEFAULT_CUDA_ARCH.to_string())
}

/// Validate that the PTX content looks structurally sound.
fn validate_ptx(ptx: &str) -> Result<(), String> {
    if !ptx.contains(".version") {
        return Err("PTX validation failed: missing .version directive".into());
    }
    if !ptx.contains(".target") {
        return Err("PTX validation failed: missing .target directive".into());
    }
    Ok(())
}

/// Load PTX content, preferring the build-time artifact. Falls back to runtime compilation.
/// In Docker environments (DOCKER_ENV set), always compile at runtime to avoid path mismatches.
pub fn load_ptx_sync() -> Result<String, String> {
    // Always compile in Docker to avoid mismatched build/runtime paths
    if std::env::var(DOCKER_ENV_VAR).is_ok() {
        return compile_ptx_fallback_sync();
    }

    if let Some(path) = get_compiled_ptx_path() {
        match fs::read_to_string(&path) {
            Ok(content) => {
                if let Err(e) = validate_ptx(&content) {
                    eprintln!("Warning: build-time PTX at {} failed validation: {}. Falling back to runtime compile.", path.display(), e);
                    compile_ptx_fallback_sync()
                } else {
                    Ok(content)
                }
            }
            Err(read_err) => {
                eprintln!("Warning: failed to read build-time PTX at {}: {}. Falling back to runtime compile.", path.display(), read_err);
                compile_ptx_fallback_sync()
            }
        }
    } else {
        eprintln!("Warning: VISIONFLOW_PTX_PATH not set at build time. Falling back to runtime compile.");
        compile_ptx_fallback_sync()
    }
}

/// Async wrapper for load_ptx_sync. Note: currently uses a direct call; upgrade to spawn_blocking if needed.
pub async fn load_ptx() -> Result<String, String> {
    // If desired, switch to tokio::task::spawn_blocking to avoid blocking executors:
    // tokio::task::spawn_blocking(|| load_ptx_sync()).await.map_err(|e| e.to_string())?
    load_ptx_sync()
}

/// Compile the CUDA source to PTX on-the-fly using nvcc.
/// Uses DEFAULT_CUDA_ARCH (sm_75) unless overridden by CUDA_ARCH env var.
pub fn compile_ptx_fallback_sync() -> Result<String, String> {
    let arch = effective_cuda_arch();

    // Locate the CUDA source relative to the crate root
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let cu_path = Path::new(manifest_dir)
        .join("src")
        .join("utils")
        .join("visionflow_unified.cu");

    if !cu_path.exists() {
        return Err(format!(
            "CUDA source not found at {}. Ensure the path is correct.",
            cu_path.display()
        ));
    }

    let out_path = std::env::temp_dir().join("visionflow_unified.ptx");

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
            "nvcc failed (code {:?}). Command: nvcc -ptx -std=c++17 -arch=sm_{} {} -o {}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            arch,
            cu_path.display(),
            out_path.display(),
            stdout,
            stderr
        ));
    }

    let ptx_content = fs::read_to_string(&out_path)
        .map_err(|e| format!("Failed to read generated PTX at {}: {}", out_path.display(), e))?;

    validate_ptx(&ptx_content)?;
    Ok(ptx_content)
}