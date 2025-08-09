use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Custom build script for GPU kernel compilation integration
/// This integrates CUDA PTX compilation into the Cargo build process

fn main() {
    println!("cargo:rerun-if-changed=src/utils/*.cu");
    println!("cargo:rerun-if-changed=scripts/compile_ptx.sh");
    
    // Get build configuration
    let is_debug = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string()) == "debug";
    let target_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "86".to_string()); // Default to A6000
    
    println!("cargo:warning=GPU Build Configuration:");
    println!("cargo:warning=  Profile: {}", if is_debug { "debug" } else { "release" });
    println!("cargo:warning=  CUDA Architecture: SM_{}", target_arch);
    
    // Check if we're in a GPU-enabled build
    let features: Vec<String> = env::var("CARGO_FEATURE_GPU")
        .map(|_| vec!["gpu".to_string()])
        .unwrap_or_else(|_| vec![]);
    
    if !features.contains(&"gpu".to_string()) && env::var("CARGO_FEATURE_GPU").is_err() {
        println!("cargo:warning=GPU support disabled. Skipping CUDA compilation.");
        return;
    }
    
    // Verify CUDA toolkit is available
    if !cuda_available() {
        println!("cargo:warning=CUDA toolkit not found. GPU kernels will not be compiled.");
        println!("cargo:warning=The application will fall back to CPU-only mode.");
        return;
    }
    
    // Get the project root directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    let project_root = Path::new(&manifest_dir);
    
    // Paths
    let compile_script = project_root.join("scripts").join("compile_ptx.sh");
    let utils_dir = project_root.join("src").join("utils");
    
    // Verify the compilation script exists
    if !compile_script.exists() {
        panic!("PTX compilation script not found: {}", compile_script.display());
    }
    
    // Get list of CUDA kernel files
    let kernel_files = get_cuda_kernels(&utils_dir);
    
    if kernel_files.is_empty() {
        println!("cargo:warning=No CUDA kernel files found in {}", utils_dir.display());
        return;
    }
    
    println!("cargo:warning=Found {} CUDA kernels to compile", kernel_files.len());
    for kernel in &kernel_files {
        println!("cargo:warning=  - {}", kernel);
    }
    
    // Check if PTX files need recompilation
    if needs_recompilation(&utils_dir, &kernel_files) {
        println!("cargo:warning=PTX files need recompilation");
        compile_ptx_kernels(&compile_script, is_debug, &target_arch);
    } else {
        println!("cargo:warning=PTX files are up to date");
    }
    
    // Verify PTX files were created successfully
    verify_ptx_outputs(&utils_dir, &kernel_files);
    
    // Set up linking for CUDA runtime if available
    setup_cuda_linking();
}

/// Check if CUDA toolkit is available
fn cuda_available() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Get list of CUDA kernel files
fn get_cuda_kernels(utils_dir: &Path) -> Vec<String> {
    let mut kernels = Vec::new();
    
    if let Ok(entries) = fs::read_dir(utils_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(extension) = path.extension() {
                if extension == "cu" {
                    if let Some(stem) = path.file_stem() {
                        kernels.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }
    }
    
    kernels.sort();
    kernels
}

/// Check if PTX files need recompilation
fn needs_recompilation(utils_dir: &Path, kernel_files: &[String]) -> bool {
    for kernel in kernel_files {
        let cu_file = utils_dir.join(format!("{}.cu", kernel));
        let ptx_file = utils_dir.join(format!("{}.ptx", kernel));
        
        // If PTX file doesn't exist, we need compilation
        if !ptx_file.exists() {
            return true;
        }
        
        // If CU file is newer than PTX file, we need recompilation
        if let (Ok(cu_metadata), Ok(ptx_metadata)) = (cu_file.metadata(), ptx_file.metadata()) {
            if let (Ok(cu_modified), Ok(ptx_modified)) = (cu_metadata.modified(), ptx_metadata.modified()) {
                if cu_modified > ptx_modified {
                    return true;
                }
            }
        }
    }
    
    false
}

/// Compile PTX kernels using the shell script
fn compile_ptx_kernels(script_path: &Path, is_debug: bool, target_arch: &str) {
    println!("cargo:warning=Compiling CUDA kernels...");
    
    let mut cmd = Command::new("bash");
    cmd.arg(script_path);
    cmd.env("CUDA_ARCH", target_arch);
    
    if is_debug {
        cmd.arg("--debug");
        cmd.env("DEBUG", "1");
    }
    
    // Run the compilation
    let output = cmd.output().expect("Failed to execute PTX compilation script");
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        eprintln!("PTX compilation failed!");
        eprintln!("STDOUT:\n{}", stdout);
        eprintln!("STDERR:\n{}", stderr);
        
        panic!("CUDA kernel compilation failed. See output above for details.");
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("cargo:warning=PTX compilation completed successfully");
    
    // Print compilation summary
    for line in stdout.lines() {
        if line.contains("INFO") {
            println!("cargo:warning={}", line);
        }
    }
}

/// Verify PTX outputs were created successfully
fn verify_ptx_outputs(utils_dir: &Path, kernel_files: &[String]) {
    let mut missing_ptx = Vec::new();
    
    for kernel in kernel_files {
        let ptx_file = utils_dir.join(format!("{}.ptx", kernel));
        if !ptx_file.exists() {
            missing_ptx.push(kernel);
        } else {
            // Check file size
            if let Ok(metadata) = ptx_file.metadata() {
                if metadata.len() == 0 {
                    missing_ptx.push(kernel);
                }
            }
        }
    }
    
    if !missing_ptx.is_empty() {
        panic!("Failed to generate PTX files for: {}", missing_ptx.join(", "));
    }
    
    println!("cargo:warning=All PTX files verified successfully");
}

/// Setup CUDA runtime linking
fn setup_cuda_linking() {
    // Try to detect CUDA installation
    let cuda_paths = [
        "/usr/local/cuda/lib64",
        "/opt/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\lib\\x64",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\lib\\x64",
    ];
    
    for cuda_path in &cuda_paths {
        let path = Path::new(cuda_path);
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_path);
            break;
        }
    }
    
    // Link CUDA runtime libraries
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    
    // Set up environment for runtime
    if let Ok(cuda_root) = env::var("CUDA_ROOT") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
        println!("cargo:rustc-env=CUDA_ROOT={}", cuda_root);
    }
    
    // Pass target architecture to the runtime
    let target_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "86".to_string());
    println!("cargo:rustc-env=CUDA_ARCH={}", target_arch);
    
    println!("cargo:warning=CUDA linking configuration complete");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_get_cuda_kernels() {
        let temp_dir = TempDir::new().unwrap();
        let utils_dir = temp_dir.path().join("utils");
        fs::create_dir_all(&utils_dir).unwrap();
        
        // Create some test .cu files
        fs::write(utils_dir.join("test1.cu"), "// test kernel").unwrap();
        fs::write(utils_dir.join("test2.cu"), "// another kernel").unwrap();
        fs::write(utils_dir.join("not_cuda.txt"), "// not a cuda file").unwrap();
        
        let kernels = get_cuda_kernels(&utils_dir);
        
        assert_eq!(kernels.len(), 2);
        assert!(kernels.contains(&"test1".to_string()));
        assert!(kernels.contains(&"test2".to_string()));
    }
    
    #[test]
    fn test_needs_recompilation() {
        let temp_dir = TempDir::new().unwrap();
        let utils_dir = temp_dir.path().join("utils");
        fs::create_dir_all(&utils_dir).unwrap();
        
        // Create a .cu file and corresponding .ptx file
        fs::write(utils_dir.join("test.cu"), "// test kernel").unwrap();
        fs::write(utils_dir.join("test.ptx"), "// compiled ptx").unwrap();
        
        let kernels = vec!["test".to_string()];
        
        // Should not need recompilation if PTX exists and is newer
        assert!(!needs_recompilation(&utils_dir, &kernels));
        
        // Should need recompilation if PTX doesn't exist
        fs::remove_file(utils_dir.join("test.ptx")).unwrap();
        assert!(needs_recompilation(&utils_dir, &kernels));
    }
}