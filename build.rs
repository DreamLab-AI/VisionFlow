use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Simplified build script for unified GPU kernel compilation
fn main() {
    println!("cargo:rerun-if-changed=src/utils/visionflow_unified.cu");
    println!("cargo:rerun-if-changed=scripts/compile_unified_ptx.sh");
    
    let is_debug = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string()) == "debug";
    let target_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "86".to_string());
    
    println!("cargo:warning=VisionFlow Unified GPU Build");
    println!("cargo:warning=  Profile: {}", if is_debug { "debug" } else { "release" });
    println!("cargo:warning=  CUDA Architecture: SM_{}", target_arch);
    
    // Check if GPU feature is enabled
    if env::var("CARGO_FEATURE_GPU").is_err() {
        println!("cargo:warning=GPU support disabled. Skipping CUDA compilation.");
        return;
    }
    
    // Verify CUDA toolkit
    if !cuda_available() {
        println!("cargo:warning=CUDA toolkit not found. GPU acceleration will not be available.");
        return;
    }
    
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let project_root = Path::new(&manifest_dir);
    
    // Check for the unified kernel
    let unified_kernel = project_root.join("src").join("utils").join("visionflow_unified.cu");
    let unified_ptx = project_root.join("src").join("utils").join("ptx").join("visionflow_unified.ptx");
    
    if !unified_kernel.exists() {
        panic!("Unified kernel not found: {}", unified_kernel.display());
    }
    
    // Check if PTX needs recompilation
    let needs_compile = if unified_ptx.exists() {
        let cu_modified = fs::metadata(&unified_kernel)
            .and_then(|m| m.modified())
            .ok();
        let ptx_modified = fs::metadata(&unified_ptx)
            .and_then(|m| m.modified())
            .ok();
        
        match (cu_modified, ptx_modified) {
            (Some(cu_time), Some(ptx_time)) => cu_time > ptx_time,
            _ => true,
        }
    } else {
        true
    };
    
    if needs_compile {
        println!("cargo:warning=Compiling unified PTX...");
        compile_unified_ptx(&project_root, is_debug, &target_arch);
    } else {
        println!("cargo:warning=Unified PTX is up to date");
    }
    
    // Verify PTX exists
    if !unified_ptx.exists() {
        panic!("Failed to generate unified PTX file");
    }
    
    let ptx_size = fs::metadata(&unified_ptx)
        .map(|m| m.len())
        .unwrap_or(0);
    
    println!("cargo:warning=Unified PTX ready: {} bytes", ptx_size);
    
    // Set up CUDA linking if available
    setup_cuda_linking();
}

fn cuda_available() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn compile_unified_ptx(project_root: &Path, is_debug: bool, target_arch: &str) {
    let utils_dir = project_root.join("src").join("utils");
    let ptx_dir = utils_dir.join("ptx");
    
    // Create PTX directory
    fs::create_dir_all(&ptx_dir).expect("Failed to create PTX directory");
    
    let cu_file = utils_dir.join("visionflow_unified.cu");
    let ptx_file = ptx_dir.join("visionflow_unified.ptx");
    
    // Build nvcc command
    let mut cmd = Command::new("nvcc");
    cmd.arg("-ptx")
        .arg("-arch").arg(format!("sm_{}", target_arch));
    
    if !is_debug {
        cmd.arg("-O3")
            .arg("--use_fast_math")
            .arg("--restrict")
            .arg("--ftz=true")
            .arg("--prec-div=false")
            .arg("--prec-sqrt=false");
    } else {
        cmd.arg("-O2");
    }
    
    cmd.arg(&cu_file)
        .arg("-o").arg(&ptx_file);
    
    println!("cargo:warning=Running: {:?}", cmd);
    
    let output = cmd.output().expect("Failed to execute nvcc");
    
    if !output.status.success() {
        eprintln!("CUDA compilation failed!");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Failed to compile unified CUDA kernel");
    }
    
    println!("cargo:warning=Successfully compiled unified kernel");
}

fn setup_cuda_linking() {
    // Try to find CUDA installation
    let cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
    ];
    
    for cuda_path in &cuda_paths {
        let path = Path::new(cuda_path);
        if path.exists() {
            let lib_path = path.join("lib64");
            if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib_path.display());
            }
            break;
        }
    }
    
    // Link CUDA runtime (optional - only if using CUDA runtime API)
    // println!("cargo:rustc-link-lib=cudart");
}