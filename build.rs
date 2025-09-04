use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Only rebuild if CUDA files change
    println!("cargo:rerun-if-changed=src/utils/visionflow_unified.cu");
    println!("cargo:rerun-if-changed=build.rs");
    
    // Get build configuration
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    // Determine CUDA architecture
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "75".to_string());
    
    // Paths
    let cuda_src = Path::new("src/utils/visionflow_unified.cu");
    let ptx_output = PathBuf::from(&out_dir).join("visionflow_unified.ptx");
    let obj_output = PathBuf::from(&out_dir).join("thrust_wrapper.o");
    
    // Compile CUDA kernel to PTX
    println!("Compiling CUDA kernel to PTX...");
    let ptx_status = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-arch", &format!("sm_{}", cuda_arch),
            "-o", ptx_output.to_str().unwrap(),
            cuda_src.to_str().unwrap(),
            "--use_fast_math",
            "-O3",
        ])
        .status()
        .expect("Failed to compile CUDA kernel to PTX");
    
    if !ptx_status.success() {
        panic!("CUDA PTX compilation failed");
    }
    
    // Compile Thrust wrapper functions to object file
    println!("Compiling Thrust wrapper functions...");
    let obj_status = Command::new("nvcc")
        .args(&[
            "-c",
            "-arch", &format!("sm_{}", cuda_arch),
            "-o", obj_output.to_str().unwrap(),
            cuda_src.to_str().unwrap(),
            "--use_fast_math",
            "-O3",
            "-Xcompiler", "-fPIC",
            "-dc",  // Enable device code linking for Thrust
        ])
        .status()
        .expect("Failed to compile Thrust wrapper");
    
    if !obj_status.success() {
        panic!("Thrust wrapper compilation failed");
    }
    
    // Device link the object file (required for Thrust)
    let dlink_output = PathBuf::from(&out_dir).join("thrust_wrapper_dlink.o");
    println!("Device linking Thrust code...");
    let dlink_status = Command::new("nvcc")
        .args(&[
            "-dlink",
            "-arch", &format!("sm_{}", cuda_arch),
            obj_output.to_str().unwrap(),
            "-o", dlink_output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to device link");
    
    if !dlink_status.success() {
        panic!("Device linking failed");
    }
    
    // Create static library from both object files
    let lib_output = PathBuf::from(&out_dir).join("libthrust_wrapper.a");
    println!("Creating static library...");
    let ar_status = Command::new("ar")
        .args(&[
            "rcs",
            lib_output.to_str().unwrap(),
            obj_output.to_str().unwrap(),
            dlink_output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to create static library");
    
    if !ar_status.success() {
        panic!("Failed to create static library");
    }
    
    // Link the static library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=thrust_wrapper");
    
    // Link CUDA libraries
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib64/stubs", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudadevrt");  // Device runtime for Thrust
    
    // Link C++ standard library for Thrust
    println!("cargo:rustc-link-lib=stdc++");
    
    // Export the PTX file path for runtime loading
    println!("cargo:rustc-env=VISIONFLOW_PTX_PATH={}", ptx_output.display());
    
    println!("CUDA build complete!");
}