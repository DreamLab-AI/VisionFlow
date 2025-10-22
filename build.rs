use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Check if GPU feature is enabled
    let gpu_enabled = env::var("CARGO_FEATURE_GPU").is_ok();

    if !gpu_enabled {
        println!("cargo:warning=GPU feature disabled, skipping CUDA compilation");
        return;
    }

    // All CUDA source files that need compilation
    let cuda_files = [
        "src/utils/visionflow_unified.cu",
        "src/utils/gpu_clustering_kernels.cu",
        "src/utils/dynamic_grid.cu",
        "src/utils/gpu_aabb_reduction.cu",
        "src/utils/gpu_landmark_apsp.cu",
        "src/utils/sssp_compact.cu",
        "src/utils/visionflow_unified_stability.cu",
        "src/utils/ontology_constraints.cu",
    ];

    // Only rebuild if CUDA files change
    for cuda_file in &cuda_files {
        println!("cargo:rerun-if-changed={}", cuda_file);
    }
    println!("cargo:rerun-if-changed=build.rs");

    // Get build configuration
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Determine CUDA architecture
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "75".to_string());

    // Compile all CUDA files to PTX
    println!("Compiling {} CUDA kernels to PTX...", cuda_files.len());

    for cuda_file in &cuda_files {
        let cuda_src = Path::new(cuda_file);
        let file_name = cuda_src.file_stem().unwrap().to_str().unwrap();
        let ptx_output = PathBuf::from(&out_dir).join(format!("{}.ptx", file_name));

        println!("Compiling {} to PTX...", file_name);
        println!(
            "NVCC Command: nvcc -ptx -arch sm_{} -o {} {} --use_fast_math -O3",
            cuda_arch,
            ptx_output.display(),
            cuda_src.display()
        );

        let nvcc_output = Command::new("nvcc")
            .args([
                "-ptx",
                "-arch",
                &format!("sm_{}", cuda_arch),
                "-o",
                ptx_output.to_str().unwrap(),
                cuda_src.to_str().unwrap(),
                "--use_fast_math",
                "-O3",
            ])
            .output()
            .expect("Failed to execute nvcc - is CUDA toolkit installed and in PATH?");

        if !nvcc_output.status.success() {
            eprintln!(
                "NVCC STDOUT: {}",
                String::from_utf8_lossy(&nvcc_output.stdout)
            );
            eprintln!(
                "NVCC STDERR: {}",
                String::from_utf8_lossy(&nvcc_output.stderr)
            );
            panic!("CUDA PTX compilation failed for {} with exit code: {:?}. Check CUDA installation and source file.",
                   file_name, nvcc_output.status.code());
        }

        // Verify the PTX file was created
        match std::fs::metadata(&ptx_output) {
            Ok(metadata) => {
                println!(
                    "PTX Build: {} created, size: {} bytes",
                    file_name,
                    metadata.len()
                );
                if metadata.len() == 0 {
                    panic!("PTX file {} was created but is empty - CUDA compilation may have failed silently", file_name);
                }

                // Export PTX path as environment variable
                let env_var = format!("{}_PTX_PATH", file_name.to_uppercase());
                println!("cargo:rustc-env={}={}", env_var, ptx_output.display());
                println!("PTX Build: Exported {}={}", env_var, ptx_output.display());
            }
            Err(e) => {
                panic!(
                    "PTX file {} was not created despite successful nvcc status: {}",
                    file_name, e
                );
            }
        }
    }

    println!("All PTX compilation successful!");

    // Compile visionflow_unified for Thrust wrapper (legacy compatibility)
    let cuda_src = Path::new("src/utils/visionflow_unified.cu");
    let obj_output = PathBuf::from(&out_dir).join("thrust_wrapper.o");

    // Compile Thrust wrapper functions to object file
    println!("Compiling Thrust wrapper functions...");
    let obj_status = Command::new("nvcc")
        .args([
            "-c",
            "-arch",
            &format!("sm_{}", cuda_arch),
            "-o",
            obj_output.to_str().unwrap(),
            cuda_src.to_str().unwrap(),
            "--use_fast_math",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "-dc", // Enable device code linking for Thrust
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
        .args([
            "-dlink",
            "-arch",
            &format!("sm_{}", cuda_arch),
            obj_output.to_str().unwrap(),
            "-o",
            dlink_output.to_str().unwrap(),
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
        .args([
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
    println!("cargo:rustc-link-lib=cudadevrt"); // Device runtime for Thrust

    // Link C++ standard library for Thrust
    println!("cargo:rustc-link-lib=stdc++");

    println!("CUDA build complete!");
}
