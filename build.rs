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
        "src/utils/semantic_forces.cu",
        "src/utils/pagerank.cu",
        "src/utils/gpu_connected_components.cu",
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
        .unwrap_or_else(|_| "/opt/cuda".to_string());

    // Determine CUDA architecture — prefer runtime GPU detection to avoid
    // toolkit/driver version mismatches (e.g. nvcc 13.1 on CUDA 13.0 driver).
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| {
        // Try to auto-detect GPU compute capability via nvidia-smi (first GPU only)
        if let Ok(output) = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader", "--id=0"])
            .output()
        {
            if output.status.success() {
                let raw = String::from_utf8_lossy(&output.stdout);
                if let Some(cap) = raw.lines().next() {
                    let cap = cap.trim();
                    // nvidia-smi returns "8.6" → we need "86"
                    let arch = cap.replace('.', "");
                    if !arch.is_empty() {
                        println!("Auto-detected GPU compute capability: {} (sm_{})", cap, arch);
                        return arch;
                    }
                }
            }
        }
        "75".to_string()
    });
    println!("Using CUDA architecture: sm_{}", cuda_arch);

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

        // Downgrade PTX ISA version 9.1 -> 9.0 for driver compatibility.
        // CUDA toolkit 13.1 emits .version 9.1 but the driver (13.0) only JITs up to 9.0.
        if let Ok(ptx_text) = std::fs::read_to_string(&ptx_output) {
            if ptx_text.contains(".version 9.1") {
                let fixed = ptx_text.replacen(".version 9.1", ".version 9.0", 1);
                std::fs::write(&ptx_output, fixed).expect("Failed to write downgraded PTX");
                println!("PTX Build: Downgraded ISA 9.1 -> 9.0 for {}", file_name);
            }
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

    // CUDA source files that export host-callable FFI symbols and need linking
    let link_sources = [
        ("src/utils/visionflow_unified.cu", "thrust_wrapper"),
        ("src/utils/semantic_forces.cu", "semantic_forces"),
        ("src/utils/pagerank.cu", "pagerank"),
        ("src/utils/gpu_connected_components.cu", "gpu_connected_components"),
    ];

    let mut obj_files: Vec<PathBuf> = Vec::new();

    for (src_path, obj_name) in &link_sources {
        let cuda_src = Path::new(src_path);
        let obj_output = PathBuf::from(&out_dir).join(format!("{}.o", obj_name));

        // Use -gencode to produce only native CUBIN (no PTX fallback).
        // This avoids Thrust/CUB JIT compilation which fails when the toolkit
        // (13.1, PTX ISA 9.1) is newer than the driver (13.0, ISA ≤ 9.0).
        let gencode_flag = format!(
            "-gencode=arch=compute_{},code=sm_{}",
            cuda_arch, cuda_arch
        );
        println!("Compiling {} to object file (gencode: {})...", obj_name, gencode_flag);
        let obj_status = Command::new("nvcc")
            .args([
                "-c",
                &gencode_flag,
                "-o",
                obj_output.to_str().unwrap(),
                cuda_src.to_str().unwrap(),
                "--use_fast_math",
                "-O3",
                "-Xcompiler",
                "-fPIC",
                "-dc", // Enable device code linking
            ])
            .status()
            .expect(&format!("Failed to compile {}", obj_name));

        if !obj_status.success() {
            panic!("{} compilation failed", obj_name);
        }

        obj_files.push(obj_output);
    }

    // Device link all object files together (required for cross-module device calls)
    let dlink_output = PathBuf::from(&out_dir).join("cuda_dlink.o");
    let dlink_gencode = format!("-gencode=arch=compute_{},code=sm_{}", cuda_arch, cuda_arch);
    println!("Device linking {} CUDA object files ({})...", obj_files.len(), dlink_gencode);
    let mut dlink_args: Vec<String> = vec![
        "-dlink".to_string(),
        dlink_gencode,
    ];
    for obj in &obj_files {
        dlink_args.push(obj.to_str().unwrap().to_string());
    }
    dlink_args.push("-o".to_string());
    dlink_args.push(dlink_output.to_str().unwrap().to_string());

    let dlink_status = Command::new("nvcc")
        .args(&dlink_args)
        .status()
        .expect("Failed to device link");

    if !dlink_status.success() {
        panic!("Device linking failed");
    }

    // Create static library from all object files + device link output
    let lib_output = PathBuf::from(&out_dir).join("libthrust_wrapper.a");
    println!("Creating static library...");
    let mut ar_args: Vec<String> = vec![
        "rcs".to_string(),
        lib_output.to_str().unwrap().to_string(),
    ];
    for obj in &obj_files {
        ar_args.push(obj.to_str().unwrap().to_string());
    }
    ar_args.push(dlink_output.to_str().unwrap().to_string());

    let ar_status = Command::new("ar")
        .args(&ar_args)
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
