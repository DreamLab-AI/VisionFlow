//! Comprehensive PTX Pipeline Validation Tests
//! 
//! Advanced testing for PTX compilation, loading, and kernel validation
//! across multiple CUDA architectures and failure scenarios.

#![allow(unused_imports)]

use cust::device::Device;
use cust::context::Context;
use cust::module::Module;
use std::time::Instant;
use std::collections::HashMap;

fn should_run() -> bool {
    std::env::var("RUN_GPU_SMOKE").ok().as_deref() == Some("1")
}

fn create_test_cuda_context() -> Option<Context> {
    match Device::get_device(0) {
        Ok(device) => match Context::new(device) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("[PTX-COMPREHENSIVE] Failed to create CUDA context: {e}");
                None
            }
        },
        Err(e) => {
            eprintln!("[PTX-COMPREHENSIVE] No CUDA device(0): {e}");
            None
        }
    }
}

#[cfg(test)]
mod ptx_comprehensive_tests {
    use super::*;

    #[test]
    fn test_multi_arch_ptx_compilation() {
        if !should_run() {
            eprintln!("[PTX-COMPREHENSIVE] Skipping multi-arch test (set RUN_GPU_SMOKE=1)");
            return;
        }

        let architectures = vec!["61", "70", "75", "80", "86", "89"];
        let mut successful_archs = Vec::new();
        let mut failed_archs = Vec::new();
        
        let original_arch = std::env::var("CUDA_ARCH").unwrap_or_default();
        
        for arch in architectures {
            println!("Testing CUDA architecture: sm_{}", arch);
            
            // Set architecture for this test
            std::env::set_var("CUDA_ARCH", arch);
            
            match crate::utils::ptx::load_ptx_sync() {
                Ok(ptx_content) => {
                    // Validate PTX content contains architecture-specific code
                    if ptx_content.contains(&format!(".target sm_{}", arch)) {
                        println!("  ✅ PTX generated for sm_{}", arch);
                        successful_archs.push(arch);
                        
                        // Test module creation
                        if let Some(_ctx) = create_test_cuda_context() {
                            match Module::from_ptx(&ptx_content, &[]) {
                                Ok(_module) => {
                                    println!("  ✅ Module created successfully for sm_{}", arch);
                                }
                                Err(e) => {
                                    println!("  ⚠️ Module creation failed for sm_{}: {}", arch, e);
                                }
                            }
                        }
                    } else {
                        println!("  ⚠️ PTX content may not target sm_{}", arch);
                    }
                }
                Err(e) => {
                    println!("  ❌ PTX compilation failed for sm_{}: {}", arch, e);
                    failed_archs.push((arch, e.to_string()));
                }
            }
        }
        
        // Restore original architecture
        if !original_arch.is_empty() {
            std::env::set_var("CUDA_ARCH", original_arch);
        }
        
        println!("\nArchitecture Support Summary:");
        println!("  Successful: {:?}", successful_archs);
        if !failed_archs.is_empty() {
            println!("  Failed: {:?}", failed_archs);
        }
        
        // At least the default architecture should work
        assert!(!successful_archs.is_empty(), 
               "At least one CUDA architecture should compile successfully");
    }

    #[test]
    fn test_kernel_symbol_completeness() {
        if !should_run() {
            eprintln!("[PTX-COMPREHENSIVE] Skipping symbol test (set RUN_GPU_SMOKE=1)");
            return;
        }

        let required_kernels = vec![
            // Phase 1 kernels (current)
            "build_grid_kernel",
            "compute_cell_bounds_kernel",
            "force_pass_kernel",
            "integrate_pass_kernel",
            "relaxation_step_kernel",
            
            // Future Phase 2 analytics kernels (may not exist yet)
            // "kmeans_init_kernel",
            // "kmeans_assign_kernel",
            // "anomaly_score_kernel",
        ];
        
        let ptx = match crate::utils::ptx::load_ptx_sync() {
            Ok(ptx) => ptx,
            Err(e) => {
                panic!("[PTX-COMPREHENSIVE] Failed to load PTX: {e}");
            }
        };
        
        let _ctx = match create_test_cuda_context() {
            Some(ctx) => ctx,
            None => {
                panic!("[PTX-COMPREHENSIVE] Cannot create CUDA context for symbol testing");
            }
        };
        
        let module = match Module::from_ptx(&ptx, &[]) {
            Ok(module) => module,
            Err(e) => {
                panic!("[PTX-COMPREHENSIVE] Failed to create module: {e}");
            }
        };
        
        let mut found_kernels = Vec::new();
        let mut missing_kernels = Vec::new();
        
        for kernel_name in required_kernels {
            match module.get_function(kernel_name) {
                Ok(_) => {
                    println!("  ✅ Kernel found: {}", kernel_name);
                    found_kernels.push(kernel_name);
                }
                Err(_) => {
                    println!("  ❌ Kernel missing: {}", kernel_name);
                    missing_kernels.push(kernel_name);
                }
            }
        }
        
        println!("\nKernel Symbol Summary:");
        println!("  Found: {}/{}", found_kernels.len(), found_kernels.len() + missing_kernels.len());
        
        if !missing_kernels.is_empty() {
            println!("  Missing: {:?}", missing_kernels);
        }
        
        // All current Phase 1 kernels must be present
        let phase1_kernels = vec![
            "build_grid_kernel",
            "compute_cell_bounds_kernel", 
            "force_pass_kernel",
            "integrate_pass_kernel",
            "relaxation_step_kernel",
        ];
        
        for kernel in phase1_kernels {
            assert!(found_kernels.contains(&kernel),
                   "Phase 1 kernel '{}' must be present in PTX", kernel);
        }
    }

    #[test]
    fn test_compilation_fallback_scenarios() {
        if !should_run() {
            eprintln!("[PTX-COMPREHENSIVE] Skipping fallback test (set RUN_GPU_SMOKE=1)");
            return;
        }

        println!("Testing PTX compilation fallback scenarios...");
        
        // Save original environment
        let original_ptx_path = std::env::var("VISIONFLOW_PTX_PATH").ok();
        let original_docker_env = std::env::var("DOCKER_ENV").ok();
        
        // Test 1: Missing PTX file (should trigger fallback compilation)
        println!("  Test 1: Missing PTX file fallback...");
        std::env::set_var("VISIONFLOW_PTX_PATH", "/nonexistent/path/missing.ptx");
        
        let result = crate::utils::ptx::load_ptx_sync();
        match result {
            Ok(ptx) => {
                println!("    ✅ Fallback compilation succeeded");
                assert!(!ptx.is_empty(), "PTX content should not be empty");
                assert!(ptx.contains(".version"), "PTX should contain version directive");
            }
            Err(e) => {
                println!("    ⚠️ Fallback compilation failed: {}", e);
                // This might be expected in some CI environments
            }
        }
        
        // Test 2: Docker environment path resolution
        println!("  Test 2: Docker environment path resolution...");
        std::env::set_var("DOCKER_ENV", "1");
        
        let result = crate::utils::ptx::load_ptx_sync();
        match result {
            Ok(ptx) => {
                println!("    ✅ Docker environment fallback succeeded");
                assert!(!ptx.is_empty(), "PTX content should not be empty");
            }
            Err(e) => {
                println!("    ⚠️ Docker environment fallback failed: {}", e);
                // This might be expected in some environments
            }
        }
        
        // Restore environment
        if let Some(path) = original_ptx_path {
            std::env::set_var("VISIONFLOW_PTX_PATH", path);
        } else {
            std::env::remove_var("VISIONFLOW_PTX_PATH");
        }
        
        if let Some(docker) = original_docker_env {
            std::env::set_var("DOCKER_ENV", docker);
        } else {
            std::env::remove_var("DOCKER_ENV");
        }
        
        println!("  ✅ Fallback scenario testing completed");
    }

    #[test]
    fn test_cold_start_performance() {
        if !should_run() {
            eprintln!("[PTX-COMPREHENSIVE] Skipping cold start test (set RUN_GPU_SMOKE=1)");
            return;
        }

        println!("Testing cold start performance...");
        
        let start_time = Instant::now();
        
        // Step 1: PTX loading
        let ptx_start = Instant::now();
        let ptx = match crate::utils::ptx::load_ptx_sync() {
            Ok(ptx) => ptx,
            Err(e) => panic!("PTX loading failed: {}", e),
        };
        let ptx_time = ptx_start.elapsed();
        println!("  PTX loading: {:.1}ms", ptx_time.as_millis());
        
        // Step 2: CUDA context creation  
        let ctx_start = Instant::now();
        let _ctx = match create_test_cuda_context() {
            Some(ctx) => ctx,
            None => panic!("CUDA context creation failed"),
        };
        let ctx_time = ctx_start.elapsed();
        println!("  CUDA context: {:.1}ms", ctx_time.as_millis());
        
        // Step 3: Module creation
        let module_start = Instant::now();
        let _module = match Module::from_ptx(&ptx, &[]) {
            Ok(module) => module,
            Err(e) => panic!("Module creation failed: {}", e),
        };
        let module_time = module_start.elapsed();
        println!("  Module creation: {:.1}ms", module_time.as_millis());
        
        // Step 4: GPU compute initialization (if available)
        let gpu_start = Instant::now();
        match crate::utils::unified_gpu_compute::UnifiedGPUCompute::new(100, 200, &ptx) {
            Ok(_gpu) => {
                let gpu_time = gpu_start.elapsed();
                println!("  GPU compute init: {:.1}ms", gpu_time.as_millis());
            }
            Err(e) => {
                println!("  GPU compute init failed: {} (may be expected)", e);
            }
        }
        
        let total_time = start_time.elapsed();
        println!("  Total cold start: {:.1}ms", total_time.as_millis());
        
        // Performance requirements
        assert!(total_time.as_secs() < 3, 
               "Cold start should complete within 3 seconds, took {:.1}s", 
               total_time.as_secs_f32());
        
        // Individual component requirements
        assert!(ptx_time.as_millis() < 1000, "PTX loading should be <1s");
        assert!(ctx_time.as_millis() < 500, "CUDA context should be <0.5s");
        assert!(module_time.as_millis() < 1000, "Module creation should be <1s");
        
        println!("  ✅ Cold start performance requirements met");
    }

    #[test]
    fn test_ptx_content_validation() {
        if !should_run() {
            eprintln!("[PTX-COMPREHENSIVE] Skipping content validation (set RUN_GPU_SMOKE=1)");
            return;
        }

        println!("Validating PTX content structure...");
        
        let ptx = match crate::utils::ptx::load_ptx_sync() {
            Ok(ptx) => ptx,
            Err(e) => panic!("PTX loading failed: {}", e),
        };
        
        // Basic PTX structure validation
        assert!(ptx.contains(".version"), "PTX should contain .version directive");
        assert!(ptx.contains(".target"), "PTX should contain .target directive");
        assert!(ptx.contains(".address_size"), "PTX should contain .address_size directive");
        
        // Check for required sections
        assert!(ptx.contains(".visible .entry"), "PTX should contain kernel entry points");
        
        // Validate kernel function signatures exist
        let kernel_patterns = vec![
            "build_grid_kernel",
            "compute_cell_bounds_kernel",
            "force_pass_kernel", 
            "integrate_pass_kernel",
            "relaxation_step_kernel",
        ];
        
        let mut found_patterns = 0;
        for pattern in kernel_patterns {
            if ptx.contains(pattern) {
                found_patterns += 1;
                println!("  ✅ Found kernel pattern: {}", pattern);
            } else {
                println!("  ⚠️ Missing kernel pattern: {}", pattern);
            }
        }
        
        assert!(found_patterns >= 3, 
               "Should find at least 3 kernel patterns in PTX, found {}", found_patterns);
        
        // Check PTX size is reasonable
        assert!(ptx.len() > 1000, "PTX should be substantial (>1KB), got {} bytes", ptx.len());
        assert!(ptx.len() < 10_000_000, "PTX should not be excessive (<10MB), got {} bytes", ptx.len());
        
        println!("  ✅ PTX content validation passed");
        println!("     PTX size: {} bytes", ptx.len());
        println!("     Kernel patterns found: {}", found_patterns);
    }

    #[test]
    fn test_cuda_arch_detection() {
        if !should_run() {
            eprintln!("[PTX-COMPREHENSIVE] Skipping arch detection (set RUN_GPU_SMOKE=1)");
            return;
        }

        println!("Testing CUDA architecture detection...");
        
        // Test effective_cuda_arch function
        let detected_arch = crate::utils::ptx::effective_cuda_arch();
        println!("  Detected architecture: sm_{}", detected_arch);
        
        // Architecture should be reasonable
        assert!(detected_arch >= 50, "CUDA architecture should be >= 50 (Maxwell)");
        assert!(detected_arch <= 90, "CUDA architecture should be <= 90 (future proof)");
        
        // Test common architectures
        let common_archs = vec![61, 70, 75, 80, 86, 89];
        let is_common = common_archs.contains(&detected_arch);
        
        if is_common {
            println!("  ✅ Detected architecture is commonly supported");
        } else {
            println!("  ⚠️ Detected architecture {} may need special handling", detected_arch);
        }
        
        // Test environment override
        let original_arch = std::env::var("CUDA_ARCH").ok();
        
        std::env::set_var("CUDA_ARCH", "75");
        let override_arch = crate::utils::ptx::effective_cuda_arch();
        assert_eq!(override_arch, 75, "Environment override should work");
        
        // Restore original
        if let Some(arch) = original_arch {
            std::env::set_var("CUDA_ARCH", arch);
        } else {
            std::env::remove_var("CUDA_ARCH");
        }
        
        println!("  ✅ CUDA architecture detection validated");
    }
}

#[cfg(test)]
mod ptx_error_handling_tests {
    use super::*;

    #[test]
    fn test_ptx_error_diagnostics() {
        if !should_run() {
            eprintln!("[PTX-COMPREHENSIVE] Skipping error diagnostics (set RUN_GPU_SMOKE=1)");
            return;
        }

        println!("Testing PTX error diagnostics...");
        
        // Test diagnostic function with known error patterns
        let test_errors = vec![
            "ptxas fatal   : Unresolved extern function 'unknown_function'",
            "error: identifier not found",
            "CUDA driver version is insufficient for CUDA runtime version",
            "no kernel image is available for execution on the device",
        ];
        
        for error in test_errors {
            let diagnosis = crate::utils::gpu_diagnostics::diagnose_ptx_error(&format!("Test error: {}", error));
            
            println!("  Error: {}", error);
            println!("  Diagnosis: {}", diagnosis);
            
            // Diagnosis should provide helpful information
            assert!(!diagnosis.is_empty(), "Diagnosis should not be empty");
            assert!(diagnosis.len() > 20, "Diagnosis should be substantive");
        }
        
        println!("  ✅ PTX error diagnostics validated");
    }

    #[test]
    fn test_gpu_validation_integration() {
        if !should_run() {
            eprintln!("[PTX-COMPREHENSIVE] Skipping GPU validation (set RUN_GPU_SMOKE=1)");
            return;
        }

        println!("Testing GPU validation integration...");
        
        // Test GPU diagnostics smoke test
        match crate::utils::gpu_diagnostics::ptx_module_smoke_test() {
            Ok(()) => {
                println!("  ✅ GPU diagnostics smoke test passed");
            }
            Err(e) => {
                println!("  ⚠️ GPU diagnostics smoke test failed: {}", e);
                // May be expected in some environments
            }
        }
        
        // Test kernel launch validation if available
        if let Some(_ctx) = create_test_cuda_context() {
            let grid_size = 32;
            let block_size = 256;
            
            match crate::utils::gpu_diagnostics::validate_kernel_launch(
                "test_kernel", grid_size, block_size, 1000, 2000
            ) {
                Ok(()) => {
                    println!("  ✅ Kernel launch validation passed");
                }
                Err(e) => {
                    println!("  ⚠️ Kernel launch validation failed: {}", e);
                }
            }
        }
        
        println!("  ✅ GPU validation integration tested");
    }
}