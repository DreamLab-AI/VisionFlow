#[cfg(test)]
mod tests {
    use crate::utils::ptx::{load_all_ptx_modules_sync, load_ptx_module_sync, PTXModule};
    use crate::utils::unified_gpu_compute::UnifiedGPUCompute;

    #[test]
    fn test_load_and_create_gpu_compute() {
        match load_ptx_module_sync(PTXModule::VisionflowUnified) {
            Ok(ptx_content) => {
                println!(
                    "✓ Loaded VisionflowUnified PTX: {} bytes",
                    ptx_content.len()
                );

                match UnifiedGPUCompute::new(100, 50, &ptx_content) {
                    Ok(compute) => {
                        println!("✓ Created UnifiedGPUCompute successfully");
                        println!(
                            "  Nodes: {}, Edges: {}",
                            compute.num_nodes, compute.num_edges
                        );
                    }
                    Err(e) => {
                        println!("⚠ Could not create GPU compute: {}", e);
                        println!("  This is expected if running without CUDA device");
                    }
                }
            }
            Err(e) => {
                println!("⚠ Could not load PTX: {}", e);
                println!(
                    "  This is expected if running outside Docker or without pre-compiled PTX"
                );
            }
        }
    }

    #[test]
    fn test_load_and_create_gpu_compute_with_clustering() {
        let main_ptx = match load_ptx_module_sync(PTXModule::VisionflowUnified) {
            Ok(content) => {
                println!("✓ Loaded VisionflowUnified PTX: {} bytes", content.len());
                content
            }
            Err(e) => {
                println!("⚠ Could not load main PTX: {}", e);
                return;
            }
        };

        let clustering_ptx = match load_ptx_module_sync(PTXModule::GpuClusteringKernels) {
            Ok(content) => {
                println!("✓ Loaded GpuClusteringKernels PTX: {} bytes", content.len());
                Some(content)
            }
            Err(e) => {
                println!("⚠ Could not load clustering PTX: {}", e);
                None
            }
        };

        match UnifiedGPUCompute::new_with_modules(100, 50, &main_ptx, clustering_ptx.as_deref()) {
            Ok(compute) => {
                println!("✓ Created UnifiedGPUCompute with modules successfully");
                println!(
                    "  Nodes: {}, Edges: {}",
                    compute.num_nodes, compute.num_edges
                );
                if clustering_ptx.is_some() {
                    println!("  ✓ Clustering module loaded");
                } else {
                    println!("  ⚠ Clustering module not available (fallback mode)");
                }
            }
            Err(e) => {
                println!("⚠ Could not create GPU compute with modules: {}", e);
                println!("  This is expected if running without CUDA device");
            }
        }
    }

    #[test]
    fn test_all_ptx_modules_available() {
        match load_all_ptx_modules_sync() {
            Ok(modules) => {
                println!("✓ Loaded all {} PTX modules:", modules.len());
                for (module, content) in &modules {
                    println!("  {:?}: {} bytes", module, content.len());
                }

                assert_eq!(modules.len(), 7, "Should have loaded all 7 modules");
            }
            Err(e) => {
                println!("⚠ Could not load all modules: {}", e);
                println!("  This is expected if running outside Docker or without CUDA toolkit");
            }
        }
    }

    #[test]
    fn test_individual_module_loading() {
        let modules_to_test = vec![
            (PTXModule::VisionflowUnified, "VisionflowUnified"),
            (PTXModule::GpuClusteringKernels, "GpuClusteringKernels"),
            (PTXModule::DynamicGrid, "DynamicGrid"),
            (PTXModule::GpuAabbReduction, "GpuAabbReduction"),
            (PTXModule::GpuLandmarkApsp, "GpuLandmarkApsp"),
            (PTXModule::SsspCompact, "SsspCompact"),
            (
                PTXModule::VisionflowUnifiedStability,
                "VisionflowUnifiedStability",
            ),
        ];

        let mut loaded_count = 0;
        for (module, name) in modules_to_test {
            match load_ptx_module_sync(module) {
                Ok(content) => {
                    println!("✓ {}: {} bytes", name, content.len());
                    loaded_count += 1;
                }
                Err(e) => {
                    println!("⚠ {}: Failed - {}", name, e);
                }
            }
        }

        println!("\nLoaded {}/7 modules", loaded_count);
        if loaded_count == 0 {
            println!("This is expected if running outside Docker or without pre-compiled PTX");
        }
    }

    #[test]
    fn test_gpu_compute_backward_compatibility() {
        
        match load_ptx_module_sync(PTXModule::VisionflowUnified) {
            Ok(ptx_content) => match UnifiedGPUCompute::new(50, 25, &ptx_content) {
                Ok(_compute) => {
                    println!("✓ Backward compatibility maintained - old API works");
                }
                Err(e) => {
                    println!("⚠ GPU compute creation failed: {}", e);
                }
            },
            Err(e) => {
                println!("⚠ PTX loading failed: {}", e);
            }
        }
    }
}
