#[cfg(test)]
mod tests {
    use crate::utils::ptx::*;
    use std::collections::HashSet;

    #[test]
    fn test_ptx_module_all_unique() {
        let modules = PTXModule::all_modules();
        let mut seen = HashSet::new();

        for module in modules {
            assert!(seen.insert(module), "Duplicate module: {:?}", module);
        }

        assert_eq!(seen.len(), 7, "Should have exactly 7 unique modules");
    }

    #[test]
    fn test_ptx_module_source_files() {
        let modules = PTXModule::all_modules();

        for module in modules {
            let source = module.source_file();
            assert!(source.ends_with(".cu"), "Source file should be .cu: {}", source);
            assert!(!source.is_empty(), "Source file should not be empty");
        }
    }

    #[test]
    fn test_ptx_module_env_vars() {
        let modules = PTXModule::all_modules();

        for module in modules {
            let env_var = module.env_var();
            assert!(env_var.ends_with("_PTX_PATH"), "Env var should end with _PTX_PATH: {}", env_var);
            assert!(!env_var.is_empty(), "Env var should not be empty");
        }
    }

    #[test]
    fn test_effective_cuda_arch_default() {
        std::env::remove_var(CUDA_ARCH_ENV);
        let arch = effective_cuda_arch();
        assert_eq!(arch, DEFAULT_CUDA_ARCH, "Should return default arch");
    }

    #[test]
    fn test_effective_cuda_arch_override() {
        std::env::set_var(CUDA_ARCH_ENV, "86");
        let arch = effective_cuda_arch();
        assert_eq!(arch, "86", "Should return overridden arch");
        std::env::remove_var(CUDA_ARCH_ENV);
    }

    #[test]
    fn test_validate_ptx_valid() {
        let valid_ptx = r#"
.version 7.5
.target sm_75
.address_size 64
        "#;

        let result = validate_ptx(valid_ptx);
        assert!(result.is_ok(), "Valid PTX should pass validation");
    }

    #[test]
    fn test_validate_ptx_missing_version() {
        let invalid_ptx = r#"
.target sm_75
.address_size 64
        "#;

        let result = validate_ptx(invalid_ptx);
        assert!(result.is_err(), "PTX without .version should fail");
        assert!(result.unwrap_err().contains("version"));
    }

    #[test]
    fn test_validate_ptx_missing_target() {
        let invalid_ptx = r#"
.version 7.5
.address_size 64
        "#;

        let result = validate_ptx(invalid_ptx);
        assert!(result.is_err(), "PTX without .target should fail");
        assert!(result.unwrap_err().contains("target"));
    }

    #[test]
    fn test_load_ptx_module_precompiled() {
        // This test will pass if pre-compiled PTX exists in src/utils/ptx/
        for module in PTXModule::all_modules() {
            match load_ptx_module_sync(module) {
                Ok(content) => {
                    println!("✓ Loaded {:?}: {} bytes", module, content.len());
                    assert!(!content.is_empty(), "PTX content should not be empty for {:?}", module);
                    assert!(validate_ptx(&content).is_ok(), "PTX should be valid for {:?}", module);
                }
                Err(e) => {
                    println!("⚠ Could not load {:?}: {}", module, e);
                    println!("  This is expected if running outside Docker or without pre-compiled PTX");
                }
            }
        }
    }

    #[test]
    fn test_load_all_ptx_modules() {
        match load_all_ptx_modules_sync() {
            Ok(modules) => {
                println!("✓ Loaded {} modules", modules.len());

                for (module, content) in modules {
                    assert!(!content.is_empty(), "PTX content should not be empty for {:?}", module);
                    assert!(validate_ptx(&content).is_ok(), "PTX should be valid for {:?}", module);
                    println!("  {:?}: {} bytes", module, content.len());
                }
            }
            Err(e) => {
                println!("⚠ Could not load all modules: {}", e);
                println!("  This is expected if running outside Docker or without CUDA toolkit");
            }
        }
    }

    #[test]
    fn test_legacy_load_ptx_sync() {
        // Test backward compatibility
        match load_ptx_sync() {
            Ok(content) => {
                println!("✓ Legacy load_ptx_sync: {} bytes", content.len());
                assert!(!content.is_empty());
                assert!(validate_ptx(&content).is_ok());
            }
            Err(e) => {
                println!("⚠ Legacy load failed: {}", e);
                println!("  This is expected if running without pre-compiled PTX");
            }
        }
    }
}
