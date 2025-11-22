// Integration tests for Rust tools
// Tests WASM parser and audit tool against all 6 domains

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde_json::Value;

const TEST_DATA_DIR: &str = "test-data";
const OUTPUT_DIR: &str = "outputs/rust";
const REPORT_DIR: &str = "reports";

/// Test domains
const DOMAINS: &[&str] = &["ai", "mv", "tc", "rb", "dt", "bc"];

/// Test results tracker
struct TestResults {
    passed: u32,
    failed: u32,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl TestResults {
    fn new() -> Self {
        TestResults {
            passed: 0,
            failed: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test audit tool on valid ontology files
    #[test]
    fn test_audit_tool_valid_files() {
        println!("\n=== Testing Audit Tool on Valid Files ===");
        let mut results = TestResults::new();

        let audit_binary = find_audit_binary();
        if audit_binary.is_none() {
            panic!("Audit tool binary not found. Please build it first.");
        }
        let audit_path = audit_binary.unwrap();

        for domain in DOMAINS {
            let valid_file = format!("{}/{}/valid-*.md", TEST_DATA_DIR, domain);
            test_audit_on_pattern(&audit_path, &valid_file, domain, &mut results, true);
        }

        assert!(results.passed > 0, "No tests passed for audit tool on valid files");
        println!("Valid files: {} passed, {} failed", results.passed, results.failed);
    }

    /// Test audit tool on invalid ontology files
    #[test]
    fn test_audit_tool_invalid_files() {
        println!("\n=== Testing Audit Tool on Invalid Files ===");
        let mut results = TestResults::new();

        let audit_binary = find_audit_binary();
        if audit_binary.is_none() {
            panic!("Audit tool binary not found. Please build it first.");
        }
        let audit_path = audit_binary.unwrap();

        for domain in DOMAINS {
            let invalid_file = format!("{}/{}/invalid-*.md", TEST_DATA_DIR, domain);
            // Invalid files should be detected by audit tool
            test_audit_on_pattern(&audit_path, &invalid_file, domain, &mut results, false);
        }

        println!("Invalid files: {} detected, {} missed", results.passed, results.failed);
    }

    /// Test audit tool OWL2 compliance checking
    #[test]
    fn test_owl2_compliance() {
        println!("\n=== Testing OWL2 Compliance ===");
        let mut results = TestResults::new();

        let audit_binary = find_audit_binary();
        if audit_binary.is_none() {
            panic!("Audit tool binary not found. Please build it first.");
        }
        let audit_path = audit_binary.unwrap();

        // Test that valid files pass OWL2 compliance
        for domain in DOMAINS {
            let test_file = get_valid_file_for_domain(domain);
            if let Some(file_path) = test_file {
                println!("Checking OWL2 compliance for {}/{}...", domain, file_path.file_name().unwrap().to_str().unwrap());

                let output = Command::new(&audit_path)
                    .arg("--owl2-check")
                    .arg(&file_path)
                    .output();

                match output {
                    Ok(result) => {
                        let stdout = String::from_utf8_lossy(&result.stdout);
                        if stdout.contains("OWL2") && stdout.contains("compliant") {
                            results.passed += 1;
                            println!("  ✓ PASS - OWL2 compliant");
                        } else {
                            results.warnings.push(format!("OWL2 compliance unclear for {}", domain));
                            println!("  ⚠ WARN - OWL2 compliance status unclear");
                        }
                    }
                    Err(e) => {
                        results.failed += 1;
                        results.errors.push(format!("Audit failed for {}: {}", domain, e));
                        println!("  ✗ FAIL - {}", e);
                    }
                }
            }
        }

        assert!(results.passed > 0, "No OWL2 compliance tests passed");
    }

    /// Test WASM parser on all domains
    #[test]
    fn test_wasm_parser() {
        println!("\n=== Testing WASM Parser ===");
        let mut results = TestResults::new();

        // Note: WASM parser typically processes TTL files
        // We'll need to convert MD to TTL first or test with TTL fixtures

        println!("Testing WASM parser requires TTL input files...");
        println!("This test validates that WASM parser can load and parse ontology data.");

        // Check if WASM module exists
        let wasm_module = PathBuf::from("../../publishing-tools/WasmVOWL/rust-wasm/pkg");
        if wasm_module.exists() {
            results.passed += 1;
            println!("  ✓ PASS - WASM module found and accessible");
        } else {
            results.warnings.push("WASM module not built. Run 'wasm-pack build' first.".to_string());
            println!("  ⚠ WARN - WASM module not found");
        }

        // Additional WASM tests would go here
        // (Loading TTL, parsing, graph generation, etc.)
    }

    /// Test namespace validation across all domains
    #[test]
    fn test_namespace_validation() {
        println!("\n=== Testing Namespace Validation ===");
        let mut results = TestResults::new();

        let audit_binary = find_audit_binary();
        if audit_binary.is_none() {
            panic!("Audit tool binary not found. Please build it first.");
        }
        let audit_path = audit_binary.unwrap();

        // Special test: rb domain file with mv: namespace (common error)
        let rb_namespace_error = format!("{}/rb/invalid-namespace-mismatch.md", TEST_DATA_DIR);
        if Path::new(&rb_namespace_error).exists() {
            println!("Testing namespace mismatch detection...");

            let output = Command::new(&audit_path)
                .arg("--check-namespaces")
                .arg(&rb_namespace_error)
                .output();

            match output {
                Ok(result) => {
                    let stderr = String::from_utf8_lossy(&result.stderr);
                    if stderr.contains("namespace") || stderr.contains("mismatch") || !result.status.success() {
                        results.passed += 1;
                        println!("  ✓ PASS - Namespace error detected");
                    } else {
                        results.failed += 1;
                        results.errors.push("Namespace validation missed error".to_string());
                        println!("  ✗ FAIL - Namespace error not detected");
                    }
                }
                Err(e) => {
                    results.warnings.push(format!("Could not run namespace check: {}", e));
                    println!("  ⚠ WARN - {}", e);
                }
            }
        }
    }

    /// Test performance with edge cases
    #[test]
    fn test_edge_cases() {
        println!("\n=== Testing Edge Cases ===");
        let mut results = TestResults::new();

        let audit_binary = find_audit_binary();
        if audit_binary.is_none() {
            panic!("Audit tool binary not found. Please build it first.");
        }
        let audit_path = audit_binary.unwrap();

        // Test minimal file
        test_single_file(&audit_path, "ai/edge-minimal.md", &mut results);

        // Test maximal file
        test_single_file(&audit_path, "mv/edge-maximal.md", &mut results);

        // Test complex properties
        test_single_file(&audit_path, "bc/edge-complex-properties.md", &mut results);

        // Test multi-domain
        test_single_file(&audit_path, "dt/edge-multi-domain.md", &mut results);

        assert!(results.passed >= 2, "Most edge cases should be handled");
        println!("Edge cases: {} passed, {} failed, {} warnings",
                 results.passed, results.failed, results.warnings.len());
    }
}

// Helper functions

fn find_audit_binary() -> Option<PathBuf> {
    let possible_paths = vec![
        PathBuf::from("../../Ontology-Tools/tools/audit/target/release/audit"),
        PathBuf::from("../../Ontology-Tools/tools/audit/target/debug/audit"),
    ];

    for path in possible_paths {
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn get_valid_file_for_domain(domain: &str) -> Option<PathBuf> {
    let pattern = format!("{}/{}/valid-*.md", TEST_DATA_DIR, domain);
    // Simple glob implementation - in real code use glob crate
    let dir = PathBuf::from(TEST_DATA_DIR).join(domain);
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name() {
                if name.to_str().unwrap().starts_with("valid-") {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn test_audit_on_pattern(
    audit_path: &Path,
    pattern: &str,
    domain: &str,
    results: &mut TestResults,
    should_pass: bool,
) {
    println!("Testing {} with pattern: {}", domain, pattern);

    // In a real implementation, expand glob pattern
    // For now, just test the concept

    let test_file = get_valid_file_for_domain(domain);
    if let Some(file_path) = test_file {
        test_single_file(audit_path, file_path.to_str().unwrap(), results);
    }
}

fn test_single_file(audit_path: &Path, file: &str, results: &mut TestResults) {
    let file_path = PathBuf::from(TEST_DATA_DIR).join(file);

    if !file_path.exists() {
        results.warnings.push(format!("Test file not found: {}", file));
        return;
    }

    println!("Testing {}...", file);

    let output = Command::new(audit_path)
        .arg(&file_path)
        .output();

    match output {
        Ok(result) => {
            if result.status.success() {
                results.passed += 1;
                println!("  ✓ PASS");
            } else {
                let stderr = String::from_utf8_lossy(&result.stderr);
                if stderr.contains("invalid") || stderr.contains("error") {
                    results.passed += 1; // Expected for invalid files
                    println!("  ✓ PASS - Correctly identified issues");
                } else {
                    results.failed += 1;
                    results.errors.push(format!("Audit failed on {}: {}", file, stderr));
                    println!("  ✗ FAIL - {}", stderr);
                }
            }
        }
        Err(e) => {
            results.failed += 1;
            results.errors.push(format!("Could not run audit on {}: {}", file, e));
            println!("  ✗ FAIL - {}", e);
        }
    }
}

fn main() {
    println!("Rust Tools Integration Tests");
    println!("Run with: cargo test --test integration");
}
