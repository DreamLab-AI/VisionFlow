use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::PathBuf;
use walkdir::WalkDir;
use regex::Regex;

#[derive(Parser)]
#[command(name = "ontology-audit")]
#[command(about = "Validate markdown files against canonical ontology format")]
struct Args {
    /// Path to markdown pages directory
    #[arg(short, long, default_value = "mainKnowledgeGraph/pages")]
    pages: PathBuf,

    /// Output JSON report path
    #[arg(short, long, default_value = "outputs/ontology-format-audit-report.json")]
    output: PathBuf,

    /// Check for IRI uniqueness across all files
    #[arg(long, default_value = "true")]
    check_iri_uniqueness: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct AuditReport {
    summary: FormatAuditSummary,
    format_validation: Vec<FileValidationResult>,
    iri_analysis: IRIAnalysis,
    files_by_domain: HashMap<String, Vec<String>>,
    issues_summary: IssuesSummary,
    recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FormatAuditSummary {
    total_files_scanned: usize,
    files_with_ontology_block: usize,
    format_compliant_files: usize,
    compliance_percentage: f64,
    files_per_domain: HashMap<String, usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FileValidationResult {
    file_path: String,
    file_name: String,
    is_valid: bool,
    ontology_block_count: usize,
    block_position: Option<usize>,
    term_id: Option<String>,
    domain: Option<String>,
    iri: Option<String>,
    errors: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct IRIAnalysis {
    total_iris: usize,
    unique_iris: usize,
    duplicate_iris: Vec<IRICollision>,
    iri_format_errors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct IRICollision {
    iri: String,
    files: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct IssuesSummary {
    multiple_blocks_per_file: Vec<String>,
    block_not_first: Vec<String>,
    missing_required_properties: Vec<MissingPropertyIssue>,
    invalid_term_ids: Vec<InvalidTermId>,
    namespace_mismatches: Vec<NamespaceMismatch>,
    invalid_public_access: Vec<String>,
    domain_classification_errors: Vec<String>,
    malformed_blocks: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MissingPropertyIssue {
    file: String,
    missing_properties: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InvalidTermId {
    file: String,
    term_id: String,
    reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct NamespaceMismatch {
    file: String,
    expected_namespace: String,
    actual_namespace: String,
}

// Required Tier 1 properties that must be in every ontology block
const REQUIRED_TIER1_PROPERTIES: &[&str] = &[
    "ontology",
    "term-id",
    "preferred-term",
    "source-domain",
    "status",
    "public-access",
    "last-updated",
    "definition",
    "owl:class",
    "owl:physicality",
    "owl:role",
    "is-subclass-of",
];

const VALID_DOMAINS: &[&str] = &["ai", "blockchain", "robotics", "metaverse", "general"];

const VALID_STATUSES: &[&str] = &["draft", "in-progress", "complete", "deprecated"];

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🔍 Starting Canonical Ontology Format Audit");
    println!("   Pages Dir: {:?}", args.pages);
    println!();

    // Phase 1: Scan all markdown files for ontology blocks
    println!("📝 Phase 1: Scanning markdown files for ontology blocks...");
    let mut file_validations = Vec::new();
    let mut all_iris: HashMap<String, Vec<String>> = HashMap::new();
    let mut files_by_domain: HashMap<String, Vec<String>> = HashMap::new();

    for entry in WalkDir::new(&args.pages).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }

        let content = match read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();

        let result = validate_ontology_block(&path.to_string_lossy().to_string(), &file_name, &content);

        // Track IRIs
        if let Some(iri) = &result.iri {
            all_iris.entry(iri.clone()).or_insert_with(Vec::new).push(path.to_string_lossy().to_string());
        }

        // Track by domain
        if let Some(domain) = &result.domain {
            files_by_domain.entry(domain.clone()).or_insert_with(Vec::new)
                .push(file_name.clone());
        }

        file_validations.push(result);
    }

    println!("   ✓ Scanned {} files", file_validations.len());

    // Phase 2: Analyze IRI uniqueness
    println!("🔗 Phase 2: Analyzing IRI uniqueness...");
    let iri_analysis = analyze_iris(&all_iris);

    // Phase 3: Analyze issues summary
    println!("⚠️  Phase 3: Analyzing issues...");
    let issues_summary = summarize_issues(&file_validations);

    // Phase 4: Generate statistics
    println!("📊 Phase 4: Computing statistics...");
    let total_files = file_validations.len();
    let compliant_files = file_validations.iter().filter(|v| v.is_valid).count();
    let with_ontology = file_validations.iter().filter(|v| v.ontology_block_count > 0).count();
    let compliance_percentage = if total_files > 0 {
        (compliant_files as f64 / total_files as f64) * 100.0
    } else {
        0.0
    };

    let summary = FormatAuditSummary {
        total_files_scanned: total_files,
        files_with_ontology_block: with_ontology,
        format_compliant_files: compliant_files,
        compliance_percentage,
        files_per_domain: files_by_domain.iter()
            .map(|(d, files)| (d.clone(), files.len()))
            .collect(),
    };

    // Phase 5: Generate recommendations
    println!("💡 Phase 5: Generating recommendations...");
    let recommendations = generate_audit_recommendations(&summary, &issues_summary);

    // Build report
    let report = AuditReport {
        summary,
        format_validation: file_validations,
        iri_analysis,
        files_by_domain,
        issues_summary,
        recommendations,
    };

    // Save report
    let report_json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.output, report_json)?;

    println!();
    println!("✅ Format Audit Complete!");
    println!("   Report: {:?}", args.output);
    print_audit_summary(&report);

    Ok(())
}

/// Validate a single ontology block in a markdown file
fn validate_ontology_block(file_path: &str, file_name: &str, content: &str) -> FileValidationResult {
    let mut result = FileValidationResult {
        file_path: file_path.to_string(),
        file_name: file_name.to_string(),
        is_valid: true,
        ontology_block_count: 0,
        block_position: None,
        term_id: None,
        domain: None,
        iri: None,
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    // Count ontology blocks
    let ontology_block_regex = Regex::new(r"### OntologyBlock").unwrap();
    let block_count = ontology_block_regex.find_iter(content).count();
    result.ontology_block_count = block_count;

    // Check: exactly one ontology block
    if block_count == 0 {
        result.errors.push("No OntologyBlock found in file".to_string());
        result.is_valid = false;
        return result;
    }

    if block_count > 1 {
        result.errors.push(format!("Multiple OntologyBlocks found ({})", block_count));
        result.is_valid = false;
    }

    // Find the position of the ontology block
    let lines: Vec<&str> = content.lines().collect();
    let mut block_line_idx = None;
    for (idx, line) in lines.iter().enumerate() {
        if line.contains("### OntologyBlock") {
            block_line_idx = Some(idx);
            break;
        }
    }

    if let Some(idx) = block_line_idx {
        result.block_position = Some(idx);

        // Check: OntologyBlock must be first (accounting for yaml frontmatter)
        let mut first_content_block = 0;
        for (i, line) in lines.iter().enumerate() {
            if !line.is_empty() && !line.starts_with("---") {
                first_content_block = i;
                break;
            }
        }

        if idx > first_content_block && !lines.iter().take(idx).all(|l| l.is_empty() || l.starts_with("---")) {
            result.warnings.push("OntologyBlock is not first content block in file".to_string());
        }
    }

    // Extract properties from the ontology block
    let properties = extract_ontology_properties(content);

    // Validate required properties
    let missing_properties: Vec<String> = REQUIRED_TIER1_PROPERTIES
        .iter()
        .filter(|&&prop| !properties.contains_key(prop))
        .map(|s| s.to_string())
        .collect();

    if !missing_properties.is_empty() {
        result.errors.push(format!("Missing required properties: {}", missing_properties.join(", ")));
        result.is_valid = false;
    }

    // Extract and validate term-id
    if let Some(term_id) = properties.get("term-id") {
        result.term_id = Some(term_id.clone());

        if !validate_term_id(term_id) {
            result.errors.push(format!("Invalid term-id format: {}", term_id));
            result.is_valid = false;
        }
    }

    // Extract domain from source-domain property
    if let Some(source_domain) = properties.get("source-domain") {
        result.domain = Some(source_domain.clone());

        if !VALID_DOMAINS.contains(&source_domain.as_str()) {
            result.errors.push(format!("Invalid source-domain: {}. Must be one of: {:?}",
                source_domain, VALID_DOMAINS));
            result.is_valid = false;
        }
    }

    // Validate status
    if let Some(status) = properties.get("status") {
        if !VALID_STATUSES.contains(&status.as_str()) {
            result.errors.push(format!("Invalid status: {}. Must be one of: {:?}",
                status, VALID_STATUSES));
            result.is_valid = false;
        }
    }

    // Validate public-access is boolean
    if let Some(public_access) = properties.get("public-access") {
        if public_access != "true" && public_access != "false" {
            result.errors.push(format!("Invalid public-access value: {} (must be true/false)", public_access));
            result.is_valid = false;
        }
    }

    // Validate owl:class and extract IRI
    if let Some(owl_class) = properties.get("owl:class") {
        result.iri = Some(owl_class.clone());

        // Check if namespace matches domain
        if let Some(domain) = &result.domain {
            let expected_namespace = get_namespace_for_domain(domain);
            if !owl_class.starts_with(&format!("{}:", expected_namespace)) {
                result.errors.push(format!(
                    "Namespace mismatch: expected {} but got {} for domain {}",
                    expected_namespace,
                    owl_class.split(':').next().unwrap_or("unknown"),
                    domain
                ));
                result.is_valid = false;
            }
        }

        // Check if class name is PascalCase
        if let Some(class_name) = owl_class.split(':').nth(1) {
            if !is_pascal_case(class_name) {
                result.warnings.push(format!("Class name '{}' is not in PascalCase", class_name));
            }
        }
    }

    // Check filename-term-id consistency
    if let Some(term_id) = &result.term_id {
        if !file_name.contains(term_id) {
            result.warnings.push(format!(
                "Filename '{}' does not match term-id '{}'",
                file_name, term_id
            ));
        }
    }

    // Validate last-updated format (ISO 8601)
    if let Some(last_updated) = properties.get("last-updated") {
        if !validate_iso_date(last_updated) {
            result.errors.push(format!("Invalid date format for last-updated: {} (expected YYYY-MM-DD)", last_updated));
            result.is_valid = false;
        }
    }

    result
}

/// Extract all ontology properties from content
fn extract_ontology_properties(content: &str) -> HashMap<String, String> {
    let mut properties = HashMap::new();
    let property_regex = Regex::new(r"^\s*-\s+([a-z][a-z:_-]*)::\s*(.+?)(?:\s*$|\s*@@|$)").unwrap();

    for line in content.lines() {
        if let Some(caps) = property_regex.captures(line) {
            let key = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let value = caps.get(2).map(|m| m.as_str().trim()).unwrap_or("");

            // Clean up page links [[...]] if present
            let cleaned_value = value
                .replace("[[", "")
                .replace("]]", "")
                .trim()
                .to_string();

            if !key.is_empty() && !cleaned_value.is_empty() {
                properties.insert(key.to_string(), cleaned_value);
            }
        }
    }

    properties
}

/// Validate term-id format: {PREFIX}-{NNNN} or numeric
fn validate_term_id(term_id: &str) -> bool {
    let term_id = term_id.trim();

    // Check for standard format: AI-NNNN, BC-NNNN, RB-NNNN
    if let Some(pos) = term_id.find('-') {
        let prefix = &term_id[..pos];
        let number = &term_id[pos + 1..];

        let valid_prefixes = ["AI", "BC", "RB"];
        if valid_prefixes.contains(&prefix) && number.len() == 4 && number.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }

    // Check for numeric format (metaverse): 20001, 20002, etc.
    if term_id.starts_with("20") && term_id.len() >= 4 && term_id.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }

    false
}

/// Get the namespace prefix for a domain
fn get_namespace_for_domain(domain: &str) -> String {
    match domain {
        "ai" => "ai",
        "blockchain" => "bc",
        "robotics" => "rb",
        "metaverse" => "mv",
        _ => "unknown",
    }.to_string()
}

/// Check if string is in PascalCase
fn is_pascal_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    // Must start with uppercase
    if !s.chars().next().unwrap().is_uppercase() {
        return false;
    }

    // Cannot contain spaces, hyphens, or underscores
    if s.contains(' ') || s.contains('-') || s.contains('_') {
        return false;
    }

    // Must contain only alphanumeric characters
    s.chars().all(|c| c.is_alphanumeric())
}

/// Validate ISO 8601 date format (YYYY-MM-DD)
fn validate_iso_date(date_str: &str) -> bool {
    let date_regex = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    date_regex.is_match(date_str)
}

/// Analyze IRIs for duplicates and format errors
fn analyze_iris(all_iris: &HashMap<String, Vec<String>>) -> IRIAnalysis {
    let mut duplicate_iris = Vec::new();
    let mut iri_format_errors = Vec::new();

    for (iri, files) in all_iris {
        if files.len() > 1 {
            duplicate_iris.push(IRICollision {
                iri: iri.clone(),
                files: files.clone(),
            });
        }

        // Check IRI format
        if !validate_iri_format(iri) {
            iri_format_errors.push(format!("Invalid IRI format: {}", iri));
        }
    }

    IRIAnalysis {
        total_iris: all_iris.len(),
        unique_iris: all_iris.len() - duplicate_iris.len(),
        duplicate_iris,
        iri_format_errors,
    }
}

/// Validate IRI format (should be namespace:ClassName)
fn validate_iri_format(iri: &str) -> bool {
    if !iri.contains(':') {
        return false;
    }

    let parts: Vec<&str> = iri.split(':').collect();
    if parts.len() != 2 {
        return false;
    }

    let namespace = parts[0];
    let class_name = parts[1];

    // Namespace should be lowercase
    if namespace != namespace.to_lowercase() {
        return false;
    }

    // Class name should be PascalCase
    if !is_pascal_case(class_name) {
        return false;
    }

    true
}

/// Summarize all issues found
fn summarize_issues(validations: &[FileValidationResult]) -> IssuesSummary {
    let mut summary = IssuesSummary {
        multiple_blocks_per_file: Vec::new(),
        block_not_first: Vec::new(),
        missing_required_properties: Vec::new(),
        invalid_term_ids: Vec::new(),
        namespace_mismatches: Vec::new(),
        invalid_public_access: Vec::new(),
        domain_classification_errors: Vec::new(),
        malformed_blocks: Vec::new(),
    };

    for validation in validations {
        if validation.ontology_block_count > 1 {
            summary.multiple_blocks_per_file.push(validation.file_name.clone());
        }

        if let Some(pos) = validation.block_position {
            if pos > 0 {
                summary.block_not_first.push(validation.file_name.clone());
            }
        }

        for error in &validation.errors {
            if error.contains("Missing required properties") {
                let missing: Vec<String> = error
                    .split(": ")
                    .nth(1)
                    .unwrap_or("")
                    .split(", ")
                    .map(|s| s.to_string())
                    .collect();
                summary.missing_required_properties.push(MissingPropertyIssue {
                    file: validation.file_name.clone(),
                    missing_properties: missing,
                });
            } else if error.contains("Invalid term-id") {
                summary.invalid_term_ids.push(InvalidTermId {
                    file: validation.file_name.clone(),
                    term_id: validation.term_id.clone().unwrap_or_default(),
                    reason: error.clone(),
                });
            } else if error.contains("Namespace mismatch") {
                summary.namespace_mismatches.push(NamespaceMismatch {
                    file: validation.file_name.clone(),
                    expected_namespace: "unknown".to_string(),
                    actual_namespace: "unknown".to_string(),
                });
            } else if error.contains("Invalid public-access") {
                summary.invalid_public_access.push(validation.file_name.clone());
            } else if error.contains("domain") {
                summary.domain_classification_errors.push(validation.file_name.clone());
            } else if error.contains("No OntologyBlock") {
                summary.malformed_blocks.push(validation.file_name.clone());
            }
        }
    }

    summary
}

/// Generate audit recommendations based on findings
fn generate_audit_recommendations(summary: &FormatAuditSummary, issues: &IssuesSummary) -> Vec<String> {
    let mut recommendations = Vec::new();

    if summary.compliance_percentage < 50.0 {
        recommendations.push("CRITICAL: Less than 50% of files are format compliant. Prioritize migration immediately.".to_string());
    } else if summary.compliance_percentage < 80.0 {
        recommendations.push("WARNING: Less than 80% compliance. Continue format migration efforts.".to_string());
    }

    if !issues.multiple_blocks_per_file.is_empty() {
        recommendations.push(format!(
            "ACTION: {} files have multiple OntologyBlocks. Each file should have exactly one block.",
            issues.multiple_blocks_per_file.len()
        ));
    }

    if !issues.missing_required_properties.is_empty() {
        recommendations.push(format!(
            "ACTION: {} files are missing required Tier 1 properties. Add: ontology, term-id, preferred-term, source-domain, status, public-access, last-updated, definition, owl:class, owl:physicality, owl:role, is-subclass-of",
            issues.missing_required_properties.len()
        ));
    }

    if !issues.invalid_term_ids.is_empty() {
        recommendations.push(format!(
            "ACTION: {} files have invalid term-ids. Format must be PREFIX-NNNN (e.g., AI-0850) or numeric (20001).",
            issues.invalid_term_ids.len()
        ));
    }

    if !issues.namespace_mismatches.is_empty() {
        recommendations.push(format!(
            "ACTION: {} files have namespace/domain mismatches. Verify owl:class namespace matches source-domain.",
            issues.namespace_mismatches.len()
        ));
    }

    if !issues.domain_classification_errors.is_empty() {
        recommendations.push(format!(
            "ACTION: {} files have domain classification errors. source-domain must be one of: ai, blockchain, robotics, metaverse, general.",
            issues.domain_classification_errors.len()
        ));
    }

    recommendations.push("RECOMMENDATION: Run this audit regularly (e.g., weekly) to track migration progress.".to_string());
    recommendations.push("NEXT STEP: Focus on Phase 1 Critical Fixes from migration rules: namespace corrections and class naming standardization.".to_string());

    recommendations
}

/// Print formatted audit summary to console
fn print_audit_summary(report: &AuditReport) {
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("        CANONICAL ONTOLOGY FORMAT AUDIT SUMMARY");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    println!("📊 Format Compliance:");
    println!("   Total Files Scanned: {}", report.summary.total_files_scanned);
    println!("   With OntologyBlock: {}", report.summary.files_with_ontology_block);
    println!("   Format Compliant: {}", report.summary.format_compliant_files);
    println!("   Compliance Rate: {:.1}%", report.summary.compliance_percentage);
    println!();

    println!("📁 Files Per Domain:");
    let mut domains: Vec<_> = report.summary.files_per_domain.iter().collect();
    domains.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (domain, count) in domains {
        println!("   {}: {} files", domain, count);
    }
    println!();

    println!("🔗 IRI Analysis:");
    println!("   Total IRIs: {}", report.iri_analysis.total_iris);
    println!("   Unique IRIs: {}", report.iri_analysis.unique_iris);
    if !report.iri_analysis.duplicate_iris.is_empty() {
        println!("   Duplicate IRIs: {}", report.iri_analysis.duplicate_iris.len());
        for collision in &report.iri_analysis.duplicate_iris {
            println!("      - {} found in {} files", collision.iri, collision.files.len());
        }
    }
    println!();

    println!("⚠️  Issues Found:");
    println!("   Multiple blocks per file: {}", report.issues_summary.multiple_blocks_per_file.len());
    println!("   Block not first: {}", report.issues_summary.block_not_first.len());
    println!("   Missing required properties: {}", report.issues_summary.missing_required_properties.len());
    println!("   Invalid term-ids: {}", report.issues_summary.invalid_term_ids.len());
    println!("   Namespace mismatches: {}", report.issues_summary.namespace_mismatches.len());
    println!("   Invalid public-access: {}", report.issues_summary.invalid_public_access.len());
    println!("   Domain classification errors: {}", report.issues_summary.domain_classification_errors.len());
    println!("   Malformed blocks: {}", report.issues_summary.malformed_blocks.len());
    println!();

    if !report.recommendations.is_empty() {
        println!("💡 Recommendations:");
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("   {}. {}", i + 1, rec);
        }
    }
    println!();
    println!("═══════════════════════════════════════════════════════════");
}
