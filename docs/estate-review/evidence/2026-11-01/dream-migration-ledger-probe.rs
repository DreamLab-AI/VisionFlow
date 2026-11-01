//! Differential input probe; execute the separately captured TypeScript oracle.
use std::io::Read;
fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(&input).unwrap();
    let output: Vec<_> = cases
        .iter()
        .map(|case| {
            let row = case["row"].as_str().unwrap();
            let rules: Vec<_> = dream_core::ledger::validate_ledger(row, "2026-09-06")
                .iter()
                .map(|v| v.rule)
                .collect();
            serde_json::json!({"id":case["id"], "rules":rules})
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
