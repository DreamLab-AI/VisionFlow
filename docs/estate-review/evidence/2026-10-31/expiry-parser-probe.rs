// Link against the inspected checkout's native nostr_bbs_core.rlib.
use nostr_bbs_core::ontology_governance::parse_rfc3339_utc;
fn main() {
    // A caught panic is the observation; suppress the default stderr hook.
    std::panic::set_hook(Box::new(|_| {}));
    for input in ["2026-10-31T00:00:00Z", "2026-10-31T00:00:0éZ", "2026-10-31T00:00:00+-1:00"] {
        let result = std::panic::catch_unwind(|| parse_rfc3339_utc(input));
        println!("input={input:?} result={result:?}");
        if input.contains('é') { assert!(result.is_err()); }
        else { assert!(result.unwrap().is_some()); }
    }
}
