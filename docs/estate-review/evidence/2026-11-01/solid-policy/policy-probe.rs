// Offline policy checks using the actual library and in-memory storage backend.
use bytes::Bytes;
use serde_json::{json, Value};
use solid_pod_rs::multitenant::{PodResolver, SubdomainResolver};
use solid_pod_rs::storage::{memory::MemoryBackend, Storage};
use solid_pod_rs::wac::{
    classify_policy_read, evaluate_access, parse_jsonld_acl_with_limits,
    parser::parse_turtle_acl_with_limit, resolve_policy_from_storage, validate_acl_document,
    AccessMode, PolicyOutcome, PolicyRead, PolicyStep,
};

const PREFIX: &str = "@prefix acl: <http://www.w3.org/ns/auth/acl#> .\n@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n";
const RULE: &str = "<#read> a acl:Authorization; acl:agentClass foaf:Agent; acl:accessTo </secret>; acl:mode acl:Read";

fn standard_turtle_accepts(body: &str) -> bool {
    oxttl::TurtleParser::new()
        .with_base_iri("https://pod.example/secret.acl")
        .unwrap()
        .for_slice(body.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .is_ok()
}

async fn conditions() -> Value {
    let suffixes = [
        (
            "valid",
            "; acl:condition [ a acl:ClientCondition; acl:client <https://allowed.example/> ] .",
        ),
        (
            "unterminated",
            "; acl:condition [ a acl:ClientCondition; acl:client <https://allowed.example/> .",
        ),
        (
            "missing_type",
            "; acl:condition [ acl:client <https://allowed.example/> ] .",
        ),
        ("empty_condition", "; acl:condition [] ."),
        (
            "unknown_type",
            "; acl:condition [ a <https://unknown.example/Condition> ] .",
        ),
    ];
    let mut results = Vec::new();
    for (name, suffix) in suffixes {
        let body = format!("{PREFIX}{RULE}{suffix}");
        let strict = standard_turtle_accepts(&body);
        let store = MemoryBackend::new();
        store
            .put("/secret.acl", Bytes::from(body), "text/turtle")
            .await
            .unwrap();
        let resolved = resolve_policy_from_storage(&store, "/secret").await;
        let PolicyOutcome::Found(doc) = resolved else {
            panic!("expected found: {resolved:?}")
        };
        let grants = evaluate_access(Some(&doc), None, "/secret", AccessMode::Read, None);
        let accepted = validate_acl_document(&doc).is_ok();
        let count = doc.graph.as_ref().unwrap()[0]
            .condition
            .as_ref()
            .map_or(0, Vec::len);
        if name == "valid" {
            assert!(strict && accepted);
            assert!(!grants);
            assert_eq!(count, 1);
        } else if name == "unknown_type" {
            assert!(strict);
            assert!(!accepted && !grants);
            assert_eq!(count, 1);
        } else {
            assert!(accepted && grants);
            assert_eq!(count, 0);
        }
        if name == "unterminated" {
            assert!(!strict);
        }
        results.push(json!({"case":name,"strict_turtle_syntax_valid":strict,"resolved":"Found","retained_conditions":count,"write_validator_accepts":accepted,"anonymous_read_granted":grants}));
    }
    json!({"id":"P1","cases":results,"http_activation_tested":false})
}

fn classifier() -> Value {
    assert!(matches!(
        classify_policy_read("/secret.acl", PolicyRead::Absent, false),
        PolicyStep::Ascend
    ));
    assert!(matches!(
        classify_policy_read(
            "/secret.acl",
            PolicyRead::Failed("offline fault".into()),
            false
        ),
        PolicyStep::Settled(PolicyOutcome::Unavailable { .. })
    ));
    assert!(matches!(
        classify_policy_read(
            "/secret.acl",
            PolicyRead::Present {
                body: b"not a policy",
                content_type: "application/json"
            },
            false
        ),
        PolicyStep::Settled(PolicyOutcome::Invalid { .. })
    ));
    let body = format!("{PREFIX}{RULE} .");
    assert!(parse_turtle_acl_with_limit(&body, body.len()).is_ok());
    assert!(parse_turtle_acl_with_limit(&body, body.len() - 1).is_err());
    assert!(parse_jsonld_acl_with_limits(b"{}", 2, 1).is_ok());
    assert!(parse_jsonld_acl_with_limits(b"{}", 1, 1).is_err());
    assert!(parse_jsonld_acl_with_limits(br#"{"x":{}}"#, 100, 1).is_err());
    json!({"id":"P2","absence_ascends":true,"failed_read_unavailable":true,"nonturtle_garbage_invalid":true,"exact_byte_limit_accepted":true,"one_byte_over_refused":true,"json_depth_refused":true})
}

fn labels() -> Value {
    let resolver = SubdomainResolver {
        base_domain: "pods.example".into(),
    };
    let valid = resolver.resolve("alice.pods.example", "/secret");
    let cleaned = resolver.resolve("al..ice.pods.example", "/secret");
    let unknown = resolver.resolve("outside.example", "/secret");
    let multi = resolver.resolve("a.b.pods.example", "/secret");
    assert_eq!(valid.pod.as_deref(), Some("alice"));
    assert_eq!(cleaned.pod, valid.pod);
    assert!(unknown.pod.is_none() && multi.pod.is_none());
    assert_eq!(unknown.storage_path, "/secret");
    json!({"id":"P3","malformed_label_normalised_to_existing_pod":cleaned.pod,"unknown_host_path_fallback":unknown.storage_path,"multi_label_path_fallback":multi.storage_path,"http_routing_or_authority_bypass_tested":false})
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"cases":[conditions().await,classifier(),labels()]}))
            .unwrap()
    );
}
