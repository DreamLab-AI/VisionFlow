use async_trait::async_trait;
use bytes::Bytes;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use solid_pod_rs::Storage;
use std::{collections::HashMap, sync::Mutex, time::Duration};
use vg_publication_audit::services::{
    ontology_generation::PublishedStorage,
    ontology_pull::{pull_once, Fetch, OntologyPullConfig, PullError, PullOutcome},
};

struct MapFetch {
    files: HashMap<String, Bytes>,
    calls: Mutex<Vec<String>>,
}
#[async_trait]
impl Fetch for MapFetch {
    async fn get(&self, url: &str) -> Result<Bytes, PullError> {
        let name = url.rsplit('/').next().unwrap();
        self.calls.lock().unwrap().push(name.into());
        self.files.get(name).cloned().ok_or_else(|| PullError::Fetch {
            url: url.into(), reason: "fixture missing".into(),
        })
    }
}
fn release(build: &str, source: &str) -> MapFetch {
    let mut files: HashMap<String, Bytes> = [
        ("visionflow.ttl", format!("<urn:{source}> <urn:p> <urn:o> .")),
        ("context.jsonld", "{}".into()),
        ("ontology.jsonld", format!("{{\"source\":\"{source}\"}}")),
        ("visionflow.stats.json", "{}".into()),
        ("index.jsonld", json!({
            "visionflow:buildSha": build, "visionflow:sourceSha": source,
            "visionflow:classes": 1, "visionflow:triples": 1,
            "ldp:contains": [{"@id":"visionflow.ttl"}, {"@id":"context.jsonld"}, {"@id":"ontology.jsonld"}]
        }).to_string()),
    ].into_iter().map(|(n, b)| (n.into(), b.into())).collect();
    let mut sums: Vec<_> = files.iter().map(|(n, b)| format!("{}  {n}\n", hex::encode(Sha256::digest(b)))).collect();
    sums.sort();
    files.insert("SHA256SUMS".into(), sums.concat().into());
    MapFetch { files, calls: Mutex::new(vec![]) }
}
fn cfg() -> OntologyPullConfig {
    OntologyPullConfig { base_url: "https://fixture.invalid".into(), enabled: true,
        timeout: Duration::from_secs(1), interval: None }
}
async fn source(store: &PublishedStorage) -> String {
    let (bytes, _) = store.get("/public/ontology/index.jsonld").await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    value["visionflow:sourceSha"].as_str().unwrap().into()
}

#[tokio::test]
async fn same_build_changed_source_is_skipped_before_hash_verification() {
    let dir = tempfile::tempdir().unwrap();
    let store = PublishedStorage::new(dir.path()).await.unwrap();
    assert!(matches!(pull_once(&release("build-1", "source-a"), &store, &cfg()).await.unwrap(), PullOutcome::Updated { .. }));
    let next = release("build-1", "source-b");
    assert!(matches!(pull_once(&next, &store, &cfg()).await.unwrap(), PullOutcome::UpToDate { .. }));
    assert_eq!(source(&store).await, "source-a");
    assert_eq!(*next.calls.lock().unwrap(), vec!["index.jsonld"]);
}

#[tokio::test]
async fn changed_build_and_source_activates_new_generation() {
    let dir = tempfile::tempdir().unwrap();
    let store = PublishedStorage::new(dir.path()).await.unwrap();
    pull_once(&release("build-1", "source-a"), &store, &cfg()).await.unwrap();
    let next = release("build-2", "source-b");
    assert!(matches!(pull_once(&next, &store, &cfg()).await.unwrap(), PullOutcome::Updated { .. }));
    assert_eq!(source(&store).await, "source-b");
    assert!(next.calls.lock().unwrap().iter().any(|n| n == "SHA256SUMS"));
}

#[tokio::test]
async fn changed_build_corrupt_bytes_refused_and_old_generation_retained() {
    let dir = tempfile::tempdir().unwrap();
    let store = PublishedStorage::new(dir.path()).await.unwrap();
    pull_once(&release("build-1", "source-a"), &store, &cfg()).await.unwrap();
    let mut next = release("build-2", "source-b");
    next.files.insert("visionflow.ttl".into(), Bytes::from_static(b"corrupt"));
    assert!(matches!(pull_once(&next, &store, &cfg()).await, Err(PullError::Digest { .. })));
    assert_eq!(source(&store).await, "source-a");
}

#[tokio::test]
async fn same_build_missing_sums_still_reports_up_to_date() {
    let dir = tempfile::tempdir().unwrap();
    let store = PublishedStorage::new(dir.path()).await.unwrap();
    pull_once(&release("build-1", "source-a"), &store, &cfg()).await.unwrap();
    let mut next = release("build-1", "source-b");
    next.files.remove("SHA256SUMS");
    assert!(matches!(pull_once(&next, &store, &cfg()).await.unwrap(), PullOutcome::UpToDate { .. }));
    assert_eq!(source(&store).await, "source-a");
}
