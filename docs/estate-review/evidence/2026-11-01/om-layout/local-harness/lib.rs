use async_trait::async_trait;
use vault_core::vocabulary::Vocabulary;

/// One markdown page offered by a [`CorpusSource`].
///
/// The field set is exactly what the sync pipeline consumes: an identity
/// (`path`, resolved against the source's base paths into a vault identity), a
/// display name, a change marker for the incremental filter, and an opaque
/// handle the source itself understands for fetching the body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusPage {
    /// File name including the `.md` extension.
    pub name: String,
    /// Source-relative path with `/` separators, e.g. `knowledge/Foo.md`.
    /// Always prefixed by one of [`CorpusSource::base_paths`].
    pub path: String,
    /// Opaque change marker: the blob SHA for GitHub, `mtime:size` on disk.
    /// Equal markers across two listings mean the page did not change.
    pub change_marker: String,
    /// Size in bytes, for telemetry only.
    pub size: u64,
    /// Handle [`CorpusSource::fetch_page`] resolves: a download URL for GitHub,
    /// an absolute filesystem path for a local directory.
    pub fetch_ref: String,
}

/// How a source describes itself in logs, statistics and the sync-identity
/// marker that triggers a forced full re-sync when the source changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDescriptor {
    /// Stable discriminator: `github` or `local`.
    pub kind: &'static str,
    /// Where the corpus is read from: `owner/repo@branch` or a vault root.
    pub location: String,
    /// The configured ingest base paths, in order.
    pub base_paths: Vec<String>,
}

impl SourceDescriptor {
    /// The identity string persisted in the sync database. A change forces a
    /// full re-sync so a store built from one source is never topped up
    /// incrementally from another.
    pub fn identity(&self) -> String {
        format!(
            "{}:{}:{}",
            self.kind,
            self.location,
            self.base_paths.join(",")
        )
    }
}

impl std::fmt::Display for SourceDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} corpus at {} (sources: {})",
            self.kind,
            self.location,
            self.base_paths.join(", ")
        )
    }
}

/// Where the knowledge corpus is read from.
///
/// Implementations must be cheap to clone behind an `Arc` and safe to call
/// concurrently — the sync pipeline fetches eight pages at a time.
#[async_trait]
pub trait CorpusSource: Send + Sync {
    /// Self-description for telemetry and the sync-identity marker.
    fn describe(&self) -> SourceDescriptor;

    /// The configured ingest base paths, in order. Page identity is resolved
    /// relative to these, so a knowledge page and its working twin share one
    /// identity (ADR-2040 §V1).
    fn base_paths(&self) -> &[String];

    /// Every markdown page under every base path, with its change marker.
    async fn list_pages(&self) -> Result<Vec<CorpusPage>, String>;

    /// Every markdown page under one prefix (a namespace such as the decisions
    /// directory). An absent prefix is an error, not an empty list, so callers
    /// can tell "no such namespace" from "namespace is empty".
    async fn list_pages_under(&self, prefix: &str) -> Result<Vec<CorpusPage>, String>;

    /// Fetch one page's body via its [`CorpusPage::fetch_ref`].
    async fn fetch_page(&self, page: &CorpusPage) -> Result<String, String>;

    /// The corpus vocabulary (`ontology/vocabulary.yaml`, contract C1), which
    /// maps every frontmatter relation key to its OWL property.
    ///
    /// `Ok(None)` when the source cannot supply one — the default, and the
    /// GitHub source's answer. The sync then ingests pages and wikilinks but
    /// emits no typed relation edges. An `Err` is a vocabulary that exists
    /// but does not load, which must not be silently ignored.
    async fn vocabulary(&self) -> Result<Option<Vocabulary>, String> {
        Ok(None)
    }
}


#[path = "../src/services/corpus_source/local.rs"]
pub mod local;
