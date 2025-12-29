//! NIP-98 HTTP Authentication for Solid Server Integration
//!
//! Generates Nostr events for HTTP authentication as defined in:
//! - NIP-98: https://nips.nostr.com/98
//! - JIP-0001: https://github.com/JavaScriptSolidServer/jips/blob/main/jip-0001.md
//!
//! Authorization header format: "Nostr <base64-encoded-event>"

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use log::{debug, error};
use nostr_sdk::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// NIP-98 HTTP Auth event kind (references RFC 7235)
const HTTP_AUTH_KIND: u16 = 27235;

/// Errors from NIP-98 operations
#[derive(Debug, Error)]
pub enum Nip98Error {
    #[error("Failed to create Nostr keys: {0}")]
    KeyCreation(String),
    #[error("Failed to build event: {0}")]
    EventBuild(String),
    #[error("Failed to sign event: {0}")]
    EventSign(String),
    #[error("Failed to serialize event: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// NIP-98 event structure for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nip98Event {
    pub id: String,
    pub pubkey: String,
    pub created_at: i64,
    pub kind: u16,
    pub tags: Vec<Vec<String>>,
    pub content: String,
    pub sig: String,
}

/// Configuration for generating NIP-98 tokens
#[derive(Debug, Clone)]
pub struct Nip98Config {
    /// Target URL for the request
    pub url: String,
    /// HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD)
    pub method: String,
    /// Optional request body for payload hash
    pub body: Option<String>,
}

/// Generate a NIP-98 authentication token for a request
///
/// Returns a base64-encoded Nostr event that can be used in the
/// Authorization header as: `Authorization: Nostr <token>`
///
/// # Arguments
/// * `keys` - The Nostr Keys (secret key) to sign with
/// * `config` - Configuration for the NIP-98 request
///
/// # Returns
/// Base64-encoded event string
pub fn generate_nip98_token(keys: &Keys, config: &Nip98Config) -> Result<String, Nip98Error> {
    let now = Timestamp::now();

    // Build tags
    let mut tags: Vec<Tag> = vec![
        Tag::custom(TagKind::Custom("u".into()), vec![config.url.clone()]),
        Tag::custom(
            TagKind::Custom("method".into()),
            vec![config.method.to_uppercase()],
        ),
    ];

    // Add payload hash if body is provided
    if let Some(body) = &config.body {
        let hash = compute_payload_hash(body);
        tags.push(Tag::custom(
            TagKind::Custom("payload".into()),
            vec![hash],
        ));
    }

    // Build the event
    let event = EventBuilder::new(Kind::Custom(HTTP_AUTH_KIND), "")
        .tags(tags)
        .sign_with_keys(keys)
        .map_err(|e| Nip98Error::EventSign(e.to_string()))?;

    // Convert to our serialization format
    let nip98_event = Nip98Event {
        id: event.id.to_hex(),
        pubkey: event.pubkey.to_hex(),
        created_at: event.created_at.as_u64() as i64,
        kind: HTTP_AUTH_KIND,
        tags: event
            .tags
            .iter()
            .map(|t| t.as_slice().iter().map(|s| s.to_string()).collect())
            .collect(),
        content: event.content.clone(),
        sig: event.sig.to_string(),
    };

    // Serialize to JSON and base64 encode
    let json = serde_json::to_string(&nip98_event)?;
    let token = BASE64.encode(json.as_bytes());

    debug!(
        "Generated NIP-98 token for {} {} (pubkey: {}...)",
        config.method,
        config.url,
        &nip98_event.pubkey[..16]
    );

    Ok(token)
}

/// Generate NIP-98 token from hex secret key
///
/// # Arguments
/// * `secret_key_hex` - 64-character hex secret key
/// * `config` - Configuration for the NIP-98 request
pub fn generate_nip98_token_from_hex(
    secret_key_hex: &str,
    config: &Nip98Config,
) -> Result<String, Nip98Error> {
    let secret_key = SecretKey::from_hex(secret_key_hex)
        .map_err(|e| Nip98Error::KeyCreation(e.to_string()))?;
    let keys = Keys::new(secret_key);
    generate_nip98_token(&keys, config)
}

/// Compute SHA256 hash of payload for the 'payload' tag
fn compute_payload_hash(body: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

/// Build the Authorization header value
///
/// # Arguments
/// * `token` - The base64-encoded NIP-98 token
///
/// # Returns
/// Full header value: "Nostr <token>"
pub fn build_auth_header(token: &str) -> String {
    format!("Nostr {}", token)
}

/// Extract pubkey from a NIP-98 token (for validation/logging)
pub fn extract_pubkey_from_token(token: &str) -> Option<String> {
    let decoded = BASE64.decode(token).ok()?;
    let json_str = String::from_utf8(decoded).ok()?;
    let event: Nip98Event = serde_json::from_str(&json_str).ok()?;
    Some(event.pubkey)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_nip98_token() {
        let keys = Keys::generate();
        let config = Nip98Config {
            url: "http://localhost:3030/pods/test/".to_string(),
            method: "GET".to_string(),
            body: None,
        };

        let token = generate_nip98_token(&keys, &config).expect("Failed to generate token");
        assert!(!token.is_empty());

        // Verify we can extract the pubkey
        let pubkey = extract_pubkey_from_token(&token).expect("Failed to extract pubkey");
        assert_eq!(pubkey, keys.public_key().to_hex());
    }

    #[test]
    fn test_generate_nip98_token_with_body() {
        let keys = Keys::generate();
        let config = Nip98Config {
            url: "http://localhost:3030/pods/test/data.jsonld".to_string(),
            method: "PUT".to_string(),
            body: Some(r#"{"@context": "https://schema.org"}"#.to_string()),
        };

        let token = generate_nip98_token(&keys, &config).expect("Failed to generate token");
        assert!(!token.is_empty());
    }

    #[test]
    fn test_payload_hash() {
        let body = r#"{"test": "data"}"#;
        let hash = compute_payload_hash(body);
        assert_eq!(hash.len(), 64); // SHA256 hex is 64 chars
    }

    #[test]
    fn test_build_auth_header() {
        let token = "dGVzdA==";
        let header = build_auth_header(token);
        assert_eq!(header, "Nostr dGVzdA==");
    }
}
