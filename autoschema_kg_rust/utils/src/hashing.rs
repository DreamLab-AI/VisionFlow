//! Hashing utilities for content deduplication and checksums

use sha2::{Digest, Sha256};

/// Calculate SHA256 hash of input text
pub fn sha256_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

/// Calculate MD5 hash of input text (for legacy compatibility)
pub fn md5_hash(input: &str) -> String {
    let digest = md5::compute(input.as_bytes());
    hex::encode(digest.0)
}

/// Generate a content ID based on text content
pub fn content_id(text: &str) -> String {
    format!("content_{}", sha256_hash(text)[..16].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_hash() {
        let hash = sha256_hash("test content");
        assert_eq!(hash.len(), 64); // SHA256 produces 64 hex characters
    }

    #[test]
    fn test_content_id() {
        let id = content_id("test content");
        assert!(id.starts_with("content_"));
        assert_eq!(id.len(), 24); // "content_" + 16 chars
    }
}