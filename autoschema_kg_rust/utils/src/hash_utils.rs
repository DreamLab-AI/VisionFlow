//! Hash ID computation utilities for generating stable identifiers

use crate::{Result, UtilsError};
use sha2::{Sha256, Digest};
use md5;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Configuration for hash generation
#[derive(Debug, Clone)]
pub struct HashConfig {
    pub algorithm: HashAlgorithm,
    pub encoding: HashEncoding,
    pub truncate_length: Option<usize>,
    pub include_salt: bool,
    pub case_sensitive: bool,
}

/// Supported hash algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Sha256,
    Md5,
    Xxh3,
    Blake3,
}

/// Supported hash encodings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashEncoding {
    Hex,
    Base64,
    Base32,
}

impl Default for HashConfig {
    fn default() -> Self {
        Self {
            algorithm: HashAlgorithm::Sha256,
            encoding: HashEncoding::Hex,
            truncate_length: Some(16),
            include_salt: false,
            case_sensitive: true,
        }
    }
}

/// Represents a hashed identifier with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashId {
    pub hash: String,
    pub algorithm: String,
    pub encoding: String,
    pub original_length: usize,
    pub truncated: bool,
}

/// Generate hash ID from string input
pub fn generate_hash_id(input: &str, config: &HashConfig) -> Result<HashId> {
    let processed_input = if config.case_sensitive {
        input.to_string()
    } else {
        input.to_lowercase()
    };

    let hash_bytes = match config.algorithm {
        HashAlgorithm::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(processed_input.as_bytes());
            hasher.finalize().to_vec()
        }
        HashAlgorithm::Md5 => {
            let digest = md5::compute(processed_input.as_bytes());
            digest.to_vec()
        }
        HashAlgorithm::Xxh3 => {
            // Note: xxhash crate would be needed for this
            return Err(UtilsError::Custom("XXH3 algorithm not implemented".to_string()));
        }
        HashAlgorithm::Blake3 => {
            // Note: blake3 crate would be needed for this
            return Err(UtilsError::Custom("Blake3 algorithm not implemented".to_string()));
        }
    };

    let encoded = match config.encoding {
        HashEncoding::Hex => hex::encode(&hash_bytes),
        HashEncoding::Base64 => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.encode(&hash_bytes)
        }
        HashEncoding::Base32 => {
            // Simple base32 implementation or use a crate
            base32_encode(&hash_bytes)
        }
    };

    let original_length = encoded.len();
    let final_hash = if let Some(length) = config.truncate_length {
        if encoded.len() > length {
            encoded[..length].to_string()
        } else {
            encoded
        }
    } else {
        encoded
    };

    Ok(HashId {
        hash: final_hash.clone(),
        algorithm: format!("{:?}", config.algorithm),
        encoding: format!("{:?}", config.encoding),
        original_length,
        truncated: final_hash.len() < original_length,
    })
}

/// Generate hash ID from multiple string components
pub fn generate_composite_hash_id(components: &[&str], config: &HashConfig) -> Result<HashId> {
    let combined = components.join("|");
    generate_hash_id(&combined, config)
}

/// Generate hash ID from structured data (HashMap)
pub fn generate_structured_hash_id(
    data: &HashMap<String, String>,
    config: &HashConfig,
) -> Result<HashId> {
    let mut sorted_pairs: Vec<_> = data.iter().collect();
    sorted_pairs.sort_by_key(|(k, _)| *k);

    let combined = sorted_pairs
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join("&");

    generate_hash_id(&combined, config)
}

/// Generate consistent hash for any serializable object
pub fn generate_object_hash_id<T: Serialize>(
    obj: &T,
    config: &HashConfig,
) -> Result<HashId> {
    let json_string = serde_json::to_string(obj)?;
    generate_hash_id(&json_string, config)
}

/// Batch generate hash IDs for multiple inputs
pub fn batch_generate_hash_ids(
    inputs: &[String],
    config: &HashConfig,
) -> Result<Vec<HashId>> {
    inputs
        .iter()
        .map(|input| generate_hash_id(input, config))
        .collect()
}

/// Generate deterministic hash for file content without loading entire file
pub fn generate_file_hash_id<P: AsRef<std::path::Path>>(
    file_path: P,
    config: &HashConfig,
) -> Result<HashId> {
    use std::io::Read;

    let mut file = std::fs::File::open(file_path)?;
    let mut hasher = match config.algorithm {
        HashAlgorithm::Sha256 => {
            let mut hasher = Sha256::new();
            let mut buffer = [0; 8192];

            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }

            hasher.finalize().to_vec()
        }
        HashAlgorithm::Md5 => {
            let mut hasher = md5::Context::new();
            let mut buffer = [0; 8192];

            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.consume(&buffer[..bytes_read]);
            }

            hasher.compute().to_vec()
        }
        _ => {
            return Err(UtilsError::Custom("Unsupported algorithm for file hashing".to_string()));
        }
    };

    let encoded = match config.encoding {
        HashEncoding::Hex => hex::encode(&hasher),
        HashEncoding::Base64 => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.encode(&hasher)
        }
        HashEncoding::Base32 => base32_encode(&hasher),
    };

    let original_length = encoded.len();
    let final_hash = if let Some(length) = config.truncate_length {
        if encoded.len() > length {
            encoded[..length].to_string()
        } else {
            encoded
        }
    } else {
        encoded
    };

    Ok(HashId {
        hash: final_hash.clone(),
        algorithm: format!("{:?}", config.algorithm),
        encoding: format!("{:?}", config.encoding),
        original_length,
        truncated: final_hash.len() < original_length,
    })
}

/// Create a namespace-aware hash ID
pub fn generate_namespaced_hash_id(
    namespace: &str,
    value: &str,
    config: &HashConfig,
) -> Result<HashId> {
    let namespaced_value = format!("{}::{}", namespace, value);
    generate_hash_id(&namespaced_value, config)
}

/// Generate hash for graph node/edge with consistent ordering
pub fn generate_graph_element_hash_id(
    element_type: &str,
    properties: &HashMap<String, String>,
    config: &HashConfig,
) -> Result<HashId> {
    let mut all_properties = properties.clone();
    all_properties.insert("_type".to_string(), element_type.to_string());

    generate_structured_hash_id(&all_properties, config)
}

/// Verify hash integrity
pub fn verify_hash_id(original: &str, hash_id: &HashId, config: &HashConfig) -> Result<bool> {
    let computed = generate_hash_id(original, config)?;
    Ok(computed.hash == hash_id.hash)
}

/// Create hash-based unique identifier with collision detection
pub fn create_unique_hash_id(
    base_value: &str,
    existing_hashes: &std::collections::HashSet<String>,
    config: &HashConfig,
) -> Result<HashId> {
    let mut attempt = 0;
    loop {
        let value = if attempt == 0 {
            base_value.to_string()
        } else {
            format!("{}-{}", base_value, attempt)
        };

        let hash_id = generate_hash_id(&value, config)?;

        if !existing_hashes.contains(&hash_id.hash) {
            return Ok(hash_id);
        }

        attempt += 1;
        if attempt > 1000 {
            return Err(UtilsError::Custom("Too many hash collisions".to_string()));
        }
    }
}

/// Generate hash lookup table for fast deduplication
pub fn create_hash_lookup_table(
    values: &[String],
    config: &HashConfig,
) -> Result<HashMap<String, Vec<usize>>> {
    let mut lookup = HashMap::new();

    for (index, value) in values.iter().enumerate() {
        let hash_id = generate_hash_id(value, config)?;
        lookup.entry(hash_id.hash).or_insert_with(Vec::new).push(index);
    }

    Ok(lookup)
}

/// Performance-optimized hash for large datasets using Rayon
#[cfg(feature = "parallel")]
pub fn parallel_generate_hash_ids(
    inputs: &[String],
    config: &HashConfig,
) -> Result<Vec<HashId>> {
    use rayon::prelude::*;

    inputs
        .par_iter()
        .map(|input| generate_hash_id(input, config))
        .collect()
}

// Helper functions

fn base32_encode(input: &[u8]) -> String {
    // Simple base32 encoding implementation
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut result = String::new();

    let mut bits = 0u64;
    let mut bit_count = 0;

    for &byte in input {
        bits = (bits << 8) | byte as u64;
        bit_count += 8;

        while bit_count >= 5 {
            bit_count -= 5;
            let index = ((bits >> bit_count) & 0x1F) as usize;
            result.push(ALPHABET[index] as char);
        }
    }

    if bit_count > 0 {
        let index = ((bits << (5 - bit_count)) & 0x1F) as usize;
        result.push(ALPHABET[index] as char);
    }

    result
}

/// Custom hasher for specific data types
pub struct CustomHasher {
    state: u64,
}

impl CustomHasher {
    pub fn new() -> Self {
        Self { state: 0 }
    }

    pub fn hash_string(&mut self, s: &str) {
        for byte in s.bytes() {
            self.state = self.state.wrapping_mul(31).wrapping_add(byte as u64);
        }
    }

    pub fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for CustomHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_hash_generation() {
        let config = HashConfig::default();
        let hash_id = generate_hash_id("test input", &config).unwrap();

        assert!(!hash_id.hash.is_empty());
        assert_eq!(hash_id.algorithm, "Sha256");
        assert_eq!(hash_id.encoding, "Hex");
    }

    #[test]
    fn test_composite_hash() {
        let config = HashConfig::default();
        let components = ["part1", "part2", "part3"];
        let hash_id = generate_composite_hash_id(&components, &config).unwrap();

        assert!(!hash_id.hash.is_empty());
    }

    #[test]
    fn test_structured_hash() {
        let mut data = HashMap::new();
        data.insert("key1".to_string(), "value1".to_string());
        data.insert("key2".to_string(), "value2".to_string());

        let config = HashConfig::default();
        let hash_id = generate_structured_hash_id(&data, &config).unwrap();

        assert!(!hash_id.hash.is_empty());
    }

    #[test]
    fn test_hash_consistency() {
        let config = HashConfig::default();
        let input = "consistent input";

        let hash1 = generate_hash_id(input, &config).unwrap();
        let hash2 = generate_hash_id(input, &config).unwrap();

        assert_eq!(hash1.hash, hash2.hash);
    }

    #[test]
    fn test_case_sensitivity() {
        let mut config = HashConfig::default();
        config.case_sensitive = false;

        let hash1 = generate_hash_id("Test", &config).unwrap();
        let hash2 = generate_hash_id("test", &config).unwrap();

        assert_eq!(hash1.hash, hash2.hash);
    }

    #[test]
    fn test_truncation() {
        let mut config = HashConfig::default();
        config.truncate_length = Some(8);

        let hash_id = generate_hash_id("test input for truncation", &config).unwrap();

        assert_eq!(hash_id.hash.len(), 8);
        assert!(hash_id.truncated);
    }

    #[test]
    fn test_namespaced_hash() {
        let config = HashConfig::default();
        let hash_id = generate_namespaced_hash_id("users", "john_doe", &config).unwrap();

        assert!(!hash_id.hash.is_empty());
    }

    #[test]
    fn test_hash_verification() {
        let config = HashConfig::default();
        let original = "verification test";
        let hash_id = generate_hash_id(original, &config).unwrap();

        assert!(verify_hash_id(original, &hash_id, &config).unwrap());
        assert!(!verify_hash_id("different input", &hash_id, &config).unwrap());
    }

    #[test]
    fn test_unique_hash_generation() {
        let config = HashConfig::default();
        let mut existing = std::collections::HashSet::new();

        let hash1 = create_unique_hash_id("test", &existing, &config).unwrap();
        existing.insert(hash1.hash.clone());

        let hash2 = create_unique_hash_id("test", &existing, &config).unwrap();

        assert_ne!(hash1.hash, hash2.hash);
    }
}