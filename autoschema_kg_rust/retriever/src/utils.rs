//! Utility functions and helpers for the retriever module

use sha2::{Sha256, Digest};
use std::collections::HashMap;

/// Generate hash for cache keys
pub fn generate_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Normalize vector to unit length
pub fn normalize_vector(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in vector.iter_mut() {
            *val /= norm;
        }
    }
}

/// Calculate Jaccard similarity between two sets
pub fn jaccard_similarity<T: std::hash::Hash + Eq>(set1: &std::collections::HashSet<T>, set2: &std::collections::HashSet<T>) -> f32 {
    let intersection = set1.intersection(set2).count();
    let union = set1.union(set2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Simple text similarity based on common words
pub fn text_similarity(text1: &str, text2: &str) -> f32 {
    let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
    let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

    jaccard_similarity(&words1, &words2)
}

/// Extract n-grams from text
pub fn extract_ngrams(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        return vec![];
    }

    words.windows(n)
        .map(|window| window.join(" "))
        .collect()
}

/// Clean and normalize text
pub fn clean_text(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Calculate edit distance between two strings
pub fn edit_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();

    let mut dp = vec![vec![0; chars2.len() + 1]; chars1.len() + 1];

    for i in 0..=chars1.len() {
        dp[i][0] = i;
    }

    for j in 0..=chars2.len() {
        dp[0][j] = j;
    }

    for i in 1..=chars1.len() {
        for j in 1..=chars2.len() {
            if chars1[i - 1] == chars2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            }
        }
    }

    dp[chars1.len()][chars2.len()]
}

/// Convert duration to human readable string
pub fn format_duration(duration_ms: u64) -> String {
    if duration_ms < 1000 {
        format!("{}ms", duration_ms)
    } else if duration_ms < 60_000 {
        format!("{:.1}s", duration_ms as f64 / 1000.0)
    } else {
        let minutes = duration_ms / 60_000;
        let seconds = (duration_ms % 60_000) / 1000;
        format!("{}m {}s", minutes, seconds)
    }
}

/// Format bytes to human readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Merge two hashmaps, combining values for duplicate keys
pub fn merge_maps<K, V>(map1: HashMap<K, V>, map2: HashMap<K, V>, combine_fn: impl Fn(V, V) -> V) -> HashMap<K, V>
where
    K: std::hash::Hash + Eq,
{
    let mut result = map1;
    for (key, value2) in map2 {
        result.entry(key)
            .and_modify(|value1| *value1 = combine_fn(std::mem::take(value1), value2))
            .or_insert(value2);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!(similarity > 0.9); // These vectors are very similar
    }

    #[test]
    fn test_normalize_vector() {
        let mut vector = vec![3.0, 4.0];
        normalize_vector(&mut vector);
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_text_similarity() {
        let text1 = "machine learning algorithms";
        let text2 = "learning algorithms and neural networks";
        let similarity = text_similarity(text1, text2);
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_extract_ngrams() {
        let text = "this is a test sentence";
        let bigrams = extract_ngrams(text, 2);
        assert_eq!(bigrams.len(), 4);
        assert_eq!(bigrams[0], "this is");
        assert_eq!(bigrams[1], "is a");
    }

    #[test]
    fn test_clean_text() {
        let text = "Hello, World! This is a test.";
        let cleaned = clean_text(text);
        assert_eq!(cleaned, "hello world this is a test");
    }

    #[test]
    fn test_edit_distance() {
        let distance = edit_distance("kitten", "sitting");
        assert_eq!(distance, 3);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(500), "500ms");
        assert_eq!(format_duration(1500), "1.5s");
        assert_eq!(format_duration(65000), "1m 5s");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
    }

    #[test]
    fn test_generate_hash() {
        let hash1 = generate_hash("test input");
        let hash2 = generate_hash("test input");
        let hash3 = generate_hash("different input");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // SHA256 produces 32 bytes = 64 hex chars
    }
}
"