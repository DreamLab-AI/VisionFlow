//! Text processing utilities

use crate::errors::{AutoSchemaError, Result};
use regex::Regex;
use std::sync::OnceLock;

static HTML_REGEX: OnceLock<Regex> = OnceLock::new();
static WHITESPACE_REGEX: OnceLock<Regex> = OnceLock::new();

/// Clean and normalize text content
pub fn normalize_text(text: &str) -> Result<String> {
    let mut cleaned = text.to_string();

    // Remove HTML tags
    cleaned = remove_html_tags(&cleaned)?;

    // Normalize whitespace
    cleaned = normalize_whitespace(&cleaned)?;

    // Normalize unicode
    cleaned = unicode_normalize(&cleaned);

    Ok(cleaned.trim().to_string())
}

/// Remove HTML tags from text
pub fn remove_html_tags(text: &str) -> Result<String> {
    let html_regex = HTML_REGEX.get_or_init(|| {
        Regex::new(r"<[^>]*>").expect("Invalid HTML regex")
    });

    Ok(html_regex.replace_all(text, " ").to_string())
}

/// Normalize whitespace (multiple spaces, tabs, newlines to single space)
pub fn normalize_whitespace(text: &str) -> Result<String> {
    let ws_regex = WHITESPACE_REGEX.get_or_init(|| {
        Regex::new(r"\s+").expect("Invalid whitespace regex")
    });

    Ok(ws_regex.replace_all(text, " ").to_string())
}

/// Normalize Unicode characters
pub fn unicode_normalize(text: &str) -> String {
    // Simple unicode normalization - can be extended with unicode-normalization crate
    text.chars()
        .map(|c| if c.is_control() && c != '\n' && c != '\t' { ' ' } else { c })
        .collect()
}

/// Split text into chunks with overlap
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Result<Vec<String>> {
    if chunk_size == 0 {
        return Err(AutoSchemaError::text_processing("Chunk size must be > 0"));
    }

    if overlap >= chunk_size {
        return Err(AutoSchemaError::text_processing("Overlap must be < chunk_size"));
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Ok(vec![]);
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);

        if end >= words.len() {
            break;
        }

        start = end - overlap;
    }

    Ok(chunks)
}

/// Extract sentences from text
pub fn extract_sentences(text: &str) -> Vec<String> {
    // Simple sentence splitting - can be improved with proper NLP libraries
    text.split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.len() > 3)
        .collect()
}

/// Calculate text similarity using simple word overlap
pub fn text_similarity(text1: &str, text2: &str) -> f32 {
    let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
    let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

    if words1.is_empty() && words2.is_empty() {
        return 1.0;
    }

    if words1.is_empty() || words2.is_empty() {
        return 0.0;
    }

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    intersection as f32 / union as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text() {
        let input = "<p>Hello   world!</p>\n\n\tTest\r\n";
        let result = normalize_text(input).unwrap();
        assert_eq!(result, "Hello world! Test");
    }

    #[test]
    fn test_chunk_text() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, 3, 1).unwrap();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], "one two three");
        assert_eq!(chunks[1], "three four five");
    }

    #[test]
    fn test_text_similarity() {
        let text1 = "hello world test";
        let text2 = "hello world example";
        let similarity = text_similarity(text1, text2);
        assert!(similarity > 0.0 && similarity < 1.0);
    }
}