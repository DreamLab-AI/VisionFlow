//! Text cleaning utilities for preprocessing and normalization

use crate::{Result, UtilsError};
use regex::Regex;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

/// Configuration for text cleaning operations
#[derive(Debug, Clone)]
pub struct TextCleaningConfig {
    pub normalize_unicode: bool,
    pub remove_html: bool,
    pub remove_urls: bool,
    pub remove_emails: bool,
    pub remove_phone_numbers: bool,
    pub normalize_whitespace: bool,
    pub convert_to_lowercase: bool,
    pub remove_punctuation: bool,
    pub remove_numbers: bool,
    pub remove_stop_words: bool,
    pub min_word_length: usize,
    pub max_word_length: usize,
    pub custom_replacements: HashMap<String, String>,
}

impl Default for TextCleaningConfig {
    fn default() -> Self {
        Self {
            normalize_unicode: true,
            remove_html: true,
            remove_urls: true,
            remove_emails: false,
            remove_phone_numbers: false,
            normalize_whitespace: true,
            convert_to_lowercase: false,
            remove_punctuation: false,
            remove_numbers: false,
            remove_stop_words: false,
            min_word_length: 1,
            max_word_length: 50,
            custom_replacements: HashMap::new(),
        }
    }
}

/// Text cleaning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningStats {
    pub original_length: usize,
    pub cleaned_length: usize,
    pub reduction_ratio: f64,
    pub words_removed: usize,
    pub urls_removed: usize,
    pub emails_removed: usize,
    pub html_tags_removed: usize,
}

/// Clean text according to configuration
pub fn clean_text(input: &str, config: &TextCleaningConfig) -> Result<(String, CleaningStats)> {
    let mut text = input.to_string();
    let original_length = text.len();

    let mut stats = CleaningStats {
        original_length,
        cleaned_length: 0,
        reduction_ratio: 0.0,
        words_removed: 0,
        urls_removed: 0,
        emails_removed: 0,
        html_tags_removed: 0,
    };

    // Unicode normalization
    if config.normalize_unicode {
        text = text.nfc().collect::<String>();
    }

    // Remove HTML tags
    if config.remove_html {
        let (cleaned, count) = remove_html_tags(&text)?;
        text = cleaned;
        stats.html_tags_removed = count;
    }

    // Remove URLs
    if config.remove_urls {
        let (cleaned, count) = remove_urls(&text)?;
        text = cleaned;
        stats.urls_removed = count;
    }

    // Remove emails
    if config.remove_emails {
        let (cleaned, count) = remove_emails(&text)?;
        text = cleaned;
        stats.emails_removed = count;
    }

    // Remove phone numbers
    if config.remove_phone_numbers {
        text = remove_phone_numbers(&text)?;
    }

    // Apply custom replacements
    for (pattern, replacement) in &config.custom_replacements {
        text = text.replace(pattern, replacement);
    }

    // Normalize whitespace
    if config.normalize_whitespace {
        text = normalize_whitespace(&text);
    }

    // Convert to lowercase
    if config.convert_to_lowercase {
        text = text.to_lowercase();
    }

    // Remove punctuation
    if config.remove_punctuation {
        text = remove_punctuation(&text);
    }

    // Remove numbers
    if config.remove_numbers {
        text = remove_numbers(&text)?;
    }

    // Word-level processing
    if config.remove_stop_words || config.min_word_length > 1 || config.max_word_length < 50 {
        let original_word_count = text.split_whitespace().count();
        text = filter_words(&text, config)?;
        let new_word_count = text.split_whitespace().count();
        stats.words_removed = original_word_count.saturating_sub(new_word_count);
    }

    // Final whitespace normalization
    text = normalize_whitespace(&text);

    stats.cleaned_length = text.len();
    stats.reduction_ratio = if original_length > 0 {
        1.0 - (stats.cleaned_length as f64 / original_length as f64)
    } else {
        0.0
    };

    Ok((text, stats))
}

/// Batch clean multiple texts
pub fn batch_clean_text(
    inputs: &[String],
    config: &TextCleaningConfig,
) -> Result<Vec<(String, CleaningStats)>> {
    inputs
        .iter()
        .map(|text| clean_text(text, config))
        .collect()
}

/// Clean text preserving sentence structure
pub fn clean_text_preserve_sentences(
    input: &str,
    config: &TextCleaningConfig,
) -> Result<Vec<String>> {
    let sentences = split_into_sentences(input)?;
    let mut cleaned_sentences = Vec::new();

    for sentence in sentences {
        let (cleaned, _) = clean_text(&sentence, config)?;
        if !cleaned.trim().is_empty() {
            cleaned_sentences.push(cleaned.trim().to_string());
        }
    }

    Ok(cleaned_sentences)
}

/// Extract and clean specific text patterns
pub fn extract_and_clean_patterns(
    input: &str,
    pattern: &str,
    config: &TextCleaningConfig,
) -> Result<Vec<String>> {
    let regex = Regex::new(pattern)
        .map_err(|e| UtilsError::Custom(format!("Invalid regex pattern: {}", e)))?;

    let mut results = Vec::new();
    for capture in regex.captures_iter(input) {
        if let Some(matched) = capture.get(0) {
            let (cleaned, _) = clean_text(matched.as_str(), config)?;
            if !cleaned.trim().is_empty() {
                results.push(cleaned.trim().to_string());
            }
        }
    }

    Ok(results)
}

/// Clean and tokenize text
pub fn clean_and_tokenize(
    input: &str,
    config: &TextCleaningConfig,
) -> Result<Vec<String>> {
    let (cleaned, _) = clean_text(input, config)?;
    Ok(tokenize(&cleaned))
}

/// Remove specific character sets
pub fn remove_character_sets(input: &str, char_sets: &[&str]) -> Result<String> {
    let mut text = input.to_string();

    for char_set in char_sets {
        match *char_set {
            "emoji" => text = remove_emojis(&text),
            "accents" => text = remove_accents(&text),
            "symbols" => text = remove_symbols(&text),
            "control" => text = remove_control_characters(&text),
            _ => return Err(UtilsError::Custom(format!("Unknown character set: {}", char_set))),
        }
    }

    Ok(text)
}

/// Detect and normalize different types of quotes
pub fn normalize_quotes(input: &str) -> String {
    input
        .replace('"', '"')
        .replace('"', '"')
        .replace(''', "'")
        .replace(''', "'")
        .replace('`', "'")
        .replace('´', "'")
}

/// Normalize line endings
pub fn normalize_line_endings(input: &str, target: &str) -> String {
    match target {
        "unix" | "\\n" => input.replace("\\r\\n", "\\n").replace('\\r', "\\n"),
        "windows" | "\\r\\n" => input.replace("\\n", "\\r\\n"),
        "mac" | "\\r" => input.replace("\\r\\n", "\\r").replace('\\n', "\\r"),
        _ => input.to_string(),
    }
}

// Helper functions

fn remove_html_tags(input: &str) -> Result<(String, usize)> {
    let regex = Regex::new(r"<[^>]*>")
        .map_err(|e| UtilsError::Custom(format!("HTML regex error: {}", e)))?;

    let count = regex.find_iter(input).count();
    let cleaned = regex.replace_all(input, "").to_string();

    Ok((cleaned, count))
}

fn remove_urls(input: &str) -> Result<(String, usize)> {
    let regex = Regex::new(r"https?://[^\\s]+|www\\.[^\\s]+|[^\\s]+\\.[a-z]{2,}(?:/[^\\s]*)?")
        .map_err(|e| UtilsError::Custom(format!("URL regex error: {}", e)))?;

    let count = regex.find_iter(input).count();
    let cleaned = regex.replace_all(input, "").to_string();

    Ok((cleaned, count))
}

fn remove_emails(input: &str) -> Result<(String, usize)> {
    let regex = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
        .map_err(|e| UtilsError::Custom(format!("Email regex error: {}", e)))?;

    let count = regex.find_iter(input).count();
    let cleaned = regex.replace_all(input, "").to_string();

    Ok((cleaned, count))
}

fn remove_phone_numbers(input: &str) -> Result<String> {
    let regex = Regex::new(r"\\b(?:\\+?1[-.]?)?\\(?[0-9]{3}\\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\\b")
        .map_err(|e| UtilsError::Custom(format!("Phone regex error: {}", e)))?;

    Ok(regex.replace_all(input, "").to_string())
}

fn normalize_whitespace(input: &str) -> String {
    let regex = Regex::new(r"\\s+").unwrap();
    regex.replace_all(input.trim(), " ").to_string()
}

fn remove_punctuation(input: &str) -> String {
    input.chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect()
}

fn remove_numbers(input: &str) -> Result<String> {
    let regex = Regex::new(r"\\b\\d+\\b")
        .map_err(|e| UtilsError::Custom(format!("Number regex error: {}", e)))?;

    Ok(regex.replace_all(input, "").to_string())
}

fn filter_words(input: &str, config: &TextCleaningConfig) -> Result<String> {
    let words: Vec<&str> = input.split_whitespace().collect();
    let mut filtered_words = Vec::new();

    for word in words {
        let word_len = word.len();

        // Length filter
        if word_len < config.min_word_length || word_len > config.max_word_length {
            continue;
        }

        // Stop words filter
        if config.remove_stop_words && is_stop_word(word) {
            continue;
        }

        filtered_words.push(word);
    }

    Ok(filtered_words.join(" "))
}

fn is_stop_word(word: &str) -> bool {
    // Basic English stop words list
    const STOP_WORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
        "will", "with", "the", "this", "but", "they", "have", "had", "what", "said",
        "each", "which", "she", "do", "how", "their", "if", "up", "out", "many",
        "then", "them", "these", "so", "some", "her", "would", "make", "like",
        "into", "him", "time", "two", "more", "go", "no", "way", "could", "my",
        "than", "first", "been", "call", "who", "oil", "sit", "now", "find",
        "down", "day", "did", "get", "come", "made", "may", "part"
    ];

    STOP_WORDS.contains(&word.to_lowercase().as_str())
}

fn split_into_sentences(input: &str) -> Result<Vec<String>> {
    let regex = Regex::new(r"[.!?]+\\s+")
        .map_err(|e| UtilsError::Custom(format!("Sentence regex error: {}", e)))?;

    let sentences: Vec<String> = regex.split(input)
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect();

    Ok(sentences)
}

fn tokenize(input: &str) -> Vec<String> {
    input.split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn remove_emojis(input: &str) -> String {
    input.chars()
        .filter(|c| {
            let code = *c as u32;
            // Basic emoji ranges (this is simplified)
            !(code >= 0x1F600 && code <= 0x1F64F) && // Emoticons
            !(code >= 0x1F300 && code <= 0x1F5FF) && // Misc Symbols
            !(code >= 0x1F680 && code <= 0x1F6FF) && // Transport
            !(code >= 0x2600 && code <= 0x26FF) &&   // Misc symbols
            !(code >= 0x2700 && code <= 0x27BF)      // Dingbats
        })
        .collect()
}

fn remove_accents(input: &str) -> String {
    input.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .collect()
}

fn remove_symbols(input: &str) -> String {
    input.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect()
}

fn remove_control_characters(input: &str) -> String {
    input.chars()
        .filter(|c| !c.is_control())
        .collect()
}

/// Advanced text normalization for better matching
pub fn normalize_for_matching(input: &str) -> String {
    let mut config = TextCleaningConfig::default();
    config.normalize_unicode = true;
    config.convert_to_lowercase = true;
    config.remove_punctuation = true;
    config.normalize_whitespace = true;

    let (normalized, _) = clean_text(input, &config).unwrap_or_else(|_| (input.to_string(), CleaningStats {
        original_length: input.len(),
        cleaned_length: input.len(),
        reduction_ratio: 0.0,
        words_removed: 0,
        urls_removed: 0,
        emails_removed: 0,
        html_tags_removed: 0,
    }));

    normalized
}

/// Extract and clean text from various file formats
pub fn extract_clean_text_from_content(
    content: &str,
    content_type: &str,
    config: &TextCleaningConfig,
) -> Result<String> {
    let extracted = match content_type.to_lowercase().as_str() {
        "html" => {
            let (text, _) = remove_html_tags(content)?;
            text
        }
        "markdown" => {
            // Basic markdown cleanup
            let mut text = content.to_string();
            // Remove markdown syntax
            text = Regex::new(r"#{1,6}\\s*")?.replace_all(&text, "").to_string();
            text = Regex::new(r"\\*\\*([^*]+)\\*\\*")?.replace_all(&text, "$1").to_string();
            text = Regex::new(r"\\*([^*]+)\\*")?.replace_all(&text, "$1").to_string();
            text = Regex::new(r"\\[([^\\]]+)\\]\\([^)]+\\)")?.replace_all(&text, "$1").to_string();
            text
        }
        "plain" | "text" | _ => content.to_string(),
    };

    let (cleaned, _) = clean_text(&extracted, config)?;
    Ok(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_text_cleaning() {
        let input = "  Hello    World!  ";
        let config = TextCleaningConfig {
            normalize_whitespace: true,
            ..Default::default()
        };

        let (cleaned, stats) = clean_text(input, &config).unwrap();
        assert_eq!(cleaned, "Hello World!");
        assert!(stats.reduction_ratio > 0.0);
    }

    #[test]
    fn test_html_removal() {
        let input = "<div>Hello <b>World</b>!</div>";
        let config = TextCleaningConfig {
            remove_html: true,
            normalize_whitespace: true,
            ..Default::default()
        };

        let (cleaned, stats) = clean_text(input, &config).unwrap();
        assert_eq!(cleaned, "Hello World!");
        assert_eq!(stats.html_tags_removed, 3);
    }

    #[test]
    fn test_url_removal() {
        let input = "Visit https://example.com for more info";
        let config = TextCleaningConfig {
            remove_urls: true,
            normalize_whitespace: true,
            ..Default::default()
        };

        let (cleaned, stats) = clean_text(input, &config).unwrap();
        assert_eq!(cleaned, "Visit for more info");
        assert_eq!(stats.urls_removed, 1);
    }

    #[test]
    fn test_email_removal() {
        let input = "Contact me at user@example.com";
        let config = TextCleaningConfig {
            remove_emails: true,
            normalize_whitespace: true,
            ..Default::default()
        };

        let (cleaned, stats) = clean_text(input, &config).unwrap();
        assert_eq!(cleaned, "Contact me at");
        assert_eq!(stats.emails_removed, 1);
    }

    #[test]
    fn test_stop_word_removal() {
        let input = "The quick brown fox jumps over the lazy dog";
        let config = TextCleaningConfig {
            remove_stop_words: true,
            ..Default::default()
        };

        let (cleaned, _) = clean_text(input, &config).unwrap();
        assert!(!cleaned.contains(" the "));
        assert!(cleaned.contains("quick"));
        assert!(cleaned.contains("brown"));
    }

    #[test]
    fn test_word_length_filtering() {
        let input = "a big extraordinary word";
        let config = TextCleaningConfig {
            min_word_length: 3,
            max_word_length: 10,
            ..Default::default()
        };

        let (cleaned, _) = clean_text(input, &config).unwrap();
        assert!(!cleaned.contains(" a "));
        assert!(!cleaned.contains("extraordinary"));
        assert!(cleaned.contains("big"));
        assert!(cleaned.contains("word"));
    }

    #[test]
    fn test_normalize_quotes() {
        let input = ""Hello" and 'world'";
        let normalized = normalize_quotes(input);
        assert_eq!(normalized, "\\"Hello\\" and 'world'");
    }

    #[test]
    fn test_line_ending_normalization() {
        let input = "line1\\r\\nline2\\nline3\\r";
        let unix = normalize_line_endings(input, "unix");
        assert_eq!(unix, "line1\\nline2\\nline3\\n");

        let windows = normalize_line_endings("line1\\nline2", "windows");
        assert_eq!(windows, "line1\\r\\nline2");
    }

    #[test]
    fn test_sentence_preservation() {
        let input = "First sentence. Second sentence! Third sentence?";
        let config = TextCleaningConfig::default();

        let sentences = clean_text_preserve_sentences(input, &config).unwrap();
        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("First"));
        assert!(sentences[1].contains("Second"));
        assert!(sentences[2].contains("Third"));
    }
}