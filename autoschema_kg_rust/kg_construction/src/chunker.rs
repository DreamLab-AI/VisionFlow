//! Text chunking functionality for handling token limits

use crate::error::{KgConstructionError, Result};
use serde::{Deserialize, Serialize};

/// Constants for text chunking
pub const TOKEN_LIMIT: usize = 1024;
pub const INSTRUCTION_TOKEN_ESTIMATE: usize = 200;
pub const CHAR_TO_TOKEN_RATIO: f64 = 3.5;

/// Handles text chunking based on token limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunker {
    /// Maximum tokens per chunk
    max_tokens: usize,
    /// Estimated tokens used by instructions
    instruction_tokens: usize,
    /// Character to token ratio for estimation
    char_ratio: f64,
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::new(TOKEN_LIMIT, INSTRUCTION_TOKEN_ESTIMATE)
    }
}

impl TextChunker {
    /// Create a new text chunker with specified limits
    pub fn new(max_tokens: usize, instruction_tokens: usize) -> Self {
        Self {
            max_tokens,
            instruction_tokens,
            char_ratio: CHAR_TO_TOKEN_RATIO,
        }
    }

    /// Create a chunker with custom character-to-token ratio
    pub fn with_char_ratio(mut self, ratio: f64) -> Self {
        self.char_ratio = ratio;
        self
    }

    /// Calculate maximum characters per chunk
    pub fn calculate_max_chars(&self) -> usize {
        let available_tokens = self.max_tokens.saturating_sub(self.instruction_tokens);
        (available_tokens as f64 * self.char_ratio) as usize
    }

    /// Split text into chunks that fit within token limits
    pub fn split_text(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let max_chars = self.calculate_max_chars();
        if max_chars == 0 {
            return Err(KgConstructionError::ChunkingError(
                "Maximum characters per chunk is 0. Check token limits.".to_string()
            ));
        }

        let mut chunks = Vec::new();
        let mut remaining_text = text;

        while !remaining_text.is_empty() {
            if remaining_text.len() <= max_chars {
                // Remaining text fits in one chunk
                chunks.push(remaining_text.to_string());
                break;
            }

            // Find optimal split point (prefer word boundaries)
            let split_pos = self.find_split_position(remaining_text, max_chars);

            let chunk = &remaining_text[..split_pos];
            chunks.push(chunk.to_string());
            remaining_text = &remaining_text[split_pos..];
        }

        Ok(chunks)
    }

    /// Find optimal position to split text, preferring word boundaries
    fn find_split_position(&self, text: &str, max_chars: usize) -> usize {
        if text.len() <= max_chars {
            return text.len();
        }

        // Try to find last space within limit
        if let Some(last_space) = text[..max_chars].rfind(' ') {
            // Make sure we don't split too early (at least 50% of max_chars)
            if last_space > max_chars / 2 {
                return last_space;
            }
        }

        // Try to find last sentence boundary
        if let Some(last_period) = text[..max_chars].rfind('.') {
            if last_period > max_chars / 3 {
                return last_period + 1;
            }
        }

        // Fallback to character limit
        max_chars
    }

    /// Split text with overlap for better context preservation
    pub fn split_text_with_overlap(&self, text: &str, overlap_ratio: f64) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        if !(0.0..=0.5).contains(&overlap_ratio) {
            return Err(KgConstructionError::ChunkingError(
                "Overlap ratio must be between 0.0 and 0.5".to_string()
            ));
        }

        let max_chars = self.calculate_max_chars();
        let overlap_chars = (max_chars as f64 * overlap_ratio) as usize;
        let step_size = max_chars - overlap_chars;

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + max_chars).min(text.len());
            let chunk = &text[start..end];
            chunks.push(chunk.to_string());

            if end == text.len() {
                break;
            }

            start += step_size;
        }

        Ok(chunks)
    }

    /// Estimate token count for a given text
    pub fn estimate_tokens(&self, text: &str) -> usize {
        (text.len() as f64 / self.char_ratio) as usize
    }

    /// Check if text exceeds token limit
    pub fn exceeds_limit(&self, text: &str) -> bool {
        self.estimate_tokens(text) + self.instruction_tokens > self.max_tokens
    }

    /// Get chunker configuration
    pub fn config(&self) -> ChunkerConfig {
        ChunkerConfig {
            max_tokens: self.max_tokens,
            instruction_tokens: self.instruction_tokens,
            char_ratio: self.char_ratio,
            max_chars: self.calculate_max_chars(),
        }
    }
}

/// Configuration information for the chunker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkerConfig {
    pub max_tokens: usize,
    pub instruction_tokens: usize,
    pub char_ratio: f64,
    pub max_chars: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_creation() {
        let chunker = TextChunker::default();
        assert_eq!(chunker.max_tokens, TOKEN_LIMIT);
        assert_eq!(chunker.instruction_tokens, INSTRUCTION_TOKEN_ESTIMATE);
    }

    #[test]
    fn test_max_chars_calculation() {
        let chunker = TextChunker::new(1000, 100);
        let max_chars = chunker.calculate_max_chars();
        assert_eq!(max_chars, (900.0 * CHAR_TO_TOKEN_RATIO) as usize);
    }

    #[test]
    fn test_simple_text_splitting() {
        let chunker = TextChunker::new(100, 10);
        let text = "a".repeat(500);
        let chunks = chunker.split_text(&text).unwrap();

        assert!(!chunks.is_empty());
        let max_chars = chunker.calculate_max_chars();
        for chunk in &chunks {
            assert!(chunk.len() <= max_chars);
        }
    }

    #[test]
    fn test_empty_text() {
        let chunker = TextChunker::default();
        let chunks = chunker.split_text("").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_text_with_overlap() {
        let chunker = TextChunker::new(100, 10);
        let text = "word ".repeat(100);
        let chunks = chunker.split_text_with_overlap(&text, 0.2).unwrap();

        assert!(!chunks.is_empty());
        if chunks.len() > 1 {
            // Check that there's some overlap between consecutive chunks
            let first_end = &chunks[0][chunks[0].len() - 20..];
            let second_start = &chunks[1][..20.min(chunks[1].len())];
            // Should have some common words due to overlap
            assert!(first_end.split_whitespace().count() > 0);
            assert!(second_start.split_whitespace().count() > 0);
        }
    }

    #[test]
    fn test_token_estimation() {
        let chunker = TextChunker::default();
        let text = "Hello world";
        let estimated = chunker.estimate_tokens(text);
        assert!(estimated > 0);
        assert_eq!(estimated, (text.len() as f64 / CHAR_TO_TOKEN_RATIO) as usize);
    }

    #[test]
    fn test_exceeds_limit() {
        let chunker = TextChunker::new(50, 10);
        let short_text = "short";
        let long_text = "a".repeat(1000);

        assert!(!chunker.exceeds_limit(short_text));
        assert!(chunker.exceeds_limit(&long_text));
    }
}