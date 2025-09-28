use serde::{Deserialize, Serialize};
use regex::Regex;
use std::collections::HashMap;
use once_cell::sync::Lazy;

use crate::{Result, LLMError, GenerationResponse, TokenUsage};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedResponse<T> {
    pub data: T,
    pub raw_response: String,
    pub confidence: f64,
    pub validation_errors: Vec<ValidationError>,
}

pub trait ResponseParser<T>: Send + Sync {
    fn parse(&self, response: &GenerationResponse) -> Result<ParsedResponse<T>>;
    fn validate(&self, data: &T) -> Vec<ValidationError>;
    fn extract_metadata(&self, response: &GenerationResponse) -> HashMap<String, serde_json::Value>;
}

// JSON response parser
pub struct JsonResponseParser<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> JsonResponseParser<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> ResponseParser<T> for JsonResponseParser<T>
where
    T: for<'de> Deserialize<'de> + Send + Sync,
{
    fn parse(&self, response: &GenerationResponse) -> Result<ParsedResponse<T>> {
        let cleaned_text = extract_json_from_text(&response.text)?;

        match serde_json::from_str::<T>(&cleaned_text) {
            Ok(data) => {
                let validation_errors = self.validate(&data);
                let confidence = calculate_json_confidence(&response.text, &cleaned_text);

                Ok(ParsedResponse {
                    data,
                    raw_response: response.text.clone(),
                    confidence,
                    validation_errors,
                })
            }
            Err(e) => Err(LLMError::InvalidResponse(format!("JSON parsing failed: {}", e))),
        }
    }

    fn validate(&self, _data: &T) -> Vec<ValidationError> {
        // Base implementation has no validation rules
        Vec::new()
    }

    fn extract_metadata(&self, response: &GenerationResponse) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("model".to_string(), serde_json::Value::String(response.model.clone()));
        metadata.insert("finish_reason".to_string(), serde_json::Value::String(response.finish_reason.clone()));
        metadata.insert("response_time_ms".to_string(), serde_json::Value::Number(
            serde_json::Number::from(response.response_time.as_millis() as u64)
        ));
        metadata.insert("total_tokens".to_string(), serde_json::Value::Number(
            serde_json::Number::from(response.usage.total_tokens)
        ));
        metadata
    }
}

// Text response parser with regex extraction
pub struct RegexResponseParser {
    patterns: HashMap<String, Regex>,
    required_fields: Vec<String>,
}

impl RegexResponseParser {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            required_fields: Vec::new(),
        }
    }

    pub fn with_pattern(mut self, name: impl Into<String>, pattern: impl AsRef<str>) -> Result<Self> {
        let regex = Regex::new(pattern.as_ref())
            .map_err(|e| LLMError::Config(format!("Invalid regex pattern: {}", e)))?;
        self.patterns.insert(name.into(), regex);
        Ok(self)
    }

    pub fn with_required_field(mut self, field: impl Into<String>) -> Self {
        self.required_fields.push(field.into());
        self
    }
}

impl ResponseParser<HashMap<String, String>> for RegexResponseParser {
    fn parse(&self, response: &GenerationResponse) -> Result<ParsedResponse<HashMap<String, String>>> {
        let mut data = HashMap::new();
        let mut confidence_sum = 0.0;
        let mut total_patterns = 0;

        for (name, regex) in &self.patterns {
            if let Some(captures) = regex.captures(&response.text) {
                if let Some(matched) = captures.get(1) {
                    data.insert(name.clone(), matched.as_str().to_string());
                    confidence_sum += 1.0;
                } else if let Some(matched) = captures.get(0) {
                    data.insert(name.clone(), matched.as_str().to_string());
                    confidence_sum += 0.8; // Lower confidence for full match without capture group
                }
            }
            total_patterns += 1;
        }

        let confidence = if total_patterns > 0 {
            confidence_sum / total_patterns as f64
        } else {
            1.0
        };

        let validation_errors = self.validate(&data);

        Ok(ParsedResponse {
            data,
            raw_response: response.text.clone(),
            confidence,
            validation_errors,
        })
    }

    fn validate(&self, data: &HashMap<String, String>) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for required_field in &self.required_fields {
            if !data.contains_key(required_field) || data[required_field].is_empty() {
                errors.push(ValidationError {
                    field: required_field.clone(),
                    message: format!("Required field '{}' is missing or empty", required_field),
                    code: "REQUIRED_FIELD_MISSING".to_string(),
                });
            }
        }

        errors
    }

    fn extract_metadata(&self, response: &GenerationResponse) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("patterns_matched".to_string(), serde_json::Value::Number(
            serde_json::Number::from(self.patterns.len())
        ));
        metadata.insert("model".to_string(), serde_json::Value::String(response.model.clone()));
        metadata
    }
}

// Code response parser
pub struct CodeResponseParser {
    language: String,
    extract_comments: bool,
}

impl CodeResponseParser {
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            extract_comments: false,
        }
    }

    pub fn with_comments(mut self) -> Self {
        self.extract_comments = true;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeResponse {
    pub code: String,
    pub language: String,
    pub comments: Vec<String>,
    pub has_syntax_errors: bool,
}

impl ResponseParser<CodeResponse> for CodeResponseParser {
    fn parse(&self, response: &GenerationResponse) -> Result<ParsedResponse<CodeResponse>> {
        let code = extract_code_block(&response.text, &self.language)?;
        let comments = if self.extract_comments {
            extract_comments(&code, &self.language)
        } else {
            Vec::new()
        };

        let has_syntax_errors = detect_syntax_errors(&code, &self.language);
        let confidence = calculate_code_confidence(&response.text, &code);

        let data = CodeResponse {
            code,
            language: self.language.clone(),
            comments,
            has_syntax_errors,
        };

        let validation_errors = self.validate(&data);

        Ok(ParsedResponse {
            data,
            raw_response: response.text.clone(),
            confidence,
            validation_errors,
        })
    }

    fn validate(&self, data: &CodeResponse) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if data.code.trim().is_empty() {
            errors.push(ValidationError {
                field: "code".to_string(),
                message: "Extracted code is empty".to_string(),
                code: "EMPTY_CODE".to_string(),
            });
        }

        if data.has_syntax_errors {
            errors.push(ValidationError {
                field: "code".to_string(),
                message: "Code contains potential syntax errors".to_string(),
                code: "SYNTAX_ERROR".to_string(),
            });
        }

        errors
    }

    fn extract_metadata(&self, response: &GenerationResponse) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        metadata.insert("language".to_string(), serde_json::Value::String(self.language.clone()));
        metadata.insert("extract_comments".to_string(), serde_json::Value::Bool(self.extract_comments));
        metadata
    }
}

// Utility functions
static JSON_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"```(?:json)?\s*(\{.*?\})\s*```").expect("Invalid JSON regex")
});

static CODE_BLOCK_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"```(\w+)?\s*(.*?)\s*```").expect("Invalid code block regex")
});

fn extract_json_from_text(text: &str) -> Result<String> {
    // First try to find JSON in code blocks
    if let Some(captures) = JSON_REGEX.captures(text) {
        if let Some(json_match) = captures.get(1) {
            return Ok(json_match.as_str().to_string());
        }
    }

    // Try to find JSON objects in the text
    let json_start = text.find('{');
    let json_end = text.rfind('}');

    if let (Some(start), Some(end)) = (json_start, json_end) {
        if end > start {
            return Ok(text[start..=end].to_string());
        }
    }

    // If no JSON found, return the original text
    Ok(text.to_string())
}

fn extract_code_block(text: &str, language: &str) -> Result<String> {
    // Look for code blocks with the specified language
    let escaped_language = language.replace("\\", "\\\\").replace(".", "\\.");
    let language_regex = Regex::new(&format!(r"```{}\s*(.*?)\s*```", escaped_language))
        .map_err(|e| LLMError::Config(format!("Invalid language regex: {}", e)))?;

    if let Some(captures) = language_regex.captures(text) {
        if let Some(code_match) = captures.get(1) {
            return Ok(code_match.as_str().to_string());
        }
    }

    // Fall back to any code block
    if let Some(captures) = CODE_BLOCK_REGEX.captures(text) {
        if let Some(code_match) = captures.get(2) {
            return Ok(code_match.as_str().to_string());
        }
    }

    // If no code block found, return the original text
    Ok(text.to_string())
}

fn extract_comments(code: &str, language: &str) -> Vec<String> {
    let comment_regex = match language {
        "rust" | "java" | "javascript" | "typescript" | "c" | "cpp" => {
            Regex::new(r"//\s*(.*)").ok()
        }
        "python" | "ruby" => {
            Regex::new(r"#\s*(.*)").ok()
        }
        _ => None,
    };

    if let Some(regex) = comment_regex {
        regex
            .captures_iter(code)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str().trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        Vec::new()
    }
}

fn detect_syntax_errors(code: &str, language: &str) -> bool {
    // Basic syntax error detection (this is simplified)
    match language {
        "rust" => {
            // Check for unmatched braces, missing semicolons, etc.
            let open_braces = code.matches('{').count();
            let close_braces = code.matches('}').count();
            open_braces != close_braces
        }
        "python" => {
            // Check for unmatched parentheses, indentation issues
            let open_parens = code.matches('(').count();
            let close_parens = code.matches(')').count();
            open_parens != close_parens
        }
        _ => false, // Default to no syntax errors for unknown languages
    }
}

fn calculate_json_confidence(original: &str, extracted: &str) -> f64 {
    if extracted.trim().is_empty() {
        return 0.0;
    }

    // Check if the extracted JSON is properly formatted
    if serde_json::from_str::<serde_json::Value>(extracted).is_ok() {
        if original.contains("```json") || original.contains("```") {
            0.95 // High confidence for explicit JSON blocks
        } else {
            0.8 // Medium confidence for extracted JSON
        }
    } else {
        0.3 // Low confidence for invalid JSON
    }
}

fn calculate_code_confidence(original: &str, extracted: &str) -> f64 {
    if extracted.trim().is_empty() {
        return 0.0;
    }

    if original.contains("```") {
        0.9 // High confidence for code blocks
    } else {
        0.6 // Medium confidence for extracted code
    }
}