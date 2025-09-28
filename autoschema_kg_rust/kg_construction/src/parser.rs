//! Output parsing functionality for structured data extraction

use crate::error::{KgConstructionError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, from_str as json_from_str};
use std::collections::HashMap;

/// Represents a triple extracted from text (Head-Relation-Tail)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Triple {
    #[serde(rename = "Head")]
    pub head: String,
    #[serde(rename = "Relation")]
    pub relation: String,
    #[serde(rename = "Tail")]
    pub tail: String,
}

/// Represents an event-entity relationship
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventEntity {
    #[serde(rename = "Event")]
    pub event: String,
    #[serde(rename = "Entity")]
    pub entity: Vec<String>,
}

/// Represents an event-relation relationship
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventRelation {
    #[serde(rename = "Head")]
    pub head: String,
    #[serde(rename = "Relation")]
    pub relation: String,
    #[serde(rename = "Tail")]
    pub tail: String,
}

/// All possible extraction types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtractionType {
    EntityRelation(Vec<Triple>),
    EventEntity(Vec<EventEntity>),
    EventRelation(Vec<EventRelation>),
}

/// Configuration for the output parser
#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub repair_json: bool,
    pub strict_validation: bool,
    pub max_retries: usize,
    pub fallback_empty: bool,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            repair_json: true,
            strict_validation: true,
            max_retries: 3,
            fallback_empty: true,
        }
    }
}

/// Parses model outputs and extracts structured data
pub struct OutputParser {
    config: ParserConfig,
    repair_stats: ParsingStats,
}

/// Statistics about parsing operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsingStats {
    pub total_parsed: usize,
    pub successful_parses: usize,
    pub json_repairs: usize,
    pub validation_failures: usize,
    pub fallback_uses: usize,
}

impl Default for OutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser {
    /// Create a new output parser with default configuration
    pub fn new() -> Self {
        Self {
            config: ParserConfig::default(),
            repair_stats: ParsingStats::default(),
        }
    }

    /// Create parser with custom configuration
    pub fn with_config(config: ParserConfig) -> Self {
        Self {
            config,
            repair_stats: ParsingStats::default(),
        }
    }

    /// Extract structured data from model outputs for entity-relation extraction
    pub fn extract_entity_relations(&mut self, outputs: &[String]) -> Result<Vec<Vec<Triple>>> {
        let mut results = Vec::with_capacity(outputs.len());

        for output in outputs {
            self.repair_stats.total_parsed += 1;

            match self.parse_entity_relation_output(output) {
                Ok(triples) => {
                    self.repair_stats.successful_parses += 1;
                    results.push(triples);
                }
                Err(e) => {
                    if self.config.fallback_empty {
                        self.repair_stats.fallback_uses += 1;
                        results.push(vec![]);
                        log::warn!("Failed to parse entity-relation output, using empty fallback: {}", e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Extract structured data from model outputs for event-entity extraction
    pub fn extract_event_entities(&mut self, outputs: &[String]) -> Result<Vec<Vec<EventEntity>>> {
        let mut results = Vec::with_capacity(outputs.len());

        for output in outputs {
            self.repair_stats.total_parsed += 1;

            match self.parse_event_entity_output(output) {
                Ok(events) => {
                    self.repair_stats.successful_parses += 1;
                    results.push(events);
                }
                Err(e) => {
                    if self.config.fallback_empty {
                        self.repair_stats.fallback_uses += 1;
                        results.push(vec![]);
                        log::warn!("Failed to parse event-entity output, using empty fallback: {}", e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Extract structured data from model outputs for event-relation extraction
    pub fn extract_event_relations(&mut self, outputs: &[String]) -> Result<Vec<Vec<EventRelation>>> {
        let mut results = Vec::with_capacity(outputs.len());

        for output in outputs {
            self.repair_stats.total_parsed += 1;

            match self.parse_event_relation_output(output) {
                Ok(relations) => {
                    self.repair_stats.successful_parses += 1;
                    results.push(relations);
                }
                Err(e) => {
                    if self.config.fallback_empty {
                        self.repair_stats.fallback_uses += 1;
                        results.push(vec![]);
                        log::warn!("Failed to parse event-relation output, using empty fallback: {}", e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Generic extraction method that returns the appropriate type
    pub fn extract_structured_data(&mut self, outputs: &[String], extraction_type: &str) -> Result<Vec<Value>> {
        match extraction_type {
            "entity_relation" => {
                let results = self.extract_entity_relations(outputs)?;
                Ok(results.into_iter().map(|r| serde_json::to_value(r).unwrap()).collect())
            }
            "event_entity" => {
                let results = self.extract_event_entities(outputs)?;
                Ok(results.into_iter().map(|r| serde_json::to_value(r).unwrap()).collect())
            }
            "event_relation" => {
                let results = self.extract_event_relations(outputs)?;
                Ok(results.into_iter().map(|r| serde_json::to_value(r).unwrap()).collect())
            }
            _ => Err(KgConstructionError::ParsingError(
                format!("Unknown extraction type: {}", extraction_type)
            ))
        }
    }

    /// Parse entity-relation output
    fn parse_entity_relation_output(&mut self, output: &str) -> Result<Vec<Triple>> {
        let json_value = self.parse_json_with_repair(output)?;

        let array = json_value.as_array().ok_or_else(|| {
            KgConstructionError::ParsingError("Expected JSON array for entity-relation output".to_string())
        })?;

        let mut triples = Vec::with_capacity(array.len());
        for (idx, item) in array.iter().enumerate() {
            let triple = self.parse_triple(item, idx)?;
            if self.config.strict_validation {
                self.validate_triple(&triple)?;
            }
            triples.push(triple);
        }

        Ok(triples)
    }

    /// Parse event-entity output
    fn parse_event_entity_output(&mut self, output: &str) -> Result<Vec<EventEntity>> {
        let json_value = self.parse_json_with_repair(output)?;

        let array = json_value.as_array().ok_or_else(|| {
            KgConstructionError::ParsingError("Expected JSON array for event-entity output".to_string())
        })?;

        let mut events = Vec::with_capacity(array.len());
        for (idx, item) in array.iter().enumerate() {
            let event = self.parse_event_entity(item, idx)?;
            if self.config.strict_validation {
                self.validate_event_entity(&event)?;
            }
            events.push(event);
        }

        Ok(events)
    }

    /// Parse event-relation output
    fn parse_event_relation_output(&mut self, output: &str) -> Result<Vec<EventRelation>> {
        let json_value = self.parse_json_with_repair(output)?;

        let array = json_value.as_array().ok_or_else(|| {
            KgConstructionError::ParsingError("Expected JSON array for event-relation output".to_string())
        })?;

        let mut relations = Vec::with_capacity(array.len());
        for (idx, item) in array.iter().enumerate() {
            let relation = self.parse_event_relation(item, idx)?;
            if self.config.strict_validation {
                self.validate_event_relation(&relation)?;
            }
            relations.push(relation);
        }

        Ok(relations)
    }

    /// Parse JSON with optional repair
    fn parse_json_with_repair(&mut self, output: &str) -> Result<Value> {
        // First try normal JSON parsing
        match json_from_str(output) {
            Ok(value) => return Ok(value),
            Err(_) if !self.config.repair_json => {
                return Err(KgConstructionError::ParsingError(
                    "Failed to parse JSON and repair is disabled".to_string()
                ));
            }
            Err(_) => {
                // Continue to repair attempt
            }
        }

        // Try to repair the JSON
        self.repair_stats.json_repairs += 1;
        let repaired = self.repair_json(output)?;

        json_from_str(&repaired).map_err(|e| {
            KgConstructionError::ParsingError(format!("Failed to parse repaired JSON: {}", e))
        })
    }

    /// Attempt to repair malformed JSON
    fn repair_json(&self, json_str: &str) -> Result<String> {
        let mut cleaned = json_str.trim().to_string();

        // Remove common prefixes/suffixes that models might add
        if let Some(start) = cleaned.find('[') {
            if let Some(end) = cleaned.rfind(']') {
                if end > start {
                    cleaned = cleaned[start..=end].to_string();
                }
            }
        } else if let Some(start) = cleaned.find('{') {
            if let Some(end) = cleaned.rfind('}') {
                if end > start {
                    cleaned = cleaned[start..=end].to_string();
                }
            }
        }

        // Fix common JSON issues
        cleaned = cleaned
            .replace("'", "\"")  // Single quotes to double quotes
            .replace("True", "true")  // Python True to JSON true
            .replace("False", "false")  // Python False to JSON false
            .replace("None", "null");  // Python None to JSON null

        // Try to fix missing quotes around keys
        cleaned = self.fix_unquoted_keys(&cleaned);

        // Try to fix trailing commas
        cleaned = self.fix_trailing_commas(&cleaned);

        Ok(cleaned)
    }

    /// Fix unquoted JSON keys
    fn fix_unquoted_keys(&self, json_str: &str) -> String {
        // This is a simple regex-based approach
        // In production, you might want a more robust JSON repair library
        use regex::Regex;

        let key_regex = Regex::new(r"(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:").unwrap();
        key_regex.replace_all(json_str, r#"$1"$2":"#).to_string()
    }

    /// Fix trailing commas in JSON
    fn fix_trailing_commas(&self, json_str: &str) -> String {
        use regex::Regex;

        let trailing_comma_regex = Regex::new(r",\s*([}\]])").unwrap();
        trailing_comma_regex.replace_all(json_str, "$1").to_string()
    }

    /// Parse a single triple from JSON value
    fn parse_triple(&self, value: &Value, index: usize) -> Result<Triple> {
        let obj = value.as_object().ok_or_else(|| {
            KgConstructionError::ParsingError(format!("Expected object at index {}", index))
        })?;

        let head = obj.get("Head")
            .or_else(|| obj.get("head"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                KgConstructionError::ParsingError(format!("Missing or invalid 'Head' field at index {}", index))
            })?
            .to_string();

        let relation = obj.get("Relation")
            .or_else(|| obj.get("relation"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                KgConstructionError::ParsingError(format!("Missing or invalid 'Relation' field at index {}", index))
            })?
            .to_string();

        let tail = obj.get("Tail")
            .or_else(|| obj.get("tail"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                KgConstructionError::ParsingError(format!("Missing or invalid 'Tail' field at index {}", index))
            })?
            .to_string();

        Ok(Triple { head, relation, tail })
    }

    /// Parse a single event-entity from JSON value
    fn parse_event_entity(&self, value: &Value, index: usize) -> Result<EventEntity> {
        let obj = value.as_object().ok_or_else(|| {
            KgConstructionError::ParsingError(format!("Expected object at index {}", index))
        })?;

        let event = obj.get("Event")
            .or_else(|| obj.get("event"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                KgConstructionError::ParsingError(format!("Missing or invalid 'Event' field at index {}", index))
            })?
            .to_string();

        let entity_array = obj.get("Entity")
            .or_else(|| obj.get("entity"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                KgConstructionError::ParsingError(format!("Missing or invalid 'Entity' field at index {}", index))
            })?;

        let entity: Vec<String> = entity_array
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();

        if entity.len() != entity_array.len() {
            return Err(KgConstructionError::ParsingError(
                format!("Some entity values are not strings at index {}", index)
            ));
        }

        Ok(EventEntity { event, entity })
    }

    /// Parse a single event-relation from JSON value
    fn parse_event_relation(&self, value: &Value, index: usize) -> Result<EventRelation> {
        let obj = value.as_object().ok_or_else(|| {
            KgConstructionError::ParsingError(format!("Expected object at index {}", index))
        })?;

        let head = obj.get("Head")
            .or_else(|| obj.get("head"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                KgConstructionError::ParsingError(format!("Missing or invalid 'Head' field at index {}", index))
            })?
            .to_string();

        let relation = obj.get("Relation")
            .or_else(|| obj.get("relation"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                KgConstructionError::ParsingError(format!("Missing or invalid 'Relation' field at index {}", index))
            })?
            .to_string();

        let tail = obj.get("Tail")
            .or_else(|| obj.get("tail"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                KgConstructionError::ParsingError(format!("Missing or invalid 'Tail' field at index {}", index))
            })?
            .to_string();

        Ok(EventRelation { head, relation, tail })
    }

    /// Validate a triple
    fn validate_triple(&mut self, triple: &Triple) -> Result<()> {
        if triple.head.trim().is_empty() {
            self.repair_stats.validation_failures += 1;
            return Err(KgConstructionError::ValidationError("Empty head in triple".to_string()));
        }
        if triple.relation.trim().is_empty() {
            self.repair_stats.validation_failures += 1;
            return Err(KgConstructionError::ValidationError("Empty relation in triple".to_string()));
        }
        if triple.tail.trim().is_empty() {
            self.repair_stats.validation_failures += 1;
            return Err(KgConstructionError::ValidationError("Empty tail in triple".to_string()));
        }
        Ok(())
    }

    /// Validate an event-entity
    fn validate_event_entity(&mut self, event_entity: &EventEntity) -> Result<()> {
        if event_entity.event.trim().is_empty() {
            self.repair_stats.validation_failures += 1;
            return Err(KgConstructionError::ValidationError("Empty event in event-entity".to_string()));
        }
        if event_entity.entity.is_empty() {
            self.repair_stats.validation_failures += 1;
            return Err(KgConstructionError::ValidationError("Empty entity list in event-entity".to_string()));
        }
        for entity in &event_entity.entity {
            if entity.trim().is_empty() {
                self.repair_stats.validation_failures += 1;
                return Err(KgConstructionError::ValidationError("Empty entity in entity list".to_string()));
            }
        }
        Ok(())
    }

    /// Validate an event-relation
    fn validate_event_relation(&mut self, event_relation: &EventRelation) -> Result<()> {
        if event_relation.head.trim().is_empty() {
            self.repair_stats.validation_failures += 1;
            return Err(KgConstructionError::ValidationError("Empty head in event-relation".to_string()));
        }
        if event_relation.relation.trim().is_empty() {
            self.repair_stats.validation_failures += 1;
            return Err(KgConstructionError::ValidationError("Empty relation in event-relation".to_string()));
        }
        if event_relation.tail.trim().is_empty() {
            self.repair_stats.validation_failures += 1;
            return Err(KgConstructionError::ValidationError("Empty tail in event-relation".to_string()));
        }
        Ok(())
    }

    /// Get parsing statistics
    pub fn stats(&self) -> &ParsingStats {
        &self.repair_stats
    }

    /// Reset parsing statistics
    pub fn reset_stats(&mut self) {
        self.repair_stats = ParsingStats::default();
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.repair_stats.total_parsed == 0 {
            0.0
        } else {
            (self.repair_stats.successful_parses as f64 / self.repair_stats.total_parsed as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = OutputParser::new();
        assert_eq!(parser.config.repair_json, true);
        assert_eq!(parser.config.strict_validation, true);
    }

    #[test]
    fn test_entity_relation_parsing() {
        let mut parser = OutputParser::new();
        let output = r#"[{"Head": "John", "Relation": "likes", "Tail": "pizza"}]"#;

        let results = parser.extract_entity_relations(&[output.to_string()]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[0][0].head, "John");
        assert_eq!(results[0][0].relation, "likes");
        assert_eq!(results[0][0].tail, "pizza");
    }

    #[test]
    fn test_event_entity_parsing() {
        let mut parser = OutputParser::new();
        let output = r#"[{"Event": "John ate pizza", "Entity": ["John", "pizza"]}]"#;

        let results = parser.extract_event_entities(&[output.to_string()]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[0][0].event, "John ate pizza");
        assert_eq!(results[0][0].entity, vec!["John", "pizza"]);
    }

    #[test]
    fn test_json_repair() {
        let mut parser = OutputParser::new();

        // Test single quotes
        let malformed = r#"[{'Head': 'John', 'Relation': 'likes', 'Tail': 'pizza'}]"#;
        let results = parser.extract_entity_relations(&[malformed.to_string()]).unwrap();
        assert_eq!(results[0][0].head, "John");

        // Test trailing comma
        let trailing_comma = r#"[{"Head": "John", "Relation": "likes", "Tail": "pizza",}]"#;
        let results = parser.extract_entity_relations(&[trailing_comma.to_string()]).unwrap();
        assert_eq!(results[0][0].head, "John");
    }

    #[test]
    fn test_validation_errors() {
        let mut parser = OutputParser::with_config(ParserConfig {
            strict_validation: true,
            fallback_empty: false,
            ..Default::default()
        });

        let invalid_output = r#"[{"Head": "", "Relation": "likes", "Tail": "pizza"}]"#;
        let result = parser.extract_entity_relations(&[invalid_output.to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fallback_behavior() {
        let mut parser = OutputParser::with_config(ParserConfig {
            fallback_empty: true,
            ..Default::default()
        });

        let invalid_output = "this is not json";
        let results = parser.extract_entity_relations(&[invalid_output.to_string()]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 0); // Should be empty due to fallback
        assert_eq!(parser.stats().fallback_uses, 1);
    }

    #[test]
    fn test_case_insensitive_keys() {
        let mut parser = OutputParser::new();
        let output = r#"[{"head": "John", "relation": "likes", "tail": "pizza"}]"#;

        let results = parser.extract_entity_relations(&[output.to_string()]).unwrap();
        assert_eq!(results[0][0].head, "John");
        assert_eq!(results[0][0].relation, "likes");
        assert_eq!(results[0][0].tail, "pizza");
    }

    #[test]
    fn test_statistics() {
        let mut parser = OutputParser::new();
        let valid_output = r#"[{"Head": "John", "Relation": "likes", "Tail": "pizza"}]"#;
        let invalid_output = "not json";

        parser.extract_entity_relations(&[
            valid_output.to_string(),
            invalid_output.to_string(),
        ]).unwrap();

        let stats = parser.stats();
        assert_eq!(stats.total_parsed, 2);
        assert_eq!(stats.successful_parses, 1);
        assert_eq!(stats.fallback_uses, 1);

        assert!(parser.success_rate() > 0.0);
        assert!(parser.success_rate() < 100.0);
    }
}