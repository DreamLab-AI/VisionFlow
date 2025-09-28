use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;
use once_cell::sync::Lazy;

use crate::{Result, LLMError, Message};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub name: String,
    pub template: String,
    pub variables: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub validation_rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub variable: String,
    pub rule_type: ValidationType,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    Required,
    MinLength(usize),
    MaxLength(usize),
    Regex(String),
    OneOf(Vec<String>),
    Custom(String), // JavaScript expression for validation
}

static VARIABLE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\{\{(\w+)\}\}").expect("Invalid regex pattern")
});

impl PromptTemplate {
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        let template = template.into();
        let variables = Self::extract_variables(&template);

        Self {
            name: name.into(),
            template,
            variables,
            metadata: HashMap::new(),
            validation_rules: Vec::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn with_validation(mut self, rule: ValidationRule) -> Self {
        self.validation_rules.push(rule);
        self
    }

    pub fn render(&self, variables: &HashMap<String, String>) -> Result<String> {
        // Validate variables first
        self.validate_variables(variables)?;

        let mut result = self.template.clone();

        for (key, value) in variables {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        // Check for any unreplaced variables
        if VARIABLE_REGEX.is_match(&result) {
            let missing: Vec<_> = VARIABLE_REGEX
                .captures_iter(&result)
                .map(|cap| cap[1].to_string())
                .collect();
            return Err(LLMError::Validation(format!(
                "Missing variables: {}",
                missing.join(", ")
            )));
        }

        Ok(result)
    }

    pub fn render_messages(&self, variables: &HashMap<String, String>) -> Result<Vec<Message>> {
        let rendered = self.render(variables)?;

        // Try to parse as structured messages first
        if let Ok(messages) = serde_json::from_str::<Vec<Message>>(&rendered) {
            return Ok(messages);
        }

        // Fall back to single user message
        Ok(vec![Message::user(rendered)])
    }

    fn extract_variables(template: &str) -> Vec<String> {
        VARIABLE_REGEX
            .captures_iter(template)
            .map(|cap| cap[1].to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    fn validate_variables(&self, variables: &HashMap<String, String>) -> Result<()> {
        for rule in &self.validation_rules {
            let value = variables.get(&rule.variable);

            match &rule.rule_type {
                ValidationType::Required => {
                    if value.is_none() || value.unwrap().is_empty() {
                        return Err(LLMError::Validation(rule.message.clone()));
                    }
                }
                ValidationType::MinLength(min) => {
                    if let Some(val) = value {
                        if val.len() < *min {
                            return Err(LLMError::Validation(rule.message.clone()));
                        }
                    }
                }
                ValidationType::MaxLength(max) => {
                    if let Some(val) = value {
                        if val.len() > *max {
                            return Err(LLMError::Validation(rule.message.clone()));
                        }
                    }
                }
                ValidationType::Regex(pattern) => {
                    if let Some(val) = value {
                        let regex = Regex::new(pattern)
                            .map_err(|e| LLMError::Config(format!("Invalid regex: {}", e)))?;
                        if !regex.is_match(val) {
                            return Err(LLMError::Validation(rule.message.clone()));
                        }
                    }
                }
                ValidationType::OneOf(options) => {
                    if let Some(val) = value {
                        if !options.contains(val) {
                            return Err(LLMError::Validation(rule.message.clone()));
                        }
                    }
                }
                ValidationType::Custom(_) => {
                    // Custom validation would require a JavaScript engine
                    // For now, we'll skip custom validation
                    log::warn!("Custom validation not implemented for variable: {}", rule.variable);
                }
            }
        }

        Ok(())
    }

    pub fn get_required_variables(&self) -> Vec<&str> {
        self.variables.iter().map(|s| s.as_str()).collect()
    }

    pub fn clone_with_name(&self, new_name: impl Into<String>) -> Self {
        let mut cloned = self.clone();
        cloned.name = new_name.into();
        cloned
    }
}

pub struct PromptBuilder {
    templates: HashMap<String, PromptTemplate>,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn add_template(&mut self, template: PromptTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    pub fn get_template(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    pub fn render_template(
        &self,
        name: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String> {
        let template = self
            .templates
            .get(name)
            .ok_or_else(|| LLMError::Config(format!("Template '{}' not found", name)))?;
        template.render(variables)
    }

    pub fn render_messages(
        &self,
        name: &str,
        variables: &HashMap<String, String>,
    ) -> Result<Vec<Message>> {
        let template = self
            .templates
            .get(name)
            .ok_or_else(|| LLMError::Config(format!("Template '{}' not found", name)))?;
        template.render_messages(variables)
    }

    pub fn list_templates(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    pub fn load_from_directory(&mut self, dir_path: &std::path::Path) -> Result<usize> {
        let mut loaded = 0;

        if !dir_path.exists() {
            return Ok(0);
        }

        for entry in std::fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension() == Some(std::ffi::OsStr::new("json")) {
                let content = std::fs::read_to_string(&path)?;
                let template: PromptTemplate = serde_json::from_str(&content)?;
                self.add_template(template);
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    pub fn save_template(&self, name: &str, dir_path: &std::path::Path) -> Result<()> {
        let template = self
            .templates
            .get(name)
            .ok_or_else(|| LLMError::Config(format!("Template '{}' not found", name)))?;

        std::fs::create_dir_all(dir_path)?;
        let file_path = dir_path.join(format!("{}.json", name));
        let content = serde_json::to_string_pretty(template)?;
        std::fs::write(file_path, content)?;

        Ok(())
    }

    pub fn with_default_templates() -> Self {
        let mut builder = Self::new();
        builder.add_default_templates();
        builder
    }

    fn add_default_templates(&mut self) {
        // System message template
        let system_template = PromptTemplate::new(
            "system",
            r#"[
                {"role": "system", "content": "{{system_message}}"},
                {"role": "user", "content": "{{user_input}}"}
            ]"#,
        )
        .with_validation(ValidationRule {
            variable: "system_message".to_string(),
            rule_type: ValidationType::Required,
            message: "System message is required".to_string(),
        })
        .with_validation(ValidationRule {
            variable: "user_input".to_string(),
            rule_type: ValidationType::Required,
            message: "User input is required".to_string(),
        });

        // Simple completion template
        let completion_template = PromptTemplate::new(
            "completion",
            "{{prompt}}"
        )
        .with_validation(ValidationRule {
            variable: "prompt".to_string(),
            rule_type: ValidationType::Required,
            message: "Prompt is required".to_string(),
        });

        // Question answering template
        let qa_template = PromptTemplate::new(
            "question_answer",
            r#"Context: {{context}}

Question: {{question}}

Please provide a detailed answer based on the context above."#,
        )
        .with_validation(ValidationRule {
            variable: "context".to_string(),
            rule_type: ValidationType::Required,
            message: "Context is required".to_string(),
        })
        .with_validation(ValidationRule {
            variable: "question".to_string(),
            rule_type: ValidationType::Required,
            message: "Question is required".to_string(),
        });

        // Code generation template
        let code_template = PromptTemplate::new(
            "code_generation",
            r#"Generate {{language}} code for the following task:

Task: {{task}}

Requirements:
{{requirements}}

Please provide clean, well-commented code."#,
        )
        .with_validation(ValidationRule {
            variable: "language".to_string(),
            rule_type: ValidationType::Required,
            message: "Programming language is required".to_string(),
        })
        .with_validation(ValidationRule {
            variable: "task".to_string(),
            rule_type: ValidationType::Required,
            message: "Task description is required".to_string(),
        });

        self.add_template(system_template);
        self.add_template(completion_template);
        self.add_template(qa_template);
        self.add_template(code_template);
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::with_default_templates()
    }
}