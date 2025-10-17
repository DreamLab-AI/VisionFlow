// Settings Validation Layer
// Schema-based validation for all settings categories

use std::collections::HashMap;
use serde_json::Value as JsonValue;

use crate::services::database_service::SettingValue;
use crate::config::PhysicsSettings;

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn invalid(error: String) -> Self {
        Self {
            is_valid: false,
            errors: vec![error],
            warnings: Vec::new(),
        }
    }

    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }
}

pub struct SettingsValidator {
    rules: HashMap<String, ValidationRule>,
}

#[derive(Clone)]
struct ValidationRule {
    value_type: ValueType,
    min: Option<f64>,
    max: Option<f64>,
    allowed_values: Option<Vec<String>>,
    pattern: Option<String>,
    required: bool,
}

#[derive(Clone, PartialEq)]
enum ValueType {
    Float,
    Integer,
    Boolean,
    String,
    Object,
    Array,
}

impl SettingsValidator {
    pub fn new() -> Self {
        let mut rules = HashMap::new();

        // Visualization settings rules
        Self::register_visualization_rules(&mut rules);

        // Physics settings rules
        Self::register_physics_rules(&mut rules);

        // System settings rules
        Self::register_system_rules(&mut rules);

        // Ontology settings rules
        Self::register_ontology_rules(&mut rules);

        Self { rules }
    }

    fn register_visualization_rules(rules: &mut HashMap<String, ValidationRule>) {
        // Rendering settings
        rules.insert(
            "visualisation.rendering.ambient_light_intensity".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(2.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.rendering.environment_intensity".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(2.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.rendering.enable_shadows".to_string(),
            ValidationRule {
                value_type: ValueType::Boolean,
                min: None,
                max: None,
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        // Glow settings
        rules.insert(
            "visualisation.glow.opacity".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(1.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.glow.intensity".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(5.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        // Node settings
        rules.insert(
            "visualisation.graphs.logseq.nodes.base_size".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.1),
                max: Some(100.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );
    }

    fn register_physics_rules(rules: &mut HashMap<String, ValidationRule>) {
        rules.insert(
            "visualisation.graphs.logseq.physics.damping".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(1.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.dt".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.001),
                max: Some(0.1),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.iterations".to_string(),
            ValidationRule {
                value_type: ValueType::Integer,
                min: Some(1.0),
                max: Some(10000.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.max_velocity".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.1),
                max: Some(100.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.max_force".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.1),
                max: Some(1000.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.repel_k".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(1000.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.spring_k".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(10.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.gravity".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(1.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.temperature".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(0.0),
                max: Some(1.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.bounds_size".to_string(),
            ValidationRule {
                value_type: ValueType::Float,
                min: Some(100.0),
                max: Some(10000.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "visualisation.graphs.logseq.physics.enabled".to_string(),
            ValidationRule {
                value_type: ValueType::Boolean,
                min: None,
                max: None,
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );
    }

    fn register_system_rules(rules: &mut HashMap<String, ValidationRule>) {
        rules.insert(
            "system.port".to_string(),
            ValidationRule {
                value_type: ValueType::Integer,
                min: Some(1024.0),
                max: Some(65535.0),
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "system.host".to_string(),
            ValidationRule {
                value_type: ValueType::String,
                min: None,
                max: None,
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );

        rules.insert(
            "system.persist_settings".to_string(),
            ValidationRule {
                value_type: ValueType::Boolean,
                min: None,
                max: None,
                allowed_values: None,
                pattern: None,
                required: true,
            },
        );
    }

    fn register_ontology_rules(rules: &mut HashMap<String, ValidationRule>) {
        rules.insert(
            "ontology.reasoner.engine".to_string(),
            ValidationRule {
                value_type: ValueType::String,
                min: None,
                max: None,
                allowed_values: Some(vec![
                    "horned-owl".to_string(),
                    "whelk".to_string(),
                    "custom".to_string(),
                ]),
                pattern: None,
                required: false,
            },
        );

        rules.insert(
            "ontology.gpu.enabled".to_string(),
            ValidationRule {
                value_type: ValueType::Boolean,
                min: None,
                max: None,
                allowed_values: None,
                pattern: None,
                required: false,
            },
        );

        rules.insert(
            "ontology.validation.mode".to_string(),
            ValidationRule {
                value_type: ValueType::String,
                min: None,
                max: None,
                allowed_values: Some(vec![
                    "quick".to_string(),
                    "full".to_string(),
                    "incremental".to_string(),
                ]),
                pattern: None,
                required: false,
            },
        );
    }

    /// Validate a single setting
    pub fn validate_setting(&self, key: &str, value: &SettingValue) -> Result<ValidationResult, String> {
        // Find matching rule
        let rule = match self.rules.get(key) {
            Some(r) => r,
            None => {
                // No explicit rule, use basic type checking
                return Ok(ValidationResult::valid().with_warning(format!(
                    "No validation rule found for key: {}",
                    key
                )));
            }
        };

        // Type validation
        if !self.validate_type(value, &rule.value_type) {
            return Ok(ValidationResult::invalid(format!(
                "Type mismatch for {}: expected {:?}, got {:?}",
                key, rule.value_type, value
            )));
        }

        // Range validation for numeric types
        if let Some(min) = rule.min {
            if let Some(num) = self.extract_number(value) {
                if num < min {
                    return Ok(ValidationResult::invalid(format!(
                        "Value {} is below minimum {} for {}",
                        num, min, key
                    )));
                }
            }
        }

        if let Some(max) = rule.max {
            if let Some(num) = self.extract_number(value) {
                if num > max {
                    return Ok(ValidationResult::invalid(format!(
                        "Value {} is above maximum {} for {}",
                        num, max, key
                    )));
                }
            }
        }

        // Allowed values validation
        if let Some(allowed) = &rule.allowed_values {
            if let SettingValue::String(s) = value {
                if !allowed.contains(s) {
                    return Ok(ValidationResult::invalid(format!(
                        "Value '{}' is not in allowed values for {}: {:?}",
                        s, key, allowed
                    )));
                }
            }
        }

        Ok(ValidationResult::valid())
    }

    /// Validate physics settings
    pub fn validate_physics_settings(&self, settings: &PhysicsSettings) -> Result<ValidationResult, String> {
        let mut result = ValidationResult::valid();

        // Validate damping
        if settings.damping < 0.0 || settings.damping > 1.0 {
            result.is_valid = false;
            result
                .errors
                .push(format!("Damping must be between 0.0 and 1.0, got {}", settings.damping));
        }

        // Validate dt
        if settings.dt < 0.001 || settings.dt > 0.1 {
            result.is_valid = false;
            result
                .errors
                .push(format!("dt must be between 0.001 and 0.1, got {}", settings.dt));
        }

        // Validate iterations
        if settings.iterations < 1 || settings.iterations > 10000 {
            result.is_valid = false;
            result.errors.push(format!(
                "Iterations must be between 1 and 10000, got {}",
                settings.iterations
            ));
        }

        // Validate max_velocity
        if settings.max_velocity <= 0.0 {
            result.is_valid = false;
            result.errors.push(format!(
                "Max velocity must be positive, got {}",
                settings.max_velocity
            ));
        }

        // Validate repel_k and spring_k (must be non-negative)
        if settings.repel_k < 0.0 {
            result.is_valid = false;
            result
                .errors
                .push(format!("Repel_k must be non-negative, got {}", settings.repel_k));
        }

        if settings.spring_k < 0.0 {
            result.is_valid = false;
            result.errors.push(format!(
                "Spring_k must be non-negative, got {}",
                settings.spring_k
            ));
        }

        // Cross-field validation: temperature and cooling_rate
        if settings.temperature > 0.5 && settings.cooling_rate < 0.0001 {
            result.warnings.push(
                "High temperature with low cooling rate may cause instability".to_string(),
            );
        }

        // Constraint validation
        if settings.constraint_max_force_per_node <= 0.0 {
            result.is_valid = false;
            result.errors.push(format!(
                "Constraint max force must be positive, got {}",
                settings.constraint_max_force_per_node
            ));
        }

        Ok(result)
    }

    fn validate_type(&self, value: &SettingValue, expected: &ValueType) -> bool {
        match (value, expected) {
            (SettingValue::Float(_), ValueType::Float) => true,
            (SettingValue::Integer(_), ValueType::Integer) => true,
            (SettingValue::Boolean(_), ValueType::Boolean) => true,
            (SettingValue::String(_), ValueType::String) => true,
            (SettingValue::Json(v), ValueType::Object) => v.is_object(),
            (SettingValue::Json(v), ValueType::Array) => v.is_array(),
            _ => false,
        }
    }

    fn extract_number(&self, value: &SettingValue) -> Option<f64> {
        match value {
            SettingValue::Float(f) => Some(*f),
            SettingValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_physics_damping() {
        let validator = SettingsValidator::new();

        // Valid damping
        let result = validator
            .validate_setting(
                "visualisation.graphs.logseq.physics.damping",
                &SettingValue::Float(0.95),
            )
            .unwrap();
        assert!(result.is_valid);

        // Invalid damping (too high)
        let result = validator
            .validate_setting(
                "visualisation.graphs.logseq.physics.damping",
                &SettingValue::Float(1.5),
            )
            .unwrap();
        assert!(!result.is_valid);

        // Invalid damping (negative)
        let result = validator
            .validate_setting(
                "visualisation.graphs.logseq.physics.damping",
                &SettingValue::Float(-0.1),
            )
            .unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_port() {
        let validator = SettingsValidator::new();

        // Valid port
        let result = validator
            .validate_setting("system.port", &SettingValue::Integer(8080))
            .unwrap();
        assert!(result.is_valid);

        // Invalid port (too low)
        let result = validator
            .validate_setting("system.port", &SettingValue::Integer(80))
            .unwrap();
        assert!(!result.is_valid);

        // Invalid port (too high)
        let result = validator
            .validate_setting("system.port", &SettingValue::Integer(70000))
            .unwrap();
        assert!(!result.is_valid);
    }
}
