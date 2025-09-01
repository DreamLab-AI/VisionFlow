// Tests for centralized validation architecture
use webxr::config::AppFullSettings;
use validator::Validate;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_validation_single_source_of_truth() {
        let mut settings = AppFullSettings::default();
        
        // Test invalid hex color
        settings.visualisation.rendering.background_color = "invalid-color".to_string();
        
        // Server validation should catch this
        let result = settings.validate();
        assert!(result.is_err(), "Server validation should reject invalid hex color");
        
        if let Err(errors) = result {
            let field_errors = errors.field_errors();
            assert!(field_errors.contains_key("visualisation"));
        }
    }
    
    #[test]
    fn test_server_validation_range_checks() {
        let mut settings = AppFullSettings::default();
        
        // Test invalid opacity (greater than 1.0)
        settings.visualisation.graphs.logseq.nodes.opacity = 1.5;
        
        // Server validation should catch this
        let result = settings.validate();
        assert!(result.is_err(), "Server validation should reject opacity > 1.0");
    }
    
    #[test]
    fn test_server_validation_passes_for_valid_data() {
        let mut settings = AppFullSettings::default();
        
        // Set valid values
        settings.visualisation.rendering.background_color = "#ff0000".to_string();
        settings.visualisation.graphs.logseq.nodes.opacity = 0.5;
        settings.visualisation.graphs.logseq.nodes.node_size = 1.0;
        
        // Server validation should pass
        let result = settings.validate();
        assert!(result.is_ok(), "Server validation should pass for valid data");
    }
    
    #[test]  
    fn test_nested_struct_validation() {
        let mut settings = AppFullSettings::default();
        
        // Test invalid port
        settings.system.network.port = 0; // Invalid port
        
        // Server validation should catch this
        let result = settings.validate();
        assert!(result.is_err(), "Server validation should reject invalid port");
        
        if let Err(errors) = result {
            let field_errors = errors.field_errors();
            // Should have validation error for the network port
            assert!(!field_errors.is_empty(), "Should have field errors");
        }
    }

    #[test]
    fn test_color_validation_patterns() {
        let mut settings = AppFullSettings::default();
        
        // Test various invalid color formats
        let invalid_colors = vec![
            "red",           // Named color
            "#ff00",         // Too short
            "#gg0000",       // Invalid hex chars
            "rgb(255,0,0)",  // CSS format
            "",              // Empty string
        ];
        
        for invalid_color in invalid_colors {
            settings.visualisation.rendering.background_color = invalid_color.to_string();
            let result = settings.validate();
            assert!(result.is_err(), "Server validation should reject color: {}", invalid_color);
        }
        
        // Test valid colors
        let valid_colors = vec!["#ff0000", "#00FF00", "#0000ff", "#123abc"];
        
        for valid_color in valid_colors {
            settings.visualisation.rendering.background_color = valid_color.to_string();
            let result = settings.validate();
            assert!(result.is_ok(), "Server validation should accept color: {}", valid_color);
        }
    }
}