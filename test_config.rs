// Simple test to show the YAML parsing issue
fn main() {
    println!("Issue Analysis:");
    println!("1. settings.yaml uses snake_case: ambient_light_intensity");
    println!("2. RenderingSettings struct uses #[serde(rename_all = \"camelCase\")] ");
    println!("3. This means serde expects camelCase in JSON but should handle snake_case in YAML");
    println!("4. BUT the config crate fallback doesn't respect serde rename attributes!");
    println!();
    
    println!("The problem:");
    println!("- Direct YAML deserialization (serde_yaml::from_str) works correctly");
    println!("- Config crate fallback tries to deserialize without respecting serde attributes"); 
    println!("- Config crate sees 'ambient_light_intensity' in YAML but struct expects 'ambientLightIntensity'");
    println!("- This causes 'missing field ambientLightIntensity' error");
    println!();
    
    println!("Solution:");
    println!("Fix the direct YAML deserialization path to handle the conversion properly");
    println!("or ensure config crate respects the serde rename attributes");
}