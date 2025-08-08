#!/usr/bin/env python3
import yaml
import json

def verify_settings_yaml():
    """Verify settings.yaml has all required fields"""
    with open('/workspace/ext/data/settings.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    print("Top-level keys in settings.yaml:")
    for key in data.keys():
        print(f"  - {key}")
    
    if 'visualisation' in data:
        print("\nKeys in visualisation section:")
        for key in data['visualisation'].keys():
            print(f"  - {key}")
        
        # Check for required legacy fields
        required_fields = ['nodes', 'edges', 'physics', 'labels', 'rendering', 'animations', 'bloom', 'hologram', 'graphs']
        missing = []
        for field in required_fields:
            if field not in data['visualisation']:
                missing.append(field)
        
        if missing:
            print(f"\n❌ Missing required fields in visualisation: {missing}")
        else:
            print("\n✅ All required fields present in visualisation section")
    else:
        print("\n❌ visualisation section not found!")
    
    # Pretty print the structure
    print("\n" + "="*60)
    print("Full YAML structure (first 2 levels):")
    print("="*60)
    
    def print_structure(data, indent=0):
        for key, value in data.items():
            print("  " * indent + f"- {key}: {type(value).__name__}")
            if isinstance(value, dict) and indent < 1:
                print_structure(value, indent + 1)
    
    print_structure(data)

if __name__ == "__main__":
    verify_settings_yaml()