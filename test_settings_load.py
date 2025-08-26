#!/usr/bin/env python3
"""Test script to validate settings loading with bloom/glow field mapping."""

import yaml
import json

# Load the settings.yaml file
with open('/workspace/ext/data/settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

# Check bloom field in visualisation
print("=== YAML Structure Check ===")
if 'visualisation' in settings:
    if 'bloom' in settings['visualisation']:
        bloom = settings['visualisation']['bloom']
        print("✓ Found bloom field in visualisation")
        print(f"  - enabled: {bloom.get('enabled')}")
        print(f"  - strength: {bloom.get('strength')}")
        print(f"  - radius: {bloom.get('radius')}")
        print(f"  - threshold: {bloom.get('threshold')}")
        print(f"  - environment_bloom_strength: {bloom.get('environment_bloom_strength')}")
        print(f"  - node_bloom_strength: {bloom.get('node_bloom_strength')}")
        print(f"  - edge_bloom_strength: {bloom.get('edge_bloom_strength')}")
    else:
        print("✗ No bloom field found")
        
    if 'glow' in settings['visualisation']:
        print("✓ Found glow field in visualisation")
    else:
        print("✗ No glow field found (expected - should be mapped from bloom)")
else:
    print("✗ No visualisation field found")

print("\n=== Required Fields Mapping ===")
print("Rust GlowSettings expects:")
print("  - intensity (mapped from bloom.strength)")
print("  - diffuse_strength (mapped from bloom.environment_bloom_strength)")
print("  - atmospheric_density (mapped from bloom.node_bloom_strength)")
print("  - volumetric_intensity (mapped from bloom.edge_bloom_strength)")

print("\n=== Field Presence Check ===")
has_all_fields = True
if 'visualisation' in settings and 'bloom' in settings['visualisation']:
    bloom = settings['visualisation']['bloom']
    
    if 'strength' not in bloom:
        print("✗ Missing bloom.strength (maps to intensity)")
        has_all_fields = False
    else:
        print("✓ bloom.strength present")
        
    if 'environment_bloom_strength' not in bloom:
        print("✗ Missing bloom.environment_bloom_strength")
        has_all_fields = False
    else:
        print("✓ bloom.environment_bloom_strength present")
        
    if 'node_bloom_strength' not in bloom:
        print("✗ Missing bloom.node_bloom_strength")
        has_all_fields = False
    else:
        print("✓ bloom.node_bloom_strength present")
        
    if 'edge_bloom_strength' not in bloom:
        print("✗ Missing bloom.edge_bloom_strength")
        has_all_fields = False
    else:
        print("✓ bloom.edge_bloom_strength present")

if has_all_fields:
    print("\n✅ All required fields are present in settings.yaml")
    print("The serde annotations should handle the mapping correctly")
else:
    print("\n❌ Some required fields are missing")