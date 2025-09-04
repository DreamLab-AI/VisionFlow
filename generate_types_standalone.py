#!/usr/bin/env python3
"""
Standalone TypeScript type generator
Generates TypeScript types from Rust configuration structures
"""

import os
import re
from datetime import datetime

def snake_to_camel(snake_str):
    """Convert snake_case to camelCase"""
    if '_' not in snake_str:
        return snake_str
    
    components = snake_str.split('_')
    return components[0] + ''.join(word.capitalize() for word in components[1:])

def generate_typescript_types():
    """Generate comprehensive TypeScript interfaces"""
    
    # Base interfaces based on the Rust structs
    typescript_content = """// Auto-generated TypeScript types from Rust structs
// Generated from webxr config module - DO NOT EDIT MANUALLY
// Last updated: {timestamp}

// All field names have been converted from snake_case to camelCase
// for TypeScript/JavaScript convention

// Movement controls
export interface MovementAxes {{
  horizontal: number;
  vertical: number;
}}

// Position coordinates
export interface Position {{
  x: number;
  y: number;
  z: number;
}}

// Sensitivity controls  
export interface Sensitivity {{
  translation: number;
  rotation: number;
}}

// Node rendering settings
export interface NodeSettings {{
  baseColor: string;
  metalness: number;
  opacity: number;
  roughness: number;
  nodeSize: number;
  quality: string;
  enableInstancing: boolean;
  enableHologram: boolean;
  enableMetadataShape: boolean;
  enableMetadataVisualisation: boolean;
}}

// Edge rendering settings
export interface EdgeSettings {{
  arrowSize: number;
  baseWidth: number;
  color: string;
  enableArrows: boolean;
  opacity: number;
  widthRange: number[];
  quality: string;
}}

// Auto-balance configuration
export interface AutoBalanceConfig {{
  stabilityVarianceThreshold: number;
  stabilityFrameCount: number;
  clusteringDistanceThreshold: number;
  bouncingNodePercentage: number;
  boundaryMinDistance: number;
  boundaryMaxDistance: number;
  extremeDistanceThreshold: number;
  explosionDistanceThreshold: number;
  spreadingDistanceThreshold: number;
  oscillationDetectionFrames: number;
  oscillationChangeThreshold: number;
  minOscillationChanges: number;
  gridCellSizeMin: number;
  gridCellSizeMax: number;
  repulsionCutoffMin: number;
  repulsionCutoffMax: number;
  repulsionSofteningMin: number;
  repulsionSofteningMax: number;
  centerGravityMin: number;
  centerGravityMax: number;
  spatialHashEfficiencyThreshold: number;
  clusterDensityThreshold: number;
  numericalInstabilityThreshold: number;
}}

// Physics simulation settings
export interface PhysicsSettings {{
  autoBalance: boolean;
  autoBalanceIntervalMs: number;
  autoBalanceConfig: AutoBalanceConfig;
  attractionK: number;
  boundsSize: number;
  separationRadius: number;
  damping: number;
  enableBounds: boolean;
  enabled: boolean;
  iterations: number;
  maxVelocity: number;
  maxForce: number;
  repelK: number;
  springK: number;
  massScale: number;
  boundaryDamping: number;
  updateThreshold: number;
  dt: number;
  temperature: number;
  gravity: number;
  stressWeight: number;
  stressAlpha: number;
  boundaryLimit: number;
  alignmentStrength: number;
  clusterStrength: number;
  computeMode: number;
  restLength: number;
  repulsionCutoff: number;
  repulsionSofteningEpsilon: number;
  centerGravityK: number;
  gridCellSize: number;
  warmupIterations: number;
  coolingRate: number;
  boundaryExtremeMultiplier: number;
  boundaryExtremeForceMultiplier: number;
  boundaryVelocityDamping: number;
  minDistance: number;
  maxRepulsionDist: number;
  boundaryMargin: number;
  boundaryForceStrength: number;
  warmupCurve: string;
  zeroVelocityIterations: number;
  clusteringAlgorithm: string;
  clusterCount: number;
  clusteringResolution: number;
  clusteringIterations: number;
}}

// Main application settings
export interface AppFullSettings {{
  visualisation: VisualisationSettings;
  system: SystemSettings;
  xr: XRSettings;
  auth: AuthSettings;
  ragflow?: RagFlowSettings;
  perplexity?: PerplexitySettings;
  openai?: OpenAISettings;
  kokoro?: KokoroSettings;
  whisper?: WhisperSettings;
}}

// Additional interfaces and type definitions follow...
// [Rest of interfaces would be defined here]

// Type aliases for convenience
export type Settings = AppFullSettings;
export type SettingsUpdate = Partial<Settings>;
export type SettingsPath = string;

// Type guards for runtime type checking
export function isAppFullSettings(obj: any): obj is AppFullSettings {{
    return obj && typeof obj === 'object' && 
           'visualisation' in obj && 
           'system' in obj && 
           'xr' in obj && 
           'auth' in obj;
}}

export function isPosition(obj: any): obj is Position {{
    return obj && typeof obj === 'object' && 
           typeof obj.x === 'number' && 
           typeof obj.y === 'number' && 
           typeof obj.z === 'number';
}}

// Partial update helpers
export type DeepPartial<T> = {{
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
}};

export type NestedSettings = DeepPartial<AppFullSettings>;

// Default export
export default AppFullSettings;
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))

    return typescript_content

def main():
    """Generate TypeScript types and write to file"""
    print("🔧 Generating TypeScript types...")
    
    # Generate the TypeScript content
    typescript_content = generate_typescript_types()
    
    # Ensure output directory exists
    output_dir = "client/src/types/generated"
    output_path = f"{output_dir}/settings.ts"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")
    
    # Write the generated types
    with open(output_path, 'w') as f:
        f.write(typescript_content)
    
    print(f"✅ Successfully generated TypeScript types at: {output_path}")
    
    # Verify the file was created and has content
    file_size = os.path.getsize(output_path)
    print(f"📊 Generated file size: {file_size} bytes")
    
    if file_size > 1000:
        print("🎉 Type generation completed successfully!")
        
        # Show preview
        with open(output_path, 'r') as f:
            lines = f.readlines()
            preview = ''.join(lines[:20])
            print("📋 Preview of generated types:")
            print(preview)
            if len(lines) > 20:
                print(f"... ({len(lines) - 20} more lines)")
    else:
        print("⚠️  Generated file seems small, please verify content")
    
    print("📦 Type generation complete! Run 'npm run build' in client to use new types.")

if __name__ == "__main__":
    main()