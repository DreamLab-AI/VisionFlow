# Testing Strategy & Validation for DualBloomPipeline Refactoring

## 🎯 Testing Infrastructure Analysis

### ✅ Current Testing Setup
- **Framework**: Vitest (v1.6.1) with React Testing Library
- **Coverage**: V8 provider with 80% threshold requirements
- **Environment**: jsdom with React support
- **Configuration**: `/client/vitest.config.ts`
- **Setup**: `/client/src/tests/setup.ts` with comprehensive mocks

### ✅ Available Testing Libraries
- `@testing-library/jest-dom` - DOM assertion utilities
- `@react-three/fiber` & `@react-three/drei` - R3F components available
- `@react-three/postprocessing` - Post-processing effects testing
- `three` - THREE.js testing utilities
- WebGL/Canvas mocking infrastructure in place

### ⚠️ Current Test Status
- **Total Tests**: ~11 existing test files
- **Focus**: Mainly store management, settings sync, API integration
- **Gap**: **No rendering or 3D component tests exist**
- **Backend**: Rust tests compile but are lengthy (2min+ compile time)

---

## 🧪 Comprehensive Test Strategy for DualBloomPipeline

### 1. **Unit Tests - Core Logic Validation**

#### A. **Layer Separation Logic Tests**
```typescript
// /tests/rendering/DualBloomPipeline.unit.test.tsx
describe('DualBloomPipeline Layer Management', () => {
  describe('Layer Visibility Control', () => {
    it('should correctly isolate Layer 1 (Graph) for graph-only rendering', () => {
      // Test selective rendering with only LAYERS.GRAPH visible
      // Verify Layer 2 objects are hidden during graph pass
    });

    it('should correctly isolate Layer 2 (Environment) for environment-only rendering', () => {
      // Test selective rendering with only LAYERS.ENVIRONMENT visible
      // Verify Layer 1 objects are hidden during environment pass
    });

    it('should restore all layer visibility after selective rendering', () => {
      // Verify cleanup restores original visibility state
    });
  });

  describe('Settings Mapping Validation', () => {
    it('should map bloom settings to Layer 1 pipeline correctly', () => {
      const settings = { visualisation: { bloom: { strength: 1.5, radius: 0.4 } } };
      // Verify UnrealBloomPass receives correct bloom parameters
    });

    it('should map glow settings to Layer 2 pipeline correctly', () => {
      const settings = { visualisation: { glow: { intensity: 2.0, radius: 0.6 } } };
      // Verify UnrealBloomPass receives correct glow parameters
    });

    it('should handle missing or invalid settings gracefully', () => {
      // Test fallback to default values
      // Test with undefined/null settings
    });
  });
});
```

#### B. **Material Integration Tests**
```typescript
// /tests/rendering/materials/BloomStandardMaterial.unit.test.ts
describe('BloomStandardMaterial', () => {
  it('should create materials optimized for post-processing', () => {
    // Verify toneMapped: false for proper bloom interaction
    // Test emissive color and intensity settings
  });

  it('should provide correct preset configurations', () => {
    // Test GraphPrimary, GraphSecondary, EnvironmentGlow presets
    // Verify each preset has appropriate emissive values
  });

  it('should update color and intensity dynamically', () => {
    // Test runtime updates to material properties
  });
});
```

### 2. **Integration Tests - Component Interaction**

#### A. **Settings-to-Pipeline Integration**
```typescript
// /tests/integration/settings-rendering.integration.test.tsx
describe('Settings → Rendering Pipeline Integration', () => {
  it('should update Layer 1 bloom effects when bloom settings change', () => {
    // Mount DualBloomPipeline with test settings
    // Update settings.visualisation.bloom
    // Verify EffectComposer receives new bloom parameters
  });

  it('should update Layer 2 glow effects when glow settings change', () => {
    // Similar test for glow settings propagation
  });

  it('should handle independent layer control', () => {
    // Enable bloom, disable glow → verify only Layer 1 renders with effects
    // Enable glow, disable bloom → verify only Layer 2 renders with effects
  });
});
```

#### B. **Component Integration with GraphCanvas**
```typescript
// /tests/integration/GraphCanvas-DualBloomPipeline.integration.test.tsx
describe('GraphCanvas → DualBloomPipeline Integration', () => {
  it('should properly integrate DualBloomPipeline into Canvas', () => {
    // Test render order and EffectComposer setup
    // Verify extend() calls for post-processing components
  });

  it('should maintain mouse interaction after post-processing', () => {
    // Critical test: verify raycasting still works
    // Test onClick events on graph nodes
  });

  it('should handle Canvas resize correctly', () => {
    // Test responsive resizing of bloom passes
  });
});
```

### 3. **Performance Tests - Optimization Validation**

#### A. **Render Performance Benchmarks**
```typescript
// /tests/performance/rendering.performance.test.ts
describe('DualBloomPipeline Performance', () => {
  it('should maintain 60fps with dual-pipeline processing', async () => {
    // Use performance.now() to measure frame times
    // Render 100 frames and verify average < 16.67ms
  });

  it('should not cause memory leaks during extended rendering', () => {
    // Monitor memory usage over 1000+ frames
    // Verify proper cleanup of WebGL resources
  });

  it('should efficiently handle layer switching', () => {
    // Benchmark selective rendering performance
    // Compare against naive full-scene rendering
  });

  it('should show performance improvement over legacy system', () => {
    // Baseline test: measure old PostProcessingEffects.tsx
    // Compare against new DualBloomPipeline performance
  });
});
```

#### B. **Memory and WebGL Resource Management**
```typescript
// /tests/performance/memory-management.test.ts
describe('WebGL Resource Management', () => {
  it('should properly dispose EffectComposer on unmount', () => {
    // Test component cleanup
    // Verify no WebGL context leaks
  });

  it('should reuse render targets efficiently', () => {
    // Test render target pooling
    // Monitor WebGL texture allocation
  });
});
```

### 4. **Visual Regression Tests - Effect Validation**

#### A. **Bloom Effect Specifications**
```typescript
// /tests/visual/bloom-effects.visual.test.tsx
describe('Bloom Visual Regression', () => {
  it('should render Layer 1 (Graph) bloom effects correctly', () => {
    // Render test scene with known graph elements
    // Compare against reference screenshots
    // Use canvas-based pixel comparison
  });

  it('should render Layer 2 (Environment) glow effects correctly', () => {
    // Test hologram environment rendering
    // Verify soft, diffuse glow appearance
  });

  it('should maintain visual consistency across settings changes', () => {
    // Test various bloom/glow parameter combinations
    // Verify gradual transitions, not abrupt changes
  });

  it('should not show visual artifacts from layer switching', () => {
    // Test for flickering, z-fighting, or rendering glitches
  });
});
```

### 5. **Rust Backend Integration Tests**

#### A. **Type Generation Validation**
```bash
# /tests/backend/type-generation.test.sh
#!/bin/bash
# Test Rust type generation pipeline
cd /workspace/ext
cargo run --bin generate_types
# Verify TypeScript types match Rust structures
```

#### B. **Settings API Integration**
```rust
// tests/integration/settings_api.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bloom_settings_api() {
        // Test settings API for bloom/glow configuration
        // Verify proper serialization/deserialization
    }

    #[tokio::test]
    async fn test_settings_validation() {
        // Test backend validation of rendering settings
        // Verify ranges and constraints
    }
}
```

---

## 🎯 Test Validation Checklist

### ✅ Layer Independence Validation
- [ ] **Layer 1 (Graph) Independence**: Bloom effects work with Layer 2 disabled
- [ ] **Layer 2 (Environment) Independence**: Glow effects work with Layer 1 disabled
- [ ] **No Cross-Layer Interference**: Changes to one layer don't affect the other
- [ ] **Settings Isolation**: `bloom` settings only affect Layer 1, `glow` only Layer 2

### ✅ Performance Validation
- [ ] **No Performance Regressions**: New system performs ≥ old PostProcessingEffects
- [ ] **Memory Stability**: No memory leaks during extended rendering
- [ ] **Frame Rate Maintenance**: Consistent 60fps under normal load
- [ ] **Resource Cleanup**: Proper WebGL resource disposal

### ✅ Mouse Interaction Restoration
- [ ] **Raycasting Works**: Mouse events reach graph elements
- [ ] **Click Events Function**: Node selection and interaction restored
- [ ] **Hover Effects Active**: Mouse-over highlighting functional
- [ ] **Performance Impact Minimal**: Interaction latency unchanged

### ✅ Visual Quality Assurance
- [ ] **Bloom Quality Maintained**: Layer 1 effects match expected quality
- [ ] **Glow Quality Improved**: Layer 2 provides softer, more atmospheric glow
- [ ] **No Visual Artifacts**: No flickering, z-fighting, or rendering errors
- [ ] **Consistent Appearance**: Stable visuals across different settings

---

## 🛠️ Testing Infrastructure Requirements

### A. **React Three Fiber Testing Utilities**
```typescript
// /tests/utils/r3f-test-utils.tsx
// Custom testing utilities for R3F components
export const renderThreeComponent = (component: ReactNode) => {
  // Setup Canvas with minimal configuration
  // Provide mock WebGL context
  // Return testing utilities for 3D scene inspection
};

export const createMockWebGLContext = () => {
  // Mock WebGL for headless testing
};

export const assertShaderUniform = (material: Material, uniform: string, expectedValue: any) => {
  // Utility for testing shader uniform values
};
```

### B. **Performance Testing Infrastructure**
```typescript
// /tests/utils/performance-utils.ts
export class PerformanceProfiler {
  measureRenderTime(renderFn: () => void): number;
  measureMemoryUsage(): MemoryInfo;
  profileFrameRate(duration: number): Promise<number>;
}
```

### C. **Visual Testing Setup**
```typescript
// /tests/utils/visual-test-utils.ts
export const captureCanvasImage = (canvas: HTMLCanvasElement): string;
export const compareImages = (imageA: string, imageB: string, threshold: number): boolean;
export const createReferenceScreenshots = (): void;
```

---

## 📋 Test Execution Strategy

### Phase 1: Unit Tests (Essential)
1. **DualBloomPipeline core logic** - Layer separation and settings mapping
2. **Material functionality** - BloomStandardMaterial and HologramNodeMaterial
3. **Settings integration** - Proper parameter propagation

### Phase 2: Integration Tests (Critical)
1. **Settings → Rendering pipeline** - End-to-end settings flow
2. **Component integration** - GraphCanvas + DualBloomPipeline
3. **Mouse interaction restoration** - Critical functionality test

### Phase 3: Performance Validation (Important)
1. **Render performance** - Frame rate and optimization verification
2. **Memory management** - Resource leak detection
3. **Comparative benchmarks** - Old vs new system performance

### Phase 4: Visual Regression (Quality Assurance)
1. **Effect quality** - Bloom and glow appearance validation
2. **Cross-browser compatibility** - Ensure consistent rendering
3. **Settings variation testing** - Visual stability across configurations

---

## 🚀 Expected Testing Outcomes

### Success Metrics
- **100% Layer Independence**: Each pipeline works correctly in isolation
- **≥95% Performance Parity**: New system matches or exceeds old performance
- **Zero Critical Regressions**: Mouse interaction and basic functionality preserved
- **Improved Visual Quality**: Better separation between graph and environment effects

### Risk Mitigation
- **Automated Testing**: Prevent regression introduction during development
- **Performance Monitoring**: Early detection of performance degradation
- **Visual Validation**: Ensure artistic intent is preserved
- **Cross-Platform Testing**: Verify compatibility across target environments

This comprehensive testing strategy ensures the DualBloomPipeline refactoring delivers on its promises while maintaining system stability and performance.