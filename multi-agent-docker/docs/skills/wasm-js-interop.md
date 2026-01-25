# WASM-JS Interop Skills

High-performance WebAssembly graphics with JavaScript interoperability for Turbo Flow Claude.

## Available Templates

| Template | Purpose | Use Case |
|----------|---------|----------|
| `voronoi-graphics` | Delaunay/Voronoi computational geometry | Hero backgrounds, generative art |
| (extensible) | Add custom templates in `templates/` | Custom WASM projects |

## Requirements

- **Rust Toolchain**: Already installed (`rustup default stable`)
- **WASM Target**: Already configured (`wasm32-unknown-unknown`)
- **wasm-pack**: Install on first use: `cargo install wasm-pack`

## Voronoi Graphics Template

### Installation

```bash
# Copy template to your project
cp -r /home/devuser/.claude/skills/wasm-js/templates/voronoi-graphics ./my-wasm-graphics

# Build WASM module
cd my-wasm-graphics
./build.sh

# Output in pkg/ directory
```

### Architecture

```
┌─────────────────────────────────────────────────────┐
│              React/TypeScript Application            │
│  ┌─────────────────────────────────────────────────┐│
│  │  - useEffect for animation loop                 ││
│  │  - useRef for canvas context                    ││
│  │  - requestAnimationFrame coordination           ││
│  └─────────────────────────────────────────────────┘│
└──────────────────────┬──────────────────────────────┘
                       │ wasm-bindgen
                       ▼
┌─────────────────────────────────────────────────────┐
│                  Rust WASM Module                    │
│  ┌─────────────────────────────────────────────────┐│
│  │  - Bowyer-Watson Delaunay triangulation         ││
│  │  - Simplex noise generation                     ││
│  │  - Golden ratio seed placement (Vogel)          ││
│  │  - Returns Float32Array for zero-copy render    ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### Key Files

| File | Description |
|------|-------------|
| `Cargo.toml` | Rust project config with wasm-bindgen |
| `src/lib.rs` | WASM module with Delaunay/Voronoi algorithms |
| `integration.tsx` | React component showing JS/WASM integration |
| `build.sh` | Build script for wasm-pack |

### Usage Examples

```typescript
// Import WASM module (lazy-loaded)
import init, { VoronoiGenerator } from './pkg/voronoi_graphics';

// Initialize WASM
await init();

// Create generator
const generator = VoronoiGenerator.new(800, 600, 50);

// Get edge positions (returns Float32Array)
const edges = generator.get_edges();

// Render on canvas
const ctx = canvas.getContext('2d');
ctx.beginPath();
for (let i = 0; i < edges.length; i += 4) {
  ctx.moveTo(edges[i], edges[i + 1]);
  ctx.lineTo(edges[i + 2], edges[i + 3]);
}
ctx.strokeStyle = 'rgba(205, 127, 50, 0.3)';
ctx.stroke();

// Animate seeds
generator.animate(deltaTime);
```

### Build Commands

```bash
# Development build (faster, larger output)
wasm-pack build --target web --dev

# Production build (optimized, smaller output)
wasm-pack build --target web --release

# With specific features
wasm-pack build --target web --release -- --features "simd"
```

### Performance Characteristics

| Metric | Target | Typical |
|--------|--------|---------|
| WASM binary | <100KB | 72KB |
| Frame time | <16ms | ~8ms |
| Memory | <10MB | ~4MB |
| Cross-boundary calls | <100/frame | ~20/frame |

## Best Practices

### 1. Minimize WASM Boundary Crossing

```rust
// BAD: Many small calls
#[wasm_bindgen]
pub fn get_x(&self, index: usize) -> f32 { ... }
#[wasm_bindgen]
pub fn get_y(&self, index: usize) -> f32 { ... }

// GOOD: Batch into typed array
#[wasm_bindgen]
pub fn get_positions(&self) -> Float32Array {
    let positions: Vec<f32> = self.points.iter()
        .flat_map(|p| vec![p.x, p.y])
        .collect();
    Float32Array::from(&positions[..])
}
```

### 2. Let JS Handle Rendering

```typescript
// WASM computes positions
const positions = wasmModule.compute();

// JS renders (has access to Canvas API, WebGL, etc.)
ctx.drawImage(...);
```

### 3. Use Feature Detection

```typescript
const hasWASM = typeof WebAssembly !== 'undefined';
const hasSIMD = await WebAssembly.validate(simdTestModule);

if (hasWASM && hasSIMD) {
  // Use SIMD-optimized WASM
} else if (hasWASM) {
  // Use standard WASM
} else {
  // Fallback to pure JS
}
```

### 4. Cargo.toml Optimization

```toml
[profile.release]
opt-level = "z"      # Optimize for size
lto = true           # Link-time optimization
codegen-units = 1    # Better optimization
panic = "abort"      # Smaller binary
```

## Troubleshooting

### WASM Module Not Loading

```bash
# Check if wasm-pack is installed
wasm-pack --version

# Rebuild with verbose output
wasm-pack build --target web --release -- --verbose

# Check browser console for errors
```

### TypeScript Type Errors

```bash
# Regenerate TypeScript definitions
wasm-pack build --target web

# Check pkg/ for .d.ts files
ls -la pkg/*.d.ts
```

### Performance Issues

```bash
# Profile WASM execution
# 1. Open Chrome DevTools
# 2. Performance tab
# 3. Record while running animation
# 4. Look for "wasm-function" entries

# Or use console.time
console.time('wasm-compute');
const result = wasmModule.compute();
console.timeEnd('wasm-compute');
```

### Memory Leaks

```typescript
// WASM memory is not automatically garbage collected
// Manually free when done
generator.free();

// Or use cleanup in useEffect
useEffect(() => {
  const gen = VoronoiGenerator.new(...);
  return () => gen.free();
}, []);
```

## Integration with Other Skills

| Skill | Integration |
|-------|-------------|
| `playwright` | Visual regression testing of WASM graphics |
| `performance-analysis` | Profile WASM execution time |
| `imagemagick` | Process output screenshots |
| `cuda` | GPU-accelerated alternatives for heavy computation |

## Container Environment

The Turbo Flow container includes:

```bash
# Rust toolchain
rustup --version          # Rust installer
rustc --version           # Rust compiler
cargo --version           # Package manager

# WASM target
rustup target list | grep wasm32-unknown-unknown

# Install wasm-pack (first time)
cargo install wasm-pack
```

## References

- [Rust and WebAssembly Book](https://rustwasm.github.io/docs/book/)
- [wasm-bindgen Guide](https://rustwasm.github.io/wasm-bindgen/)
- [wasm-pack Documentation](https://rustwasm.github.io/wasm-pack/)
- [Bowyer-Watson Algorithm](https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm)
- [Vogel's Model (Golden Angle)](https://en.wikipedia.org/wiki/Golden_angle)
