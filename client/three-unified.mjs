// Unified Three.js entry: WebGPU superset + WebGL compatibility exports.
//
// Resolves the dual-module duplication that causes the WebGPU drawIndexed crash.
// Both Three.WebGPU.js and WebGLRenderer.js share the same Three.Core.js source,
// so Vite/Rollup produces a SINGLE copy of every core class (Scene, Mesh, etc.).
//
// Three.js (WebGL entry) = Three.Core.js + 7 WebGL-specific exports
// Three.WebGPU.js        = Three.Core.js + WebGPU renderer + NodeMaterials + TSL
// This file              = Three.WebGPU.js + the 6 non-conflicting WebGL extras
//
// PMREMGenerator is intentionally excluded — Three.WebGPU.js exports its own
// WebGPU-compatible version from renderers/common/extras/PMREMGenerator.js.
//
// See: https://github.com/mrdoob/three.js/issues/32142

// WebGPU superset: Core + NodeMaterials + WebGPU renderer + TSL
export * from 'three/src/Three.WebGPU.js';

// WebGL-specific exports (not in Three.WebGPU.js, required by R3F + three-stdlib)
export { WebGLRenderer } from 'three/src/renderers/WebGLRenderer.js';
export { WebGLUtils } from 'three/src/renderers/webgl/WebGLUtils.js';
export { ShaderChunk } from 'three/src/renderers/shaders/ShaderChunk.js';
export { ShaderLib } from 'three/src/renderers/shaders/ShaderLib.js';
export { UniformsLib } from 'three/src/renderers/shaders/UniformsLib.js';
export { UniformsUtils } from 'three/src/renderers/shaders/UniformsUtils.js';
