/**
 * scene-effects-bridge.ts
 *
 * TypeScript bridge for the scene-effects WASM module.
 * Provides typed wrappers around the raw WASM exports and creates
 * zero-copy Float32Array / Uint8Array views over WASM linear memory.
 *
 * Usage:
 *   const wasm = await initSceneEffects();
 *   const particles = wasm.createParticleField(256);
 *   particles.update(0.016, 0, 0, 5);
 *   const positions = particles.getPositions(); // Float32Array view
 */

// The WASM module is loaded dynamically, so these types mirror the
// generated .d.ts but we define them here to avoid import-time failures
// when WASM is unavailable.

export interface SceneEffectsModule {
  memory: WebAssembly.Memory;
  ParticleField: new (count: number) => WasmParticleField;
  AtmosphereField: new (width: number, height: number) => WasmAtmosphereField;
  EnergyWisps: new (count: number) => WasmEnergyWisps;
  version: () => string;
}

/** Raw WASM ParticleField handle. */
interface WasmParticleField {
  update(dt: number, camera_x: number, camera_y: number, camera_z: number): void;
  get_positions_ptr(): number;
  get_positions_len(): number;
  get_opacities_ptr(): number;
  get_opacities_len(): number;
  get_sizes_ptr(): number;
  get_sizes_len(): number;
  particle_count(): number;
  free(): void;
}

/** Raw WASM EnergyWisps handle. */
interface WasmEnergyWisps {
  update(dt: number, camera_x: number, camera_y: number, camera_z: number): void;
  set_drift_speed(speed: number): void;
  get_positions_ptr(): number;
  get_positions_len(): number;
  get_opacities_ptr(): number;
  get_opacities_len(): number;
  get_sizes_ptr(): number;
  get_sizes_len(): number;
  get_hues_ptr(): number;
  get_hues_len(): number;
  wisp_count(): number;
  free(): void;
}

/** Raw WASM AtmosphereField handle. */
interface WasmAtmosphereField {
  update(dt: number): void;
  get_pixels_ptr(): number;
  get_pixels_len(): number;
  get_width(): number;
  get_height(): number;
  set_frequency(freq: number): void;
  set_speed(speed: number): void;
  free(): void;
}

/**
 * Wrapped particle field that exposes typed array views over WASM memory.
 * The Float32Array views are created fresh on each access to account for
 * potential WASM memory growth, but the underlying buffer is zero-copy.
 */
export class ParticleFieldBridge {
  private inner: WasmParticleField;
  private memory: WebAssembly.Memory;
  private _disposed = false;

  constructor(inner: WasmParticleField, memory: WebAssembly.Memory) {
    this.inner = inner;
    this.memory = memory;
  }

  /** True after dispose() has been called. Guards against use-after-free. */
  get isDisposed(): boolean { return this._disposed; }

  /** Advance simulation. Call once per frame. */
  update(dt: number, cameraX: number, cameraY: number, cameraZ: number): void {
    if (this._disposed) return;
    this.inner.update(dt, cameraX, cameraY, cameraZ);
  }

  /** Zero-copy Float32Array view of particle positions [x,y,z, x,y,z, ...]. */
  getPositions(): Float32Array {
    if (this._disposed) return new Float32Array(0);
    const ptr = this.inner.get_positions_ptr();
    const len = this.inner.get_positions_len();
    const byteOffset = ptr;
    const byteLength = len * 4;
    if (byteOffset + byteLength > this.memory.buffer.byteLength) {
      throw new Error('WASM pointer out of bounds');
    }
    return new Float32Array(this.memory.buffer, ptr, len);
  }

  /** Zero-copy Float32Array view of per-particle opacities. */
  getOpacities(): Float32Array {
    if (this._disposed) return new Float32Array(0);
    const ptr = this.inner.get_opacities_ptr();
    const len = this.inner.get_opacities_len();
    const byteOffset = ptr;
    const byteLength = len * 4;
    if (byteOffset + byteLength > this.memory.buffer.byteLength) {
      throw new Error('WASM pointer out of bounds');
    }
    return new Float32Array(this.memory.buffer, ptr, len);
  }

  /** Zero-copy Float32Array view of per-particle sizes. */
  getSizes(): Float32Array {
    if (this._disposed) return new Float32Array(0);
    const ptr = this.inner.get_sizes_ptr();
    const len = this.inner.get_sizes_len();
    const byteOffset = ptr;
    const byteLength = len * 4;
    if (byteOffset + byteLength > this.memory.buffer.byteLength) {
      throw new Error('WASM pointer out of bounds');
    }
    return new Float32Array(this.memory.buffer, ptr, len);
  }

  /** Number of particles in this field. */
  get count(): number {
    if (this._disposed) return 0;
    return this.inner.particle_count();
  }

  /** Release WASM resources. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.inner.free();
  }
}

/**
 * Wrapped atmosphere field that provides a Uint8Array RGBA texture view.
 */
export class AtmosphereFieldBridge {
  private inner: WasmAtmosphereField;
  private memory: WebAssembly.Memory;
  private _disposed = false;

  constructor(inner: WasmAtmosphereField, memory: WebAssembly.Memory) {
    this.inner = inner;
    this.memory = memory;
  }

  /** True after dispose() has been called. */
  get isDisposed(): boolean { return this._disposed; }

  /** Advance the atmosphere texture. Call once per frame. */
  update(dt: number): void {
    if (this._disposed) return;
    this.inner.update(dt);
  }

  /** Zero-copy Uint8Array view of RGBA pixel data. */
  getPixels(): Uint8Array {
    if (this._disposed) return new Uint8Array(0);
    const ptr = this.inner.get_pixels_ptr();
    const len = this.inner.get_pixels_len();
    if (ptr + len > this.memory.buffer.byteLength) {
      throw new Error('WASM pointer out of bounds');
    }
    return new Uint8Array(this.memory.buffer, ptr, len);
  }

  /** Texture width in pixels. */
  get width(): number {
    if (this._disposed) return 0;
    return this.inner.get_width();
  }

  /** Texture height in pixels. */
  get height(): number {
    if (this._disposed) return 0;
    return this.inner.get_height();
  }

  /** Set noise frequency (higher = finer detail). */
  setFrequency(freq: number): void {
    if (this._disposed) return;
    this.inner.set_frequency(freq);
  }

  /** Set animation speed multiplier. */
  setSpeed(speed: number): void {
    if (this._disposed) return;
    this.inner.set_speed(speed);
  }

  /** Release WASM resources. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.inner.free();
  }
}

/**
 * Wrapped energy wisps field: ephemeral glowing orbs with lifecycle fade.
 */
export class WispFieldBridge {
  private inner: WasmEnergyWisps;
  private memory: WebAssembly.Memory;
  private _disposed = false;

  constructor(inner: WasmEnergyWisps, memory: WebAssembly.Memory) {
    this.inner = inner;
    this.memory = memory;
  }

  /** True after dispose() has been called. */
  get isDisposed(): boolean { return this._disposed; }

  /** Advance simulation. Call once per frame. */
  update(dt: number, cameraX: number, cameraY: number, cameraZ: number): void {
    if (this._disposed) return;
    this.inner.update(dt, cameraX, cameraY, cameraZ);
  }

  /** Set drift speed multiplier (default 1.0). */
  setDriftSpeed(speed: number): void {
    if (this._disposed) return;
    this.inner.set_drift_speed(speed);
  }

  /** Zero-copy Float32Array view of wisp positions [x,y,z, ...]. */
  getPositions(): Float32Array {
    if (this._disposed) return new Float32Array(0);
    const ptr = this.inner.get_positions_ptr();
    const len = this.inner.get_positions_len();
    if (ptr + len * 4 > this.memory.buffer.byteLength) {
      throw new Error('WASM pointer out of bounds');
    }
    return new Float32Array(this.memory.buffer, ptr, len);
  }

  /** Zero-copy Float32Array view of per-wisp opacities. */
  getOpacities(): Float32Array {
    if (this._disposed) return new Float32Array(0);
    const ptr = this.inner.get_opacities_ptr();
    const len = this.inner.get_opacities_len();
    if (ptr + len * 4 > this.memory.buffer.byteLength) {
      throw new Error('WASM pointer out of bounds');
    }
    return new Float32Array(this.memory.buffer, ptr, len);
  }

  /** Zero-copy Float32Array view of per-wisp sizes. */
  getSizes(): Float32Array {
    if (this._disposed) return new Float32Array(0);
    const ptr = this.inner.get_sizes_ptr();
    const len = this.inner.get_sizes_len();
    if (ptr + len * 4 > this.memory.buffer.byteLength) {
      throw new Error('WASM pointer out of bounds');
    }
    return new Float32Array(this.memory.buffer, ptr, len);
  }

  /** Zero-copy Float32Array view of per-wisp hue offsets (0..1). */
  getHues(): Float32Array {
    if (this._disposed) return new Float32Array(0);
    const ptr = this.inner.get_hues_ptr();
    const len = this.inner.get_hues_len();
    if (ptr + len * 4 > this.memory.buffer.byteLength) {
      throw new Error('WASM pointer out of bounds');
    }
    return new Float32Array(this.memory.buffer, ptr, len);
  }

  /** Number of wisps. */
  get count(): number {
    if (this._disposed) return 0;
    return this.inner.wisp_count();
  }

  /** Release WASM resources. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.inner.free();
  }
}

/**
 * Public API returned from initSceneEffects().
 */
export interface SceneEffectsAPI {
  /** Create a particle field with up to `count` particles (max 512). */
  createParticleField(count: number): ParticleFieldBridge;
  /** Create an atmosphere texture generator of the given resolution. */
  createAtmosphereField(width: number, height: number): AtmosphereFieldBridge;
  /** Create an energy wisps field with up to `count` wisps (max 128). */
  createWispField(count: number): WispFieldBridge;
  /** WASM module version string. */
  version: string;
}

// Module-level singleton to prevent double-init
let cachedAPI: SceneEffectsAPI | null = null;
let initPromise: Promise<SceneEffectsAPI> | null = null;

/**
 * Initialize the WASM scene effects module.
 *
 * Safe to call multiple times -- returns the cached instance after the
 * first successful init. Returns null-like API if WASM fails to load
 * (graceful degradation).
 */
export async function initSceneEffects(): Promise<SceneEffectsAPI> {
  if (cachedAPI) return cachedAPI;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      // Dynamic import of the wasm-pack generated glue code.
      // The path is relative to where this bridge file sits in the build output.
      const wasmModule = await import('./scene-effects/scene_effects.js');
      const initOutput = await wasmModule.default();
      const memory = initOutput.memory;

      cachedAPI = {
        createParticleField(count: number): ParticleFieldBridge {
          const inner = new wasmModule.ParticleField(count);
          return new ParticleFieldBridge(inner, memory);
        },
        createAtmosphereField(width: number, height: number): AtmosphereFieldBridge {
          const inner = new wasmModule.AtmosphereField(width, height);
          return new AtmosphereFieldBridge(inner, memory);
        },
        createWispField(count: number): WispFieldBridge {
          const inner = new wasmModule.EnergyWisps(count);
          return new WispFieldBridge(inner, memory);
        },
        version: wasmModule.version(),
      };

      // eslint-disable-next-line no-console
      console.info(`[scene-effects] WASM loaded, version ${cachedAPI.version}`);
      return cachedAPI;
    } catch (err) {
      // Reset initPromise so subsequent calls can retry (e.g., after a
      // transient network error loading the .wasm file).
      initPromise = null;
      // eslint-disable-next-line no-console
      console.warn('[scene-effects] WASM failed to load, effects disabled:', err);
      throw err;
    }
  })();

  return initPromise;
}
