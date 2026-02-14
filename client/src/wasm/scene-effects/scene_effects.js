/* @ts-self-types="./scene_effects.d.ts" */

/**
 * Atmospheric density field that generates RGBA texture data.
 *
 * The texture evolves over time using 3D fBm noise (2D position + time),
 * producing an organic, gently shifting nebula background.
 */
export class AtmosphereField {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AtmosphereFieldFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_atmospherefield_free(ptr, 0);
    }
    /**
     * Texture height.
     * @returns {number}
     */
    get_height() {
        const ret = wasm.atmospherefield_get_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of bytes in the pixel buffer.
     * @returns {number}
     */
    get_pixels_len() {
        const ret = wasm.atmospherefield_get_pixels_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Raw pointer to the RGBA pixel buffer for zero-copy access.
     * Layout: [r0, g0, b0, a0, r1, g1, b1, a1, ...] (width * height * 4 bytes)
     * @returns {number}
     */
    get_pixels_ptr() {
        const ret = wasm.atmospherefield_get_pixels_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Texture width.
     * @returns {number}
     */
    get_width() {
        const ret = wasm.atmospherefield_get_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Create a new atmosphere texture generator.
     *
     * * `width` - Texture width (0 defaults to 128)
     * * `height` - Texture height (0 defaults to 128)
     * @param {number} width
     * @param {number} height
     */
    constructor(width, height) {
        const ret = wasm.atmospherefield_new(width, height);
        this.__wbg_ptr = ret >>> 0;
        AtmosphereFieldFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set the noise frequency. Higher values produce finer detail.
     * @param {number} freq
     */
    set_frequency(freq) {
        wasm.atmospherefield_set_frequency(this.__wbg_ptr, freq);
    }
    /**
     * Set the animation speed multiplier.
     * @param {number} speed
     */
    set_speed(speed) {
        wasm.atmospherefield_set_speed(this.__wbg_ptr, speed);
    }
    /**
     * Advance the atmosphere by `dt` seconds and regenerate the texture.
     *
     * This is the main per-frame call. It writes RGBA data into the
     * internal pixel buffer which can then be read via `get_pixels_ptr`.
     * @param {number} dt
     */
    update(dt) {
        wasm.atmospherefield_update(this.__wbg_ptr, dt);
    }
}
if (Symbol.dispose) AtmosphereField.prototype[Symbol.dispose] = AtmosphereField.prototype.free;

export class EnergyWisps {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EnergyWispsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_energywisps_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get_hues_len() {
        const ret = wasm.energywisps_get_hues_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_hues_ptr() {
        const ret = wasm.energywisps_get_hues_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_opacities_len() {
        const ret = wasm.energywisps_get_opacities_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_opacities_ptr() {
        const ret = wasm.energywisps_get_opacities_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_positions_len() {
        const ret = wasm.energywisps_get_positions_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_positions_ptr() {
        const ret = wasm.energywisps_get_positions_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_sizes_len() {
        const ret = wasm.energywisps_get_sizes_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get_sizes_ptr() {
        const ret = wasm.energywisps_get_sizes_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Create a new wisp field with the given count (clamped to MAX_WISPS).
     * @param {number} count
     */
    constructor(count) {
        const ret = wasm.energywisps_new(count);
        this.__wbg_ptr = ret >>> 0;
        EnergyWispsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set the drift speed multiplier (default 1.0).
     * @param {number} speed
     */
    set_drift_speed(speed) {
        wasm.energywisps_set_drift_speed(this.__wbg_ptr, speed);
    }
    /**
     * Advance simulation by `dt` seconds.
     *
     * Camera position is used for depth-aware opacity, same as ParticleField.
     * @param {number} dt
     * @param {number} camera_x
     * @param {number} camera_y
     * @param {number} camera_z
     */
    update(dt, camera_x, camera_y, camera_z) {
        wasm.energywisps_update(this.__wbg_ptr, dt, camera_x, camera_y, camera_z);
    }
    /**
     * @returns {number}
     */
    wisp_count() {
        const ret = wasm.energywisps_wisp_count(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) EnergyWisps.prototype[Symbol.dispose] = EnergyWisps.prototype.free;

/**
 * Particle field managing positions, velocities, visual properties.
 *
 * All buffers are contiguous f32 arrays suitable for direct Float32Array
 * views from JavaScript without any copying.
 */
export class ParticleField {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ParticleFieldFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_particlefield_free(ptr, 0);
    }
    /**
     * Number of f32 values in the opacities buffer.
     * @returns {number}
     */
    get_opacities_len() {
        const ret = wasm.particlefield_get_opacities_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Raw pointer to the opacities buffer.
     * Layout: [o0, o1, o2, ...] (count floats)
     * @returns {number}
     */
    get_opacities_ptr() {
        const ret = wasm.particlefield_get_opacities_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of f32 values in the positions buffer.
     * @returns {number}
     */
    get_positions_len() {
        const ret = wasm.particlefield_get_positions_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Raw pointer to the positions buffer for zero-copy Float32Array.
     * Layout: [x0, y0, z0, x1, y1, z1, ...] (count * 3 floats)
     * @returns {number}
     */
    get_positions_ptr() {
        const ret = wasm.particlefield_get_positions_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of f32 values in the sizes buffer.
     * @returns {number}
     */
    get_sizes_len() {
        const ret = wasm.particlefield_get_sizes_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Raw pointer to the sizes buffer.
     * @returns {number}
     */
    get_sizes_ptr() {
        const ret = wasm.particlefield_get_sizes_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Create a new particle field with the given number of particles.
     * Clamped to MAX_PARTICLES.
     * @param {number} count
     */
    constructor(count) {
        const ret = wasm.particlefield_new(count);
        this.__wbg_ptr = ret >>> 0;
        ParticleFieldFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Current particle count.
     * @returns {number}
     */
    particle_count() {
        const ret = wasm.particlefield_particle_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Advance the particle simulation by `dt` seconds.
     *
     * Camera position is used for depth-aware opacity: particles near the
     * camera fade out (to avoid visual clutter) while distant particles
     * have gentle luminosity.
     * @param {number} dt
     * @param {number} camera_x
     * @param {number} camera_y
     * @param {number} camera_z
     */
    update(dt, camera_x, camera_y, camera_z) {
        wasm.particlefield_update(this.__wbg_ptr, dt, camera_x, camera_y, camera_z);
    }
}
if (Symbol.dispose) ParticleField.prototype[Symbol.dispose] = ParticleField.prototype.free;

/**
 * Initialize the WASM module. Call once before using any other exports.
 * Sets up the panic hook for better error messages in the browser console.
 */
export function init() {
    wasm.init();
}

/**
 * Diagnostic: returns the library version string.
 * @returns {string}
 */
export function version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_9a7fe3f932034cde: function(arg0) {
            console.error(arg0);
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./scene_effects_bg.js": import0,
    };
}

const AtmosphereFieldFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_atmospherefield_free(ptr >>> 0, 1));
const EnergyWispsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_energywisps_free(ptr >>> 0, 1));
const ParticleFieldFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_particlefield_free(ptr >>> 0, 1));

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('scene_effects_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
