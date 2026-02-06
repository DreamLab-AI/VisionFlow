//! Scene Effects - WASM-powered background visuals for Three.js knowledge graphs.
//!
//! This crate provides two main systems:
//!
//! - **ParticleField**: Ambient drifting particles with noise-based motion
//!   and depth-aware opacity. Exposes raw f32 buffers for zero-copy
//!   Float32Array views in JavaScript.
//!
//! - **AtmosphereField**: Procedural nebula-like background texture that
//!   evolves slowly over time. Produces RGBA pixel data suitable for
//!   uploading to a Three.js DataTexture.
//!
//! All math uses f32 for WASM performance. The noise module provides
//! 2D/3D simplex noise and fractal Brownian motion.

#[allow(dead_code)]
mod noise;
mod particles;
mod atmosphere;
mod energy_wisps;

// Re-export the WASM-bindgen types so they are accessible from JS
pub use particles::ParticleField;
pub use atmosphere::AtmosphereField;
pub use energy_wisps::EnergyWisps;

use wasm_bindgen::prelude::*;

/// Initialize the WASM module. Call once before using any other exports.
/// Sets up the panic hook for better error messages in the browser console.
#[wasm_bindgen(start)]
pub fn init() {
    // Minimal panic hook: log panics to console.error
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        web_sys::console::error_1(&msg.into());
    }));
}

/// Diagnostic: returns the library version string.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
