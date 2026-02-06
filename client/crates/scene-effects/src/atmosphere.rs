/// Atmospheric fog/glow texture generator computed in WASM.
///
/// Produces a slowly evolving RGBA texture driven by fractal noise.
/// The color palette is deep blue-black to soft indigo with faint nebula
/// wisps, designed as a subtle background for a knowledge graph.

use wasm_bindgen::prelude::*;
use crate::noise::fbm3d;

/// Default texture resolution.
const DEFAULT_RES: usize = 128;

/// Atmospheric density field that generates RGBA texture data.
///
/// The texture evolves over time using 3D fBm noise (2D position + time),
/// producing an organic, gently shifting nebula background.
#[wasm_bindgen]
pub struct AtmosphereField {
    width: usize,
    height: usize,
    /// RGBA pixel buffer: width * height * 4 bytes
    pixels: Vec<u8>,
    /// Accumulated time for animation
    time: f32,
    /// Noise frequency scale
    frequency: f32,
    /// Animation speed multiplier
    speed: f32,
}

/// Color palette anchors for the nebula gradient.
/// Each entry is (r, g, b) in 0..255 range.
const PALETTE: [(f32, f32, f32); 5] = [
    (10.0, 10.0, 30.0),    // Deep void (#0a0a1e)
    (15.0, 15.0, 45.0),    // Dark indigo (#0f0f2d)
    (26.0, 26.0, 62.0),    // Soft indigo (#1a1a3e)
    (35.0, 20.0, 70.0),    // Purple wisp (#231446)
    (20.0, 30.0, 80.0),    // Blue nebula (#141e50)
];

/// Interpolate between two palette colors.
#[inline]
fn lerp_color(a: (f32, f32, f32), b: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
        a.2 + (b.2 - a.2) * t,
    )
}

/// Map a noise value in [-1, 1] to a color from the palette.
fn noise_to_color(noise_val: f32) -> (f32, f32, f32) {
    // Remap from [-1, 1] to [0, 1]
    let t = (noise_val * 0.5 + 0.5).clamp(0.0, 1.0);

    // Map to palette segments
    let segments = (PALETTE.len() - 1) as f32;
    let scaled = t * segments;
    let idx = (scaled as usize).min(PALETTE.len() - 2);
    let frac = scaled - idx as f32;

    lerp_color(PALETTE[idx], PALETTE[idx + 1], frac)
}

#[wasm_bindgen]
impl AtmosphereField {
    /// Create a new atmosphere texture generator.
    ///
    /// * `width` - Texture width (0 defaults to 128)
    /// * `height` - Texture height (0 defaults to 128)
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        let w = if width == 0 { DEFAULT_RES } else { width as usize };
        let h = if height == 0 { DEFAULT_RES } else { height as usize };

        Self {
            width: w,
            height: h,
            pixels: vec![0u8; w * h * 4],
            time: 0.0,
            frequency: 1.5,
            speed: 0.08,
        }
    }

    /// Set the noise frequency. Higher values produce finer detail.
    pub fn set_frequency(&mut self, freq: f32) {
        self.frequency = freq.max(0.1);
    }

    /// Set the animation speed multiplier.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.max(0.0);
    }

    /// Advance the atmosphere by `dt` seconds and regenerate the texture.
    ///
    /// This is the main per-frame call. It writes RGBA data into the
    /// internal pixel buffer which can then be read via `get_pixels_ptr`.
    pub fn update(&mut self, dt: f32) {
        self.time += dt * self.speed;
        let t = self.time;
        let freq = self.frequency;

        let inv_w = 1.0 / self.width as f32;
        let inv_h = 1.0 / self.height as f32;

        for y in 0..self.height {
            let ny = y as f32 * inv_h;
            for x in 0..self.width {
                let nx = x as f32 * inv_w;
                let idx = (y * self.width + x) * 4;

                // Primary large-scale density field
                let density = fbm3d(
                    nx * freq,
                    ny * freq,
                    t,
                    4,   // octaves
                    2.0, // lacunarity
                    0.5, // persistence
                );

                // Secondary wisp layer at different frequency/offset
                let wisp = fbm3d(
                    nx * freq * 2.3 + 100.0,
                    ny * freq * 2.3 + 100.0,
                    t * 0.7 + 50.0,
                    3,
                    2.2,
                    0.4,
                );

                // Combine: primary density with subtle wisp highlights
                let combined = density * 0.7 + wisp * 0.3;

                let (r, g, b) = noise_to_color(combined);

                // Subtle luminosity boost for bright areas (nebula glow)
                let glow = ((combined * 0.5 + 0.5) * 1.2).clamp(0.0, 1.0);
                let glow_boost = glow * glow * 15.0;

                self.pixels[idx] = (r + glow_boost).clamp(0.0, 255.0) as u8;
                self.pixels[idx + 1] = (g + glow_boost * 0.8).clamp(0.0, 255.0) as u8;
                self.pixels[idx + 2] = (b + glow_boost * 1.5).clamp(0.0, 255.0) as u8;

                // Alpha: mostly opaque for background, slight transparency
                // at the edges to allow blending
                let edge_x = (nx * 2.0 - 1.0).abs();
                let edge_y = (ny * 2.0 - 1.0).abs();
                let edge_fade = (1.0 - edge_x.max(edge_y)).clamp(0.0, 1.0);
                let alpha = (edge_fade * 200.0 + 30.0).clamp(0.0, 255.0);
                self.pixels[idx + 3] = alpha as u8;
            }
        }
    }

    /// Raw pointer to the RGBA pixel buffer for zero-copy access.
    /// Layout: [r0, g0, b0, a0, r1, g1, b1, a1, ...] (width * height * 4 bytes)
    pub fn get_pixels_ptr(&self) -> *const u8 {
        self.pixels.as_ptr()
    }

    /// Number of bytes in the pixel buffer.
    pub fn get_pixels_len(&self) -> usize {
        self.width * self.height * 4
    }

    /// Texture width.
    pub fn get_width(&self) -> usize {
        self.width
    }

    /// Texture height.
    pub fn get_height(&self) -> usize {
        self.height
    }
}
