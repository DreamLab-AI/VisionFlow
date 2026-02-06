/// Ambient particle system computed entirely in WASM.
///
/// Manages a field of gently drifting particles with noise-based perturbation.
/// Exposes raw f32 buffers for zero-copy Float32Array views in JavaScript.

use wasm_bindgen::prelude::*;
use crate::noise::simplex3d;

/// Maximum number of particles the system can handle.
const MAX_PARTICLES: usize = 512;

/// Spread radius for initial particle placement.
const SPAWN_RADIUS: f32 = 50.0;

/// A pseudo-random number generator (xorshift32) that avoids pulling in rand.
struct Xorshift {
    state: u32,
}

impl Xorshift {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Returns a value in [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        (self.state as f32) / (u32::MAX as f32)
    }

    /// Returns a value in [-1, 1).
    fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Particle field managing positions, velocities, visual properties.
///
/// All buffers are contiguous f32 arrays suitable for direct Float32Array
/// views from JavaScript without any copying.
#[wasm_bindgen]
pub struct ParticleField {
    count: usize,
    /// Interleaved position buffer: [x0, y0, z0, x1, y1, z1, ...]
    positions: Vec<f32>,
    /// Per-particle velocity: [vx0, vy0, vz0, ...]
    velocities: Vec<f32>,
    /// Per-particle opacity (0..1)
    opacities: Vec<f32>,
    /// Per-particle size
    sizes: Vec<f32>,
    /// Accumulated time for noise evolution
    time: f32,
}

#[wasm_bindgen]
impl ParticleField {
    /// Create a new particle field with the given number of particles.
    /// Clamped to MAX_PARTICLES.
    #[wasm_bindgen(constructor)]
    pub fn new(count: u32) -> Self {
        let count = (count as usize).min(MAX_PARTICLES);
        let mut rng = Xorshift::new(0xDEAD_BEEF);

        let mut positions = vec![0.0f32; count * 3];
        let mut velocities = vec![0.0f32; count * 3];
        let mut opacities = vec![0.0f32; count];
        let mut sizes = vec![0.0f32; count];

        for i in 0..count {
            let idx = i * 3;
            // Distribute particles in a spherical volume
            let theta = rng.next_f32() * std::f32::consts::TAU;
            let phi = (rng.next_f32() * 2.0 - 1.0).acos();
            let r = rng.next_f32().cbrt() * SPAWN_RADIUS;

            positions[idx] = r * phi.sin() * theta.cos();
            positions[idx + 1] = r * phi.sin() * theta.sin();
            positions[idx + 2] = r * phi.cos();

            // Very gentle base drift
            velocities[idx] = rng.next_signed() * 0.02;
            velocities[idx + 1] = rng.next_signed() * 0.02;
            velocities[idx + 2] = rng.next_signed() * 0.02;

            opacities[i] = rng.next_f32() * 0.3 + 0.1;
            sizes[i] = rng.next_f32() * 0.8 + 0.2;
        }

        Self {
            count,
            positions,
            velocities,
            opacities,
            sizes,
            time: 0.0,
        }
    }

    /// Advance the particle simulation by `dt` seconds.
    ///
    /// Camera position is used for depth-aware opacity: particles near the
    /// camera fade out (to avoid visual clutter) while distant particles
    /// have gentle luminosity.
    pub fn update(&mut self, dt: f32, camera_x: f32, camera_y: f32, camera_z: f32) {
        self.time += dt;
        let t = self.time;

        for i in 0..self.count {
            let idx = i * 3;
            let px = self.positions[idx];
            let py = self.positions[idx + 1];
            let pz = self.positions[idx + 2];

            // Noise-based perturbation: use particle position scaled down
            // and offset by time to create gentle swirling motion
            let noise_scale = 0.03;
            let time_scale = 0.15;
            let nx = simplex3d(
                px * noise_scale,
                py * noise_scale,
                t * time_scale,
            ) * 0.5;
            let ny = simplex3d(
                py * noise_scale + 100.0,
                pz * noise_scale,
                t * time_scale,
            ) * 0.5;
            let nz = simplex3d(
                pz * noise_scale + 200.0,
                px * noise_scale,
                t * time_scale,
            ) * 0.5;

            // Apply velocity + noise perturbation
            self.positions[idx] += (self.velocities[idx] + nx) * dt;
            self.positions[idx + 1] += (self.velocities[idx + 1] + ny) * dt;
            self.positions[idx + 2] += (self.velocities[idx + 2] + nz) * dt;

            // Soft boundary: gently pull particles back if they drift too far
            let dist_sq = self.positions[idx] * self.positions[idx]
                + self.positions[idx + 1] * self.positions[idx + 1]
                + self.positions[idx + 2] * self.positions[idx + 2];
            let max_dist_sq = SPAWN_RADIUS * SPAWN_RADIUS * 1.5;

            if dist_sq > max_dist_sq {
                let factor = 0.98;
                self.positions[idx] *= factor;
                self.positions[idx + 1] *= factor;
                self.positions[idx + 2] *= factor;
            }

            // Depth-aware opacity: distance from camera
            let dx = self.positions[idx] - camera_x;
            let dy = self.positions[idx + 1] - camera_y;
            let dz = self.positions[idx + 2] - camera_z;
            let cam_dist = (dx * dx + dy * dy + dz * dz).sqrt();

            // Fade near camera (< 3 units) and far away (> 40 units)
            let near_fade = ((cam_dist - 1.5) / 3.0).clamp(0.0, 1.0);
            let far_fade = ((50.0 - cam_dist) / 15.0).clamp(0.0, 1.0);

            // Base opacity with breathing (slow noise modulation)
            let breath = (simplex3d(
                (i as f32) * 0.1,
                t * 0.3,
                0.0,
            ) * 0.5 + 0.5) * 0.4 + 0.1;

            self.opacities[i] = breath * near_fade * far_fade;

            // Size also varies gently
            let base_size = 0.3 + (i as f32 % 5.0) * 0.15;
            let size_pulse = simplex3d(
                (i as f32) * 0.2 + 50.0,
                t * 0.2,
                0.0,
            ) * 0.1 + 1.0;
            self.sizes[i] = base_size * size_pulse;
        }
    }

    /// Raw pointer to the positions buffer for zero-copy Float32Array.
    /// Layout: [x0, y0, z0, x1, y1, z1, ...] (count * 3 floats)
    pub fn get_positions_ptr(&self) -> *const f32 {
        self.positions.as_ptr()
    }

    /// Number of f32 values in the positions buffer.
    pub fn get_positions_len(&self) -> usize {
        self.count * 3
    }

    /// Raw pointer to the opacities buffer.
    /// Layout: [o0, o1, o2, ...] (count floats)
    pub fn get_opacities_ptr(&self) -> *const f32 {
        self.opacities.as_ptr()
    }

    /// Number of f32 values in the opacities buffer.
    pub fn get_opacities_len(&self) -> usize {
        self.count
    }

    /// Raw pointer to the sizes buffer.
    pub fn get_sizes_ptr(&self) -> *const f32 {
        self.sizes.as_ptr()
    }

    /// Number of f32 values in the sizes buffer.
    pub fn get_sizes_len(&self) -> usize {
        self.count
    }

    /// Current particle count.
    pub fn particle_count(&self) -> usize {
        self.count
    }
}
