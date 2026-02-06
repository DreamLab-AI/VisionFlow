/// Ephemeral energy wisps: short-lived glowing orbs that drift through
/// the scene on noise-driven paths, fading in and out of existence.
///
/// Each wisp has a random lifespan (2-8 seconds). During the first 20%
/// of life it fades in; during the last 30% it fades out. When a wisp
/// dies it respawns at a new random position. The result is a gentle,
/// living background of flickering motes.
///
/// All buffers are contiguous f32 for zero-copy Float32Array views.

use wasm_bindgen::prelude::*;
use crate::noise::simplex3d;

/// Maximum wisps the system supports.
const MAX_WISPS: usize = 128;

/// Spawn radius for wisp birth positions.
const SPAWN_RADIUS: f32 = 45.0;

/// Minimum wisp lifespan in seconds.
const MIN_LIFE: f32 = 2.0;
/// Maximum wisp lifespan in seconds.
const MAX_LIFE: f32 = 8.0;

/// Deterministic xorshift PRNG (avoids pulling in `rand`).
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

#[wasm_bindgen]
pub struct EnergyWisps {
    count: usize,
    /// Interleaved positions: [x0, y0, z0, x1, y1, z1, ...]
    positions: Vec<f32>,
    /// Per-wisp opacity (0..1), includes lifecycle fade
    opacities: Vec<f32>,
    /// Per-wisp size
    sizes: Vec<f32>,
    /// Per-wisp color hue offset (0..1)
    hues: Vec<f32>,
    /// Per-wisp current age (seconds)
    ages: Vec<f32>,
    /// Per-wisp total lifespan (seconds)
    lifespans: Vec<f32>,
    /// Per-wisp base velocity
    velocities: Vec<f32>,
    /// Accumulated time
    time: f32,
    /// RNG state persisted across frames for respawn
    rng_state: u32,
    /// Drift speed multiplier (from settings)
    drift_speed: f32,
}

#[wasm_bindgen]
impl EnergyWisps {
    /// Create a new wisp field with the given count (clamped to MAX_WISPS).
    #[wasm_bindgen(constructor)]
    pub fn new(count: u32) -> Self {
        let count = (count as usize).min(MAX_WISPS);
        let mut rng = Xorshift::new(0xCAFE_BABE);

        let mut positions = vec![0.0f32; count * 3];
        let mut opacities = vec![0.0f32; count];
        let mut sizes = vec![0.0f32; count];
        let mut hues = vec![0.0f32; count];
        let mut ages = vec![0.0f32; count];
        let mut lifespans = vec![0.0f32; count];
        let mut velocities = vec![0.0f32; count * 3];

        for i in 0..count {
            let idx = i * 3;

            // Spawn in a spherical volume
            let theta = rng.next_f32() * std::f32::consts::TAU;
            let phi = (rng.next_f32() * 2.0 - 1.0).acos();
            let r = rng.next_f32().cbrt() * SPAWN_RADIUS;

            positions[idx] = r * phi.sin() * theta.cos();
            positions[idx + 1] = r * phi.sin() * theta.sin();
            positions[idx + 2] = r * phi.cos();

            // Gentle base drift
            velocities[idx] = rng.next_signed() * 0.8;
            velocities[idx + 1] = rng.next_signed() * 0.8;
            velocities[idx + 2] = rng.next_signed() * 0.8;

            // Stagger initial ages so they don't all spawn/die together
            lifespans[i] = MIN_LIFE + rng.next_f32() * (MAX_LIFE - MIN_LIFE);
            ages[i] = rng.next_f32() * lifespans[i];

            sizes[i] = 0.4 + rng.next_f32() * 0.6;
            hues[i] = rng.next_f32();
        }

        Self {
            count,
            positions,
            opacities,
            sizes,
            hues,
            ages,
            lifespans,
            velocities,
            time: 0.0,
            rng_state: rng.state,
            drift_speed: 1.0,
        }
    }

    /// Set the drift speed multiplier (default 1.0).
    pub fn set_drift_speed(&mut self, speed: f32) {
        self.drift_speed = speed.max(0.0);
    }

    /// Advance simulation by `dt` seconds.
    ///
    /// Camera position is used for depth-aware opacity, same as ParticleField.
    pub fn update(&mut self, dt: f32, camera_x: f32, camera_y: f32, camera_z: f32) {
        self.time += dt;
        let t = self.time;
        let mut rng = Xorshift::new(self.rng_state);
        let drift = self.drift_speed;

        for i in 0..self.count {
            let idx = i * 3;

            // Advance age
            self.ages[i] += dt;

            // Respawn if wisp has expired
            if self.ages[i] >= self.lifespans[i] {
                self.ages[i] = 0.0;
                self.lifespans[i] = MIN_LIFE + rng.next_f32() * (MAX_LIFE - MIN_LIFE);

                // New random position
                let theta = rng.next_f32() * std::f32::consts::TAU;
                let phi = (rng.next_f32() * 2.0 - 1.0).acos();
                let r = rng.next_f32().cbrt() * SPAWN_RADIUS;

                self.positions[idx] = r * phi.sin() * theta.cos();
                self.positions[idx + 1] = r * phi.sin() * theta.sin();
                self.positions[idx + 2] = r * phi.cos();

                // New drift direction
                self.velocities[idx] = rng.next_signed() * 0.8;
                self.velocities[idx + 1] = rng.next_signed() * 0.8;
                self.velocities[idx + 2] = rng.next_signed() * 0.8;

                // New appearance
                self.sizes[i] = 0.4 + rng.next_f32() * 0.6;
                self.hues[i] = rng.next_f32();
            }

            // Lifecycle opacity: fade in first 20%, fade out last 30%
            let life_frac = self.ages[i] / self.lifespans[i];
            let fade_in = (life_frac / 0.2).min(1.0);
            let fade_out = ((1.0 - life_frac) / 0.3).min(1.0);
            let life_alpha = fade_in * fade_out;

            // Noise-driven motion
            let px = self.positions[idx];
            let py = self.positions[idx + 1];
            let pz = self.positions[idx + 2];

            let noise_scale = 0.04;
            let time_scale = 0.25;

            let nx = simplex3d(
                px * noise_scale + 300.0,
                py * noise_scale,
                t * time_scale,
            ) * 1.5;
            let ny = simplex3d(
                py * noise_scale + 400.0,
                pz * noise_scale,
                t * time_scale,
            ) * 1.5;
            let nz = simplex3d(
                pz * noise_scale + 500.0,
                px * noise_scale,
                t * time_scale,
            ) * 1.5;

            // Apply velocity + noise, scaled by drift speed
            self.positions[idx] += (self.velocities[idx] + nx) * dt * drift;
            self.positions[idx + 1] += (self.velocities[idx + 1] + ny) * dt * drift;
            self.positions[idx + 2] += (self.velocities[idx + 2] + nz) * dt * drift;

            // Depth-aware opacity (distance from camera)
            let dx = self.positions[idx] - camera_x;
            let dy = self.positions[idx + 1] - camera_y;
            let dz = self.positions[idx + 2] - camera_z;
            let cam_dist = (dx * dx + dy * dy + dz * dz).sqrt();

            let near_fade = ((cam_dist - 2.0) / 4.0).clamp(0.0, 1.0);
            let far_fade = ((55.0 - cam_dist) / 15.0).clamp(0.0, 1.0);

            // Gentle flicker from noise
            let flicker = simplex3d(
                (i as f32) * 0.3 + 700.0,
                t * 0.8,
                0.0,
            ) * 0.15 + 0.85;

            self.opacities[i] = life_alpha * near_fade * far_fade * flicker;

            // Size pulses over lifetime
            let base_size = self.sizes[i];
            let pulse = 1.0 + simplex3d(
                (i as f32) * 0.15 + 800.0,
                t * 0.5,
                0.0,
            ) * 0.2;
            self.sizes[i] = base_size * pulse;

            // Slowly shift hue over lifetime
            self.hues[i] = (self.hues[i] + dt * 0.02) % 1.0;
        }

        // Persist RNG state for next frame
        self.rng_state = rng.state;
    }

    // ---- Zero-copy buffer accessors ----

    pub fn get_positions_ptr(&self) -> *const f32 {
        self.positions.as_ptr()
    }
    pub fn get_positions_len(&self) -> usize {
        self.count * 3
    }

    pub fn get_opacities_ptr(&self) -> *const f32 {
        self.opacities.as_ptr()
    }
    pub fn get_opacities_len(&self) -> usize {
        self.count
    }

    pub fn get_sizes_ptr(&self) -> *const f32 {
        self.sizes.as_ptr()
    }
    pub fn get_sizes_len(&self) -> usize {
        self.count
    }

    pub fn get_hues_ptr(&self) -> *const f32 {
        self.hues.as_ptr()
    }
    pub fn get_hues_len(&self) -> usize {
        self.count
    }

    pub fn wisp_count(&self) -> usize {
        self.count
    }
}
