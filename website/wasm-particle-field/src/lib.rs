use wasm_bindgen::prelude::*;
use web_sys::CanvasRenderingContext2d;

const PARTICLE_COUNT: usize = 200;
const BRONZE: (u8, u8, u8) = (205, 127, 50);
const GOLD: (u8, u8, u8) = (212, 165, 116);
const BRIGHT_GOLD: (u8, u8, u8) = (255, 215, 0);

struct SimplexNoise {
    perm: [u8; 512],
}

impl SimplexNoise {
    fn new(seed: u32) -> Self {
        let mut perm = [0u8; 512];
        let mut p: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut rng = seed;
        for i in (1..256).rev() {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng >> 16) as usize % (i + 1);
            p.swap(i, j);
        }
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Self { perm }
    }

    fn noise2d(&self, x: f64, y: f64) -> f64 {
        const F2: f64 = 0.3660254037844386;
        const G2: f64 = 0.21132486540518713;
        let s = (x + y) * F2;
        let i = (x + s).floor();
        let j = (y + s).floor();
        let t = (i + j) * G2;
        let x0 = x - (i - t);
        let y0 = y - (j - t);
        let (i1, j1) = if x0 > y0 { (1.0, 0.0) } else { (0.0, 1.0) };
        let x1 = x0 - i1 + G2;
        let y1 = y0 - j1 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;
        let ii = (i as i32 & 255) as usize;
        let jj = (j as i32 & 255) as usize;
        let gi0 = self.perm[ii + self.perm[jj] as usize] % 12;
        let gi1 = self.perm[ii + i1 as usize + self.perm[jj + j1 as usize] as usize] % 12;
        let gi2 = self.perm[ii + 1 + self.perm[jj + 1] as usize] % 12;
        let grad = |gi: u8, gx: f64, gy: f64| -> f64 {
            match gi {
                0 => gx + gy, 1 => -gx + gy, 2 => gx - gy, 3 => -gx - gy,
                4 => gx, 5 => -gx, 6 => gy, 7 => -gy,
                8 => gx + gy, 9 => -gx + gy, 10 => gx - gy, _ => -gx - gy,
            }
        };
        let contrib = |g: u8, cx: f64, cy: f64| -> f64 {
            let t = 0.5 - cx * cx - cy * cy;
            if t > 0.0 { t * t * t * t * grad(g, cx, cy) } else { 0.0 }
        };
        70.0 * (contrib(gi0, x0, y0) + contrib(gi1, x1, y1) + contrib(gi2, x2, y2))
    }
}

struct Particle {
    x: f64,
    y: f64,
    base_speed: f64,
    size: f64,
    seed_offset: f64,
}

#[wasm_bindgen]
pub struct ParticleField {
    particles: Vec<Particle>,
    noise: SimplexNoise,
    width: f64,
    height: f64,
    time: f64,
}

#[wasm_bindgen]
impl ParticleField {
    #[wasm_bindgen(constructor)]
    pub fn new(width: f64, height: f64) -> Self {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let noise = SimplexNoise::new(137);
        let mut particles = Vec::with_capacity(PARTICLE_COUNT);
        let mut rng: u32 = 31415;

        for _ in 0..PARTICLE_COUNT {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let x = (rng as f64 / u32::MAX as f64) * width;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let y = (rng as f64 / u32::MAX as f64) * height;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let speed = 0.08 + (rng as f64 / u32::MAX as f64) * 0.15;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let size = 0.5 + (rng as f64 / u32::MAX as f64) * 1.5;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let seed_offset = (rng as f64 / u32::MAX as f64) * 1000.0;

            particles.push(Particle { x, y, base_speed: speed, size, seed_offset });
        }

        Self { particles, noise, width, height, time: 0.0 }
    }

    pub fn tick(&mut self, dt: f64, scroll_offset: f64) {
        self.time += dt * 0.00004;
        let scroll_shift = scroll_offset * 0.05;

        for p in &mut self.particles {
            let nx = self.noise.noise2d(p.x * 0.001 + p.seed_offset, self.time) * 0.3;
            let ny = self.noise.noise2d(p.y * 0.001 + p.seed_offset, self.time + 50.0);

            p.x += nx;
            p.y -= p.base_speed * (dt * 0.06) + ny * 0.1 - scroll_shift * 0.01;

            if p.y < -10.0 {
                p.y = self.height + 10.0;
                p.x = (p.x + self.width * 0.3) % self.width;
            }
            if p.y > self.height + 10.0 {
                p.y = -10.0;
            }
            if p.x < -10.0 { p.x = self.width + 10.0; }
            if p.x > self.width + 10.0 { p.x = -10.0; }
        }
    }

    pub fn render(&self, ctx: &CanvasRenderingContext2d) {
        ctx.clear_rect(0.0, 0.0, self.width, self.height);

        for p in &self.particles {
            let y_ratio = p.y / self.height;
            let color = if y_ratio < 0.33 {
                BRIGHT_GOLD
            } else if y_ratio < 0.66 {
                GOLD
            } else {
                BRONZE
            };

            let alpha = 0.1 + (1.0 - y_ratio) * 0.25;
            let twinkle = (self.time * 800.0 + p.seed_offset).sin() * 0.08;

            ctx.begin_path();
            ctx.arc(p.x, p.y, p.size, 0.0, std::f64::consts::TAU).unwrap();
            ctx.set_fill_style_str(&format!(
                "rgba({},{},{},{:.3})",
                color.0, color.1, color.2,
                (alpha + twinkle).clamp(0.05, 0.4)
            ));
            ctx.fill();
        }
    }

    pub fn resize(&mut self, width: f64, height: f64) {
        let sx = width / self.width;
        let sy = height / self.height;
        for p in &mut self.particles {
            p.x *= sx;
            p.y *= sy;
        }
        self.width = width;
        self.height = height;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_particles() {
        let field = ParticleField::new(1920.0, 1080.0);
        assert_eq!(field.particles.len(), PARTICLE_COUNT);
    }

    #[test]
    fn tick_wraps_particles() {
        let mut field = ParticleField::new(100.0, 100.0);
        for _ in 0..10000 {
            field.tick(16.0, 0.0);
        }
        for p in &field.particles {
            assert!(p.x >= -10.0 && p.x <= 110.0);
        }
    }
}
