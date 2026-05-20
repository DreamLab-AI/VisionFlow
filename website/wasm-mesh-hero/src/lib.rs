use wasm_bindgen::prelude::*;
use web_sys::CanvasRenderingContext2d;

const TAU: f64 = std::f64::consts::TAU;
const BRIGHT_GOLD: (u8, u8, u8) = (255, 215, 0);
const GOLD: (u8, u8, u8) = (212, 165, 116);
const BRONZE: (u8, u8, u8) = (205, 127, 50);
const CRIMSON: (u8, u8, u8) = (233, 69, 96);

const PRIMARY_NAMES: [&str; 5] = [
    "VisionClaw",
    "Agentbox",
    "solid-pod-rs",
    "nostr-forum",
    "DreamLab Edge",
];

const SECONDARY_COUNT: usize = 25;
const EDGE_REST_LEN: f64 = 120.0;
const SPRING_K: f64 = 0.0003;
const DAMPING: f64 = 0.97;

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
                0 => gx + gy,
                1 => -gx + gy,
                2 => gx - gy,
                3 => -gx - gy,
                4 => gx,
                5 => -gx,
                6 => gy,
                7 => -gy,
                8 => gx + gy,
                9 => -gx + gy,
                10 => gx - gy,
                _ => -gx - gy,
            }
        };

        let contrib = |g: u8, cx: f64, cy: f64| -> f64 {
            let t = 0.5 - cx * cx - cy * cy;
            if t > 0.0 { t * t * t * t * grad(g, cx, cy) } else { 0.0 }
        };

        70.0 * (contrib(gi0, x0, y0) + contrib(gi1, x1, y1) + contrib(gi2, x2, y2))
    }
}

#[derive(Clone)]
struct Node {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    base_x: f64,
    base_y: f64,
    radius: f64,
    primary: bool,
    parent: usize,
}

#[wasm_bindgen]
pub struct MeshHero {
    nodes: Vec<Node>,
    edges: Vec<(usize, usize)>,
    noise: SimplexNoise,
    width: f64,
    height: f64,
    time: f64,
    mouse_x: f64,
    mouse_y: f64,
    pulse_phase: f64,
}

#[wasm_bindgen]
impl MeshHero {
    #[wasm_bindgen(constructor)]
    pub fn new(width: f64, height: f64) -> Self {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let cx = width / 2.0;
        let cy = height / 2.0;
        let orbit_r = width.min(height) * 0.28;

        let mut nodes = Vec::with_capacity(5 + SECONDARY_COUNT);
        for i in 0..5 {
            let angle = (i as f64 / 5.0) * TAU - std::f64::consts::FRAC_PI_2;
            let x = cx + orbit_r * angle.cos();
            let y = cy + orbit_r * angle.sin();
            nodes.push(Node {
                x, y, vx: 0.0, vy: 0.0,
                base_x: x, base_y: y,
                radius: 6.0,
                primary: true,
                parent: i,
            });
        }

        let noise = SimplexNoise::new(42);
        let secondary_orbit = orbit_r * 0.4;
        for i in 0..SECONDARY_COUNT {
            let parent = i % 5;
            let angle_offset = (i as f64 / 5.0) * TAU * 0.618;
            let dist = secondary_orbit * (0.5 + noise.noise2d(i as f64 * 0.7, 0.0).abs() * 0.5);
            let px = nodes[parent].base_x;
            let py = nodes[parent].base_y;
            let x = px + dist * angle_offset.cos();
            let y = py + dist * angle_offset.sin();
            nodes.push(Node {
                x, y, vx: 0.0, vy: 0.0,
                base_x: x, base_y: y,
                radius: 2.0 + noise.noise2d(i as f64 * 1.3, 5.0).abs() * 2.0,
                primary: false,
                parent,
            });
        }

        let mut edges = Vec::new();
        for i in 0..5 {
            edges.push((i, (i + 1) % 5));
            edges.push((i, (i + 2) % 5));
        }
        for i in 5..nodes.len() {
            edges.push((nodes[i].parent, i));
            let neighbour = 5 + ((i - 5 + 5) % SECONDARY_COUNT);
            if neighbour != i {
                edges.push((i, neighbour));
            }
        }

        Self {
            nodes,
            edges,
            noise,
            width,
            height,
            time: 0.0,
            mouse_x: -1000.0,
            mouse_y: -1000.0,
            pulse_phase: 0.0,
        }
    }

    pub fn set_mouse(&mut self, x: f64, y: f64) {
        self.mouse_x = x;
        self.mouse_y = y;
    }

    pub fn tick(&mut self, dt: f64) {
        self.time += dt * 0.00003;
        self.pulse_phase += dt * 0.002;

        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            let nx = self.noise.noise2d(node.base_x * 0.002, self.time) * 12.0;
            let ny = self.noise.noise2d(node.base_y * 0.002, self.time + 100.0) * 12.0;
            let target_x = node.base_x + nx;
            let target_y = node.base_y + ny;

            let dx = target_x - node.x;
            let dy = target_y - node.y;

            self.nodes[i].vx += dx * 0.001;
            self.nodes[i].vy += dy * 0.001;
        }

        for e_idx in 0..self.edges.len() {
            let (a, b) = self.edges[e_idx];
            let dx = self.nodes[b].x - self.nodes[a].x;
            let dy = self.nodes[b].y - self.nodes[a].y;
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            let force = (dist - EDGE_REST_LEN) * SPRING_K;
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;
            self.nodes[a].vx += fx;
            self.nodes[a].vy += fy;
            self.nodes[b].vx -= fx;
            self.nodes[b].vy -= fy;
        }

        for node in &mut self.nodes {
            let mdx = node.x - self.mouse_x;
            let mdy = node.y - self.mouse_y;
            let md = (mdx * mdx + mdy * mdy).sqrt();
            if md < 150.0 && md > 1.0 {
                let repel = (150.0 - md) * 0.00005;
                node.vx += (mdx / md) * repel;
                node.vy += (mdy / md) * repel;
            }

            node.vx *= DAMPING;
            node.vy *= DAMPING;
            node.x += node.vx;
            node.y += node.vy;
        }
    }

    pub fn render(&self, ctx: &CanvasRenderingContext2d) {
        ctx.clear_rect(0.0, 0.0, self.width, self.height);

        let pulse = (self.pulse_phase.sin() * 0.5 + 0.5) * 0.15;

        for &(a, b) in &self.edges {
            let na = &self.nodes[a];
            let nb = &self.nodes[b];
            let mx = (na.x + nb.x) / 2.0;
            let my = (na.y + nb.y) / 2.0;
            let cx = self.width / 2.0;
            let cy = self.height / 2.0;
            let dist_center = ((mx - cx).powi(2) + (my - cy).powi(2)).sqrt();
            let max_r = self.width.min(self.height) * 0.45;
            let ratio = (dist_center / max_r).min(1.0);

            let both_primary = na.primary && nb.primary;
            let alpha = if both_primary {
                0.3 + pulse
            } else {
                0.08 + (1.0 - ratio) * 0.15 + pulse * 0.5
            };

            let c = lerp_color(BRIGHT_GOLD, BRONZE, ratio);
            let color = if both_primary { CRIMSON } else { c };

            ctx.begin_path();
            ctx.move_to(na.x, na.y);
            ctx.line_to(nb.x, nb.y);
            ctx.set_stroke_style_str(&rgba(color, alpha));
            ctx.set_line_width(if both_primary { 1.5 } else { 0.6 });
            ctx.stroke();
        }

        for (i, node) in self.nodes.iter().enumerate() {
            let md = ((node.x - self.mouse_x).powi(2) + (node.y - self.mouse_y).powi(2)).sqrt();
            let glow = if md < 120.0 { (120.0 - md) / 120.0 * 0.4 } else { 0.0 };

            if node.primary {
                let alpha = 0.6 + glow + pulse;
                ctx.begin_path();
                ctx.arc(node.x, node.y, node.radius + 3.0, 0.0, TAU).unwrap();
                ctx.set_fill_style_str(&rgba(CRIMSON, 0.08 + glow * 0.3));
                ctx.fill();

                ctx.begin_path();
                ctx.arc(node.x, node.y, node.radius, 0.0, TAU).unwrap();
                ctx.set_fill_style_str(&rgba(BRIGHT_GOLD, alpha));
                ctx.fill();

                ctx.set_font("11px Inter, system-ui, sans-serif");
                ctx.set_fill_style_str(&rgba(GOLD, 0.7 + glow));
                ctx.set_text_align("center");
                ctx.fill_text(PRIMARY_NAMES[i], node.x, node.y + node.radius + 16.0).unwrap();
            } else {
                let alpha = 0.25 + glow + pulse * 0.5;
                ctx.begin_path();
                ctx.arc(node.x, node.y, node.radius, 0.0, TAU).unwrap();
                ctx.set_fill_style_str(&rgba(BRONZE, alpha));
                ctx.fill();
            }
        }
    }

    pub fn resize(&mut self, width: f64, height: f64) {
        let old_cx = self.width / 2.0;
        let old_cy = self.height / 2.0;
        let new_cx = width / 2.0;
        let new_cy = height / 2.0;
        let sx = width / self.width;
        let sy = height / self.height;

        for node in &mut self.nodes {
            node.base_x = new_cx + (node.base_x - old_cx) * sx;
            node.base_y = new_cy + (node.base_y - old_cy) * sy;
            node.x = new_cx + (node.x - old_cx) * sx;
            node.y = new_cy + (node.y - old_cy) * sy;
        }

        self.width = width;
        self.height = height;
    }
}

fn rgba(c: (u8, u8, u8), a: f64) -> String {
    format!("rgba({},{},{},{:.3})", c.0, c.1, c.2, a.clamp(0.0, 1.0))
}

fn lerp_color(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    let l = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t).round() as u8;
    (l(a.0, b.0), l(a.1, b.1), l(a.2, b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_correct_node_count() {
        let hero = MeshHero::new(1200.0, 800.0);
        assert_eq!(hero.nodes.len(), 5 + SECONDARY_COUNT);
        assert_eq!(hero.nodes.iter().filter(|n| n.primary).count(), 5);
    }

    #[test]
    fn noise_range() {
        let noise = SimplexNoise::new(42);
        for i in 0..100 {
            let v = noise.noise2d(i as f64 * 0.1, i as f64 * 0.2);
            assert!((-1.0..=1.0).contains(&v), "noise out of range: {}", v);
        }
    }

    #[test]
    fn tick_does_not_panic() {
        let mut hero = MeshHero::new(800.0, 600.0);
        for _ in 0..100 {
            hero.tick(16.0);
        }
    }
}
