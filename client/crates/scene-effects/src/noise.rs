/// Fast simplex noise implementation optimized for WASM (f32 throughout).
///
/// Provides 2D and 3D simplex noise plus fractal Brownian motion (fBm)
/// for organic, fluid-like motion in the scene background.
///
/// The 2D variants are available for external consumers even though the
/// internal crate currently only uses the 3D versions.

// Skew/unskew constants for 2D simplex
const F2: f32 = 0.366_025_4; // (sqrt(3) - 1) / 2
const G2: f32 = 0.211_324_87; // (3 - sqrt(3)) / 6

// Skew/unskew constants for 3D simplex
const F3: f32 = 1.0 / 3.0;
const G3: f32 = 1.0 / 6.0;

/// Gradient vectors for 2D noise
static GRAD2: [[f32; 2]; 8] = [
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
];

/// Gradient vectors for 3D noise
static GRAD3: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// Permutation table (duplicated for wrapping without modulo)
static PERM: [u8; 512] = {
    const BASE: [u8; 256] = [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30,
        69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62,
        94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125,
        136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111,
        229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65,
        25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130,
        116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
        5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182,
        189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101,
        155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185,
        112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
        81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84,
        204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24,
        72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    ];
    let mut table = [0u8; 512];
    let mut i = 0;
    while i < 512 {
        table[i] = BASE[i & 255];
        i += 1;
    }
    table
};

/// Fast floor for f32 (avoids std::f32::floor for WASM code size)
#[inline(always)]
fn fast_floor(x: f32) -> i32 {
    let xi = x as i32;
    if x < xi as f32 {
        xi - 1
    } else {
        xi
    }
}

/// Dot product of a gradient vector and (x, y)
#[inline(always)]
fn dot2(g: &[f32; 2], x: f32, y: f32) -> f32 {
    g[0] * x + g[1] * y
}

/// Dot product of a gradient vector and (x, y, z)
#[inline(always)]
fn dot3(g: &[f32; 3], x: f32, y: f32, z: f32) -> f32 {
    g[0] * x + g[1] * y + g[2] * z
}

/// 2D simplex noise. Returns a value in approximately [-1, 1].
pub fn simplex2d(x: f32, y: f32) -> f32 {
    // Skew input space to determine which simplex cell we are in
    let s = (x + y) * F2;
    let i = fast_floor(x + s);
    let j = fast_floor(y + s);

    let t = (i + j) as f32 * G2;
    let x0 = x - (i as f32 - t);
    let y0 = y - (j as f32 - t);

    // Determine which simplex triangle we are in
    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

    let x1 = x0 - i1 as f32 + G2;
    let y1 = y0 - j1 as f32 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;

    // Calculate contributions from three corners
    let mut n = 0.0f32;

    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if t0 > 0.0 {
        let t0 = t0 * t0;
        let gi0 = PERM[ii + PERM[jj] as usize] as usize % 8;
        n += t0 * t0 * dot2(&GRAD2[gi0], x0, y0);
    }

    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if t1 > 0.0 {
        let t1 = t1 * t1;
        let gi1 = PERM[ii + i1 + PERM[jj + j1] as usize] as usize % 8;
        n += t1 * t1 * dot2(&GRAD2[gi1], x1, y1);
    }

    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if t2 > 0.0 {
        let t2 = t2 * t2;
        let gi2 = PERM[ii + 1 + PERM[jj + 1] as usize] as usize % 12;
        n += t2 * t2 * dot2(&GRAD2[gi2 % 8], x2, y2);
    }

    // Scale to [-1, 1]
    70.0 * n
}

/// 3D simplex noise. Returns a value in approximately [-1, 1].
pub fn simplex3d(x: f32, y: f32, z: f32) -> f32 {
    let s = (x + y + z) * F3;
    let i = fast_floor(x + s);
    let j = fast_floor(y + s);
    let k = fast_floor(z + s);

    let t = (i + j + k) as f32 * G3;
    let x0 = x - (i as f32 - t);
    let y0 = y - (j as f32 - t);
    let z0 = z - (k as f32 - t);

    // Determine which simplex we are in
    let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
        if y0 >= z0 {
            (1, 0, 0, 1, 1, 0)
        } else if x0 >= z0 {
            (1, 0, 0, 1, 0, 1)
        } else {
            (0, 0, 1, 1, 0, 1)
        }
    } else if y0 < z0 {
        (0, 0, 1, 0, 1, 1)
    } else if x0 < z0 {
        (0, 1, 0, 0, 1, 1)
    } else {
        (0, 1, 0, 1, 1, 0)
    };

    let x1 = x0 - i1 as f32 + G3;
    let y1 = y0 - j1 as f32 + G3;
    let z1 = z0 - k1 as f32 + G3;
    let x2 = x0 - i2 as f32 + 2.0 * G3;
    let y2 = y0 - j2 as f32 + 2.0 * G3;
    let z2 = z0 - k2 as f32 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    let ii = (i & 255) as usize;
    let jj = (j & 255) as usize;
    let kk = (k & 255) as usize;

    let mut n = 0.0f32;

    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if t0 > 0.0 {
        let t0 = t0 * t0;
        let gi0 = PERM[ii + PERM[jj + PERM[kk] as usize] as usize] as usize % 12;
        n += t0 * t0 * dot3(&GRAD3[gi0], x0, y0, z0);
    }

    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if t1 > 0.0 {
        let t1 = t1 * t1;
        let gi1 =
            PERM[ii + i1 + PERM[jj + j1 + PERM[kk + k1] as usize] as usize] as usize % 12;
        n += t1 * t1 * dot3(&GRAD3[gi1], x1, y1, z1);
    }

    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if t2 > 0.0 {
        let t2 = t2 * t2;
        let gi2 =
            PERM[ii + i2 + PERM[jj + j2 + PERM[kk + k2] as usize] as usize] as usize % 12;
        n += t2 * t2 * dot3(&GRAD3[gi2], x2, y2, z2);
    }

    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if t3 > 0.0 {
        let t3 = t3 * t3;
        let gi3 = PERM[ii + 1 + PERM[jj + 1 + PERM[kk + 1] as usize] as usize] as usize % 12;
        n += t3 * t3 * dot3(&GRAD3[gi3], x3, y3, z3);
    }

    32.0 * n
}

/// Fractal Brownian motion using 2D simplex noise.
///
/// Layers multiple octaves of noise at increasing frequency and decreasing
/// amplitude to produce rich, organic textures.
///
/// * `octaves` - Number of noise layers (2-8 recommended)
/// * `lacunarity` - Frequency multiplier per octave (typically 2.0)
/// * `persistence` - Amplitude decay per octave (typically 0.5)
pub fn fbm2d(x: f32, y: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut max_value = 0.0f32;

    for _ in 0..octaves {
        value += amplitude * simplex2d(x * frequency, y * frequency);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    // Normalize to [-1, 1]
    value / max_value
}

/// Fractal Brownian motion using 3D simplex noise.
///
/// Same as `fbm2d` but with a third dimension, useful for time-varying
/// 2D fields (pass time as z).
pub fn fbm3d(
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut max_value = 0.0f32;

    for _ in 0..octaves {
        value += amplitude * simplex3d(x * frequency, y * frequency, z * frequency);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex2d_is_bounded() {
        for i in 0..1000 {
            let x = (i as f32) * 0.1;
            let y = (i as f32) * 0.07;
            let v = simplex2d(x, y);
            assert!(v >= -1.5 && v <= 1.5, "simplex2d out of range: {v}");
        }
    }

    #[test]
    fn simplex3d_is_bounded() {
        for i in 0..1000 {
            let x = (i as f32) * 0.1;
            let y = (i as f32) * 0.07;
            let z = (i as f32) * 0.03;
            let v = simplex3d(x, y, z);
            assert!(v >= -1.5 && v <= 1.5, "simplex3d out of range: {v}");
        }
    }

    #[test]
    fn fbm2d_returns_reasonable_values() {
        let v = fbm2d(1.0, 2.0, 4, 2.0, 0.5);
        assert!(v >= -1.0 && v <= 1.0, "fbm2d out of range: {v}");
    }
}
