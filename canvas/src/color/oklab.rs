use crate::color_matrix::ColMatrix;
use crate::math::powf;

const M1: ColMatrix = ColMatrix([
    [0.8189330101, 0.0329845436, 0.0482003018],
    [0.3618667424, 0.9293118715, 0.2643662691],
    [-0.1288597137, 0.0361456387, 0.6338517070],
]);

const M2: ColMatrix = ColMatrix([
    [0.2104542553, 1.9779984951, 0.0259040371],
    [0.7936177850, -2.4285922050, 0.7827717662],
    [-0.0040720468, 0.4505937099, -0.8086757660],
]);

pub fn oklab_from_xyz(xyza: [f32; 4]) -> [f32; 4] {
    let [x, y, z, a] = xyza;

    // The OKLab transformation.
    let lms = M1.mul_vec([x, y, z]);
    // We can't use pow outright for negative components.
    let lms_star = f_lms(lms);
    let [l, ca, cb] = M2.mul_vec(lms_star);

    // Write this as our 'linear color' (preserve alpha).
    [l, ca, cb, a]
}

pub fn from_xyz_slice(xyz: &[[f32; 4]], pixel: &mut [[f32; 4]]) {
    // Loop-split version of the single-instance case.
    for (pixel, &[x, y, z, a]) in pixel.iter_mut().zip(xyz) {
        let [l, m, s] = M1.mul_vec([x, y, z]);
        *pixel = [l, m, s, a];
    }

    for lms in pixel.iter_mut() {
        let [l, m, s, a] = *lms;
        let [ls, ms, ss] = f_lms([l, m, s]);
        *lms = [ls, ms, ss, a];
    }

    for lms in pixel.iter_mut() {
        let [ls, ms, ss, a] = *lms;
        let [l, ca, cb] = M2.mul_vec([ls, ms, ss]);
        *lms = [l, ca, cb, a];
    }
}

#[allow(non_snake_case)]
pub fn oklab_to_xyz([l, ca, cb, a]: [f32; 4]) -> [f32; 4] {
    let M2_INV: ColMatrix = M2.inv().to_col();
    let M1_INV: ColMatrix = M1.inv().to_col();

    // The OKLab transformation.
    let lms_star = M2_INV.mul_vec([l, ca, cb]);
    // Not using pow because that would be undefined for negative components.
    let lms = f_lms_inv(lms_star);
    let [x, y, z] = M1_INV.mul_vec(lms);

    [x, y, z, a]
}

#[allow(non_snake_case)]
pub fn to_xyz_slice(pixel: &[[f32; 4]], xyz: &mut [[f32; 4]]) {
    let M2_INV: ColMatrix = M2.inv().to_col();
    let M1_INV: ColMatrix = M1.inv().to_col();

    // Loop-split version of the single-instance case.
    for (xyz, &[l, ca, cb, a]) in xyz.iter_mut().zip(pixel) {
        // The OKLab transformation.
        let [l, m, s] = M2_INV.mul_vec([l, ca, cb]);
        *xyz = [l, m, s, a];
    }

    for xyz in xyz.iter_mut() {
        let [ls, ms, ss, a] = *xyz;
        // Not using pow because that would be undefined for negative components.
        let [l, m, s] = f_lms_inv([ls, ms, ss]);
        *xyz = [l, m, s, a];
    }

    for xyz in xyz.iter_mut() {
        let [l, m, s, a] = *xyz;
        let [x, y, z] = M1_INV.mul_vec([l, m, s]);
        *xyz = [x, y, z, a]
    }
}

pub(crate) fn f_lms(lms: [f32; 3]) -> [f32; 3] {
    copysign(pow(abs(lms), 1.0 / 3.0), lms)
}

pub(crate) fn f_lms_inv(lms: [f32; 3]) -> [f32; 3] {
    pow3(lms)
}

fn pow([a, b, c]: [f32; 3], exp: f32) -> [f32; 3] {
    [powf(a, exp), powf(b, exp), powf(c, exp)]
}

fn copysign([a, b, c]: [f32; 3], [sa, sb, sc]: [f32; 3]) -> [f32; 3] {
    [a.copysign(sa), b.copysign(sb), c.copysign(sc)]
}

fn abs([a, b, c]: [f32; 3]) -> [f32; 3] {
    [a.abs(), b.abs(), c.abs()]
}

fn pow3([a, b, c]: [f32; 3]) -> [f32; 3] {
    [a * a * a, b * b * b, c * c * c]
}

#[test]
fn inverse() {
    const XYZA: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let xyza = oklab_to_xyz(oklab_from_xyz(XYZA));
    // FIXME: assert component-wise up to some expected diff.
}
