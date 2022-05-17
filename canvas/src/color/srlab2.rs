use crate::color::Whitepoint;
use crate::color_matrix::ColMatrix;

#[rustfmt::skip]
const M_CAT02: ColMatrix = ColMatrix([
    [ 0.7328,-0.7036, 0.0030],
    [ 0.4296, 1.6975, 0.0136],
    [-0.1624, 0.0061, 0.9834],
]);

/// Hunt-Pointer-Estevez fundamentals, or LMS cones.
#[rustfmt::skip]
const M_HPE: ColMatrix = ColMatrix([
    [ 0.38971,-0.22981, 0.00000],
    [ 0.68898, 1.18340, 0.00000],
    [-0.07868, 0.04641, 1.00000],
]);

pub fn srlab_from_xyz(xyz: [f32; 3], whitepoint: Whitepoint) -> [f32; 3] {
    // First step: correct whitepoint adaptation.
    // TODO: Note this is a linear operation. For specific whitepoint we could do a pre-calculation
    // and then do this much more efficiently. The website, for example, has values for D65 (???)
    // and then it combines with the `M_HPE` multiplication below, too.
    let [rw, gw, bw] = M_CAT02.mul_vec(whitepoint.to_xyz());
    let [r, g, b] = M_CAT02.mul_vec(xyz);
    let xyz_w = M_CAT02.inv().mul_vec([r / rw, g / gw, b / bw]);

    // Second step: non-linearity
    let nl = non_linearity(M_HPE.mul_vec(xyz_w));
    let [xe, ye, ze] = M_HPE.inv().mul_vec(nl);

    // Third step: lightness, chroma encoding. Confusingly, the paper as well as C-code multiplies
    // everything by 100 here for unexplained reasons. We don't because this wouldn't fit in our
    // Unorm range for storing as bits. Do at your own risk.
    [ye, (xe - ye) * 5.0 / 1.16, (ze - ye) * 2.0 / 1.16]
}

pub fn srlab_to_xyz([li, a, b]: [f32; 3], whitepoint: Whitepoint) -> [f32; 3] {
    // First step, undo lightness and chroma encoding.
    let xyz = [a * 1.16 / 5.0 + li, li, b * 1.16 / 2.0 + li];

    // Second step: non-linearity
    let nl = M_HPE.mul_vec(xyz);
    let xyz_w = M_HPE.inv().mul_vec(non_linearity_inv(nl));

    // Third step: Undo whitepoint adaptation.
    let [rw, gw, bw] = M_CAT02.mul_vec(whitepoint.to_xyz());
    let [r, g, b] = M_CAT02.mul_vec(xyz_w);
    M_CAT02.inv().mul_vec([r * rw, g * gw, b * bw])
}

fn non_linearity(lms: [f32; 3]) -> [f32; 3] {
    fn adjust(v: f32) -> f32 {
        // 6**3 / 29**3
        if v.abs() < 216.0 / 24389.0 {
            // Limited to 0.08 precisely
            v * 24389.0 / 2700.0
        } else {
            1.16 * v.powf(1.0 / 3.0) - 0.16
        }
    }

    lms.map(adjust)
}

fn non_linearity_inv(lms: [f32; 3]) -> [f32; 3] {
    fn adjust(v: f32) -> f32 {
        if v.abs() < 0.08 {
            v * 2700.0 / 24389.0
        } else {
            ((v + 0.16) / 1.16).powf(3.0)
        }
    }

    lms.map(adjust)
}
