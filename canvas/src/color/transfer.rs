/// To emulate the syntax used in GLSL more closely.
#[inline]
fn pow(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}

pub fn transfer_oe_bt709(val: f32) -> f32 {
    // TODO: is there a numerically better way?
    if val >= 0.018 {
        1.099 * pow(val, 0.45) - 0.099
    } else {
        4.500 * val
    }
}

// Used Reference: BT.709-6, Section 1.2, inverted.
pub fn transfer_eo_bt709(val: f32) -> f32 {
    // TODO: is there a numerically better way?
    if val >= transfer_oe_bt709(0.018) {
        pow((val + 0.099) / 1.099, 1.0 / 0.45)
    } else {
        val / 4.500
    }
}

// Used Reference: BT.470, Table 1, Item 5
pub fn transfer_oe_bt470m(val: f32) -> f32 {
    pow(val, 1.0 / 2.200)
}

// Used Reference: BT.470, Table 1, Item 5
pub fn transfer_eo_bt470m(val: f32) -> f32 {
    pow(val, 2.200)
}

// Used Reference: BT.601-7, Section 2.6.4
pub fn transfer_oe_bt601(val: f32) -> f32 {
    transfer_eo_bt709(val)
}

// Used Reference: BT.601-7, Section 2.6.4
pub fn transfer_eo_bt601(val: f32) -> f32 {
    transfer_oe_bt709(val)
}

// Used Reference:
// <https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-smpte-240m-v4l2-colorspace-smpte240m>
pub fn transfer_oe_smpte240(val: f32) -> f32 {
    if val < 0.0228 {
        4.0 * val
    } else {
        1.1115 * pow(val, 0.45) - 0.1115
    }
}

// Used Reference:
// <https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-smpte-240m-v4l2-colorspace-smpte240m>
pub fn transfer_eo_smpte240(val: f32) -> f32 {
    if val < 0.0913 {
        val / 4.0
    } else {
        pow((val - 0.1115) / 1.1115, 1.0 / 0.45)
    }
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#
// Transfer function. Note that negative values for L are only used by the Y’CbCr conversion.
pub fn transfer_oe_srgb(val: f32) -> f32 {
    if val < -0.0031308 {
        -1.055 * pow(-val, 1.0 / 2.4) + 0.055
    } else if val <= 0.0031308 {
        val * 12.92
    } else {
        1.055 * pow(val, 1.0 / 2.4) - 0.055
    }
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html
pub fn transfer_eo_srgb(val: f32) -> f32 {
    if val < -0.04045 {
        -pow((-val + 0.055) / 1.055, 2.4)
    } else if val <= 0.04045 {
        return val / 12.92;
    } else {
        pow((val + 0.055) / 1.055, 2.4)
    }
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-bt-2020-v4l2-colorspace-bt2020
pub fn transfer_oe_bt2020_10b(val: f32) -> f32 {
    transfer_oe_bt709(val)
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-bt-2020-v4l2-colorspace-bt2020
pub fn transfer_eo_bt2020_10b(val: f32) -> f32 {
    transfer_eo_bt709(val)
}

// Used Reference: BT.2100-2, Table 4, Reference PQ EOTF
const SMPTE2084_M1: f32 = 2610.0 / 16384.0;
const SMPTE2084_M2: f32 = 2523.0 / 4096.0;
const SMPTE2084_C1: f32 = 3424.0 / 4096.0;
const SMPTE2084_C2: f32 = 2413.0 / 128.0;
const SMPTE2084_C3: f32 = 2392.0 / 128.0;

// Used Reference: BT.2100-2, Table 4, Reference PQ EOTF
// Note: the output is _display_ color value Y and _not_ scene luminance.
#[allow(non_snake_case)] // for conformity with reference.
pub fn transfer_eo_smpte2084(val: f32) -> f32 {
    let N = pow(val, 1.0 / SMPTE2084_M2);
    let nom = if N - SMPTE2084_C1 > 0.0 {
        N - SMPTE2084_C1
    } else {
        0.0
    };
    let denom = SMPTE2084_C2 - SMPTE2084_C3 * N;
    pow(nom / denom, 1.0 / SMPTE2084_M1)
}
// Used Reference: BT.2100-2, Table 4, Reference PQ OETF
// Note: the input is _display_ color value Y and _not_ scene luminance.
#[allow(non_snake_case)] // for conformity with reference.
pub fn transfer_eo_inv_smpte2084(val: f32) -> f32 {
    let Y = pow(val, SMPTE2084_M1);
    let nom = SMPTE2084_C1 + SMPTE2084_C2 * Y;
    let denom = SMPTE2084_C3 * Y + 1.0;
    pow(nom / denom, SMPTE2084_M2)
}

// Used Reference: BT.2100-2, Table 4, Reference PQ OOTF
// Used Reference: Python `colour science`: https://github.com/colour-science/colour/blob/a196f9536c44e2101cde53446550d64303c0ab46/colour/models/rgb/transfer_functions/itur_bt_2100.py#L276
// IMPORTANT: we map to a normalized linear color range Y, and _not_ to display luminance F_D.
pub fn transfer_scene_display_smpte2084(val: f32) -> f32 {
    let e_prime = transfer_oe_bt709(59.5208 * val);
    pow(e_prime, 2.4) / 100.0
}

// Used Reference: BT.2100-2, Table 4, Reference PQ OOTF
pub fn transfer_display_scene_smpte2084(val: f32) -> f32 {
    let e_prime = pow(val * 100.0, 1.0 / 2.4);
    transfer_eo_bt709(e_prime) / 59.5208
}

pub fn transfer_oe_smpte2084(val: f32) -> f32 {
    transfer_eo_inv_smpte2084(transfer_scene_display_smpte2084(val))
}
pub fn transfer_oe_inv_smpte2084(val: f32) -> f32 {
    transfer_display_scene_smpte2084(transfer_eo_smpte2084(val))
}

// TODO: https://github.com/colour-science/colour/blob/a196f9536c44e2101cde53446550d64303c0ab46/colour/models/rgb/transfer_functions/arib_std_b67.py#L108
// vec3 transfer_scene_display_bt2100hlg(vec3 rgb) {
// return vec3(0.0);
//}
