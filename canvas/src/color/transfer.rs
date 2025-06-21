/// To emulate the syntax used in GLSL more closely.
#[inline]
fn pow(base: f32, exp: f32) -> f32 {
    libm::powf(base, exp)
}

// Validated in `colour_test_vectors`.
pub fn transfer_oe_bt709(val: f32) -> f32 {
    if val >= 0.018 {
        1.099 * pow(val, 0.45) - 0.099
    } else {
        4.500 * val
    }
}

// Used Reference: BT.709-6, Section 1.2, inverted.
//
// Validated in `colour_test_vectors`.
pub fn transfer_eo_bt709(val: f32) -> f32 {
    // TODO: is there a numerically better way?
    if val >= transfer_oe_bt709(0.018) {
        pow((val + 0.099) / 1.099, 1.0 / 0.45)
    } else {
        val / 4.500
    }
}

// Used Reference: BT.470-5, Table 1, Item 5
pub fn transfer_oe_bt470m(val: f32) -> f32 {
    pow(val, 1.0 / 2.200)
}

// Used Reference: BT.470-5, Table 1, Item 5
pub fn transfer_eo_bt470m(val: f32) -> f32 {
    pow(val, 2.200)
}

// Used Reference: BT.470-5, Table 1, Item 5
//
// Validated in `colour_test_vectors`.
pub fn transfer_oe_bt470(val: f32) -> f32 {
    pow(val, 1.0 / 2.800)
}

// Used Reference: BT.470-5, Table 1, Item 5
//
// Validated in `colour_test_vectors`.
pub fn transfer_eo_bt470(val: f32) -> f32 {
    pow(val, 2.800)
}

// Used Reference: BT.601-7, Section 2.6.4
pub fn transfer_oe_bt601(val: f32) -> f32 {
    transfer_oe_bt709(val)
}

// Used Reference: BT.601-7, Section 2.6.4
pub fn transfer_eo_bt601(val: f32) -> f32 {
    transfer_eo_bt709(val)
}

// Used Reference:
// <https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-smpte-240m-v4l2-colorspace-smpte240m>
//
// Validated in `colour_test_vectors`.
pub fn transfer_oe_smpte240(val: f32) -> f32 {
    if val < 0.0228 {
        4.0 * val
    } else {
        1.1115 * pow(val, 0.45) - 0.1115
    }
}

// Used Reference:
// <https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#colorspace-smpte-240m-v4l2-colorspace-smpte240m>
//
// Validated in `colour_test_vectors`.
pub fn transfer_eo_smpte240(val: f32) -> f32 {
    if val < 0.0913 {
        val / 4.0
    } else {
        pow((val + 0.1115) / 1.1115, 1.0 / 0.45)
    }
}

// Used Reference: https://www.kernel.org/doc/html/v4.11/media/uapi/v4l/pixfmt-007.html#
// Transfer function. Note that negative values for L are only used by the Y’CbCr conversion.
//
// Validated in `colour_test_vectors`.
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
//
// Validated in `colour_test_vectors`.
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

#[expect(dead_code)]
pub fn transfer_oe_inv_smpte2084(val: f32) -> f32 {
    transfer_display_scene_smpte2084(transfer_eo_smpte2084(val))
}

// TODO: https://github.com/colour-science/colour/blob/a196f9536c44e2101cde53446550d64303c0ab46/colour/models/rgb/transfer_functions/arib_std_b67.py#L108
// vec3 transfer_scene_display_bt2100hlg(vec3 rgb) {
// return vec3(0.0);
//}

#[test]
fn colour_test_vectors() {
    struct TestVector {
        name: &'static str,
        eotf: fn(f32) -> f32,
        oetf: fn(f32) -> f32,
        data: &'static [(f32, f32)],
    }

    // If you're contributing and you have an LLM available, give yourself the laugh of having it
    // try to autocomplete the floating point pairs in the data tables below. Well, in a few years
    // maybe you can actually do this reliably. But in the meantime keep in mind that the
    // maintainer will have to fully review any additions to these tables by executing the
    // statements to verify numbers. So please keep accurate comments as to how each of these are
    // derived.
    const VECTORS: &[TestVector] = &[
        TestVector {
            // # colour-science    0.4.6
            name: "sRGB",
            eotf: transfer_eo_srgb,
            oetf: transfer_oe_srgb,
            data: &[
                // b = colour.EOTFS['sRGB'](a)
                (0.0, 0.0),
                (1.0, 1.0),
                (0.5, 0.21404114048223255),
                (0.25, 0.050876088171556789),
                (0.75, 0.52252155396839206),
                (0.0031308, 0.00024232198142414861),
            ],
        },
        TestVector {
            // # colour-science    0.4.6
            name: "BT.709",
            eotf: transfer_eo_bt709,
            oetf: transfer_oe_bt709,
            data: &[
                // b = colour.RGB_COLOURSPACES['ITU-R BT.709'].cctf_decoding(a)
                (0.0, 0.0),
                (1.0, 1.0),
                (0.5, 0.25958940050628576),
                (0.25, 0.07815387594543223),
                (0.75, 0.56352229924287789),
                (0.01, 0.0022222222222222222),
            ],
        },
        TestVector {
            // # colour-science    0.4.6
            name: "BT.601",
            // Same parameters as then later defined in BT709
            eotf: transfer_eo_bt601,
            oetf: transfer_oe_bt601,
            data: &[
                // b = colour.RGB_COLOURSPACES['ITU-R BT.709'].cctf_decoding(a)
                (0.0, 0.0),
                (1.0, 1.0),
                (0.5, 0.25958940050628576),
                (0.25, 0.07815387594543223),
                (0.75, 0.56352229924287789),
                (0.01, 0.0022222222222222222),
            ],
        },
        TestVector {
            // # colour-science    0.4.6
            name: "SMPTE 240M",
            eotf: transfer_eo_smpte240,
            oetf: transfer_oe_smpte240,
            data: &[
                // b = colour.EOTFS['SMPTE 240M'](a)
                (0.0, 0.0),
                (1.0, 1.0),
                (0.5, 0.26503573357867721),
                (0.25, 0.082413320052187017),
                (0.75, 0.56767766904658656),
            ],
        },
        TestVector {
            // # colour-science    0.4.6
            name: "BT470",
            eotf: transfer_eo_bt470,
            oetf: transfer_oe_bt470,
            data: &[
                // b = colour.RGB_COLOURSPACES['ITU-R BT.470 - 525'].cctf_decoding(a)
                (0.0, 0.0),
                (1.0, 1.0),
                (0.5, 0.14358729437462939),
                (0.25, 0.020617311105826479),
                (0.75, 0.44686005794246769),
                // a = colour.RGB_COLOURSPACES['ITU-R BT.470 - 525'].cctf_encoding(b)
                (0.90235831092596908, 0.75),
                (0.78070918215571006, 0.5),
                (0.60950682710223769, 0.25),
            ],
        },
    ];

    for vector in VECTORS {
        for (a, b) in vector.data {
            let eotf_result = (vector.eotf)(*a);
            let oetf_result = (vector.oetf)(*b);
            assert!(
                (eotf_result - *b).abs() < 1e-6,
                "{} failed for eotf {}: expected {}, got {}",
                vector.name,
                a,
                b,
                eotf_result
            );
            assert!(
                (oetf_result - *a).abs() < 1e-6,
                "{} failed for oetf {}: expected {}, got {}",
                vector.name,
                b,
                a,
                oetf_result
            );
        }
    }
}
