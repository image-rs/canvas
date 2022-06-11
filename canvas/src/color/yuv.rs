use crate::color_matrix::RowMatrix;

use super::{Differencing, Transfer};

/// A digital differencing scheme, i.e. where non-linear values are used.
///
/// Also referred to as using 'gamma pre-corrected' values. The main difference of these schemes is
/// the domain of the output signal. Mostly, Y (luma) is assumed [0; 1] while U, V (or Cb, Cr) are
/// in a range [-Um; Um], [-Vm, Vm].
///
/// For analog use these are 'quantized' to a positive range with some symbols reserved above and
/// below. For example, SMPTE-125M leaves head- and toe-room to compensate for errors due to analog
/// errors from frequency analysis. For digital use, the chroma samples are sometimes also shifted
/// to occupy the [0; 1) range (up to quantization accuracy) instead.
trait DigitalDifferencing {
    const DIFF: RowMatrix;
}

trait DifferencingAction<R = ()> {
    fn digital<D: DigitalDifferencing>(self) -> R;
    fn quantized<D: DigitalDifferencing>(self) -> R;
}

pub fn from_rgb_slice(rgb: &mut [[f32; 4]], transfer: Transfer, differencing: Differencing) {
    struct EncodeAction<'data>(&'data mut [[f32; 4]]);

    impl DifferencingAction for EncodeAction<'_> {
        fn digital<D: DigitalDifferencing>(self) {
            from_slice_digital::<D>(self.0)
        }

        fn quantized<D: DigitalDifferencing>(self) { /* todo */
        }
    }

    if let Some(oe_transfer) = transfer.from_optical_display_slice() {
        oe_transfer(rgb);
    } else {
        for rgb in rgb.iter_mut() {
            *rgb = transfer.from_optical_display(*rgb);
        }
    }

    differencing.action(EncodeAction(rgb));
}

pub fn to_rgb_slice(rgb: &mut [[f32; 4]], transfer: Transfer, differencing: Differencing) {
    struct EncodeAction<'data>(&'data mut [[f32; 4]]);

    impl DifferencingAction for EncodeAction<'_> {
        fn digital<D: DigitalDifferencing>(self) {
            to_slice_digital::<D>(self.0)
        }

        fn quantized<D: DigitalDifferencing>(self) { /* todo */
        }
    }

    differencing.action(EncodeAction(rgb));

    if let Some(eo_transfer) = transfer.to_optical_display_slice_inplace() {
        eo_transfer(rgb);
    } else {
        for rgb in rgb.iter_mut() {
            *rgb = transfer.to_optical_display(*rgb);
        }
    }
}

fn from_slice_digital<T: DigitalDifferencing>(pix: &mut [[f32; 4]]) {
    let diff = T::DIFF;
    for pix in pix {
        let [r, g, b, a] = *pix;
        let [y, u, v] = diff.mul_vec([r, g, b]);
        *pix = [y, u, v, a];
    }
}

fn to_slice_digital<T: DigitalDifferencing>(rgb: &mut [[f32; 4]]) {
    let diff = T::DIFF.inv();
    for pix in rgb {
        let [r, g, b, a] = *pix;
        let [y, u, v] = diff.mul_vec([r, g, b]);
        *pix = [y, u, v, a];
    }
}

impl Differencing {
    fn action<R>(self, act: impl DifferencingAction<R>) -> R {
        match self {
            Differencing::Bt407MPal => act.digital::<Bt407MPal>(),
            Differencing::Bt407MPalPrecise => act.digital::<Bt407MPalPrecise>(),
            Differencing::Bt601 => act.digital::<Bt601>(),
            Differencing::Bt601Quantized => act.quantized::<Bt601Quantized>(),
            Differencing::Bt601FullSwing => act.quantized::<Bt601FullSwing>(),
            Differencing::Bt709 => act.digital::<Bt709>(),
            Differencing::Bt709Quantized => act.quantized::<Bt709Quantized>(),
            Differencing::Bt709FullSwing => act.quantized::<Bt709FullSwing>(),
            Differencing::YDbDr => act.digital::<YDbDr>(),
            Differencing::Bt2020 => act.digital::<Bt2020>(),
            Differencing::Bt2100 => act.digital::<Bt2100>(),
            Differencing::YCoCg => act.digital::<YCoCg>(),
        }
    }
}

struct Bt407MPal;
struct Bt407MPalPrecise;
struct Pal525;
struct Pal625;
struct Bt601;
struct Bt601Quantized;
struct Bt601FullSwing;
struct Bt709;
struct Bt709Quantized;
struct Bt709FullSwing;
struct Bt2020;
struct Bt2100;
struct YDbDr;
struct YCoCg;
struct YCoCgR;

// We derive the coefficients from scratch, from their definition.
// This makes it more simlar to the BT601 formulas. Any complaints?
const UDIV: f32 = 0.886 / 0.436; // ~1/0.492, but typo'd to 0.493 in the standard...
const VDIV: f32 = 0.701 / 0.615;
impl DigitalDifferencing for Bt407MPalPrecise {
    // Source: https://www.itu.int/rec/R-REC-BT.470-6-199811-S/en
    //
    // Note: revision 7 refers us to Rec. 1700 instead.
    // The main Rec1700 doesn't contain a neat table listing all combinations while its
    // supplementary S170m-2004 contains cool history on the derivation of NTSC parameters with
    // all assumptions and calculations for the analog signal, but again no combination table.
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        -0.299/UDIV, -0.587/UDIV, 0.886/UDIV,
        0.701/VDIV, -0.587/VDIV, -0.114/VDIV,
    ]);
}

impl DigitalDifferencing for Bt407MPal {
    const DIFF: RowMatrix = <Pal525 as DigitalDifferencing>::DIFF;
}

// Extra info: assumed under Illuminant C
// Same as Bt407MPal but with a typos in the standard.
impl DigitalDifferencing for Pal525 {
    // Source: Rec1700-e
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        -0.299*0.493, -0.587*0.943, 0.866*0.493,
        0.701*0.877, -0.587*0.877, -0.114*0.877,
    ]);
}

// Extra info: assumed under Illuminant D65
impl DigitalDifferencing for Pal625 {
    // Source: Rec1700-e
    const DIFF: RowMatrix = <Pal525 as DigitalDifferencing>::DIFF;
}

// Same Y coefficients as Bt470 but normalized to [0.5, 0.5] in both U and V.
impl DigitalDifferencing for Bt601 {
    // Source: https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        0.701/1.402, -0.587/1.402, -0.114/1.402,
        -0.299/1.772, -0.587/1.722, 0.866/1.722,
    ]);
}

impl DigitalDifferencing for Bt601Quantized {
    // Source: https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en
    // FIXME(color): wrong quantization: multiply then round: 219/256, 224/256
    #[rustfmt::skip]
    const DIFF: RowMatrix = panic!();
}

impl DigitalDifferencing for Bt601FullSwing {
    // Source: https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en
    // FIXME(color): wrong quantization: add 0.5
    #[rustfmt::skip]
    const DIFF: RowMatrix = panic!();
}

impl DigitalDifferencing for Bt709 {
    // Source: https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
    // quantization: multiply then round: 219/256, 224/256
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.2126, 0.7152, 0.0722,
        -0.2126/1.8556, -0.7152/1.8556, 0.9278/1.8556,
        0.7874/1.5748, -0.7152/1.5748, -0.0722/1.5748,
    ]);
}

impl DigitalDifferencing for Bt709Quantized {
    // Source: https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
    // FIXME(color): wrong quantization: multiply then round: 219/256, 224/256
    #[rustfmt::skip]
    const DIFF: RowMatrix = panic!();
}

impl DigitalDifferencing for Bt709FullSwing {
    // Source: https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
    // FIXME(color): wrong quantization: add 0.5
    #[rustfmt::skip]
    const DIFF: RowMatrix = panic!();
}

impl DigitalDifferencing for YDbDr {
    // Source: Rec1700-e, YDbDr 625-line SECAM
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        -0.299*1.505, -0.587*1.505, 0.886*1.505,
        -0.701*-1.902, -0.587*-1.902, -0.114*-1.902,
    ]);
}

/* FIXME: these are YIQ.
 *
 * In Rec 1700 we find matrices for YDbDr 625-line SECAM instead. Did it mean the S170 attached?
impl DigitalDifferencing for Ntsc1953 {
    // Source: Wikipedia,
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        0.5959, -0.2746, -0.3213,
        0.2115, -0.5227, 0.3112,
    ]);
}

impl DigitalDifferencing for SmpteC {
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        0.599, -0.2773, -0.3217,
        0.213, -0.5251, 0.3121,
    ]);
}
*/

impl DigitalDifferencing for YCoCg {
    // Source: Wikipedia.
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.25, 0.5, 0.25,
        0.5, 0.0, -0.5,
        -0.25, 0.5, -0.25,
    ]);
}

impl DigitalDifferencing for YCoCgR {
    // Source: Wikipedia.
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.25, 0.5, 0.25,
        1.0, 0.0, -1.0,
        -0.5, 1.0, -0.5,
    ]);
}

// Note: regularize U and V to [-0.5, 0.5], same as previous standards.
// 1.8814 = 2*(1-0.0593)
// 1.4746 = 2*(1-0.2627)
// FIXME(color): provide the C'BC or constant luminance one. That has a discontinuity where
// positive chroma differences are weighted different from negative ones.
impl DigitalDifferencing for Bt2020 {
    // Source: https://www.itu.int/rec/R-REC-BT.2020-0-201208-S/en
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.2627, 0.6780, 0.0593,
        -0.2627/1.8814, -0.6780/1.8814, 0.9407/1.8814,
        0.7373/1.4746, -0.6780/1.4746, -0.0593/1.4746,
    ]);
}

impl DigitalDifferencing for Bt2100 {
    // Source: https://www.itu.int/rec/R-REC-BT.2100-2-201807-I/en
    const DIFF: RowMatrix = <Bt2020 as DigitalDifferencing>::DIFF;
}

// FIXME(color): 10bit and 12bit quantized matrices.
