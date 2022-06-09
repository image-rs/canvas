use crate::color_matrix::RowMatrix;

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

struct Bt407MPal;
struct Bt601;
struct Bt601Quantized;
struct Bt709;
struct YDbDr;
struct Ntsc1953;
struct SmpteC;
struct YCoCg;
struct YCoCgR;

// We derive the coefficients from scratch, from their definition.
// This makes it more simlar to the BT601 formulas. Any complaints?
const UDIV: f32 = 0.886 / 0.436;
const VDIV: f32 = 0.701 / 0.615;
impl DigitalDifferencing for Bt407MPal {
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        -0.299/UDIV, -0.587/UDIV, 0.886/UDIV,
        0.701/VDIV, -0.587/VDIV, -0.114/VDIV,
    ]);
}

impl DigitalDifferencing for Bt601 {
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        0.701/1.402, -0.587/1.402, -0.114/1.402,
        -0.299/1.772, -0.587/1.722, 0.866/1.722,
    ]);
}

impl DigitalDifferencing for Bt601Quantized {
    // FIXME(color): quantization: multiply then round: 219/256, 224/256
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        0.701/1.402, -0.587/1.402, -0.114/1.402,
        -0.299/1.772, -0.587/1.722, 0.866/1.722,
    ]);
}

impl DigitalDifferencing for Bt709 {
    // quantization: multiply then round: 219/256, 224/256
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.2126, 0.7152, 0.0722,
        -0.2126/1.8556, -0.7152/1.8556, 0.9278/1.8556,
        0.7874/1.5748, -0.7152/1.5748, -0.0722/1.5748,
    ]);
}

impl DigitalDifferencing for YDbDr {
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.299, 0.587, 0.114,
        -0.450, -0.883, 1.333,
        -1.333, 1.116, 0.217,
    ]);
}

impl DigitalDifferencing for Ntsc1953 {
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

impl DigitalDifferencing for YCoCg {
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.25, 0.5, 0.25,
        0.5, 0.0, -0.5,
        -0.25, 0.5, -0.25,
    ]);
}

impl DigitalDifferencing for YCoCgR {
    #[rustfmt::skip]
    const DIFF: RowMatrix = RowMatrix([
        0.25, 0.5, 0.25,
        1.0, 0.0, -1.0,
        -0.5, 1.0, -0.5,
    ]);
}
