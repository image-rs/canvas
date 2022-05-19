mod oklab;
mod srlab2;
mod transfer;

use crate::color_matrix::{ColMatrix, RowMatrix};

/// Identifies a color representation.
///
/// This names the model by which the numbers in the channels relate to a physical model. How
/// exactly depends on the variant as presented below. Some of them can be customized further with
/// parameters.
///
/// Notably, there are _NOT_ the numbers which we will use in image operations. Generally, we will
/// use an associated _linear_ representation of those colors instead. The choice here depends on
/// the color and is documented for each variants. It is chosen to provide models for faithful
/// linear operations on these colors such as mixing etc.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Color {
    /// An rgb-ish, additive model based on the CIE 1931 XYZ observers.
    ///
    /// The _linear_ representation is the screen space linear RGB, which depends on primaries,
    /// whitepoint and reference luminance. It is derived from the encoded form through the
    /// transfer function.
    Rgb {
        primary: Primaries,
        transfer: Transfer,
        whitepoint: Whitepoint,
        luminance: Luminance,
    },
    /// The simple but perceptual space Oklab by Björn Ottoson.
    ///
    /// The _linear_ representation of this color is Lab but its quantized components are may be
    /// either Lab or LCh.
    ///
    /// It's based on a combination of two linear transforms and one non-linear power-function
    /// between them. Coefficients of these transforms are based on optimization against matching
    /// pairs in the detailed CAM16 model, trying to predict the parameters in those pairs as
    /// precisely as possible. For details see [the post's derivation][derivation].
    ///
    /// Reference: <https://bottosson.github.io/posts/oklab/>
    ///
    /// [derivation]: https://bottosson.github.io/posts/oklab/#how-oklab-was-derived
    Oklab,
    /// A group of scalar values, with no assigned relation to physical quantities.
    ///
    /// The purpose of this color is to simplify the process of creating color ramps and sampling
    /// functions, which do not have any interpretation themselves but are just coefficients to be
    /// used somewhere else.
    ///
    /// The only `SampleParts` that are allowed to be paired with this are `XYZ`.
    ///
    /// Additionally, you might use the images created with this color as an input or an
    /// intermediate step of a `transmute` to create images with chosen values in the linear
    /// representation without the need to manually calculate their texel encoding.
    Scalars {
        /// The transfer to use for points, as if they are RGB-ish colors.
        /// You can simply use `Linear` if you do not want to encode and rgb texel.
        transfer: Transfer,
    },
    /// A LAB space based on contemporary perceptual understanding.
    ///
    /// > The newly defined SRLAB2 color model is a compromise between the simplicity of CIELAB and
    /// the correctness of CIECAM02.
    ///
    /// By combining whitepoint adaption in the (more) precise model of CIECAM02 while performing
    /// the transfer function in the cone response space, this achieves a good uniformity by
    /// simply modelling the human perception properly. It just leaves out the surround luminance
    /// model in the vastly more complex CIECAM02.
    ///
    /// Reference: <https://www.magnetkern.de/srlab2.html>
    SrLab2 { whitepoint: Whitepoint },
}

/// How to interpret channels as physical quantities.
///
/// Each color model consists of a set of color channels, each of which may occur or be omitted in
/// buffers using that model. Each model defines one canonical _channel order_. This is the order
/// they appear in within 'shader units' when pixels are decoded from texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorChannelModel {
    /// An additive model consisting of a redish, greenish, blueish channel.
    ///
    /// Not all models have truly red, green, or blue channels. More specifically we refer to any
    /// color representation that uses three observer functions (weight functions on the visible
    /// portion of the spectrum of light) and represents color as the simplex defined from mixing
    /// them.
    ///
    /// The most common, or nearly universal in computer imagery, choice is a linear combination of
    /// the three CIE XYZ standard observers at 2° center of vision.
    ///
    /// Example: sRGB, CIE XYZ.
    Rgb,
    /// A lightness, and two color difference components.
    ///
    /// Also sometimes called YUV but that is easily confused is the specific color model called
    /// 'YUV', a common analog encoding for several PAL systems (now outdated). Don't confuse with
    /// CIE Yuv (1960) or CIE L*u*v* which is different thing entirely. Yes, confusing.
    ///
    /// Based on an Rgb color spaces, with a linear transform to express the color in terms of a
    /// total luminance and the difference of blue, red luminance relative to the total one. The
    /// linear transform is most often applied to non-linear (aka. gamma pre-corrected, or
    /// electric) R'G'B' values but sometimes (Rec.709) such correct is applied after
    /// transformation. Coefficients differ between systems.
    ///
    /// As can be read from the terms, the intensity is given as a _photometric_ definition in
    /// terms of luminance and not as a perceptual 'lightness' which differentiates it from Lab/Lch
    /// as defined below.
    // TODO: figure out if we want to call ICtCp `Yuv`.. After all there is a non-linear transform
    // involved that is not evaluated independently for each channel. But we do not _need_ to add a
    // corresponding `Color` variant that captures all the models.
    // ICtCp
    Yuv,
    /// A lightness, and two chroma components.
    ///
    /// Differs from xyY spaces by a non-linear transform, commonly with the goal of generating
    /// perceptually uniform values. Example: CIE La*b*.
    ///
    /// The uniformity permits a perceptual distance metric as Euclidean distance, although this
    /// proves imprecise under in-depth investigation. Good for a decent estimate though.
    Lab,
    /// A lightness and two chroma components as polar coordinates.
    ///
    /// Polar transform of a Lab model. Example: Oklab
    Lch,
    /// A subtractive model consisting of fours inks defining absorbed colors.
    ///
    /// Example: ISO 2846 (Euroskala)
    Cmyk,
    // Deprecate as a joke?
    /// HSV (Hue, saturation, value).
    ///
    /// On closer inspection, a model that is neither physical nor perceptual nor based on
    /// correctness merits and its use should be strongly reconsidered in favor of a proper Lab-like
    /// color model. Please stop, please. <https://en.wikipedia.org/wiki/HSL_and_HSV#Disadvantages>
    Hsv,
    /// HSL (Hue, saturation, lightness).
    ///
    /// Careful, lightness means neither luminance nor perceptual lightness and is a mere
    /// arithmetic mean of color values. Some recommend using Luma (based on primary weights)
    /// instead but neglect to mention a specific standard. Really research what definition was
    /// used when the pixel color was computed. Good luck.
    ///
    /// On closer inspection, a model that is neither physical nor perceptual nor based on
    /// correctness merits and its use should be strongly reconsidered in favor of a proper Lab-like
    /// color model. Please stop, please. <https://en.wikipedia.org/wiki/HSL_and_HSV#Disadvantages>
    Hsl,
}

/// Describes a single channel from an image.
///
/// This can be thought of as an index into a vector of channels relating to a color. Combine with
/// a concrete [`ColorChannelModel`] for the canonical index in a 4-sample color representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorChannel {
    /// The weight of the red primary.
    R,
    /// The weight of the green primary.
    G,
    /// The weight of the blue primary.
    B,
    /// A luminescence.
    /// Note that `YCbCr` will be composed of Luma and Cb, Cr. This avoids the gnarly overlap
    /// between it and `Y` as the standard observer (even though this Y is often used to define the
    /// Luma relative to standard illuminant).
    Luma,
    /// An alpha/translucence component.
    Alpha,
    /// Blue-channel difference.
    Cb,
    /// Red-channel difference.
    Cr,
    /// Lightness. Not to be confused with luminescence as this is perceptual.
    L,
    /// The component a (green/red) of a LAB color.
    LABa,
    /// The component b (green/red) of a LAB color.
    LABb,
    /// Chroma of a LAB color, polar distance, `hypot(a, b)`.
    C,
    /// Hue of a LAB based color, polar angle, `atan2(b, a).
    LABh,
    /// The first CIE standard observer.
    X,
    /// The second CIE standard observer.
    Y,
    /// The second CIE standard observer.
    Z,
    Scalar0,
    Scalar1,
    Scalar2,
}

/// Transfer functions from encoded chromatic samples to physical quantity.
///
/// Ignoring viewing environmental effects, this describes a pair of functions that are each others
/// inverse: An electro-optical transfer (EOTF) and opto-electronic transfer function (OETF) that
/// describes how scene lighting is encoded as an electric signal. These are applied to each
/// stimulus value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Transfer {
    /// Non-linear electrical data of Bt.709
    Bt709,
    Bt470M,
    /// Non-linear electrical data of Bt.601
    Bt601,
    /// Non-linear electrical data of Smpte-240
    Smpte240,
    /// Linear color in display luminance.
    Linear,
    /// Non-linear electrical data of Srgb
    ///
    /// Technically, we're implementing scRGB since we handle negative primaries just well enough.
    Srgb,
    /// Non-linear electrical data of Bt2020 that was 10-bit quantized
    Bt2020_10bit,
    /// Non-linear electrical data of Bt2020 that was 12-bit quantized
    /// FIXME(color): not yet supported, panics on use.
    Bt2020_12bit,
    /// Non-linear electrical data of Smpte-2048
    Smpte2084,
    /// Another name for Smpte2084.
    /// FIXME(color): not yet supported, panics on use.
    Bt2100Pq,
    /// Non-linear electrical data of Bt2100 Hybrid-Log-Gamma.
    /// FIXME(color): not yet supported, panics on use.
    Bt2100Hlg,
    /// Linear color in scene luminance of Bt2100.
    /// This is perfect for an artistic composition pipeline. The rest of the type system will
    /// ensure this is not accidentally and unwittingly mixed with `Linear` but otherwise this is
    /// treated as `Linear`. You might always transmute.
    /// FIXME(color): not yet supported, panics on use.
    Bt2100Scene,
}

/// The reference brightness of the color specification.
///
/// FIXME(color): scaling to reference luminance doesn't have an interface yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Luminance {
    /// 100cd/m².
    Sdr,
    /// 10_000cd/m².
    /// Known as high-dynamic range.
    Hdr,
    /// 160cd/m².
    AdobeRgb,
}

/// The relative stimuli of the three corners of a triangular RGBish gamut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Primaries {
    /// The CIE XYZ 'primaries'.
    /// FIXME(color): does this really make sense?
    Xyz,
    /// First set of primaries specified in Bt/Rec.601.
    ///
    /// These are actually the same as in SMPTE240M.
    Bt601_525,
    /// Second set of primaries specified in Bt/Rec.601.
    Bt601_625,
    /// Primaries specified in Bt/Rec.709.
    Bt709,
    /// Primaries specified in SMPTE240-M.
    ///
    /// There are actually the same as BT.601.
    Smpte240,
    /// Primaries specified in Bt/Rec.2020.
    ///
    /// Also known as Wide Color Gamut.
    Bt2020,
    /// Primaries specified in Bt/Rec.2100.
    ///
    /// Also known as Wide Color Gamut. See Bt.2020.
    Bt2100,
}

/// The whitepoint/standard illuminant.
///
/// | Illuminant | X       | Y       | Z       |
/// |------------|---------|---------|---------|
/// | A          | 1.09850 | 1.00000 | 0.35585 |
/// | B          | 0.99072 | 1.00000 | 0.85223 |
/// | C          | 0.98074 | 1.00000 | 1.18232 |
/// | D50        | 0.96422 | 1.00000 | 0.82521 |
/// | D55        | 0.95682 | 1.00000 | 0.92149 |
/// | D65        | 0.95047 | 1.00000 | 1.08883 |
/// | D75        | 0.94972 | 1.00000 | 1.22638 |
/// | E          | 1.00000 | 1.00000 | 1.00000 |
/// | F2         | 0.99186 | 1.00000 | 0.67393 |
/// | F7         | 0.95041 | 1.00000 | 1.08747 |
/// | F11        | 1.00962 | 1.00000 | 0.64350 |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Whitepoint {
    A,
    B,
    C,
    D50,
    D55,
    D65,
    D75,
    E,
    F2,
    F7,
    F11,
}

impl Color {
    pub const SRGB: Color = Color::Rgb {
        luminance: Luminance::Sdr,
        primary: Primaries::Bt709,
        transfer: Transfer::Srgb,
        whitepoint: Whitepoint::D65,
    };

    pub const BT709: Color = Color::Rgb {
        luminance: Luminance::Sdr,
        primary: Primaries::Bt709,
        transfer: Transfer::Bt709,
        whitepoint: Whitepoint::D65,
    };

    pub(crate) fn to_xyz_slice(&self, pixel: &[[f32; 4]], xyz: &mut [[f32; 4]]) {
        // We can do shared pre-processing.
        if let Color::Rgb {
            primary,
            transfer,
            whitepoint,
            luminance: _,
        } = self
        {
            let to_xyz = primary.to_xyz(*whitepoint);

            if let Some(eo_transfer) = transfer.to_optical_display_slice() {
                eo_transfer(pixel, xyz);

                for target_xyz in xyz {
                    let [r, g, b, a] = *target_xyz;
                    let [x, y, z] = to_xyz.mul_vec([r, g, b]);
                    *target_xyz = [x, y, z, a];
                }
            } else {
                for (target_xyz, src_pix) in xyz.iter_mut().zip(pixel) {
                    *target_xyz = transfer.to_optical_display(*src_pix);
                }

                for target_xyz in xyz {
                    let [r, g, b, a] = *target_xyz;
                    let [x, y, z] = to_xyz.mul_vec([r, g, b]);
                    *target_xyz = [x, y, z, a];
                }
            }

            return;
        } else if let Color::Oklab {} = self {
            return oklab::to_xyz_slice(pixel, xyz);
        } else if let Color::SrLab2 { whitepoint } = self {
            return srlab2::to_xyz_slice(pixel, xyz, *whitepoint);
        }

        // Fallback path in all cases.
        for (src_pix, target_xyz) in pixel.iter().zip(xyz) {
            *target_xyz = self.to_xyz_once(*src_pix)
        }
    }

    pub(crate) fn from_xyz_slice(&self, xyz: &[[f32; 4]], pixel: &mut [[f32; 4]]) {
        if let Color::Rgb {
            primary,
            transfer,
            whitepoint,
            luminance: _,
        } = self
        {
            let from_xyz = primary.to_xyz(*whitepoint).inv();

            if let Some(oe_transfer) = transfer.from_optical_display_slice() {
                for (target_pix, src_xyz) in pixel.iter_mut().zip(xyz) {
                    let [x, y, z, a] = *src_xyz;
                    let [r, g, b] = from_xyz.mul_vec([x, y, z]);
                    *target_pix = [r, g, b, a];
                }

                oe_transfer(pixel);
            } else {
                for (target_pix, src_xyz) in pixel.iter_mut().zip(xyz) {
                    let [x, y, z, a] = *src_xyz;
                    let [r, g, b] = from_xyz.mul_vec([x, y, z]);
                    *target_pix = [r, g, b, a];
                }

                for target_pix in pixel {
                    *target_pix = transfer.from_optical_display(*target_pix);
                }
            }

            return;
        } else if let Color::Oklab {} = self {
            return oklab::from_xyz_slice(xyz, pixel);
        } else if let Color::SrLab2 { whitepoint } = self {
            return srlab2::from_xyz_slice(xyz, pixel, *whitepoint);
        }

        for (target_pix, src_xyz) in pixel.iter_mut().zip(xyz) {
            *target_pix = self.from_xyz_once(*src_xyz)
        }
    }

    pub(crate) fn to_xyz_once(&self, value: [f32; 4]) -> [f32; 4] {
        match self {
            Color::Oklab => oklab::oklab_to_xyz(value),
            Color::Rgb {
                primary,
                transfer,
                whitepoint,
                luminance: _,
            } => {
                let [r, g, b, a] = transfer.to_optical_display(value);
                let to_xyz = primary.to_xyz(*whitepoint);
                let [x, y, z] = to_xyz.mul_vec([r, g, b]);
                [x, y, z, a]
            }
            Color::Scalars { transfer } => transfer.to_optical_display(value),
            Color::SrLab2 { whitepoint } => {
                let [x, y, z, a] = value;
                let [x, y, z] = srlab2::srlab_to_xyz([x, y, z], *whitepoint);
                [x, y, z, a]
            }
        }
    }

    pub(crate) fn from_xyz_once(&self, value: [f32; 4]) -> [f32; 4] {
        match self {
            Color::Oklab => oklab::oklab_from_xyz(value),
            Color::Rgb {
                primary,
                transfer,
                whitepoint,
                luminance: _,
            } => {
                let [x, y, z, a] = value;
                let from_xyz = primary.to_xyz(*whitepoint).inv();
                let [r, g, b] = from_xyz.mul_vec([x, y, z]);
                transfer.from_optical_display([r, g, b, a])
            }
            Color::Scalars { transfer } => transfer.from_optical_display(value),
            Color::SrLab2 { whitepoint } => {
                let [x, y, z, a] = value;
                let [x, y, z] = srlab2::srlab_from_xyz([x, y, z], *whitepoint);
                [x, y, z, a]
            }
        }
    }

    pub fn model(&self) -> Option<ColorChannelModel> {
        Some(match self {
            Color::Rgb { .. } => ColorChannelModel::Rgb,
            Color::Oklab | Color::SrLab2 { .. } => ColorChannelModel::Lab,
            Color::Scalars { .. } => return None,
        })
    }
}

impl Transfer {
    /// Convert to optical (=linear) display intensity.
    ///
    /// The difference between display and scene light only matters for very recent HDR content,
    /// just regard it as electro-optical transfer application.
    pub(crate) fn to_optical_display(self, value: [f32; 4]) -> [f32; 4] {
        use self::transfer::*;

        let [r, g, b, a] = value;
        let rgb = [r, g, b];

        let [r, g, b] = match self {
            Transfer::Bt709 => rgb.map(transfer_eo_bt709),
            Transfer::Bt470M => rgb.map(transfer_eo_bt470m),
            Transfer::Bt601 => rgb.map(transfer_eo_bt601),
            Transfer::Smpte240 => rgb.map(transfer_eo_smpte240),
            Transfer::Linear => rgb,
            Transfer::Srgb => rgb.map(transfer_eo_srgb),
            Transfer::Bt2020_10bit => rgb.map(transfer_eo_bt2020_10b),
            Transfer::Bt2020_12bit => {
                // FIXME(color): implement.
                todo!()
            }
            Transfer::Smpte2084 => rgb.map(transfer_eo_smpte2084),
            Transfer::Bt2100Pq => {
                // FIXME(color): implement.
                todo!()
            }
            Transfer::Bt2100Hlg => {
                // FIXME(color): implement.
                todo!()
            }
            Transfer::Bt2100Scene => {
                // FIXME(color): implement.
                todo!()
            }
        };

        [r, g, b, a]
    }

    pub(crate) fn from_optical_display(self, value: [f32; 4]) -> [f32; 4] {
        use self::transfer::*;

        let [r, g, b, a] = value;
        let rgb = [r, g, b];

        let [r, g, b] = match self {
            Transfer::Bt709 => rgb.map(transfer_oe_bt709),
            Transfer::Bt470M => rgb.map(transfer_oe_bt470m),
            Transfer::Bt601 => rgb.map(transfer_oe_bt601),
            Transfer::Smpte240 => rgb.map(transfer_oe_smpte240),
            Transfer::Linear => rgb,
            Transfer::Srgb => rgb.map(transfer_oe_srgb),
            Transfer::Bt2020_10bit => rgb.map(transfer_oe_bt2020_10b),
            Transfer::Bt2020_12bit => {
                // FIXME(color): implement.
                todo!()
            }
            Transfer::Smpte2084 => rgb.map(transfer_oe_smpte2084),
            Transfer::Bt2100Pq => {
                // FIXME(color): implement.
                todo!()
            }
            Transfer::Bt2100Hlg => {
                // FIXME(color): implement.
                todo!()
            }
            Transfer::Bt2100Scene => {
                // FIXME(color): implement.
                todo!()
            }
        };

        [r, g, b, a]
    }

    pub(crate) fn to_optical_display_slice(self) -> Option<fn(&[[f32; 4]], &mut [[f32; 4]])> {
        macro_rules! optical_by_display {
            ($what:ident: $($pattern:pat => $transfer:path,)*) => {
                match $what {
                    $($pattern => return optical_by_display! {@ $transfer },)*
                    _ => return None,
                }
            };
            (@ $transfer:path) => {
                Some(|texels: &[[f32; 4]], pixels: &mut [[f32; 4]]| {
                    for (texel, target_pix) in texels.iter().zip(pixels) {
                        let [r, g, b, a] = *texel;
                        let [r, g, b] = [r, g, b].map($transfer);
                        *target_pix = [r, g, b, a];
                    }
                })
            };
        }

        if let Transfer::Linear = self {
            return Some(|x, y| y.copy_from_slice(x));
        }

        use self::transfer::*;
        optical_by_display!(self:
            Transfer::Bt709 => transfer_eo_bt709,
            Transfer::Bt470M => transfer_eo_bt470m,
            Transfer::Bt601 => transfer_eo_bt601,
            Transfer::Smpte240 => transfer_eo_smpte240,
            Transfer::Srgb => transfer_eo_srgb,
            Transfer::Bt2020_10bit => transfer_eo_bt2020_10b,
        );
    }

    pub(crate) fn from_optical_display_slice(self) -> Option<fn(&mut [[f32; 4]])> {
        macro_rules! optical_by_display {
            ($what:ident: $($pattern:pat => $transfer:path,)*) => {
                match $what {
                    $($pattern => return optical_by_display! {@ $transfer },)*
                    _ => return None,
                }
            };
            (@ $transfer:path) => {
                Some(|pixels: &mut [[f32; 4]]| {
                    for target_pix in pixels.iter_mut() {
                        let [r, g, b, a] = *target_pix;
                        let [r, g, b] = [r, g, b].map($transfer);
                        *target_pix = [r, g, b, a];
                    }
                })
            };
        }

        if let Transfer::Linear = self {
            return Some(|_| {});
        }

        use self::transfer::*;
        optical_by_display!(self:
            Transfer::Bt709 => transfer_oe_bt709,
            Transfer::Bt470M => transfer_oe_bt470m,
            Transfer::Bt601 => transfer_oe_bt601,
            Transfer::Smpte240 => transfer_oe_smpte240,
            Transfer::Srgb => transfer_oe_srgb,
            Transfer::Bt2020_10bit => transfer_oe_bt2020_10b,
        );
    }
}

impl Whitepoint {
    pub(crate) fn to_xyz(self) -> [f32; 3] {
        use Whitepoint::*;
        match self {
            A => [1.09850, 1.00000, 0.35585],
            B => [0.99072, 1.00000, 0.85223],
            C => [0.98074, 1.00000, 1.18232],
            D50 => [0.96422, 1.00000, 0.82521],
            D55 => [0.95682, 1.00000, 0.92149],
            D65 => [0.95047, 1.00000, 1.08883],
            D75 => [0.94972, 1.00000, 1.22638],
            E => [1.00000, 1.00000, 1.00000],
            F2 => [0.99186, 1.00000, 0.67393],
            F7 => [0.95041, 1.00000, 1.08747],
            F11 => [1.00962, 1.00000, 0.64350],
        }
    }
}

#[rustfmt::skip]
impl Primaries {
    /// Convert to XYZ, or back if you invert the matrix.
    ///
    /// This is done with the 'wrong' van Kries transform, under given illuminant, where the CIE
    /// XYZ are scaled to match the whitepoint individually. This is in accordance to the
    /// specification for sRGB et.al even though it isn't very correct in a perceptual sense.
    ///
    /// See: Mark D. Fairchild, Color Appearance Models, 2nd Edition,
    /// Or: SRLAB2 <https://www.magnetkern.de/srlab2.html> for a color model that is perceptually
    /// more correct with regards to illuminants, or the complex CIECAM02.
    pub(crate) fn to_xyz(self, white: Whitepoint) -> RowMatrix {
        use Primaries::*;
        // Rec.BT.601
        // https://en.wikipedia.org/wiki/Color_spaces_with_RGB_primaries#Specifications_with_RGB_primaries
        let xy: [[f32; 2]; 3] = match self {
            Bt601_525 | Smpte240 => [[0.63, 0.34], [0.31, 0.595], [0.155, 0.07]],
            Bt601_625 => [[0.64, 0.33], [0.29, 0.6], [0.15, 0.06]],
            Bt709 => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
            Bt2020 | Bt2100 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
            Xyz => todo!(),
        };

        // A column of CIE XYZ intensities for that primary.
        let xyz = |[x, y]: [f32; 2]| {
            [x / y, 1.0, (1.0 - x - y)/y]
        };

        let xyz_r = xyz(xy[0]);
        let xyz_g = xyz(xy[1]);
        let xyz_b = xyz(xy[2]);

        // Virtually, N = [xyz_r | xyz_g | xyz_b]
        // As the unweighted conversion matrix for:
        //  XYZ = N · RGB
        let n1 = ColMatrix([xyz_r, xyz_g, xyz_b]).inv();

        // http://www.brucelindbloom.com/index.html
        let w = white.to_xyz();

        // s is the weights that give the whitepoint when converted to xyz.
        // That is we're solving:
        //  W = N · S
        let s = n1.mul_vec(w);

        RowMatrix([
            s[0]*xyz_r[0], s[1]*xyz_g[0], s[2]*xyz_b[0],
            s[0]*xyz_r[1], s[1]*xyz_g[1], s[2]*xyz_b[1],
            s[0]*xyz_r[2], s[1]*xyz_g[2], s[2]*xyz_b[2],
        ])
    }
}

#[test]
fn inverse() {
    const RGBA: [f32; 4] = [1.0, 1.0, 0.0, 1.0];
    let color = Color::SRGB;
    let rgba = color.from_xyz_once(color.to_xyz_once(RGBA));
}
