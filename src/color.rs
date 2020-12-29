// Distributed under The MIT License (MIT)
//
// Copyright (c) 2020, 2021 The `image-rs` developers

/// A color space of a specific color model.
///
/// This specifies an interpretation of numerical values of channels such that they describe a
/// specific perception. The gamut (set of colors that can be represented) is a volume in the space
/// of all possible tri-stimuli that the human eye can interpret¹.
///
/// The simple additive color spaces define three primary colors and mix all other colors as an
/// affine combination of those, thus its gamuts are a triangle. Commonly, the linear values are
/// individually transformed in a non-linear manner. The goal of this gamma-correction is ensuring
/// that the perceptual difference of changes in values is more uniform.
///
/// ---
///
/// ¹Not all humans have three cones. Keep in mind that a cone reacts to incoming light at more
/// than one wavelength, it behaves more as a integral measure of incoming light after that was
/// filtered by the optical components. (People lacking retinas perceive ultraviolet light). The
/// sensitivity curves of the standard human cones overlap to a large degree. There is also
/// evidence that we additional cells to react to luminance in very dim lighting conditions.
///
/// However, no finite set of measure functions can fully measure the full distribution of incoming
/// light. Thus, the stimuli of one observer are not enough to transfer into stimuli for another
/// observer as large amounts of information on the exact wavelengths is already lost by sampling.
/// Keep this in mind when designing systems for color perception impaired people.
///
/// Some 8% of the population with a Y chromosome, and 0.5% of those without have a form of
/// dichromacy, that is a disability to pick up some colors, due to a deficiency of one of the
/// cones. Each cone might be affected with differing probability, the rarest being a deficiency of
/// the blue, short wavelengths, the most common being the long, red wavelengths.
///
/// There is also a separate expression where an additional form of cone is present, with estimates
/// ranging up to 50% of the population. This cone peaks between green and red, providing another,
/// non-redundant stimulus and thus greater potential perception. For people actually able to
/// perceive that additional dimension, no RGB image can accurately express the full reality of an
/// image. At least an additional yellow light source/color signal would be required.
#[non_exhaustive]
pub enum ColorSpace {
    /// Linear RGB, aka. CIE XYZ.
    /// Directly models the stimuli of the three common receptors of the human eye.
    Xyz,
    /// The common sRGB primaries and gamma correction.
    Srgb,
    /// The Adobe RGB encoding (1998) with a large gamut.
    AdobeRgb,
    /// The ProPhoto chromaticities (Kodak) with a wide gamut, even outside human perception.
    ProPhotoRgb,
    /// BT.601 or Rec.601 or CCIR 601, a gamut for old television.
    Bt601,
    /// Rec.709, BT.709, and ITU 709 as defined for film and HDTV, used for h264 as well.
    Bt709,
    /// DCI-P3, a color space used in film.
    DciP3,
    /// Adobe Wide Gamut RGB.
    AdobeWideGamut,
    /// A wide gamut for UHDTV.
    Bt2020,
    /// Rec.2100, a wide gamut with high dynamic range, with Perceptual Quantization.
    Bt2100PQ,
    /// Rec.2100, a wide gamut with high dynamic range, with Hybrid Log-Gamma.
    Bt2100HLG,
    /// Hue, saturation, value; an alternative representation for linear rgb.
    Hsv,
    /// Hue, saturation, lightness; an alternative representation for linear rgb.
    Hsl,
    /// A very early (1947) model with the goal of perceptual uniformity.
    /// The main problem is that no closed form reverse transformation has been published.
    OsaUcs,
    /// This CIE model is intended to be perceptually uniform.
    /// Contrary to most of the above, its gamut is very much not a simple polygon.
    CieLab,
    /// This CIE model is a cylindrical version of Lab, with chroma and hue instead of stimuli.
    CieLch,
    /// This CIE model takes into account the viewing environment.
    CieCam02,
    /// An additive color model for printing ink.
    /// This very much depends on the color profiles (ICC) of your device.
    Cmyk,
    Munsell,
    /// The Natural Color System.
    Ncs,
}

pub struct Input<'buf> {
    inner: InputKind<'buf>,
}

enum InputKind<'buf> {
    Rgb8 {
        buffer: &'buf [[u8; 3]],
        space: RgbSpace,
    },
}

enum RgbSpace {
    Srgb,
    Rec709,
    //
}

pub struct Output<'buf> {
    inner: OutputKind<'buf>,
}

enum OutputKind<'buf> {
    Rgb8(&'buf [[u8; 3]]),
}

impl<'buf> Input<'buf> {
    pub fn rgb8(_: &'buf [[u8; 3]]) -> Self {
        todo!()
    }
}

impl<'buf> Output<'buf> {}

impl ColorSpace {}
