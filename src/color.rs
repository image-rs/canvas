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

/// Next, define the various input kinds we support. The concern is the data layout, not the color
/// space interpretation. In short we don't want to have any generic type parameters as that does
/// not help code generation (except if we had full specialization but I'm convinced the actual
/// benefit is marginal). Instead, the conversion interface needs to be just generic enough such
/// that other kinds can be defined. Support for an `f32` channel type should be sufficient as the
/// accuracy suffices for all relevant physical cases.
///
/// We do this as a macro to handle the input/output const/mut case and the constructors.
macro_rules! color_layouts {
    ($($($ul:ty),+ as $ref_name:ident, mut $mut_name:ident, enum $kind_name:ident => {
        $($color_name:ident = $const_name:ident),* $(,)?
    }),*$(,)?) => {
        enum InputKind<'buf> {
            $($ref_name($ref_name <'buf>)),*
        }

        enum OutputKind<'buf> {
            $($ref_name($mut_name <'buf>)),*
        }

        $(
            color_layouts!(@impl $($ul),* as $ref_name,$mut_name,$kind_name => {
                $($color_name = $const_name),*
            });
        )*
    };
    // Terminal rule, one value type.
    (@impl $ul:ty as $ref_name:ident,$mut_name:ident,$kind_name:ident => {
        $($color_name:ident = $const_name:ident),*
    }) => {
        $(
            impl<'buf> Input<'buf> {
                pub fn $const_name (buffer: &'buf [$ul]) -> Self {
                    Input {
                        inner: InputKind::$ref_name($ref_name {
                            buffer,
                            kind: $kind_name :: $color_name,
                        })
                    }
                }
            }
        )*

        struct $ref_name <'buf> {
            buffer: &'buf [$ul],
            kind: $kind_name,
        }

        struct $mut_name <'buf> {
            buffer: &'buf mut [$ul],
            kind: $kind_name,
        }

        enum $kind_name {
            $($color_name),*
        }
    };
    // Terminal rule for two planes.
    (@impl $p0:ty, $p1:ty as $ref_name:ident,$mut_name:ident,$kind_name:ident => {
        $($color_name:ident),*
    }) => {
        struct $ref_name <'buf> {
            plane0: &'buf [$p0],
            plane1: &'buf [$p1],
            kind: $kind_name,
        }

        struct $mut_name <'buf> {
            plane0: &'buf mut [$p0],
            plane1: &'buf mut [$p1],
            kind: $kind_name,
        }

        enum $kind_name {
            $($color_name),*
        }
    };
    // Terminal rule for three planes.
    (@impl $p0:ty, $p1:ty, $p2:ty as $ref_name:ident,$mut_name:ident,$kind_name:ident => {
        $($color_name:ident),*
    }) => {
        struct $ref_name <'buf> {
            plane0: &'buf [$p0],
            plane1: &'buf [$p1],
            plane2: &'buf [$p2],
            kind: $kind_name,
        }

        struct $mut_name <'buf> {
            plane0: &'buf mut [$p0],
            plane1: &'buf mut [$p1],
            plane2: &'buf mut [$p2],
            kind: $kind_name,
        }

        enum $kind_name {
            $($color_name),*
        }
    };
}

color_layouts! {
    u8 as U8, mut U8Mut, enum U8Kind => {
        Gray8 = gray8,
        Rgb332 = rgb332,
        Bgr332 = bgr332,
    },
    u16 as U16, mut U16Mut, enum U16Kind => {
        Gray16 = gray16,
        Rgb565 = rgb565,
        Bgr565 = bgr565,
    },
    [u8; 3] as U8x3, mut U8x3Mut, enum U8x3Kind => {
        Rgb8 = rgb8,
        Bgr8 = bgr8,
    },
    [u8; 4] as U8x4, mut U8x4Mut, enum U8x4Kind => {
        Xrgb8 = xrgb8,
        Xbgr8 = xbgr8,
        Argb8 = argb8,
        Abgr8 = abgr8,
    },
    [u16; 3] as U16x3, mut U16x3Mut, enum U16x3Kind => {
        Rgb16 = rgb16,
        Bgr16 = bgr16,
    },
    [u16; 4] as U16x4, mut U16x4Mut, enum U16x4Kind => {
        Xrgb16 = xrgb16,
        Xbgr16 = xbgr16,
        Argb16 = argb16,
        Abgr16 = abgr16,
        ArgbF16 = argbf16,
        AbgrF16 = abgrf16,
    },
    u32 as U32, mut U32Mut, enum U32Kind => {
        Gray32 = gray32,
    },
    f32 as F32, mut F32Mut, enum F32Kind => {
        GrayF32 = grayf32,
    },
    [f32; 3] as F32x3, mut F32x3Mut, enum F32x3Kind => {
        RgbF32 = rgbf32,
        BgrF32 = bgrf32,
    },
    [f32; 4] as F32x4, mut F32x4Mut, enum F32x4Kind => {
        ArgbF32 = argbf32,
        AbgrF32 = abgrf32,
    },
    u8,u8 as U8pU8, mut U8pU8Mut, enum U8pU8Kind => {
    },
    u16,u8 as U16pU8, mut U16pU8Mut, enum U16pU8Kind => {
    },
    u8,u16 as U8pU16, mut U8pU16Mut, enum U8pU16Kind => {
    },
    u8,u8,u8 as U8pU8pU8, mut U8pU8pU8Mut, enum U8pU8pU8Kind => {
    },
}

pub struct Input<'buf> {
    inner: InputKind<'buf>,
}

pub struct Output<'buf> {
    inner: OutputKind<'buf>,
}

impl ColorSpace {}
