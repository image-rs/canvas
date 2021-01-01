// Distributed under The MIT License (MIT)
//
// Copyright (c) 2020, 2021 The `image-rs` developers
use alloc::boxed::Box;
use palette::{luma::Luma, FromColor, Srgb, Xyz, Xyza};

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
#[derive(Clone, Copy)]
pub enum ColorSpace {
    /// Linear RGB, aka. CIE XYZ.
    ///
    /// In standard notation: xYz, that is red activation, luminance, blue activation.
    /// Directly models the stimuli of the three common receptors of the human eye.
    /// This can also be used for gray scale colors where it is interpreted as using the white
    /// point D65 with a linear transfer function.
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

        impl InputKind<'_> {
            fn pixels(&self) -> usize {
                match self {
                    $(
                        InputKind::$ref_name(kind) => kind.pixels(),
                    )*
                }
            }
        }

        impl OutputKind<'_> {
            fn pixels(&self) -> usize {
                match self {
                    $(
                        OutputKind::$ref_name(kind) => kind.pixels(),
                    )*
                }
            }
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

            impl<'buf> Output<'buf> {
                pub fn $const_name (buffer: &'buf mut [$ul]) -> Self {
                    Output {
                        inner: OutputKind::$ref_name($mut_name {
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

        impl $ref_name<'_> {
            fn pixels(&self) -> usize {
                self.buffer.len()
            }
        }

        impl $mut_name<'_> {
            fn pixels(&self) -> usize {
                self.buffer.len()
            }
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

        impl $ref_name<'_> {
            fn pixels(&self) -> usize {
                self.plane0.len()
            }
        }

        impl $mut_name<'_> {
            fn pixels(&self) -> usize {
                self.plane0.len()
            }
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

        impl $ref_name<'_> {
            fn pixels(&self) -> usize {
                self.plane0.len()
            }
        }

        impl $mut_name<'_> {
            fn pixels(&self) -> usize {
                self.plane0.len()
            }
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

/// Describes one conversion between color spaces.
///
/// Constructing this does not yet execute the conversion. Use the `run` method to execute it in
/// a single work thread.
pub struct Convert<'buf> {
    input: InputKind<'buf>,
    in_space: ColorSpace,
    output: OutputKind<'buf>,
    out_space: ColorSpace,
}

impl<'buf> Convert<'buf> {
    /// Prepare a conversion between two buffers.
    pub fn new(
        input: Input<'buf>,
        in_space: ColorSpace,
        output: Output<'buf>,
        out_space: ColorSpace,
    ) -> Option<Self> {
        if input.inner.pixels() != output.inner.pixels() {
            return None;
        }

        // TODO: What other checks are required?

        Some(Convert {
            input: input.inner,
            in_space,
            output: output.inner,
            out_space,
        })
    }

    /// Return the number of involved pixels.
    pub fn num_pixels(&self) -> usize {
        todo!()
    }

    /// Write the converted pixels into the output buffer.
    pub fn run(mut self) {
        if let Some(fallback_xyz) = self.input.iter_xyz(self.in_space) {
            self.output.write_xyz(self.out_space, fallback_xyz);
        }
    }
}

impl ColorSpace {}

type XyzTransfer = palette::encoding::linear::Linear<palette::white_point::D65>;

impl InputKind<'_> {
    fn iter_xyz(&self, color: ColorSpace) -> Option<Box<dyn Iterator<Item = Xyza> + '_>> {
        if let Some(convert) = color.convert_from_rgba() {
            return self.iter_rgb(convert);
        }

        // All the less generic color space cases...
        Some(match (self, color) {
            (
                InputKind::U8(U8 {
                    buffer,
                    kind: U8Kind::Gray8,
                }),
                ColorSpace::Xyz,
            ) => Box::new(buffer.iter().map(|&ch| {
                let luma = Luma::<XyzTransfer, f32>::new(f32::from(ch) / 255.0);
                Xyza {
                    color: Xyz::from_luma(luma),
                    alpha: 1.0,
                }
            })),
            (
                InputKind::U8(U8 {
                    buffer,
                    kind: U8Kind::Gray8,
                }),
                ColorSpace::Srgb,
            ) => {
                Box::new(buffer.iter().map(|&ch| {
                    // Default parameters are for srgb.
                    let luma = Luma::new(f32::from(ch));
                    Xyza {
                        color: Xyz::from_luma(luma),
                        alpha: 1.0,
                    }
                }))
            }
            _ => return None,
        })
    }

    fn iter_rgb(
        &self,
        rgba: impl (Fn([f32; 4]) -> Xyza) + 'static,
    ) -> Option<Box<dyn Iterator<Item = Xyza> + '_>> {
        Some(match self {
            InputKind::U8(U8 {
                buffer,
                kind: U8Kind::Rgb332,
            }) => Box::new(buffer.iter().map(move |&ch| {
                let (r, g, b) = (ch >> 5, (ch >> 2) & 0x7, ch & 0x3);
                let (r, g, b) = (f32::from(r) / 7.0, f32::from(g) / 7.0, f32::from(b) / 3.0);
                rgba([r, g, b, 1.0])
            })),
            InputKind::U8x3(U8x3 {
                buffer,
                kind: U8x3Kind::Rgb8,
            }) => Box::new(buffer.iter().map(move |&[r, g, b]| {
                let (r, g, b) = scale_u8x3_to_f32([r, g, b]);
                rgba([r, g, b, 1.0])
            })),
            InputKind::U8x3(U8x3 {
                buffer,
                kind: U8x3Kind::Bgr8,
            }) => Box::new(buffer.iter().map(move |&[b, g, r]| {
                let (r, g, b) = scale_u8x3_to_f32([r, g, b]);
                rgba([r, g, b, 1.0])
            })),
            InputKind::U8x4(U8x4 {
                buffer,
                kind: U8x4Kind::Argb8,
            }) => Box::new(buffer.iter().map(move |&[alpha, r, g, b]| {
                let (r, g, b) = scale_u8x3_to_f32([r, g, b]);
                rgba([r, g, b, scale_u8_to_f32(alpha)])
            })),
            InputKind::U8x4(U8x4 {
                buffer,
                kind: U8x4Kind::Abgr8,
            }) => Box::new(buffer.iter().map(move |&[alpha, b, g, r]| {
                let (r, g, b) = scale_u8x3_to_f32([r, g, b]);
                rgba([r, g, b, scale_u8_to_f32(alpha)])
            })),
            _ => return None,
        })
    }
}

impl OutputKind<'_> {
    fn write_xyz(&mut self, color: ColorSpace, from: Box<dyn Iterator<Item = Xyza> + '_>) {
        if let Some(convert) = color.convert_to_rgba() {
            return self.write_rgb(from, convert);
        }

        // All the less generic color space cases...
        match (self, color) {
            (
                OutputKind::U8(U8Mut {
                    buffer,
                    kind: U8Kind::Gray8,
                }),
                ColorSpace::Xyz,
            ) => {
                for (into, from) in buffer.iter_mut().zip(from) {
                    let luma = Luma::<XyzTransfer, f32>::from_xyz(from.color);
                    *into = (luma.luma * 255.0) as u8;
                }
            }
            (
                OutputKind::U8(U8Mut {
                    buffer,
                    kind: U8Kind::Gray8,
                }),
                ColorSpace::Srgb,
            ) => {
                for (into, from) in buffer.iter_mut().zip(from) {
                    let luma: Luma = Luma::from_xyz(from.color);
                    *into = (luma.luma * 255.0) as u8;
                }
            }
            _ => {}
        }
    }

    fn write_rgb(
        &mut self,
        from: Box<dyn Iterator<Item = Xyza> + '_>,
        rgba: impl Fn(Xyza) -> [f32; 4],
    ) {
        match self {
            OutputKind::U8(U8Mut {
                buffer,
                kind: U8Kind::Rgb332,
            }) => {
                for (into, from) in buffer.iter_mut().zip(from) {
                    let [r, g, b, _] = rgba(from);
                    let (r, g, b) = ((r * 7.0) as u8, (g * 7.0) as u8, (b * 3.0) as u8);
                    *into = r << 5 | g << 2 | b;
                }
            }
            OutputKind::U8x3(U8x3Mut {
                buffer,
                kind: U8x3Kind::Rgb8,
            }) => {
                for (into, from) in buffer.iter_mut().zip(from) {
                    let [r, g, b, _] = rgba(from);
                    let [r, g, b] = scale_u8x3_from_f32((r, g, b));
                    *into = [r, g, b];
                }
            }
            OutputKind::U8x3(U8x3Mut {
                buffer,
                kind: U8x3Kind::Bgr8,
            }) => {
                for (into, from) in buffer.iter_mut().zip(from) {
                    let [r, g, b, _] = rgba(from);
                    let [r, g, b] = scale_u8x3_from_f32((r, g, b));
                    *into = [b, g, r];
                }
            }
            OutputKind::U8x4(U8x4Mut {
                buffer,
                kind: U8x4Kind::Argb8,
            }) => {
                for (into, from) in buffer.iter_mut().zip(from) {
                    let [r, g, b, alpha] = rgba(from);
                    let [r, g, b] = scale_u8x3_from_f32((r, g, b));
                    *into = [scale_u8_from_f32(alpha), r, g, b];
                }
            }
            OutputKind::U8x4(U8x4Mut {
                buffer,
                kind: U8x4Kind::Abgr8,
            }) => {
                for (into, from) in buffer.iter_mut().zip(from) {
                    let [r, g, b, alpha] = rgba(from);
                    let [r, g, b] = scale_u8x3_from_f32((r, g, b));
                    *into = [scale_u8_from_f32(alpha), b, g, r];
                }
            }
            _ => {}
        }
    }
}

/// The rgba spaces we support conversion from/to.
#[derive(Clone, Copy)]
enum RgbaSpace {
    Xyza,
    Srgb,
}

impl RgbaSpace {
    fn from_components(self, [r, g, b, alpha]: [f32; 4]) -> Xyza {
        match self {
            RgbaSpace::Xyza => {
                let color = Xyz::new(r, g, b);
                Xyza { color, alpha }
            }
            RgbaSpace::Srgb => {
                let color = Xyz::from(Srgb::new(r, g, b));
                Xyza { color, alpha }
            }
        }
    }

    fn to_components(self, xyza: Xyza) -> [f32; 4] {
        match self {
            RgbaSpace::Xyza => {
                let (x, y, z) = xyza.color.into_components();
                [x, y, z, xyza.alpha]
            }
            RgbaSpace::Srgb => {
                let (r, g, b) = Srgb::from(xyza.color).into_components();
                [r, g, b, xyza.alpha]
            }
        }
    }
}

impl ColorSpace {
    fn convert_to_rgba(self) -> Option<impl Fn(Xyza) -> [f32; 4]> {
        let space = match self {
            ColorSpace::Xyz => RgbaSpace::Xyza,
            ColorSpace::Srgb => RgbaSpace::Srgb,
            _ => return None,
        };

        Some(move |xyza: Xyza| space.to_components(xyza))
    }

    fn convert_from_rgba(self) -> Option<impl Fn([f32; 4]) -> Xyza> {
        let space = match self {
            ColorSpace::Xyz => RgbaSpace::Xyza,
            ColorSpace::Srgb => RgbaSpace::Srgb,
            _ => return None,
        };

        Some(move |xyza: [f32; 4]| space.from_components(xyza))
    }
}

fn scale_u8_to_f32(alpha: u8) -> f32 {
    f32::from(alpha) / 255.0
}

fn scale_u8x3_to_f32([r, g, b]: [u8; 3]) -> (f32, f32, f32) {
    (
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
    )
}

fn scale_u8_from_f32(alpha: f32) -> u8 {
    (alpha * 255.0) as u8
}

fn scale_u8x3_from_f32((r, g, b): (f32, f32, f32)) -> [u8; 3] {
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

#[test]
fn transpose() {
    let inarr = [[0u8, 1u8, 2u8]];
    let mut outarr = [[0xffu8; 3]];

    let input = Input::rgb8(&inarr);
    let output = Output::bgr8(&mut outarr);

    let convert = Convert::new(input, ColorSpace::Xyz, output, ColorSpace::Xyz)
        .expect("Valid and possible conversion");
    convert.run();

    assert_eq!(outarr[0], [2u8, 1u8, 0u8]);
}

#[test]
fn with_alpha() {
    let inarr = [[0xff, 0u8, 1u8, 2u8]];
    let mut outarr = [[0xff; 4]];

    let input = Input::argb8(&inarr);
    let output = Output::abgr8(&mut outarr);

    let convert = Convert::new(input, ColorSpace::Xyz, output, ColorSpace::Xyz)
        .expect("Valid and possible conversion");
    convert.run();

    assert_eq!(outarr[0], [0xff, 2u8, 1u8, 0u8]);
}
