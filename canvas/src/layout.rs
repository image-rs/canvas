//! Defines layout and buffer of our images.
use crate::color::*;

use image_texel::image::{Coord, ImageRef};
use image_texel::layout::{
    Decay, Layout as ImageLayout, MatrixBytes, Raster, StrideSpec, StridedBytes, Strides,
    TexelLayout,
};

/// The byte layout of a buffer.
///
/// An inner invariant is that the layout fits in memory, and in particular into a `usize`, while
/// at the same time also fitting inside a `u64` of bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ByteLayout {
    /// The number of pixels along our width.
    pub(crate) width: u32,
    /// The number of pixels along our height.
    pub(crate) height: u32,
    /// The number of bytes per row.
    /// This is a u32 for compatibility with `wgpu`.
    pub(crate) bytes_per_row: u32,
}

/// The layout of a full frame, with all planes and color.
#[derive(Clone, Debug, PartialEq)]
pub struct CanvasLayout {
    // TODO: planarity..
    /// The primary layout descriptor of the image itself.
    /// When no explicit planes are given then this describes the sole plane as well.
    pub(crate) bytes: ByteLayout,
    /// The texel representing merged layers.
    /// When no explicit planes are given then this describes the sole plane as well.
    pub(crate) texel: Texel,
    /// How the numbers relate to physical quantities, important for conversion.
    pub(crate) color: Option<Color>,
    /// Additional color planes, if any.
    pub(crate) planes: Box<[Plane]>,
}

/// The layout of a single color plane, internal.
///
/// This isn't a full descriptor as width and height in numbers of texels can be derived from the
/// underlying byte layout.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plane {
    pub(crate) bytes_per_row: u32,
    /// Representation of the partial texel of the full frame.
    pub(crate) texel: Texel,
}

/// The strides of uniformly spaced (color) channels.
pub struct ChannelSpec {
    pub channels: u8,
    pub channel_stride: usize,
    pub height: u32,
    pub height_stride: usize,
    pub width: u32,
    pub width_stride: usize,
}

/// A layout with uniformly spaced (color) channels.
#[derive(Clone, PartialEq)]
pub struct ChannelBytes {
    /// Based on a strided layout of the texel matrix.
    pub(crate) inner: StridedBytes,
    /// The texel associated with the layout.
    pub(crate) texel: Texel,
    /// Channels also have a uniform stride.
    pub(crate) channel_stride: usize,
    /// Provides the number of channels.
    /// Assume that `channels * channel_stride` is at most the with stride of the underlying layout.
    pub(crate) channels: u8,
}

/// A typed layout with uniform spaced (color) channels.
#[derive(Clone, PartialEq)]
pub struct ChannelLayout<T> {
    pub(crate) channel: image_texel::Texel<T>,
    pub(crate) inner: ChannelBytes,
}

/// The byte matrix layout of a single plane of the image.
pub struct PlaneBytes {
    /// The texel in this plane.
    texel: Texel,
    /// Offset within the image in bytes.
    offset_in_texels: usize,
    /// The matrix descriptor of this plane.
    matrix: StridedBytes,
}

/// The typed matrix layout of a single plane of the image.
///
/// Note that this is _not_ fully public like the related [`PlaneBytes`] as we have invariants
/// relating the texel type to alignment and offset of the matrix within the image.
pub struct PlanarLayout<T> {
    /// The texel in this plane.
    texel: Texel,
    /// Offset within the image in bytes.
    offset_in_texels: usize,
    /// The matrix descriptor of this plane.
    matrix: Strides<T>,
}

/// Describe a row-major rectangular matrix layout.
///
/// This is only concerned with byte-buffer compatibility and not type or color space semantics of
/// texels. It assumes a row-major layout without space between texels of a row as that is the most
/// efficient and common such layout.
///
/// For usage as an actual image buffer, to convert it to a `CanvasLayout` by calling
/// [`CanvasLayout::with_row_layout`].
#[derive(Clone, Debug, PartialEq)]
pub struct RowLayoutDescription {
    pub width: u32,
    pub height: u32,
    pub row_stride: u64,
    pub texel: Texel,
}

/// One Unit of bytes in a texture.
#[derive(Clone, Debug, PartialEq)]
pub struct Texel {
    /// Which part of the image a single texel refers to.
    pub block: Block,
    /// How the values are encoded as bits in the bytes.
    pub bits: SampleBits,
    /// Which values are encoded, which controls the applicable color spaces.
    pub parts: SampleParts,
}

/// How many pixels are described by a single texel unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u8)]
pub enum Block {
    /// Each texel is a single pixel.
    Pixel = 0,
    /// Each texel refers to two pixels across width.
    Sub1x2 = 1,
    /// Each texel refers to four pixels across width.
    Sub1x4 = 2,
    /// Each texel refers to a two-by-two block.
    Sub2x2 = 3,
    /// Each texel refers to a two-by-four block.
    Sub2x4 = 4,
    /// Each texel refers to a four-by-four block.
    Sub4x4 = 5,
}

/// Describes which values are present in a texel.
///
/// This is some set of channels that describe the color point precisely, given a color space.
/// Depending on the chosen color there may be multiple ways in which case this names which of the
/// canonical encodings to use. For example, `CIELAB` may be represented as `Lab` (Lightness,
/// red/green, blue/yellow) or `LCh` (Lightness, Chroma, Hue; the polar cooordinate form of the
/// previous).
///
/// FIXME(color): describe YUV, ASTC and BC block formats? Other? We surely can handle planar data
/// properly?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleParts {
    pub(crate) parts: [Option<ColorChannel>; 4],
    /// The position of each channel as a 2-bit number.
    /// This is the index into which the channel is written.
    pub(crate) color_index: u8,
    /// How the samples are recovered from a texel.
    pub(crate) pitch: SamplePitch,
}

/// Defines the method by which parts from a texel are expanded into bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SamplePitch {
    /// Parts are expanded by extracting bit fields from the texel.
    PixelBits,
    // Subsampled chroma, in order uyvy.
    Yuv422,
    // Like Yuv422 but yuyv.
    Yuy2,
    // Subsampled chroma, uyyvyy.
    Yuv411,
    // Subsampled blocks.
    Yuv420,
    Bc1,
}

macro_rules! sample_parts {
    ( $($(#[$attr:meta])* $color:ident: $name:ident = $($ch:path),+;)* ) => {
        $(sample_parts! { @$color: $(#[$attr])* $name = $($ch),* })*

        impl SampleParts {
            $(sample_parts! { @$color: $(#[$attr])* $name = $($ch),* })*
        }
    };
    (@$color:ident: $(#[$attr:meta])* $name:ident = $ch0:path) => {
        $(#[$attr])*
        pub const $name: SampleParts = SampleParts {
            parts: [Some($ch0), None, None, None],
            color_index: (
                $ch0.canonical_index_in_surely(ColorChannelModel::$color)
            ),
            pitch: SamplePitch::PixelBits,
        };
    };
    (@$color:ident: $(#[$attr:meta])* $name:ident = $ch0:path,$ch1:path) => {
        $(#[$attr])*
        pub const $name: SampleParts = SampleParts {
            parts: [Some($ch0), Some($ch1), None, None],
            color_index: (
                $ch0.canonical_index_in_surely(ColorChannelModel::$color)
                | $ch1.canonical_index_in_surely(ColorChannelModel::$color) << 2
            ),
            pitch: SamplePitch::PixelBits,
        };
    };
    (@$color:ident: $(#[$attr:meta])* $name:ident = $ch0:path,$ch1:path,$ch2:path) => {
        $(#[$attr])*
        pub const $name: SampleParts = SampleParts {
            parts: [Some($ch0), Some($ch1), Some($ch2), None],
            color_index: (
                $ch0.canonical_index_in_surely(ColorChannelModel::$color)
                | $ch1.canonical_index_in_surely(ColorChannelModel::$color) << 2
                | $ch2.canonical_index_in_surely(ColorChannelModel::$color) << 4
            ),
            pitch: SamplePitch::PixelBits,
        };
    };
    (@$color:ident: $(#[$attr:meta])* $name:ident = $ch0:path,$ch1:path,$ch2:path,$ch3:path) => {
        $(#[$attr])*
        pub const $name: SampleParts = SampleParts {
            parts: [Some($ch0), Some($ch1), Some($ch2), Some($ch3)],
            color_index: (
                $ch0.canonical_index_in_surely(ColorChannelModel::$color)
                | $ch1.canonical_index_in_surely(ColorChannelModel::$color) << 2
                | $ch2.canonical_index_in_surely(ColorChannelModel::$color) << 4
                | $ch3.canonical_index_in_surely(ColorChannelModel::$color) << 6
            ),
            pitch: SamplePitch::PixelBits,
        };
    };
}

#[allow(non_upper_case_globals)]
// We use items here just as a glob-import.
// They are duplicated as constants to the struct then.
#[allow(unused)]
mod sample_parts {
    type Cc = super::ColorChannel;
    use super::ColorChannelModel;
    use super::SampleParts;
    use super::SamplePitch;

    sample_parts! {
        /// A pure alpha part.
        Rgb: A = Cc::Alpha;
        /// A pure red part.
        Rgb: R = Cc::R;
        Rgb: G = Cc::G;
        Rgb: B = Cc::B;
        Yuv: Luma = Cc::Luma;
        Yuv: LumaA = Cc::Luma,Cc::Alpha;
        Rgb: Rgb = Cc::R,Cc::G,Cc::B;
        Rgb: RgbA = Cc::R,Cc::G,Cc::B,Cc::Alpha;
        Rgb: ARgb = Cc::Alpha,Cc::R,Cc::G,Cc::B;
        Rgb: Bgr = Cc::B,Cc::G,Cc::R;
        Rgb: BgrA = Cc::B,Cc::G,Cc::R,Cc::Alpha;
        Rgb: ABgr = Cc::Alpha,Cc::B,Cc::G,Cc::R;
        Yuv: Yuv = Cc::L,Cc::Cb,Cc::Cr;
        Lab: Lab = Cc::L,Cc::LABa,Cc::LABb;
        Lab: LabA = Cc::L,Cc::LABa,Cc::LABb,Cc::Alpha;
        Lab: Lch = Cc::L,Cc::C,Cc::LABh;
        Lab: LchA = Cc::L,Cc::C,Cc::LABh,Cc::Alpha;
    }

    /*
    impl SampleParts {
        Rgb_ = 9,
        Bgr_ = 11,
        _Rgb = 13,
        _Bgr = 15,
    }
    */
}

/// The bit-placement of samples within a texel.
///
/// We start with low-order bits in a little-endian representation of the surrounding numbers. So,
/// for example, Int332 has the first sample in the three lowest bits of a u8 (byte-order
/// independent) and a Int565 has its first channel in the first 5 low-order bits of a u16 little
/// endian interpretation of the bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum SampleBits {
    /// A single 8-bit signed integer.
    Int8,
    /// A single 8-bit integer.
    UInt8,
    /// Three packed integer.
    UInt332,
    /// Three packed integer.
    UInt233,
    /// A single 8-bit signed integer.
    Int16,
    /// A single 16-bit integer.
    UInt16,
    /// Four packed integer.
    UInt4x4,
    /// Four packed integer, one component ignored.
    UInt_444,
    /// Four packed integer, one component ignored.
    UInt444_,
    /// Three packed integer.
    UInt565,
    /// Two 8-bit integers.
    UInt8x2,
    /// Three 8-bit integer.
    UInt8x3,
    /// Four 8-bit integer.
    UInt8x4,
    /// Six 8-bit integer.
    UInt8x6,
    /// Two 8-bit integers.
    Int8x2,
    /// Three 8-bit integer.
    Int8x3,
    /// Four 8-bit integer.
    Int8x4,
    /// Two 16-bit integers.
    UInt16x2,
    /// Three 16-bit integer.
    UInt16x3,
    /// Four 16-bit integer.
    UInt16x4,
    /// Two 16-bit signed integers.
    Int16x2,
    /// Three 16-bit integer.
    Int16x3,
    /// Four 16-bit integer.
    Int16x4,
    /// Six 16-bit integer.
    UInt16x6,
    /// Four packed integer.
    UInt1010102,
    /// Four packed integer.
    UInt2101010,
    /// Three packed integer, one component ignored.
    UInt101010_,
    /// Three packed integer, one component ignored.
    UInt_101010,
    /// Four half-floats.
    Float16x4,
    /// A single floating-point channel.
    Float32,
    /// Two float channels.
    Float32x2,
    /// Three float channels.
    Float32x3,
    /// Four floats.
    Float32x4,
    /// Six floats.
    Float32x6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BitEncoding {
    Opaque,
    UInt,
    Int,
    Float,
}

/// Error that occurs when constructing a layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutError {
    inner: (),
}

impl Texel {
    pub fn new_u8(parts: SampleParts) -> Self {
        use SampleBits::*;
        Self::pixel_from_bits(parts, [UInt8, UInt8x2, UInt8x3, UInt8x4])
    }

    pub fn new_u16(parts: SampleParts) -> Self {
        use SampleBits::*;
        Self::pixel_from_bits(parts, [UInt16, UInt16x2, UInt16x3, UInt16x4])
    }

    pub fn new_f32(parts: SampleParts) -> Self {
        use SampleBits::*;
        Self::pixel_from_bits(parts, [Float32, Float32x2, Float32x3, Float32x4])
    }

    fn pixel_from_bits(parts: SampleParts, bits: [SampleBits; 4]) -> Self {
        Texel {
            block: Block::Pixel,
            bits: bits[(parts.num_components() - 1) as usize],
            parts,
        }
    }

    /// Get the texel describing a single channel.
    /// Returns None if the channel is not contained, or if it can not be extracted on its own.
    pub fn channel_texel(&self, channel: ColorChannel) -> Option<Texel> {
        use sample_parts::*;
        use Block::*;
        use SampleBits::*;
        #[allow(non_upper_case_globals)]
        let parts = match self.parts {
            Rgb => match channel {
                ColorChannel::R => R,
                ColorChannel::G => G,
                ColorChannel::B => B,
                _ => return None,
            },
            RgbA | BgrA | ABgr | ARgb => match channel {
                ColorChannel::R => R,
                ColorChannel::G => G,
                ColorChannel::B => B,
                ColorChannel::Alpha => A,
                _ => return None,
            },
            _ => return None,
        };
        let bits = match self.bits {
            UInt8 | UInt8x3 | UInt8x4 => UInt8,
            _ => return None,
        };
        let block = match self.block {
            Pixel | Sub1x2 | Sub1x4 | Sub2x2 | Sub2x4 | Sub4x4 => self.block,
        };
        Some(Texel { bits, parts, block })
    }
}

impl ColorChannel {
    /// The color model of the channel.
    ///
    /// Returns `None` if it does not belong to any singular color model.
    pub fn in_model(self, model: ColorChannelModel) -> bool {
        self.canonical_index_in(model).is_some()
    }

    // Figure out how to expose this.. Return type is not entirely clear.
    const fn canonical_index_in(self, model: ColorChannelModel) -> Option<u8> {
        use ColorChannel::*;
        use ColorChannelModel::*;
        Some(match (self, model) {
            (R | X, Rgb) => 0,
            (G | Y, Rgb) => 1,
            (B | Z, Rgb) => 2,
            (Luma, Yuv) => 0,
            (Cb, Yuv) => 1,
            (Cr, Yuv) => 2,
            (L, Lab) => 0,
            (LABa | C, Lab) => 1,
            (LABb | LABh, Lab) => 2,
            // Alpha allowed anywhere, as the last component.
            (Alpha, _) => 3,
            // FIXME: Scalar0, Scalar1, Scalar2
            _ => return None,
        })
    }

    const fn canonical_index_in_surely(self, model: ColorChannelModel) -> u8 {
        match Self::canonical_index_in(self, model) {
            Some(idx) => idx,
            None => 0,
        }
    }
}

impl SampleBits {
    /// Determine the number of bytes for texels containing these samples.
    pub fn bytes(self) -> u16 {
        use SampleBits::*;
        #[allow(non_upper_case_globals)]
        match self {
            Int8 | UInt8 | UInt332 | UInt233 => 1,
            Int8x2 | UInt8x2 | Int16 | UInt16 | UInt565 | UInt4x4 | UInt444_ | UInt_444 => 2,
            Int8x3 | UInt8x3 => 3,
            Int8x4 | UInt8x4 | Int16x2 | UInt16x2 | UInt1010102 | UInt2101010 | UInt101010_
            | UInt_101010 | Float32 => 4,
            UInt8x6 | Int16x3 | UInt16x3 => 6,
            Int16x4 | UInt16x4 | Float16x4 | Float32x2 => 8,
            UInt16x6 | Float32x3 => 12,
            Float32x4 => 16,
            Float32x6 => 24,
        }
    }

    fn as_array(self) -> Option<(TexelLayout, u8)> {
        use image_texel::AsTexel;
        use SampleBits::*;
        Some(match self {
            UInt8 | UInt8x2 | UInt8x3 | UInt8x4 | UInt8x6 => {
                (u8::texel().into(), self.bytes() as u8)
            }
            Int8 | Int8x2 | Int8x3 | Int8x4 => (i8::texel().into(), self.bytes() as u8),
            UInt16 | UInt16x2 | UInt16x3 | UInt16x4 | UInt16x6 => {
                (u16::texel().into(), self.bytes() as u8 / 2)
            }
            Int16 | Int16x2 | Int16x3 | Int16x4 => (i16::texel().into(), self.bytes() as u8 / 2),
            Float32 | Float32x2 | Float32x3 | Float32x4 | Float32x6 => {
                (u32::texel().into(), self.bytes() as u8 / 4)
            }
            _ => return None,
        })
    }

    fn layout(self) -> TexelLayout {
        use crate::shader::{GenericTexelAction, TexelKind};
        struct ToLayout;

        impl GenericTexelAction<TexelLayout> for ToLayout {
            fn run<T>(self, texel: image_texel::Texel<T>) -> TexelLayout {
                texel.into()
            }
        }

        TexelKind::from(self).action(ToLayout)
    }

    pub(crate) fn bit_encoding(self) -> ([BitEncoding; 6], u8) {
        use SampleBits::*;
        match self {
            UInt8 | UInt8x2 | UInt8x3 | UInt8x4 | UInt8x6 => {
                ([BitEncoding::UInt; 6], self.bytes() as u8)
            }
            Int8 | Int8x2 | Int8x3 | Int8x4 => ([BitEncoding::Int; 6], self.bytes() as u8),
            UInt16 | UInt16x2 | UInt16x3 | UInt16x4 | UInt16x6 => {
                ([BitEncoding::UInt; 6], self.bytes() as u8 / 2)
            }
            Int16 | Int16x2 | Int16x3 | Int16x4 => ([BitEncoding::Int; 6], self.bytes() as u8 / 2),
            Float32 | Float32x2 | Float32x3 | Float32x4 | Float32x6 => {
                ([BitEncoding::Float; 6], self.bytes() as u8 / 4)
            }
            UInt332 | UInt233 | UInt565 => ([BitEncoding::UInt; 6], 3),
            SampleBits::UInt4x4 => ([BitEncoding::UInt; 6], 4),
            SampleBits::UInt_444 | SampleBits::UInt444_ => ([BitEncoding::UInt; 6], 3),
            SampleBits::UInt101010_ | UInt_101010 => ([BitEncoding::Float; 6], 3),
            SampleBits::UInt1010102 | UInt2101010 => ([BitEncoding::Float; 6], 4),
            SampleBits::Float16x4 => ([BitEncoding::Float; 6], 4),
        }
    }
}

impl SampleParts {
    /// Create from up to four color channels.
    ///
    /// This is suitable for describing the channels of a single pixel, and relating it to the bit
    /// parts in the corresponding texel.
    ///
    /// The order of parts will be remembered. All color channels must belong to a common color
    /// representation.
    pub fn new(parts: [Option<ColorChannel>; 4], model: ColorChannelModel) -> Option<Self> {
        let color_index = Self::color_index(&parts, model)?;

        Some(SampleParts {
            parts,
            color_index,
            pitch: SamplePitch::PixelBits,
        })
    }

    fn color_index(parts: &[Option<ColorChannel>; 4], model: ColorChannelModel) -> Option<u8> {
        let mut unused = [true; 4];
        let mut color_index = [0; 4];
        for (part, pos) in parts.into_iter().zip(&mut color_index) {
            if let Some(p) = part {
                let idx = p.canonical_index_in(model)?;
                if !core::mem::take(&mut unused[idx as usize]) {
                    return None;
                }
                *pos = idx;
            }
        }

        let color_index = color_index
            .into_iter()
            .enumerate()
            .fold(0u8, |acc, (idx, pos)| acc | pos << (2 * idx));

        Some(color_index)
    }

    /// Create parts that describe 4:2:2 subsampled color channels.
    ///
    /// These parts represent a 1x2 block, with 4 channels total.
    pub fn with_yuv_422(
        parts: [Option<ColorChannel>; 3],
        model: ColorChannelModel,
    ) -> Option<Self> {
        let parts = [parts[0], parts[1], parts[2], None];
        let color_index = Self::color_index(&parts, model)?;

        Some(SampleParts {
            parts,
            color_index,
            pitch: SamplePitch::Yuv422,
        })
    }

    /// Create parts that describe 4:1:1 subsampled color channels.
    ///
    /// These parts represent a 1x4 block, with 6 channels total.
    pub fn with_yuv_411(
        parts: [Option<ColorChannel>; 3],
        model: ColorChannelModel,
    ) -> Option<Self> {
        todo!()
    }

    pub fn num_components(self) -> u8 {
        self.parts.iter().map(|ch| u8::from(ch.is_some())).sum()
    }

    pub(crate) fn channels(&self) -> impl '_ + Iterator<Item = (Option<ColorChannel>, u8)> {
        (0..4).map(|i| (self.parts[i], (self.color_index >> (2 * i)) & 0x3))
    }
}

impl Block {
    pub fn width(&self) -> u32 {
        use Block::*;
        match self {
            Pixel => 1,
            Sub1x2 | Sub2x2 => 2,
            Sub1x4 | Sub2x4 | Sub4x4 => 4,
        }
    }

    pub fn height(&self) -> u32 {
        use Block::*;
        match self {
            Pixel | Sub1x2 | Sub1x4 => 1,
            Sub2x2 | Sub2x4 => 2,
            Sub4x4 => 3,
        }
    }
}

impl CanvasLayout {
    /// Construct a full frame from a single plane.
    pub fn with_plane(bytes: PlaneBytes) -> Self {
        let stride = bytes.matrix.spec();

        CanvasLayout {
            bytes: ByteLayout {
                width: stride.width as u32,
                height: stride.height as u32,
                bytes_per_row: stride.width_stride as u32,
            },
            planes: Box::default(),
            texel: bytes.texel,
            color: None,
        }
    }

    /// Create from a list of planes, and the texel they describe when merged.
    pub fn with_planes(layers: &[PlaneBytes], texel: Texel) -> Result<Self, LayoutError> {
        if layers.len() == 0 {
            return Err(LayoutError::NO_INFO);
        }

        if layers.len() > 1 {
            // FIXME(planar): should support validation of this.
            return Err(LayoutError::NO_INFO);
        }

        let spec = layers[0].matrix.spec();
        let width = spec.width.try_into().map_err(LayoutError::width_error)?;
        let height = spec.height.try_into().map_err(LayoutError::height_error)?;

        Self::validate(CanvasLayout {
            bytes: ByteLayout {
                width,
                height,
                bytes_per_row: (spec.width_stride as u32) * width,
            },
            planes: Box::default(),
            texel,
            color: None,
        })
        .ok_or(LayoutError::NO_INFO)
    }

    /// Create a buffer layout given the layout of a simple, strided matrix.
    pub fn with_row_layout(rows: &RowLayoutDescription) -> Result<Self, LayoutError> {
        let bytes_per_texel = rows.texel.bits.bytes();
        let bytes_per_row = usize::try_from(rows.row_stride).map_err(LayoutError::width_error)?;

        let stride = StrideSpec {
            offset: 0,
            width: rows.width as usize,
            height: rows.height as usize,
            element: rows.texel.bits.layout(),
            height_stride: bytes_per_row,
            width_stride: bytes_per_texel.into(),
        };

        let bytes = PlaneBytes {
            texel: rows.texel.clone(),
            offset_in_texels: 0,
            matrix: StridedBytes::new(stride).map_err(LayoutError::stride_error)?,
        };

        Self::with_planes(&[bytes], rows.texel.clone())
    }

    /// Create a buffer layout from a texel and dimensions.
    ///
    /// This is a simplification of `with_row_layout` which itself is a simplified `new`.
    pub fn with_texel(texel: &Texel, width: u32, height: u32) -> Result<Self, LayoutError> {
        let texel_stride = u64::from(texel.bits.bytes());
        Self::with_row_layout(&RowLayoutDescription {
            width,
            height,
            // Note: with_row_layout will do an overflow check anyways.
            row_stride: u64::from(width) * texel_stride,
            texel: texel.clone(),
        })
    }

    /// Returns the index of a texel in a slice of planar image data.
    pub fn texel_index(&self, x: u32, y: u32) -> u64 {
        let bytes_per_texel = self.texel.bits.bytes();
        let byte_index = u64::from(x) * u64::from(self.bytes.bytes_per_row)
            + u64::from(y) * u64::from(bytes_per_texel);
        byte_index / u64::from(bytes_per_texel)
    }

    pub(crate) fn fill_texel_indices_impl(&self, idx: &mut [usize], iter: &[Coord]) {
        match self.texel.bits.bytes() {
            0 => unreachable!("No texel with zero bytes"),
            _ if self.bytes.bytes_per_row % self.texel.bits.bytes() as u32 == 0 => {
                let pitch = self.bytes.bytes_per_row / self.texel.bits.bytes() as u32;
                return Self::fill_indices_constant_size(idx, iter, pitch);
            }
            // FIXME(perf): do we need common divisors? perf shows that a significant time is spent
            // on division by `bytes_per_texel` but the common cases (1, 2, 3, 4, 8, etc) should
            // all optimize a lot better.
            _ => {}
        }

        for (&Coord(x, y), idx) in iter.iter().zip(idx) {
            *idx = self.texel_index(x, y) as usize;
        }
    }

    fn fill_indices_constant_size(idx: &mut [usize], iter: &[Coord], pitch: u32) {
        let pitch = u64::from(pitch);
        for (&Coord(x, y), idx) in iter.iter().zip(idx) {
            let texindex = u64::from(x) * pitch + u64::from(y);
            *idx = texindex as usize;
        }
    }

    /// Returns a matrix descriptor that can store all bytes.
    ///
    /// Note: for the moment, all layouts are row-wise matrices. This will be relaxed in the future
    /// to also permit the construction from planar image layouts. In this case, the method will
    /// return a descriptor that does _not_ equal this layout. Instead, an image buffer shaped like
    /// the returned descriptor can be used to re-arrange all bytes into a simple matrix form.
    pub fn as_row_layout(&self) -> RowLayoutDescription {
        RowLayoutDescription {
            width: self.bytes.width,
            height: self.bytes.height,
            texel: self.texel.clone(),
            row_stride: u64::from(self.bytes.bytes_per_row),
        }
    }

    /// Returns the width of the underlying image in pixels.
    pub fn width(&self) -> u32 {
        self.bytes.width
    }

    /// Returns the height of the underlying image in pixels.
    pub fn height(&self) -> u32 {
        self.bytes.height
    }

    /// Returns the memory usage as a `u64`.
    pub fn u64_len(&self) -> u64 {
        // No overflow due to inner invariant.
        u64::from(self.bytes.bytes_per_row) * u64::from(self.bytes.height)
    }

    /// Returns the memory usage as a `usize`.
    pub fn byte_len(&self) -> usize {
        // No overflow due to inner invariant.
        (self.bytes.bytes_per_row as usize) * (self.bytes.height as usize)
    }

    /// Set the color of the layout, if compatible with the texel.
    pub fn set_color(&mut self, color: Color) -> Result<(), LayoutError> {
        let model = color.model().ok_or(LayoutError::NO_INFO)?;

        for (channel, idx) in self.texel.parts.channels() {
            if let Some(channel) = channel {
                let other_idx = match channel.canonical_index_in(model) {
                    Some(idx) => idx,
                    None => return Err(LayoutError::NO_INFO),
                };

                if other_idx != idx {
                    return Err(LayoutError::NO_INFO);
                }
            }
        }

        self.color = Some(color);
        Ok(())
    }

    pub(crate) fn plane(&self, idx: u8) -> Option<PlaneBytes> {
        if !self.planes.is_empty() {
            // FIXME(planar): should support returning layout of this.
            return None;
        }

        if idx != 0 {
            return None;
        }

        let matrix = StridedBytes::with_row_major(
            MatrixBytes::from_width_height(
                self.texel.bits.layout(),
                self.bytes.width as usize,
                self.bytes.height as usize,
            )
            .unwrap(),
        );

        Some(PlaneBytes {
            texel: self.texel.clone(),
            offset_in_texels: 0,
            matrix,
        })
    }

    pub(crate) fn num_planes(&self) -> u8 {
        if self.planes.is_empty() {
            1
        } else {
            self.planes.len() as u8
        }
    }

    pub fn as_plane(&self) -> Option<PlaneBytes> {
        // Not only a single plane.
        if !self.planes.is_empty() {
            return None;
        }

        self.plane(0)
    }

    /// Verify that the byte-length is below `isize::MAX`.
    fn validate(this: Self) -> Option<Self> {
        let mut start = 0;
        // For now, validation requires that planes are successive.
        // This can probably stay true for quite a while..
        for plane in 0..this.num_planes() {
            // Require that the number of planes actually works..
            let plane = this.plane(plane)?;
            let spec = plane.matrix.spec();
            // TODO: should we require that planes are aligned to MAX_ALIGN?
            // Probably useful for some methods but that's something for planar layouts.
            // FIXME(planar): decide on this issue.
            let offset = plane.offset_in_texels.checked_mul(spec.element.size())?;
            let plane_end = offset.checked_add(plane.matrix.byte_len())?;

            let texel_layout = plane.texel.bits.layout();
            if !spec.element.superset_of(texel_layout) {
                return None;
            }

            if start > offset {
                return None;
            }

            start = plane_end;
        }

        let lines = usize::try_from(this.bytes.width).ok()?;
        let height = usize::try_from(this.bytes.height).ok()?;
        let ok = height
            .checked_mul(lines)
            .map_or(false, |len| len < isize::MAX as usize);

        Some(this).filter(|_| ok)
    }
}

impl PlaneBytes {
    pub(crate) fn as_channel_bytes(&self) -> Option<ChannelBytes> {
        let (channel_layout, channels) = self.texel.bits.as_array()?;
        Some(ChannelBytes {
            channel_stride: channel_layout.size(),
            channels,
            inner: self.matrix.clone(),
            texel: self.texel.clone(),
        })
    }

    pub(crate) fn is_compatible<T>(&self, texel: image_texel::Texel<T>) -> Option<PlanarLayout<T>> {
        use image_texel::layout::TryMend;
        Some(PlanarLayout {
            texel: self.texel.clone(),
            offset_in_texels: self.offset_in_texels,
            matrix: texel.try_mend(&self.matrix).ok()?,
        })
    }
}

impl ChannelBytes {
    pub(crate) fn is_compatible<T>(
        &self,
        texel: image_texel::Texel<T>,
    ) -> Option<ChannelLayout<T>> {
        if self.channel_stride == texel.size() {
            Some(ChannelLayout {
                channel: texel,
                inner: self.clone(),
            })
        } else {
            None
        }
    }
}

impl<T> ChannelLayout<T> {
    fn from_planar_assume_u8<const N: usize>(from: PlanarLayout<[T; N]>) -> Self {
        let channel = from.matrix.texel().array_element();
        let inner = StridedBytes::decay(from.matrix);
        ChannelLayout {
            channel,
            inner: ChannelBytes {
                texel: from.texel,
                channels: N as u8,
                channel_stride: channel.size(),
                inner,
            },
        }
    }
}

impl LayoutError {
    const NO_INFO: Self = LayoutError { inner: () };

    fn width_error(_: core::num::TryFromIntError) -> Self {
        Self::NO_INFO
    }

    fn height_error(_: core::num::TryFromIntError) -> Self {
        Self::NO_INFO
    }

    fn stride_error(_: image_texel::layout::BadStrideError) -> Self {
        Self::NO_INFO
    }
}

impl ImageLayout for CanvasLayout {
    fn byte_len(&self) -> usize {
        CanvasLayout::byte_len(self)
    }
}

impl<T> ImageLayout for PlanarLayout<T> {
    fn byte_len(&self) -> usize {
        self.matrix.byte_len()
    }
}

impl ImageLayout for PlaneBytes {
    fn byte_len(&self) -> usize {
        self.matrix.byte_len()
    }
}

impl<T> Raster<T> for PlanarLayout<T> {
    fn dimensions(&self) -> Coord {
        let StrideSpec { width, height, .. } = self.matrix.spec();
        // The PlanarLayout should only be constructed from u32 width and height, guaranteeing that
        // this conversion works. If it doesn't, these should be traced to the constructor.
        debug_assert!(u32::try_from(width).is_ok(), "Invalid dimension: {}", width);
        debug_assert!(
            u32::try_from(height).is_ok(),
            "Invalid dimension: {}",
            height
        );
        Coord(width as u32, height as u32)
    }

    fn get(from: ImageRef<&Self>, at: Coord) -> Option<T> {
        let (x, y) = at.xy();
        let layout = from.layout();
        let matrix = &layout.matrix;
        let texel = matrix.texel();
        // TODO: should we add a method to `canvas::Matrix`?
        let StrideSpec { width_stride, .. } = matrix.spec();

        debug_assert!(
            width_stride % texel.size() == 0,
            "Invalid stride: {} not valid for {:?}",
            width_stride,
            texel
        );

        let idx = y as usize * (width_stride / texel.size()) + x as usize;
        let slice = from.as_texels(texel);
        let value = slice.get(layout.offset_in_texels..)?.get(idx)?;
        Some(texel.copy_val(value))
    }
}

impl<T> Decay<PlanarLayout<T>> for PlaneBytes {
    fn decay(from: PlanarLayout<T>) -> Self {
        PlaneBytes {
            texel: from.texel,
            offset_in_texels: from.offset_in_texels,
            matrix: StridedBytes::decay(from.matrix),
        }
    }
}

impl<T> ImageLayout for ChannelLayout<T> {
    fn byte_len(&self) -> usize {
        self.inner.byte_len()
    }
}

impl ImageLayout for ChannelBytes {
    fn byte_len(&self) -> usize {
        self.inner.byte_len()
    }
}

impl<T> Decay<PlanarLayout<[T; 1]>> for ChannelLayout<T> {
    fn decay(from: PlanarLayout<[T; 1]>) -> Self {
        ChannelLayout::from_planar_assume_u8(from)
    }
}

impl<T> Decay<PlanarLayout<[T; 2]>> for ChannelLayout<T> {
    fn decay(from: PlanarLayout<[T; 2]>) -> Self {
        ChannelLayout::from_planar_assume_u8(from)
    }
}

impl<T> Decay<PlanarLayout<[T; 3]>> for ChannelLayout<T> {
    fn decay(from: PlanarLayout<[T; 3]>) -> Self {
        ChannelLayout::from_planar_assume_u8(from)
    }
}

impl<T> Decay<PlanarLayout<[T; 4]>> for ChannelLayout<T> {
    fn decay(from: PlanarLayout<[T; 4]>) -> Self {
        ChannelLayout::from_planar_assume_u8(from)
    }
}

impl<T> Decay<ChannelLayout<T>> for ChannelBytes {
    fn decay(from: ChannelLayout<T>) -> Self {
        from.inner
    }
}
