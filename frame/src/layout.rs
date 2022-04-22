//! Defines layout and buffer of our images.
use crate::color::*;
use canvas::layout::{Layout as CanvasLayout, Raster};

/// The byte layout of a buffer.
///
/// An inner invariant is that the layout fits in memory, and in particular into a `usize`, while
/// at the same time also fitting inside a `u64` of bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ByteLayout {
    /// The number of texels along our width.
    pub(crate) width: u32,
    /// The number of texels along our height.
    pub(crate) height: u32,
    /// The number of bytes per texel.
    /// We need to be able to infallibly convert to both `usize` and `u32`. Thus we have chosen
    /// `u8` for now because no actual texel is that large. However, we could use some other type
    /// to represent the intersection of our two target types (i.e. the `index-ext` crate has
    /// `mem::Umem32` with those exact semantics).
    pub(crate) bytes_per_texel: u8,
    /// The number of bytes per row.
    /// This is a u32 for compatibility with `wgpu`.
    pub(crate) bytes_per_row: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub(crate) bytes: ByteLayout,
    /// The texel representing merged layers.
    pub(crate) texel: Texel,
    // TODO: planarity..
    /// How the numbers relate to physical quantities, important for conversion.
    pub(crate) color: Color,
}

/// Describe a row-major rectangular matrix layout.
///
/// This is only concerned with byte-buffer compatibility and not type or color space semantics of
/// texels. It assumes a row-major layout without space between texels of a row as that is the most
/// efficient and common such layout.
///
/// For usage as an actual image buffer, to convert it to a `Layout` by calling
/// [`Layout::with_row_layout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowLayoutDescription {
    pub width: u32,
    pub height: u32,
    pub texel_stride: u64,
    pub row_stride: u64,
}

/// Describes an image semantically.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutDescriptor {
    /// The byte and physical layout of the buffer.
    pub layout: ByteLayout,
    /// Describe how each single texel is interpreted.
    pub texel: Texel,
    /// How the numbers relate to physical quantities, important for conversion.
    pub color: Color,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleParts {
    parts: [Option<ColorChannel>; 4],
}

macro_rules! sample_parts {
    ( $($(#[$attr:meta])* $name:ident = $($ch:path),+;)* ) => {
        $(sample_parts! { $(#[$attr])* $name = $($ch),* })*

        impl SampleParts {
            $(sample_parts! { $(#[$attr])* $name = $($ch),* })*
        }
    };
    ($(#[$attr:meta])* $name:ident = $ch0:path) => {
        $(#[$attr])*
        pub const $name: SampleParts = SampleParts {
            parts: [Some($ch0), None, None, None]
        };
    };
    ($(#[$attr:meta])* $name:ident = $ch0:path,$ch1:path) => {
        $(#[$attr])*
        pub const $name: SampleParts = SampleParts {
            parts: [Some($ch0), Some($ch1), None, None]
        };
    };
    ($(#[$attr:meta])* $name:ident = $ch0:path,$ch1:path,$ch2:path) => {
        $(#[$attr])*
        pub const $name: SampleParts = SampleParts {
            parts: [Some($ch0), Some($ch1), Some($ch2), None]
        };
    };
    ($(#[$attr:meta])* $name:ident = $ch0:path,$ch1:path,$ch2:path,$ch3:path) => {
        $(#[$attr])*
        pub const $name: SampleParts = SampleParts {
            parts: [Some($ch0), Some($ch1), Some($ch2), Some($ch3)]
        };
    };
}

#[allow(non_upper_case_globals)]
mod sample_parts {
    type Cc = super::ColorChannel;
    use super::SampleParts;

    sample_parts! {
        /// A pure alpha part.
        A = Cc::Alpha;
        /// A pure red part.
        R = Cc::R;
        G = Cc::G;
        B = Cc::B;
        Luma = Cc::Luma;
        LumaA = Cc::Luma,Cc::Alpha;
        Rgb = Cc::R,Cc::G,Cc::B;
        RgbA = Cc::R,Cc::G,Cc::B,Cc::Alpha;
        ARgb = Cc::Alpha,Cc::R,Cc::G,Cc::B;
        Bgr = Cc::B,Cc::G,Cc::R;
        BgrA = Cc::B,Cc::G,Cc::R,Cc::Alpha;
        ABgr = Cc::Alpha,Cc::B,Cc::G,Cc::R;
        Yuv = Cc::L,Cc::Cb,Cc::Cr;
        Lab = Cc::L,Cc::LABa,Cc::LABb;
        LabA = Cc::L,Cc::LABa,Cc::LABb,Cc::Alpha;
        Lch = Cc::L,Cc::C,Cc::LABh;
        LchA = Cc::L,Cc::C,Cc::LABh,Cc::Alpha;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum SampleBits {
    /// A single 8-bit integer.
    Int8,
    /// Three packed integer.
    Int332,
    /// Three packed integer.
    Int233,
    /// A single 16-bit integer.
    Int16,
    /// Four packed integer.
    Int4x4,
    /// Four packed integer, one component ignored.
    Int_444,
    /// Four packed integer, one component ignored.
    Int444_,
    /// Three packed integer.
    Int565,
    /// Two 8-bit integers.
    Int8x2,
    /// Three 8-bit integer.
    Int8x3,
    /// Four 8-bit integer.
    Int8x4,
    /// Two 16-bit integers.
    Int16x2,
    /// Three 16-bit integer.
    Int16x3,
    /// Four 16-bit integer.
    Int16x4,
    /// Four packed integer.
    Int1010102,
    /// Four packed integer.
    Int2101010,
    /// Three packed integer, one component ignored.
    Int101010_,
    /// Three packed integer, one component ignored.
    Int_101010,
    /// Four half-floats.
    Float16x4,
    /// Four floats.
    Float32x4,
}

/// Denotes the 'position' of a channel in the sample parts.
///
/// This is private for now because the constructor might be a bit confusing. In actuality, we are
/// interested in the position of a channel in the _linear_ color representation. For example, all
/// RGB-ish colors (including the variant `Bgra`) are mapped to a `vec4` in the order `rgba` in the
/// shader execution. Thus, the 'position' of the `R` channel is _always_ `First` in these cases.
///
/// This can only make sense with internal knowledge about how we remap color representations into
/// the texture during the Staging phase of loading a color image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ChannelPosition {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
}

impl LayoutDescriptor {
    pub const EMPTY: Self = LayoutDescriptor {
        layout: ByteLayout {
            width: 0,
            height: 0,
            bytes_per_texel: 4,
            bytes_per_row: 0,
        },
        texel: Texel {
            block: Block::Pixel,
            bits: SampleBits::Int8x4,
            parts: sample_parts::RgbA,
        },
        color: Color::SRGB,
    };

    fn with_texel(texel: Texel, color: Color, width: u32, height: u32) -> Option<Self> {
        let layout = ByteLayout::with_texel(&texel, width, height)?;
        Some(LayoutDescriptor {
            layout,
            texel,
            color,
        })
    }

    /// Get the texel describing a single channel.
    /// Returns None if the channel is not contained, or if it can not be extracted on its own.
    pub fn channel_texel(&self, channel: ColorChannel) -> Option<Texel> {
        self.texel.channel_texel(channel)
    }

    /// Check if the descriptor is consistent.
    ///
    /// A consistent descriptor makes inherent sense. That is, the different fields contain values
    /// that are not contradictory. For example, the color channels parts and the color model
    /// correspond to each other, and the sample parts and sample bits field is correct, and the
    /// texel descriptor has the same number of bytes as the layout, etc.
    pub fn is_consistent(&self) -> bool {
        // FIXME: other checks.
        self.texel.bits.bytes() == usize::from(self.layout.bytes_per_texel)
    }

    /// Calculate the total number of pixels in width of this layout.
    pub fn pixel_width(&self) -> u32 {
        self.layout.width * self.texel.block.width()
    }

    /// Calculate the total number of pixels in height of this layout.
    pub fn pixel_height(&self) -> u32 {
        self.layout.height * self.texel.block.height()
    }

    /// Calculate the number of texels in width and height dimension.
    pub fn size(&self) -> (u32, u32) {
        (self.layout.width, self.layout.height)
    }
}

impl Texel {
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
            Int8 | Int8x3 | Int8x4 => Int8,
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
    fn canonical_index_in(self, model: ColorChannelModel) -> Option<u8> {
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
}

impl SampleBits {
    /// Determine the number of bytes for texels containing these samples.
    pub fn bytes(self) -> usize {
        use SampleBits::*;
        #[allow(non_upper_case_globals)]
        match self {
            Int8 | Int332 | Int233 => 1,
            Int8x2 | Int16 | Int565 | Int4x4 | Int444_ | Int_444 => 2,
            Int8x3 => 3,
            Int8x4 | Int16x2 | Int1010102 | Int2101010 | Int101010_ | Int_101010 => 4,
            Int16x3 => 6,
            Int16x4 | Float16x4 => 8,
            Float32x4 => 16,
        }
    }
}

impl SampleParts {
    /// Create from up to four color channels.
    ///
    /// The order of parts will be remembered. All color channels must belong to a common color
    /// representation.
    pub fn new(parts: [Option<ColorChannel>; 4], model: ColorChannelModel) -> Option<Self> {
        let mut unused = [true; 4];
        for part in parts {
            if let Some(p) = part {
                let idx = p.canonical_index_in(model)?;
                if !core::mem::take(&mut unused[idx as usize]) {
                    return None;
                }
            }
        }

        Some(SampleParts { parts })
    }

    pub fn num_components(self) -> u8 {
        self.parts.iter().map(|ch| u8::from(ch.is_some())).sum()
    }
}

impl Color {
    pub fn model(&self) -> Option<ColorChannelModel> {
        Some(match self {
            Color::Rgb { .. } => ColorChannelModel::Rgb,
            Color::Oklab => ColorChannelModel::Lab,
            Color::Scalars { .. } => return None,
        })
    }

    /// Check if this color space contains the sample parts.
    ///
    /// For example, an Xyz color is expressed in terms of a subset of Rgb while HSV color spaces
    /// contains the Hsv parts (duh!) and CIECAM and similar spaces have a polar representation of
    /// hue etc.
    ///
    /// Note that one can always combine a color space with an alpha component.
    #[allow(non_upper_case_globals)]
    pub fn is_consistent(&self, parts: SampleParts) -> bool {
        use sample_parts::*;
        matches!(
            (self, parts),
            (Color::Rgb { .. }, R | G | B | Rgb | RgbA)
            //  | Rgb_ | _Rgb | Bgr_ | _Bgr
            | (Color::Oklab, Lch | LchA)
            // With scalars pseudo color, everything goes.
            // Essentially, the user assigns which meaning each channel has.
            | (Color::Scalars { .. }, _)
        )
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

impl ByteLayout {
    /// Create a buffer layout given the layout of a simple, strided matrix.
    pub fn with_row_layout(rows: RowLayoutDescription) -> Option<Self> {
        let bytes_per_texel = u8::try_from(rows.texel_stride).ok()?;
        let bytes_per_row = u32::try_from(rows.row_stride).ok()?;

        // Enforce that the layout makes sense and does not alias.
        let _ = u32::from(bytes_per_texel)
            .checked_mul(rows.width)
            .filter(|&bwidth| bwidth <= bytes_per_row)?;

        // Enforce our inner invariant.
        let u64_len = u64::from(rows.height).checked_mul(rows.row_stride)?;
        let _ = usize::try_from(u64_len).ok()?;

        Some(ByteLayout {
            width: rows.width,
            height: rows.height,
            bytes_per_texel,
            bytes_per_row,
        })
    }

    /// Create a buffer layout from a texel and dimensions.
    pub fn with_texel(texel: &Texel, width: u32, height: u32) -> Option<Self> {
        let texel_stride = u64::try_from(texel.bits.bytes()).ok()?;
        Self::with_row_layout(RowLayoutDescription {
            width,
            height,
            texel_stride,
            // Note: with_row_layout will do an overflow check anyways.
            row_stride: u64::from(width) * texel_stride,
        })
    }

    /// Returns the width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the memory usage as a `u64`.
    pub fn u64_len(&self) -> u64 {
        // No overflow due to inner invariant.
        u64::from(self.bytes_per_row) * u64::from(self.height)
    }

    /// Returns the memory usage as a `usize`.
    pub fn byte_len(&self) -> usize {
        // No overflow due to inner invariant.
        (self.bytes_per_row as usize) * (self.height as usize)
    }

    /// Returns the index of a texel in a slice of planar image data.
    pub fn texel_index(&self, x: u32, y: u32) -> u64 {
        let byte_index = u64::from(y) * u64::from(self.bytes_per_row)
            + u64::from(x) * u64::from(self.bytes_per_texel);
        byte_index / u64::from(self.bytes_per_texel)
    }

    /// Returns a matrix descriptor that can store all bytes.
    ///
    /// Note: for the moment, all layouts are row-wise matrices. This will be relaxed in the future
    /// to also permit the construction from planar image layouts. In this case, the method will
    /// return a descriptor that does _not_ equal this layout. Instead, an image buffer shaped like
    /// the returned descriptor can be used to re-arrange all bytes into a simple matrix form.
    pub fn as_row_layout(&self) -> RowLayoutDescription {
        RowLayoutDescription {
            width: self.width,
            height: self.height,
            texel_stride: u64::from(self.bytes_per_texel),
            row_stride: u64::from(self.bytes_per_row),
        }
    }
}

impl CanvasLayout for Layout {
    fn byte_len(&self) -> usize {
        ByteLayout::byte_len(&self.bytes)
    }
}
