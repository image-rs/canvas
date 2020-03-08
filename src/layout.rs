//! A module for different pixel layouts.
use crate::AsPixel;

/// Describes the byte layout of an element, untyped.
///
/// This is not so different from `Pixel` and `Layout` but is a combination of both. It has the
/// same invariants on alignment as the former which being untyped like the latter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Element {
    size: usize,
    align: usize,
}

/// A descriptor of the layout of image bytes.
///
/// There is no color space and no strict type interpretation here, just some mapping to required
/// bytes for such a fixed buffer and a width and height of the described image. This means that
/// the byte usage for a particular buffer needs to be independent of the content, in particular
/// can not be based on compressibility.
///
/// There is one more thing that differentiates an image from an encoded format. It is expected
/// that the image can be unfolded into some matrix of independent pixels (with potentially
/// multiple channels) without any arithmetic or conversion function. Independent here means that,
/// when supplied with the missing color space and type information, there should exist an
/// `Fn(U) -> T` that can map these pixels independently into some linear color space.
///
/// This property holds for any packed, strided or planar RGB/YCbCr/HSV format as well as chroma
/// subsampled YUV images and even raw Bayer filtered images.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    pub(crate) repr: LayoutRepr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LayoutRepr {
    Matrix(Matrix),
    Yuv420p(Yuv420p),
}

/// A matrix of packed pixels (channel groups).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Matrix {
    element: Element,
    first_dim: usize,
    second_dim: usize,
}

/// Planar chroma 2×2 block-wise sub-sampled image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Yuv420p {
    channel: Element,
    width: u32,
    height: u32,
}

impl Element {
    pub fn from_pixel<P: AsPixel>() -> Self {
        let pix = P::pixel();
        Element {
            size: pix.size(),
            align: pix.align(),
        }
    }

    pub const fn size(self) -> usize {
        self.size
    }

    pub const fn align(self) -> usize {
        self.size
    }
}

impl Layout {
    pub fn byte_len(self) -> usize {
        match self.repr {
            LayoutRepr::Matrix(matrix) => matrix.byte_len(),
            LayoutRepr::Yuv420p(matrix) => matrix.byte_len(),
        }
    }
}

impl Matrix {
    pub fn empty(element: Element) -> Self {
        Matrix {
            element,
            first_dim: 0,
            second_dim: 0,
        }
    }

    pub fn from_width_height(
        element: Element,
        first_dim: usize,
        second_dim: usize,
    ) -> Option<Self> {
        let max_index = first_dim.checked_mul(second_dim)?;
        let _ = max_index.checked_mul(element.size)?;

        Some(Matrix {
            element,
            first_dim,
            second_dim,
        })
    }

    /// Get the required bytes for this layout.
    pub const fn byte_len(self) -> usize {
        // Exactly this does not overflow due to construction.
        self.element.size * self.len()
    }

    /// The number of pixels in this layout
    pub const fn len(self) -> usize {
        self.first_dim * self.second_dim
    }

    pub fn offset(self, coord1: usize, coord2: usize) -> Option<usize> {
        if self.first_dim >= coord1 || self.second_dim >= coord2 {
            None
        } else {
            Some(self.offset_unchecked(coord1, coord2))
        }
    }

    pub const fn offset_unchecked(self, coord1: usize, coord2: usize) -> usize {
        coord1 + coord2 * self.first_dim
    }

    pub fn byte_offset(self, coord1: usize, coord2: usize) -> Option<usize> {
        if self.first_dim >= coord1 || self.second_dim >= coord2 {
            None
        } else {
            Some(self.byte_offset_unchecked(coord1, coord2))
        }
    }

    pub const fn byte_offset_unchecked(self, coord1: usize, coord2: usize) -> usize {
        (coord1 + coord2 * self.first_dim) * self.element.size
    }
}

impl Yuv420p {
    pub fn from_width_height(channel: Element, width: u32, height: u32) -> Option<Self> {
        use core::convert::TryFrom;
        if width % 2 != 0 || height % 2 != 0 {
            return None;
        }

        let mwidth = usize::try_from(width).ok()?;
        let mheight = usize::try_from(height).ok()?;

        let y_count = mwidth.checked_mul(mheight)?;
        let uv_count = y_count / 2;

        let count = y_count.checked_add(uv_count)?;
        let total = count.checked_mul(channel.size)?;
        Some(Yuv420p {
            channel,
            width,
            height,
        })
    }

    pub const fn byte_len(self) -> usize {
        let ylen = (self.width as usize) * (self.height as usize) * self.channel.size;
        ylen + ylen / 2
    }
}

impl From<Matrix> for Layout {
    fn from(matrix: Matrix) -> Self {
        Layout {
            repr: LayoutRepr::Matrix(matrix),
        }
    }
}

impl From<Yuv420p> for Layout {
    fn from(matrix: Yuv420p) -> Self {
        Layout {
            repr: LayoutRepr::Yuv420p(matrix),
        }
    }
}
