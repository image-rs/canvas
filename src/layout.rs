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

/// A matrix of packed pixels (channel groups).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Matrix {
    element: Element,
    first_dim: usize,
    second_dim: usize,
}

/// A column major image of packed pixels (channel groups).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColumnMajor {
    element: Element,
    backing: Matrix,
    view: Matrix,
}

/// A row major image of packed pixels (channel groups).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowMajor {
    matrix: Matrix,
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

impl Matrix {
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
