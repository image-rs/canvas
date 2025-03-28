//! Different styles of matrices.
use crate::image::{ImageMut, ImageRef};
use crate::layout::{
    Coord, Decay, Layout, MismatchedPixelError, Raster, RasterMut, SliceLayout, Take, TexelLayout,
    TryMend,
};

use crate::Texel;

/// A matrix of packed texels (channel groups).
///
/// This is a simple layout of exactly width·height homogeneous pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixBytes {
    pub(crate) element: TexelLayout,
    pub(crate) first_dim: usize,
    pub(crate) second_dim: usize,
}

/// A matrix of packed texels (channel groups).
///
/// The underlying buffer may have more data allocated than this region and cause the overhead to
/// be reused when resizing the image. All ways to construct this already check that all pixels
/// within the resulting image can be addressed via an index.
pub struct Matrix<P> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) pixel: Texel<P>,
}

/// A layout that's a matrix of elements.
pub trait MatrixLayout: Layout {
    /// The valid matrix specification of this layout.
    ///
    /// This call should not fail, or panic. Otherwise, prefer an optional getter for the
    /// [`StridedBytes`] and have the caller decay their own buffer.
    fn matrix(&self) -> MatrixBytes;
}

impl MatrixBytes {
    pub fn empty(element: TexelLayout) -> Self {
        MatrixBytes {
            element,
            first_dim: 0,
            second_dim: 0,
        }
    }

    pub fn from_width_height(
        element: TexelLayout,
        first_dim: usize,
        second_dim: usize,
    ) -> Option<Self> {
        let max_index = first_dim.checked_mul(second_dim)?;
        let _ = max_index.checked_mul(element.size)?;

        Some(MatrixBytes {
            element,
            first_dim,
            second_dim,
        })
    }

    /// Get the element type of this matrix.
    pub const fn element(&self) -> TexelLayout {
        self.element
    }

    /// Get the width of this matrix.
    pub const fn width(&self) -> usize {
        self.first_dim
    }

    /// Get the height of this matrix.
    pub const fn height(&self) -> usize {
        self.second_dim
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
}

impl Layout for MatrixBytes {
    fn byte_len(&self) -> usize {
        MatrixBytes::byte_len(*self)
    }
}

impl Take for MatrixBytes {
    fn take(&mut self) -> Self {
        core::mem::replace(self, MatrixBytes::empty(self.element))
    }
}

impl<P> MatrixLayout for Matrix<P> {
    fn matrix(&self) -> MatrixBytes {
        self.into_matrix_bytes()
    }
}

/// Remove the strong typing for dynamic channel type information.
impl<L: MatrixLayout> Decay<L> for MatrixBytes {
    fn decay(from: L) -> MatrixBytes {
        from.matrix()
    }
}

/// Try to use the matrix with a specific pixel type.
impl<P> TryMend<MatrixBytes> for Texel<P> {
    type Into = Matrix<P>;
    type Err = MismatchedPixelError;

    fn try_mend(self, matrix: &MatrixBytes) -> Result<Matrix<P>, Self::Err> {
        Matrix::with_matrix(self, *matrix).ok_or_else(MismatchedPixelError::default)
    }
}

impl<P> From<Matrix<P>> for MatrixBytes {
    fn from(mat: Matrix<P>) -> Self {
        MatrixBytes {
            element: mat.pixel().into(),
            first_dim: mat.width(),
            second_dim: mat.height(),
        }
    }
}

/// Note: on 64-bit targets only the first `u32::MAX` dimensions appear accessible.
impl<P> Raster<P> for Matrix<P> {
    fn dimensions(&self) -> Coord {
        use core::convert::TryFrom;
        let width = u32::try_from(self.width()).unwrap_or(u32::MAX);
        let height = u32::try_from(self.height()).unwrap_or(u32::MAX);
        Coord(width, height)
    }

    fn get(from: ImageRef<&Self>, Coord(x, y): Coord) -> Option<P> {
        if from.layout().in_bounds(x as usize, y as usize) {
            let index = from.layout().index_of(x as usize, y as usize);
            let texel = from.layout().sample();
            from.as_slice().get(index).map(|v| texel.copy_val(v))
        } else {
            None
        }
    }
}

impl<P> RasterMut<P> for Matrix<P> {
    fn put(into: ImageMut<&mut Self>, Coord(x, y): Coord, val: P) {
        if into.layout().in_bounds(x as usize, y as usize) {
            let index = into.layout().index_of(x as usize, y as usize);
            if let Some(dst) = into.into_mut_slice().get_mut(index) {
                *dst = val;
            }
        }
    }
}
