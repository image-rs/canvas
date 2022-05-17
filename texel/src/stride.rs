//! Byte-based, stride operations on an image.
//!
//! This is the most general, uniform source of pixel data. The design allows pixels to alias each
//! other even for mutable operations. The result is always as if performing pixel wise operations
//! row-for-row and column-by-column, except where otherwise noted.
//!
//! In comparison to the standard `Canvas`, the reference types do not need to rely on the
//! container and can be constructed from (suitably aligned) byte data. This makes it possible
//! initialize an image, for example. They internally contain a simple byte slice which allows
//! viewing any source buffer as a strided matrix even when it was not allocated with the special
//! allocator.
use crate::image::Image;
use crate::layout;
use crate::layout::{Layout, MismatchedPixelError, TexelLayout, TryMend};
use crate::texel::{AsTexel, Texel};
use core::ops::Range;

/// A simple layout describing some pixels as a byte matrix.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StrideSpec {
    /// The number of pixels in width direction.
    pub width: usize,
    /// The number of pixels in height direction.
    pub height: usize,
    /// The number of bytes of a single pixel.
    ///
    /// If this differs from both `width_stride` and `height_stride` the any copy must loop over
    /// individual pixels. Otherwise, whole rows or columns of contiguous data may be inspected.
    pub element: TexelLayout,
    /// The number of bytes to go one pixel along the width.
    pub width_stride: usize,
    /// The number of bytes to go one pixel along the height.
    pub height_stride: usize,
    /// Offset of this matrix from the start.
    pub offset: usize,
}

/// A validated layout of a rectangular matrix of pixels, treated as bytes.
///
/// The invariants are that the whole layout fits into memory, additionally ensuring that all
/// indices within have proper indices into the byte slice containing the data.
///
/// The related containers [`ByteCanvasRef`] and [`ByteCanvasMut`] can be utilized to setup
/// efficient initialization of data from different stride sources. Since they require only the
/// alignment according to their elements, not according to the maximum alignment, they may be used
/// for external data that is copied to an image.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StridedBytes {
    spec: StrideSpec,
    /// The total number of bytes, as proof of calculation basically.
    total: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StridedTexels<T> {
    inner: StridedBytes,
    texel: Texel<T>,
}

/// Error that occurs when a [`StrideSpec`] is invalid.
#[derive(Debug)]
pub struct BadStrideError {
    kind: BadStrideKind,
}

#[derive(Debug)]
enum BadStrideKind {
    UnalignedOffset,
    UnalignedWidthStride,
    UnalignedHeightStride,
    OutOfMemory,
}

/// A reference to byte of a strided matrix.
pub struct ByteCanvasRef<'data> {
    layout: StridedBytes,
    data: &'data [u8],
}

/// A reference to mutable byte of a strided matrix.
///
/// This can be constructed from a mutably borrowed image that is currently set to a strided
/// layout such as a matrix. It can be regarded as a generalization to the standard matrix layout.
/// Alternatively, it can be constructed directly from a mutable reference to raw bytes.
///
/// # Usage
///
/// Here is an example of filling a matrix-like image with a constant value.
///
/// ```
/// use image_texel::layout::Matrix;
/// use image_texel::image::{ByteCanvasRef, ByteCanvasMut, Image};
///
/// let layout = Matrix::<u32>::width_and_height(4, 4).unwrap();
/// let mut image = Image::new(layout);
///
/// let fill = ByteCanvasRef::with_repeated_element(&0x42u32, 4, 4);
/// ByteCanvasMut::new(&mut image).copy_from_image(fill);
///
/// assert_eq!(image.as_slice(), &[0x42; 16]);
/// ```
pub struct ByteCanvasMut<'data> {
    layout: StridedBytes,
    data: &'data mut [u8],
}

impl StrideSpec {
    /// Compare sizes without taking into account the offset or strides.
    fn matches(&self, other: &Self) -> bool {
        self.element.size() == other.element.size()
            && self.width == other.width
            && self.height == other.height
    }

    fn has_contiguous_rows(&self) -> bool {
        self.element.size() == self.width_stride
    }

    fn has_contiguous_cols(&self) -> bool {
        self.element.size() == self.height_stride
    }

    fn element_start(&self, row: usize, col: usize) -> usize {
        (row * self.height_stride) + (col * self.width_stride) + self.offset
    }

    fn element(&self, row: usize, col: usize) -> Range<usize> {
        let start = self.element_start(row, col);
        start..start + self.element.size()
    }

    fn contiguous_row(&self, row: usize) -> Range<usize> {
        let start = self.element_start(row, 0);
        let length = self.width * self.element.size();
        start..start + length
    }

    fn contiguous_col(&self, col: usize) -> Range<usize> {
        let start = self.element_start(0, col);
        let length = self.height * self.element.size();
        start..start + length
    }

    fn end(&self) -> Option<usize> {
        if self.height == 0 || self.width == 0 {
            return Some(self.offset);
        }

        let max_w = self.width - 1;
        let max_h = self.height - 1;

        let max_w_offset = max_w.checked_mul(self.width_stride)?;
        let max_h_offset = max_h.checked_mul(self.height_stride)?;

        let relative_past_end = self
            .element
            .size()
            .checked_add(max_h_offset)?
            .checked_add(max_w_offset)?;

        // We wouldn't need to validated if there are no elements. However, this is basically the
        // caller's responsibility. It's more consistent if we keep the offset. For future
        // additions such as calculating free space (?) this would also be required.
        let total = relative_past_end.checked_add(self.offset)?;
        Some(total)
    }
}

impl StridedBytes {
    /// Try to create a new layout from a specification.
    ///
    /// This fails if the specification does not describe a valid layout. The reasons for this
    /// include the element being misaligned according to the provided offsets/strides or the
    /// layout not describing a memory size expressible on the current architecture.
    pub fn new(spec: StrideSpec) -> Result<Self, BadStrideError> {
        if spec.offset % spec.element.align() != 0 {
            return Err(BadStrideKind::UnalignedOffset.into());
        }

        if spec.width_stride % spec.element.align() != 0 {
            return Err(BadStrideKind::UnalignedWidthStride.into());
        }

        if spec.height_stride % spec.element.align() != 0 {
            return Err(BadStrideKind::UnalignedHeightStride.into());
        }

        let total = spec.end().ok_or(BadStrideKind::OutOfMemory)?;

        Ok(StridedBytes { spec, total })
    }

    /// Construct a layout with zeroed strides, repeating one element.
    pub fn with_repeated_width_and_height(
        element: TexelLayout,
        width: usize,
        height: usize,
    ) -> Self {
        StridedBytes {
            spec: StrideSpec {
                element,
                width,
                height,
                height_stride: 0,
                width_stride: 0,
                offset: 0,
            },
            total: element.size(),
        }
    }

    /// Construct from a packed matrix of elements in column major layout.
    ///
    /// This is guaranteed to succeed and will construct the strides such that a packed column
    /// major matrix of elements at offset zero is described.
    pub fn with_column_major(matrix: layout::MatrixBytes) -> Self {
        StridedBytes {
            spec: StrideSpec {
                element: matrix.element(),
                width: matrix.width(),
                height: matrix.height(),
                height_stride: matrix.element().size(),
                // Overflow can't happen because all of `matrix` fits in memory according to its own
                // internal invariant.
                width_stride: matrix.height() * matrix.element().size(),
                offset: 0,
            },
            total: matrix.byte_len(),
        }
    }

    /// Construct from a packed matrix of elements in row major layout.
    ///
    /// This is guaranteed to succeed and will construct the strides such that a packed row major
    /// matrix of elements at offset zero is described.
    pub fn with_row_major(matrix: layout::MatrixBytes) -> Self {
        StridedBytes {
            spec: StrideSpec {
                element: matrix.element(),
                width: matrix.width(),
                height: matrix.height(),
                // Overflow can't happen because all of `matrix` fits in memory according to its own
                // internal invariant.
                height_stride: matrix.width() * matrix.element().size(),
                width_stride: matrix.element().size(),
                offset: 0,
            },
            total: matrix.byte_len(),
        }
    }

    /// Get the specification of this matrix.
    pub fn spec(&self) -> StrideSpec {
        self.spec
    }

    /// Shrink the element's size or alignment.
    ///
    /// This is always valid since the new layout is strictly contained within the old one.
    pub fn shrink_element(&mut self, new: TexelLayout) {
        self.spec.element = self.spec.element.infimum(new);
    }

    fn matches(&self, other: &Self) -> bool {
        self.spec.matches(&other.spec)
    }

    fn contiguous_rows(&self) -> Option<impl Iterator<Item = Range<usize>> + '_> {
        if self.spec.has_contiguous_rows() {
            Some((0..self.spec.height).map(move |row| self.spec.contiguous_row(row)))
        } else {
            None
        }
    }

    fn contiguous_columns(&self) -> Option<impl Iterator<Item = Range<usize>> + '_> {
        if self.spec.has_contiguous_cols() {
            Some((0..self.spec.width).map(move |row| self.spec.contiguous_col(row)))
        } else {
            None
        }
    }

    fn pixel(&self, x: usize, y: usize) -> Range<usize> {
        self.spec.element(x, y)
    }
}

impl<T> StridedTexels<T> {
    /// Upgrade a byte specification to a strong typed texel one.
    ///
    /// Requires that the element is _exactly_ equivalent to the provided texel.
    pub fn with_texel(texel: Texel<T>, bytes: StridedBytes) -> Option<Self> {
        if TexelLayout::from(texel) == bytes.spec.element {
            Some(StridedTexels {
                inner: bytes,
                texel,
            })
        } else {
            None
        }
    }

    pub fn spec(&self) -> StrideSpec {
        self.inner.spec()
    }

    pub fn texel(&self) -> Texel<T> {
        self.texel
    }
}

impl<'data> ByteCanvasRef<'data> {
    /// Construct a reference to a strided image buffer.
    pub fn new(image: &'data Image<impl StridedLayout>) -> Self {
        let layout = image.layout().strided();
        let data = &image.as_bytes()[..layout.total];
        ByteCanvasRef { layout, data }
    }

    /// View bytes under a certain strided layout.
    ///
    /// Unlike an image, the data need only be aligned to the `element` mentioned in the layout and
    /// not to the maximum alignment.
    pub fn with_bytes(layout: StridedBytes, content: &'data [u8]) -> Option<Self> {
        let data = content
            .get(..layout.total)
            .filter(|data| data.as_ptr() as usize % layout.spec.element.align() == 0)?;
        Some(ByteCanvasRef { layout, data })
    }

    pub fn with_repeated_element<T: AsTexel>(el: &'data T, width: usize, height: usize) -> Self {
        let texel = T::texel();
        let layout = StridedBytes::with_repeated_width_and_height(texel.into(), width, height);
        let data = texel.to_bytes(core::slice::from_ref(el));
        ByteCanvasRef { layout, data }
    }

    /// Shrink the element's size or alignment.
    pub fn shrink_element(&mut self, new: TexelLayout) -> TexelLayout {
        self.layout.shrink_element(new);
        self.layout.spec.element
    }

    /// Borrow this as a reference to a strided byte matrix.
    pub fn as_ref(&self) -> ByteCanvasRef<'_> {
        ByteCanvasRef {
            layout: self.layout,
            data: &*self.data,
        }
    }
}

impl<'data> ByteCanvasMut<'data> {
    /// Construct a mutable reference to a strided image buffer.
    pub fn new(image: &'data mut Image<impl StridedLayout>) -> Self {
        let layout = image.layout().strided();
        let data = &mut image.as_bytes_mut()[..layout.total];
        ByteCanvasMut { layout, data }
    }

    /// View bytes mutably under a certain strided layout.
    ///
    /// Unlike an image, the data need only be aligned to the `element` mentioned in the layout and
    /// not to the maximum alignment.
    pub fn with_bytes(layout: StridedBytes, content: &'data mut [u8]) -> Option<Self> {
        let data = content
            .get_mut(..layout.total)
            .filter(|data| data.as_ptr() as usize % layout.spec.element.align() == 0)?;
        Some(ByteCanvasMut { layout, data })
    }

    /// Shrink the element's size or alignment.
    pub fn shrink_element(&mut self, new: TexelLayout) -> TexelLayout {
        self.layout.shrink_element(new);
        self.layout.spec.element
    }

    /// Copy the bytes from another image.
    ///
    /// The source must have the same width, height, and element size.
    pub fn copy_from_image(&mut self, source: ByteCanvasRef<'_>) {
        assert!(self.layout.matches(&source.layout), "Mismatching layouts.");
        // FIXME: Special case copying for 100% contiguous layouts.

        if let Some(rows) = self.layout.contiguous_rows() {
            if let Some(src_rows) = source.layout.contiguous_rows() {
                for (row, src) in rows.zip(src_rows) {
                    self.data[row].copy_from_slice(&source.data[src]);
                }
                return;
            }
        }

        if let Some(cols) = self.layout.contiguous_columns() {
            if let Some(src_cols) = source.layout.contiguous_columns() {
                for (col, src) in cols.zip(src_cols) {
                    self.data[col].copy_from_slice(&source.data[src]);
                }
                return;
            }
        }

        // Panics: we've validated that the widths and heights match.
        for x in 0..self.layout.spec.width {
            for y in 0..self.layout.spec.height {
                let into = self.layout.pixel(x, y);
                let from = source.layout.pixel(x, y);
                // Panics: we've validated that the element sizes match.
                self.data[into].copy_from_slice(&source.data[from]);
            }
        }
    }

    /// Borrow this as a reference to an immutable byte matrix.
    pub fn as_ref(&self) -> ByteCanvasRef<'_> {
        ByteCanvasRef {
            layout: self.layout,
            data: &*self.data,
        }
    }

    /// Convert this into a reference to an immutable byte matrix.
    pub fn into_ref(self) -> ByteCanvasRef<'data> {
        ByteCanvasRef {
            layout: self.layout,
            data: self.data,
        }
    }
}

/// A layout that is a strided matrix of elements.
///
/// Like all layout traits, implementations should ensure that the layout returned in these methods
/// occupied a subset of pixels of their original layout.
pub trait StridedLayout: Layout {
    /// The valid strided specification of this layout.
    ///
    /// This call should not fail, or panic. Otherwise, prefer an optional getter for the
    /// `StridedBytes` and have the caller decay their own buffer.
    fn strided(&self) -> StridedBytes;
}

impl Layout for StridedBytes {
    fn byte_len(&self) -> usize {
        self.total
    }
}

impl StridedLayout for StridedBytes {
    fn strided(&self) -> StridedBytes {
        *self
    }
}

impl<T: StridedLayout> StridedLayout for &'_ T {
    fn strided(&self) -> StridedBytes {
        (**self).strided()
    }
}

impl<T: StridedLayout> StridedLayout for &'_ mut T {
    fn strided(&self) -> StridedBytes {
        (**self).strided()
    }
}

impl<T: StridedLayout> layout::Decay<T> for StridedBytes {
    fn decay(from: T) -> Self {
        from.strided()
    }
}

impl<P: AsTexel> StridedLayout for layout::Matrix<P> {
    fn strided(&self) -> StridedBytes {
        let matrix: layout::MatrixBytes = self.clone().into();
        StridedBytes::with_row_major(matrix)
    }
}

impl<P> Layout for StridedTexels<P> {
    fn byte_len(&self) -> usize {
        self.inner.total
    }
}

impl<P> StridedLayout for StridedTexels<P> {
    fn strided(&self) -> StridedBytes {
        self.inner.clone()
    }
}

impl From<BadStrideKind> for BadStrideError {
    fn from(kind: BadStrideKind) -> Self {
        BadStrideError { kind }
    }
}

impl From<&'_ StridedBytes> for StrideSpec {
    fn from(layout: &'_ StridedBytes) -> Self {
        layout.spec()
    }
}

/// Try to use the matrix with a specific pixel type.
impl<P> TryMend<StridedBytes> for Texel<P> {
    type Into = StridedTexels<P>;
    type Err = MismatchedPixelError;

    fn try_mend(self, matrix: &StridedBytes) -> Result<StridedTexels<P>, Self::Err> {
        StridedTexels::with_texel(self, *matrix).ok_or_else(MismatchedPixelError::default)
    }
}

#[test]
fn align_validation() {
    // Setup a good base specification.
    let matrix = layout::MatrixBytes::from_width_height(TexelLayout::from_pixel::<u16>(), 2, 2)
        .expect("Valid matrix");
    let layout = StridedBytes::with_row_major(matrix);

    let bad_offset = StrideSpec {
        offset: 1,
        ..layout.spec
    };
    assert!(StridedBytes::new(bad_offset).is_err());
    let bad_pitch = StrideSpec {
        width_stride: 5,
        ..layout.spec
    };
    assert!(StridedBytes::new(bad_pitch).is_err());
}

#[test]
fn image_copies() {
    let matrix = layout::MatrixBytes::from_width_height(TexelLayout::from_pixel::<u8>(), 2, 2)
        .expect("Valid matrix");
    let row_layout = StridedBytes::with_row_major(matrix);
    let col_layout = StridedBytes::with_column_major(matrix);

    let src = Image::with_bytes(row_layout, &[0u8, 1, 2, 3]);

    let mut dst = Image::new(row_layout);
    ByteCanvasMut::new(&mut dst).copy_from_image(ByteCanvasRef::new(&src));
    assert_eq!(dst.as_bytes(), &[0u8, 1, 2, 3], "Still in same order");

    let mut dst = Image::new(col_layout);
    ByteCanvasMut::new(&mut dst).copy_from_image(ByteCanvasRef::new(&src));
    assert_eq!(
        dst.as_bytes(),
        &[0u8, 2, 1, 3],
        "In transposed matrix order"
    );
}
