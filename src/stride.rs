//! Byte-based, stride operations on a canvas.
//!
//! This is the most general, uniform source of pixel data. The design allows pixels to alias each
//! other even for mutable operations. The result is always as if performing pixel wise operations
//! row-for-row and column-by-column, except where otherwise noted.
//!
//! The container type `Strides` is a simple wrapper around a `Canvas` that ensures that the
//! backing buffer corresponds to the layout and offers additional operations that are only valid
//! for the stride layout. Note that it ensures more strictly that the buffer is accurately sized
//! as the raw methods for editing the layout are not exposed. It can always be converted to its
//! general matrix form (by `From`) for such modifications but then the constructor is fallible.
//!
//! In comparison, the reference types do not have an interface for conversion to a borrowed
//! canvas. They internally contain a simple byte slice which allows viewing any source buffer as a
//! strided matrix even when it was not allocated with the special allocator.
use crate::canvas::Canvas;
use crate::layout::Layout;
use crate::texel::AsTexel;
use crate::{layout, matrix};
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
    pub element: layout::TexelLayout,
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
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StrideLayout {
    spec: StrideSpec,
    /// The total number of bytes, as proof of calculation basically.
    total: usize,
}

/// An untyped matrix of pixels treated as pure bytes.
///
/// This leverages the invariants of a `StrideLayout` and additionally ensures that the allocated
/// buffer of the matrix conforms to the requirements of the layout, which would not be strictly
/// ensured by the `Canvas` wrapper itself.
pub struct Strides {
    inner: Canvas<StrideLayout>,
}

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

pub struct ByteCanvasRef<'data> {
    layout: StrideLayout,
    data: &'data [u8],
}

pub struct ByteCanvasMut<'data> {
    layout: StrideLayout,
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

impl StrideLayout {
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

        Ok(StrideLayout { spec, total })
    }

    /// Construct from a packed matrix of elements in column major layout.
    ///
    /// This is guaranteed to succeed and will construct the strides such that a packed column
    /// major matrix of elements at offset zero is described.
    pub fn with_column_major(matrix: layout::Matrix) -> Self {
        StrideLayout {
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
    pub fn with_row_major(matrix: layout::Matrix) -> Self {
        StrideLayout {
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
    pub fn shrink_element(&mut self, new: layout::TexelLayout) {
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

impl Strides {
    /// Create a matrix with a specific layout.
    pub fn new(layout: StrideLayout) -> Self {
        Self::with_canvas(Canvas::new(layout))
    }

    /// Construct from a canvas.
    ///
    /// This will assert that the bytes reserved by the canvas correspond to the layout. This
    /// should already be the case but `Canvas` does not require it.
    pub fn with_canvas(inner: Canvas<StrideLayout>) -> Self {
        let layout = inner.layout();
        assert!(
            inner.as_bytes().get(..layout.total).is_some(),
            "Contract violation, canvas smaller than required by layout"
        );
        Strides { inner }
    }

    /// Shrink the element's size or alignment.
    ///
    /// This operation never reallocates the buffer.
    pub fn shrink_element(&mut self, new: layout::TexelLayout) {
        self.inner.layout_mut_unguarded().shrink_element(new)
    }

    /// Borrow this as a reference to an immutable byte matrix.
    pub fn as_ref(&self) -> ByteCanvasRef<'_> {
        ByteCanvasRef {
            layout: *self.inner.layout(),
            data: self.inner.as_bytes(),
        }
    }

    /// Borrow this as a reference to a mutable byte matrix.
    pub fn as_mut(&mut self) -> ByteCanvasMut<'_> {
        ByteCanvasMut {
            layout: *self.inner.layout(),
            data: self.inner.as_bytes_mut(),
        }
    }
}

/// Unwrap the inner matrix.
///
/// This drops the strong assertion that the matrix buffer corresponds to the correct layout but
/// allows reuse for a potentially unrelated layout.
impl From<Strides> for Canvas<StrideLayout> {
    fn from(strides: Strides) -> Canvas<StrideLayout> {
        strides.inner
    }
}

impl<'data> ByteCanvasRef<'data> {
    /// Construct a reference to a strided canvas buffer.
    pub fn new(canvas: &'data Canvas<impl Strided>) -> Self {
        let layout = canvas.layout().strided();
        let data = &canvas.as_bytes()[..layout.total];
        ByteCanvasRef { layout, data }
    }

    /// Shrink the element's size or alignment.
    ///
    /// This operation never reallocates the buffer.
    pub fn shrink_element(&mut self, new: layout::TexelLayout) {
        self.layout.shrink_element(new)
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
    /// Construct a mutable reference to a strided canvas buffer.
    pub fn new(canvas: &'data mut Canvas<impl Strided>) -> Self {
        let layout = canvas.layout().strided();
        let data = &mut canvas.as_bytes_mut()[..layout.total];
        ByteCanvasMut { layout, data }
    }

    /// Shrink the element's size or alignment.
    ///
    /// This operation never reallocates the buffer.
    pub fn shrink_element(&mut self, new: layout::TexelLayout) {
        self.layout.shrink_element(new)
    }

    /// Copy the bytes from another canvas.
    ///
    /// The source must have the same width, height, and element size.
    pub fn copy_from_canvas(&mut self, source: ByteCanvasRef<'_>) {
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

/// Describes a rectangular matrix of pixels.
pub trait Strided: Layout {
    fn strided(&self) -> StrideLayout;
}

impl Layout for StrideLayout {
    fn byte_len(&self) -> usize {
        self.total
    }
}

impl Strided for StrideLayout {
    fn strided(&self) -> StrideLayout {
        *self
    }
}

impl<P: AsTexel> Strided for matrix::Layout<P> {
    fn strided(&self) -> StrideLayout {
        let matrix = layout::Matrix::from_width_height(
            layout::TexelLayout::from_pixel::<P>(),
            self.width(),
            self.height(),
        );
        let matrix = matrix.expect("Fits into memory");
        StrideLayout::with_row_major(matrix)
    }
}

impl From<BadStrideKind> for BadStrideError {
    fn from(kind: BadStrideKind) -> Self {
        BadStrideError { kind }
    }
}

#[test]
fn align_validation() {
    // Setup a good base specification.
    let matrix = layout::Matrix::from_width_height(layout::TexelLayout::from_pixel::<u16>(), 2, 2)
        .expect("Valid matrix");
    let layout = StrideLayout::with_row_major(matrix);

    let bad_offset = StrideSpec {
        offset: 1,
        ..layout.spec
    };
    assert!(StrideLayout::new(bad_offset).is_err());
    let bad_pitch = StrideSpec {
        width_stride: 5,
        ..layout.spec
    };
    assert!(StrideLayout::new(bad_pitch).is_err());
}

#[test]
fn canvas_copies() {
    let matrix = layout::Matrix::from_width_height(layout::TexelLayout::from_pixel::<u8>(), 2, 2)
        .expect("Valid matrix");
    let row_layout = StrideLayout::with_row_major(matrix);
    let col_layout = StrideLayout::with_column_major(matrix);

    let src = Canvas::with_bytes(row_layout, &[0u8, 1, 2, 3]);

    let mut dst = Canvas::new(row_layout);
    ByteCanvasMut::new(&mut dst).copy_from_canvas(ByteCanvasRef::new(&src));
    assert_eq!(dst.as_bytes(), &[0u8, 1, 2, 3], "Still in same order");

    let mut dst = Canvas::new(col_layout);
    ByteCanvasMut::new(&mut dst).copy_from_canvas(ByteCanvasRef::new(&src));
    assert_eq!(
        dst.as_bytes(),
        &[0u8, 2, 1, 3],
        "In transposed matrix order"
    );
}
