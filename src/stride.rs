//! Byte-based, stride operations on a canvas.
use crate::canvas::Canvas;
use crate::layout::Layout;
use crate::pixel::AsPixel;
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
    pub element_size: usize,
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
        self.element_size == other.element_size
            && self.width == other.width
            && self.height == other.height
    }

    fn has_contiguous_rows(&self) -> bool {
        self.element_size == self.width_stride
    }

    fn has_contiguous_cols(&self) -> bool {
        self.element_size == self.height_stride
    }

    fn element_start(&self, row: usize, col: usize) -> usize {
        (row * self.height_stride) + (col * self.width_stride) + self.offset
    }

    fn element(&self, row: usize, col: usize) -> Range<usize> {
        let start = self.element_start(row, col);
        start..start + self.element_size
    }

    fn contiguous_row(&self, row: usize) -> Range<usize> {
        let start = self.element_start(row, 0);
        let length = self.width * self.element_size;
        start..start + length
    }

    fn contiguous_col(&self, col: usize) -> Range<usize> {
        let start = self.element_start(0, col);
        let length = self.height * self.element_size;
        start..start + length
    }
}

impl StrideLayout {
    pub fn new(spec: StrideSpec) -> Option<Self> {
        let relative_past_end = if spec.height > 0 && spec.width > 0 {
            let max_w = spec.width - 1;
            let max_h = spec.height - 1;

            let max_w_offset = max_w.checked_mul(spec.width_stride)?;
            let max_h_offset = max_h.checked_mul(spec.height_stride)?;

            spec.element_size
                .checked_add(max_h_offset)?
                .checked_add(max_w_offset)?
        } else {
            0
        };

        // We wouldn't need to validated if there are no elements. However, this is basically the
        // caller's responsibility. It's more consistent if we keep the offset. For future
        // additions such as calculating free space (?) this would also be required.
        let total = relative_past_end.checked_add(spec.offset)?;

        Some(StrideLayout { spec, total })
    }

    /// Construct from a packed matrix of elements in column major layout.
    pub fn with_column_major(matrix: layout::Matrix) -> Self {
        StrideLayout {
            spec: StrideSpec {
                element_size: matrix.element().size(),
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
    pub fn with_row_major(matrix: layout::Matrix) -> Self {
        StrideLayout {
            spec: StrideSpec {
                element_size: matrix.element().size(),
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
    pub fn new(inner: Canvas<StrideLayout>) -> Self {
        let layout = inner.layout();
        assert!(
            inner.as_bytes().get(..layout.total).is_some(),
            "Contract violation, canvas smaller than required by layout"
        );
        Strides { inner }
    }
}

impl<'data> ByteCanvasRef<'data> {
    pub fn new(canvas: &'data Canvas<impl Strided>) -> Self {
        let layout = canvas.layout().strided();
        let data = &canvas.as_bytes()[..layout.total];
        ByteCanvasRef { layout, data }
    }
}

impl<'data> ByteCanvasMut<'data> {
    pub fn new(canvas: &'data mut Canvas<impl Strided>) -> Self {
        let layout = canvas.layout().strided();
        let data = &mut canvas.as_bytes_mut()[..layout.total];
        ByteCanvasMut { layout, data }
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

impl<P: AsPixel> Strided for matrix::Layout<P> {
    fn strided(&self) -> StrideLayout {
        let matrix = layout::Matrix::from_width_height(
            layout::Element::from_pixel::<P>(),
            self.width(),
            self.height(),
        );
        let matrix = matrix.expect("Fits into memory");
        StrideLayout::with_row_major(matrix)
    }
}

#[test]
fn canvas_copies() {
    let matrix = layout::Matrix::from_width_height(layout::Element::from_pixel::<u8>(), 2, 2)
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
