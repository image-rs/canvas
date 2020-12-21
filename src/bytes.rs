//! Byte-based operations on a canvas.
use crate::canvas::Canvas;
use crate::layout::{self, Layout};
use core::{convert::TryFrom, ops::Range};

/// A simple layout describing some pixels as a byte matrix.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ByteMatrixSpec {
    /// The number of pixels in width direction.
    pub width: u32,
    /// The number of pixels in height direction.
    pub height: u32,
    /// The number of bytes of a single pixel.
    ///
    /// If this differs from both `w_stride` and `h_stride` the any copy must loop over individual
    /// pixels. Otherwise, whole rows or columns of contiguous data may be inspected.
    pub elsize: usize,
    /// The number of bytes to go one pixel along the width.
    pub w_stride: usize,
    /// The number of bytes to go one pixel along the height.
    pub h_stride: usize,
    /// Offset of this matrix from the start.
    pub offset: usize,
}

/// A validated layout of a rectangular matrix of pixels, treated as bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ByteMatrix {
    spec: ByteMatrixSpec,
    /// The total number of bytes, as proof of calculation basically.
    total: usize,
}

/// An untyped matrix of pixels treated as pure bytes.
pub struct ByteCanvas {
    inner: Canvas<ByteMatrix>,
}

pub struct ByteCanvasRef<'data> {
    layout: ByteMatrix,
    data: &'data [u8],
}

pub struct ByteCanvasMut<'data> {
    layout: ByteMatrix,
    data: &'data mut [u8],
}

impl ByteMatrixSpec {
    /// Compare sizes without taking into account the offset or strides.
    fn matches(&self, other: &Self) -> bool {
        self.elsize == other.elsize && self.width == other.width && self.height == other.height
    }

    fn has_contiguous_rows(&self) -> bool {
        self.elsize == self.w_stride
    }

    fn has_contiguous_cols(&self) -> bool {
        self.elsize == self.h_stride
    }

    fn element_start(&self, row: u32, col: u32) -> usize {
        (row as usize * self.h_stride) + (col as usize * self.w_stride) + self.offset
    }

    fn element(&self, row: u32, col: u32) -> Range<usize> {
        let start = self.element_start(row, col);
        start..start + self.elsize
    }

    fn contiguous_row(&self, row: u32) -> Range<usize> {
        let start = self.element_start(row, 0);
        let length = self.width as usize * self.elsize;
        start..start + length
    }

    fn contiguous_col(&self, col: u32) -> Range<usize> {
        let start = self.element_start(0, col);
        let length = self.height as usize * self.elsize;
        start..start + length
    }
}

impl ByteMatrix {
    pub fn new(spec: ByteMatrixSpec) -> Option<Self> {
        let relative_past_end = if spec.height > 0 && spec.width > 0 {
            let max_w = usize::try_from(spec.width - 1).ok()?;
            let max_h = usize::try_from(spec.height - 1).ok()?;

            let max_w_offset = max_w.checked_mul(spec.w_stride)?;
            let max_h_offset = max_h.checked_mul(spec.h_stride)?;

            spec.elsize
                .checked_add(max_h_offset)?
                .checked_add(max_w_offset)?
        } else {
            0
        };

        // We wouldn't need to validated if there are no elements. However, this is basically the
        // caller's responsibility. It's more consistent if we keep the offset. For future
        // additions such as calculating free space (?) this would also be required.
        let total = relative_past_end.checked_add(spec.offset)?;

        Some(ByteMatrix { spec, total })
    }

    /// Construct from a packed matrix of elements in column major layout.
    pub fn with_column_major(matrix: layout::Matrix) -> Option<Self> {
        Self::new(ByteMatrixSpec {
            elsize: matrix.element().size(),
            width: u32::try_from(matrix.width()).ok()?,
            height: u32::try_from(matrix.height()).ok()?,
            h_stride: matrix.element().size(),
            // Overflow can't happen because all of `matrix` fits in memory according to its own
            // internal invariant.
            w_stride: matrix.height() * matrix.element().size(),
            offset: 0,
        })
    }

    /// Construct from a packed matrix of elements in row major layout.
    pub fn with_row_major(matrix: layout::Matrix) -> Option<Self> {
        Self::new(ByteMatrixSpec {
            elsize: matrix.element().size(),
            width: u32::try_from(matrix.width()).ok()?,
            height: u32::try_from(matrix.height()).ok()?,
            // Overflow can't happen because all of `matrix` fits in memory according to its own
            // internal invariant.
            h_stride: matrix.width() * matrix.element().size(),
            w_stride: matrix.element().size(),
            offset: 0,
        })
    }

    /// Get the specification of this matrix.
    pub fn spec(&self) -> ByteMatrixSpec {
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

    fn pixel(&self, x: u32, y: u32) -> Range<usize> {
        self.spec.element(x, y)
    }
}

impl ByteCanvas {
    pub fn new(inner: Canvas<ByteMatrix>) -> Self {
        let layout = inner.layout();
        assert!(
            inner.as_bytes().get(..layout.total).is_some(),
            "Contract violation, canvas smaller than required by layout"
        );
        ByteCanvas { inner }
    }
}

impl<'data> ByteCanvasRef<'data> {
    pub fn new(canvas: &'data Canvas<impl Rectangular>) -> Self {
        let layout = canvas.layout().rectangular();
        let data = &canvas.as_bytes()[..layout.total];
        ByteCanvasRef { layout, data }
    }
}

impl<'data> ByteCanvasMut<'data> {
    pub fn new(canvas: &'data mut Canvas<impl Rectangular>) -> Self {
        let layout = canvas.layout().rectangular();
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
pub trait Rectangular: Layout {
    fn rectangular(&self) -> ByteMatrix;
}

impl Layout for ByteMatrix {
    fn byte_len(&self) -> usize {
        self.total
    }
}

impl Rectangular for ByteMatrix {
    fn rectangular(&self) -> ByteMatrix {
        *self
    }
}

#[test]
fn canvas_copies() {
    let matrix = layout::Matrix::from_width_height(layout::Element::from_pixel::<u8>(), 2, 2)
        .expect("Valid matrix");
    let row_layout = ByteMatrix::with_row_major(matrix).expect("Valid layout");
    let col_layout = ByteMatrix::with_column_major(matrix).expect("Valid layout");

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
