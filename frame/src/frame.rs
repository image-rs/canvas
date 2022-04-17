//! A byte-buffer based image descriptor.
use canvas::canvas::{CanvasMut, CanvasRef};
use canvas::layout::{Layout as CanvasLayout, Matrix, MatrixBytes, MatrixLayout, Raster};

use crate::layout::{ByteLayout, Layout, SampleBits, Texel};

/// A byte buffer with dynamic color contents.
pub struct Frame {
    inner: canvas::Canvas<Layout>,
}

/// Represents a single matrix like layer of an image.
pub struct LayerRef<'data, T> {
    inner: CanvasRef<'data, PlanarLayout<T>>,
}

/// Represents a single mutable matrix like layer of an image.
pub struct LayerMut<'data, T> {
    inner: CanvasMut<'data, PlanarLayout<T>>,
}

/// Denotes the type we pass to the color decoder.
///
/// This is an internal type due to the type assigned to each color being an implementation detail.
/// Consider that rgb565 can be passed as u16 or a special wrapper type for example. Or that
/// `[f16; 2]` can be a `u32` or a `[u16; 2]` or a wrapper. Until there's indication that this
/// needs stabilization it's kept secret.
///
/// For a valid layout it also fits to the indicated color components. There may be more than one
/// pixel in each texel.
#[derive(Clone, Copy)]
pub enum TexelKind {
    U8,
    U8x2,
    U8x3,
    U8x4,
    U16,
    U16x2,
    U16x3,
    U16x4,
    F32,
    F32x2,
    F32x3,
    F32x4,
}

pub struct SampleLayout {
    pub channels: u8,
    pub channel_stride: usize,
    pub height: u32,
    pub height_stride: usize,
    pub width: u32,
    pub width_stride: usize,
}

pub struct PlanarBytes {
    pub offset: usize,
    pub matrix: MatrixBytes,
}

pub struct PlanarLayout<T> {
    pub offset: usize,
    pub matrix: Matrix<T>,
}

impl Layout {
    // Verify that the byte-length is below `isize::MAX`.
    fn validate(this: Self) -> Option<Self> {
        let lines = usize::try_from(this.bytes.width).ok()?;
        let height = usize::try_from(this.bytes.height).ok()?;
        let ok = height
            .checked_mul(lines)
            .map_or(false, |len| len < isize::MAX as usize);
        Some(this).filter(|_| ok)
    }

    /// A layout representing scanlines of a colored pixel.
    ///
    /// The texel is chosen based on color. Returns `None` if the number of bytes used by the
    /// layout is larger than `usize::MAX`, i.e. the layout can not be allocated.
    pub fn for_scanlines(texel: TexelKind, width: u32, height: u32) -> Option<Self> {
        Self::validate(Layout { ..todo!() })
    }

    /// Returns the width of the underlying image in pixels.
    pub fn width(&self) -> u32 {
        self.bytes.width
    }

    /// Returns the height of the underlying image in pixels.
    pub fn height(&self) -> u32 {
        self.bytes.height
    }

    fn flat_layout(&self) -> Option<SampleLayout> {
        Some(SampleLayout {
            channels: todo!(),
            ..todo!()
        })
    }
}

impl Frame {
    /// Create an empty image that will use the indicated texels.
    ///
    /// This will _not_ allocate.
    pub fn empty(texel: TexelKind) -> Self {
        Frame {
            inner: canvas::Canvas::new(todo!()),
        }
    }

    /// Get a reference to the layout of this frame.
    pub fn layout(&self) -> &Layout {
        self.inner.layout()
    }

    /// Overwrite the layout, allocate if necessary, and clear the image.
    pub fn set_layout(&mut self, layout: Layout) {
        self.set_layout_conservative(layout);
        self.inner.as_bytes_mut().fill(0);
    }

    /// Overwrite the layout, allocate if necessary, _do not_ clear the image.
    pub fn set_layout_conservative(&mut self, layout: Layout) {
        *self.inner.layout_mut_unguarded() = layout;
        self.inner.ensure_layout();
    }

    /// Return this image's pixels as a native endian byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Return this image's pixels as a mutable native endian byte slice.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// Get the matrix-like sample descriptor if the samples are `u8`.
    pub fn as_flat_samples_u8(&self) -> Option<LayerRef<u8>> {
        None
    }

    /// Get the matrix-like sample descriptor if the samples are `u16`.
    pub fn as_flat_samples_u16(&self) -> Option<LayerRef<u16>> {
        None
    }

    /// Get the matrix-like sample descriptor if the samples are `f32`.
    pub fn as_flat_samples_f32(&self) -> Option<LayerRef<f32>> {
        None
    }

    /// Return this image's pixels as a byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        // We can not reuse the allocation of `canvas`.
        self.as_bytes().to_owned()
    }

    pub(crate) fn as_ref(&self) -> CanvasRef<'_, &'_ Layout> {
        self.inner.as_ref()
    }

    pub(crate) fn as_mut(&mut self) -> CanvasMut<'_, &'_ mut Layout> {
        self.inner.as_mut()
    }
}

impl TexelKind {
    fn byte_len(&self) -> usize {
        use TexelKind::*;
        match self {
            U8 => 1,
            U8x2 => 2,
            U8x3 => 3,
            U8x4 => 4,
            U16 => 2,
            U16x2 => 4,
            U16x3 => 6,
            U16x4 => 8,
            F32 => 4,
            F32x2 => 8,
            F32x3 => 12,
            F32x4 => 16,
        }
    }
}

impl From<Texel> for TexelKind {
    fn from(texel: Texel) -> Self {
        Self::from(texel.bits)
    }
}

impl From<SampleBits> for TexelKind {
    fn from(bits: SampleBits) -> Self {
        match bits {
            SampleBits::Int8 | SampleBits::Int332 | SampleBits::Int233 => TexelKind::U8,
            SampleBits::Int16
            | SampleBits::Int4x4
            | SampleBits::Int_444
            | SampleBits::Int444_
            | SampleBits::Int565 => TexelKind::U16,
            SampleBits::Int8x2 => TexelKind::U8x2,
            SampleBits::Int8x3 => TexelKind::U8x3,
            SampleBits::Int8x4 => TexelKind::U8x4,
            SampleBits::Int16x2 => TexelKind::U16x2,
            SampleBits::Int16x3 => TexelKind::U16x3,
            SampleBits::Int16x4 => TexelKind::U16x4,
            SampleBits::Int1010102
            | SampleBits::Int2101010
            | SampleBits::Int101010_
            | SampleBits::Int_101010 => TexelKind::U16x2,
            SampleBits::Float16x4 => TexelKind::U16x4,
            SampleBits::Float32x4 => TexelKind::F32x3,
        }
    }
}
