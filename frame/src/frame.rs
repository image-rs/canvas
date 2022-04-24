//! A byte-buffer based image descriptor.
use canvas::canvas::{CanvasMut, CanvasRef};
use canvas::layout::{Layout as CanvasLayout, MatrixLayout, Raster};

use crate::layout::{
    ByteLayout, ChannelLayout, FrameLayout, PlanarBytes, PlanarLayout, SampleBits, Texel,
};

/// A byte buffer with dynamic color contents.
pub struct Frame {
    inner: canvas::Canvas<FrameLayout>,
}

/// Represents a single matrix like layer of an image.
pub struct LayerRef<'data, T> {
    inner: CanvasRef<'data, PlanarLayout<T>>,
}

/// Represents a single mutable matrix like layer of an image.
pub struct LayerMut<'data, T> {
    inner: CanvasMut<'data, PlanarLayout<T>>,
}

impl FrameLayout {
    // Verify that the byte-length is below `isize::MAX`.
    fn validate(this: Self) -> Option<Self> {
        let lines = usize::try_from(this.bytes.width).ok()?;
        let height = usize::try_from(this.bytes.height).ok()?;
        let ok = height
            .checked_mul(lines)
            .map_or(false, |len| len < isize::MAX as usize);
        Some(this).filter(|_| ok)
    }

    /// Returns the width of the underlying image in pixels.
    pub fn width(&self) -> u32 {
        self.bytes.width
    }

    /// Returns the height of the underlying image in pixels.
    pub fn height(&self) -> u32 {
        self.bytes.height
    }

    fn flat_layout(&self) -> Option<ChannelLayout> {
        Some(ChannelLayout {
            channels: todo!(),
            ..todo!()
        })
    }
}

impl Frame {
    /// Get a reference to the layout of this frame.
    pub fn layout(&self) -> &FrameLayout {
        self.inner.layout()
    }

    /// Overwrite the layout, allocate if necessary, and clear the image.
    pub fn set_layout(&mut self, layout: FrameLayout) {
        self.set_layout_conservative(layout);
        self.inner.as_bytes_mut().fill(0);
    }

    /// Overwrite the layout, allocate if necessary, _do not_ clear the image.
    pub fn set_layout_conservative(&mut self, layout: FrameLayout) {
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

    pub(crate) fn as_ref(&self) -> CanvasRef<'_, &'_ FrameLayout> {
        self.inner.as_ref()
    }

    pub(crate) fn as_mut(&mut self) -> CanvasMut<'_, &'_ mut FrameLayout> {
        self.inner.as_mut()
    }
}
