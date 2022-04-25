//! A byte-buffer based image descriptor.
use canvas::canvas::{CanvasMut, CanvasRef};

use crate::layout::{
    ByteLayout, ChannelLayout, FrameLayout, PlanarBytes, PlanarLayout, SampleBits, Texel,
};

/// A byte buffer with dynamic color contents.
pub struct Frame {
    inner: canvas::Canvas<FrameLayout>,
}

/// A byte buffer containing a single plane.
pub struct Plane {
    inner: canvas::Canvas<FrameLayout>,
}

/// Represents a single matrix like layer of an image.
pub struct PlaneBytes<'data> {
    inner: CanvasRef<'data, PlanarBytes>,
}

/// Represents a single matrix like layer of an image.
pub struct PlaneBytesMut<'data> {
    inner: CanvasMut<'data, PlanarBytes>,
}

/// Represents a single matrix like layer of an image.
pub struct PlaneRef<'data, T> {
    inner: CanvasRef<'data, PlanarLayout<T>>,
}

/// Represents a single mutable matrix like layer of an image.
pub struct PlaneMut<'data, T> {
    inner: CanvasMut<'data, PlanarLayout<T>>,
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
    pub fn as_flat_samples_u8(&self) -> Option<PlaneRef<u8>> {
        None
    }

    /// Get the matrix-like sample descriptor if the samples are `u16`.
    pub fn as_flat_samples_u16(&self) -> Option<PlaneRef<u16>> {
        None
    }

    /// Get the matrix-like sample descriptor if the samples are `f32`.
    pub fn as_flat_samples_f32(&self) -> Option<PlaneRef<f32>> {
        None
    }

    /// Return this image's pixels as a byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        // We can not reuse the allocation of `canvas`.
        self.as_bytes().to_owned()
    }

    pub fn plane(&self, idx: usize) -> Option<PlaneBytes<'_>> {
        let layout = self.layout().plane(idx)?;
        Some(PlaneBytes {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    pub fn plane_mut(&mut self, idx: usize) -> Option<PlaneBytesMut<'_>> {
        let layout = self.layout().plane(idx)?;
        Some(PlaneBytesMut {
            inner: self.inner.as_mut().with_layout(layout)?,
        })
    }

    pub(crate) fn as_ref(&self) -> CanvasRef<'_, &'_ FrameLayout> {
        self.inner.as_ref()
    }

    pub(crate) fn as_mut(&mut self) -> CanvasMut<'_, &'_ mut FrameLayout> {
        self.inner.as_mut()
    }
}

impl<'data> PlaneBytes<'data> {
    /// Upgrade to a view with strongly typed texel type.
    pub fn as_texels<T>(self, texel: canvas::Texel<T>) -> Option<PlaneRef<'data, T>> {
        if let Some(layout) = self.inner.layout().is_compatible(texel) {
            Some(PlaneRef {
                inner: self.inner.with_layout(layout).unwrap(),
            })
        } else {
            None
        }
    }
}

impl<'data> PlaneBytesMut<'data> {
    /// Upgrade to a view with strongly typed texel type.
    pub fn as_texels<T>(self, texel: canvas::Texel<T>) -> Option<PlaneRef<'data, T>> {
        if let Some(layout) = self.inner.layout().is_compatible(texel) {
            Some(PlaneRef {
                inner: self.inner.into_ref().with_layout(layout).unwrap(),
            })
        } else {
            None
        }
    }

    /// Upgrade to a mutable view with strongly typed texel type.
    pub fn as_mut_texels<T>(self, texel: canvas::Texel<T>) -> Option<PlaneMut<'data, T>> {
        if let Some(layout) = self.inner.layout().is_compatible(texel) {
            Some(PlaneMut {
                inner: self.inner.with_layout(layout).unwrap(),
            })
        } else {
            None
        }
    }
}

impl<'data, T> From<PlaneRef<'data, T>> for PlaneBytes<'data> {
    fn from(plane: PlaneRef<'data, T>) -> Self {
        PlaneBytes {
            inner: plane.inner.decay().unwrap(),
        }
    }
}

impl<'data, T> From<PlaneMut<'data, T>> for PlaneBytesMut<'data> {
    fn from(plane: PlaneMut<'data, T>) -> Self {
        PlaneBytesMut {
            inner: plane.inner.decay().unwrap(),
        }
    }
}
