//! A byte-buffer based image descriptor.
use canvas::canvas::{CanvasMut, CanvasRef};

use crate::layout::{ByteLayout, ChannelLayout, FrameLayout, PlanarBytes, PlanarLayout};

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
    /// Create a frame by its layout.
    ///
    /// # Usage
    ///
    /// ```
    /// use image_framebuf::{Frame, FrameLayout, SampleParts, Texel};
    ///
    /// // Define what type of color we want to store...
    /// let texel = Texel::new_u8(SampleParts::RgbA);
    /// // and which dimensions to use, chooses a stride for us.
    /// let layout = FrameLayout::with_texel(&texel, 32, 32)?;
    ///
    /// let frame = Frame::new(layout);
    /// # use image_framebuf::LayoutError;
    /// # Ok::<(), LayoutError>(())
    /// ```
    pub fn new(layout: FrameLayout) -> Self {
        Frame {
            inner: canvas::Canvas::new(layout),
        }
    }

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
    ///
    /// See also [`Self::as_bytes_mut`].
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Return this image's pixels as a mutable native endian byte slice.
    ///
    /// The exact interpretation of each byte depends on the layout, which also contains a
    /// descriptor of the color types used. This method is for purposes of foreign-function
    /// interfaces and as a fallback to process bytes regardless of any restrictions placed by the
    /// limits of what other methods are offered.
    ///
    /// Mind that the library __guarantees__ that the byte slice is actually aligned with an
    /// alignment larger than `1`. The exact details depend on the underlying platform but are at
    /// least `8`.
    ///
    /// Still, do __not__ write uninitialized bytes to the buffer. This is UB by Rust's semantics.
    /// (Writing into the buffer by syscalls or assembly most likely never counts as writing
    /// uninitialized bytes but take this with a grain of salt).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// Get the matrix-like sample descriptor if the samples are `u8`.
    ///
    /// FIXME: returns Some only when texel is exactly `u8` compatible. However, we'd rather match
    /// on the SampleBits and allow multiple channels?
    pub fn as_flat_samples_u8(&self) -> Option<PlaneRef<u8>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane.is_compatible(<u8 as canvas::AsTexel>::texel())?;
        Some(PlaneRef {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    /// Get the matrix-like sample descriptor if the samples are `u16`.
    ///
    /// FIXME: see as_flat_samples_u8.
    pub fn as_flat_samples_u16(&self) -> Option<PlaneRef<u16>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane.is_compatible(<u16 as canvas::AsTexel>::texel())?;
        Some(PlaneRef {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    /// Get the matrix-like sample descriptor if the samples are `f32`.
    ///
    /// FIXME: see as_flat_samples_f32.
    pub fn as_flat_samples_f32(&self) -> Option<PlaneRef<f32>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane.is_compatible(<f32 as canvas::AsTexel>::texel())?;
        Some(PlaneRef {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    /// Return this image's pixels as a byte vector.
    pub fn into_bytes(self) -> Vec<u8> {
        // We can not reuse the allocation of `canvas`.
        self.as_bytes().to_owned()
    }

    /// Get the untyped descriptor of the texel matrix.
    ///
    /// Returns `None` if the image contains data that can not be described as a single texel
    /// plane, e.g. multiple planes or if the plane is not a matrix.
    pub fn plane(&self, idx: u8) -> Option<PlaneBytes<'_>> {
        let layout = self.layout().plane(idx)?;
        Some(PlaneBytes {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    pub fn plane_mut(&mut self, idx: u8) -> Option<PlaneBytesMut<'_>> {
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
