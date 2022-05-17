//! A byte-buffer based image descriptor.
use canvas::canvas::{CanvasMut, CanvasRef};

use crate::color::Color;
use crate::layout::{CanvasLayout, ChannelLayout, LayoutError, PlanarLayout, PlaneBytes};
use crate::shader::Converter;

/// A byte buffer with dynamic color contents.
pub struct Canvas {
    inner: canvas::Canvas<CanvasLayout>,
}

/// A byte buffer containing a single plane.
pub struct Plane {
    inner: canvas::Canvas<CanvasLayout>,
}

/// Represents a single matrix like layer of an image.
pub struct BytePlane<'data> {
    inner: CanvasRef<'data, PlaneBytes>,
}

/// Represents a single matrix like layer of an image.
pub struct BytePlaneMut<'data> {
    inner: CanvasMut<'data, PlaneBytes>,
}

/// Represents a single matrix like layer of an image.
pub struct PlaneRef<'data, T> {
    inner: CanvasRef<'data, PlanarLayout<T>>,
}

/// Represents a single mutable matrix like layer of an image.
pub struct PlaneMut<'data, T> {
    inner: CanvasMut<'data, PlanarLayout<T>>,
}

/// Represent a single matrix with uniform channel type.
pub struct ChannelsRef<'data, T> {
    inner: CanvasRef<'data, ChannelLayout<T>>,
}

/// Represent a single matrix with uniform channel type.
pub struct ChannelsMut<'data, T> {
    inner: CanvasMut<'data, ChannelLayout<T>>,
}

impl Canvas {
    /// Create a frame by its layout.
    ///
    /// # Usage
    ///
    /// ```
    /// use image_canvas::{Canvas, CanvasLayout, SampleParts, Texel};
    ///
    /// // Define what type of color we want to store...
    /// let texel = Texel::new_u8(SampleParts::RgbA);
    /// // and which dimensions to use, chooses a stride for us.
    /// let layout = CanvasLayout::with_texel(&texel, 32, 32)?;
    ///
    /// let frame = Canvas::new(layout);
    /// # use image_canvas::LayoutError;
    /// # Ok::<(), LayoutError>(())
    /// ```
    pub fn new(layout: CanvasLayout) -> Self {
        Canvas {
            inner: canvas::Canvas::new(layout),
        }
    }

    /// Get a reference to the layout of this frame.
    pub fn layout(&self) -> &CanvasLayout {
        self.inner.layout()
    }

    /// Overwrite the layout, allocate if necessary, and clear the image.
    pub fn set_layout(&mut self, layout: CanvasLayout) {
        self.set_layout_conservative(layout);
        self.inner.as_bytes_mut().fill(0);
    }

    /// Overwrite the layout, allocate if necessary, _do not_ clear the image.
    pub fn set_layout_conservative(&mut self, layout: CanvasLayout) {
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

    /// Return the bytes making up this image as a slice of arbitrary elements.
    pub fn as_texels<T>(&self, texel: canvas::Texel<T>) -> &[T] {
        self.inner.as_texels(texel)
    }

    /// Return the bytes making up this image as a slice of arbitrary elements.
    pub fn as_texels_mut<T>(&mut self, texel: canvas::Texel<T>) -> &mut [T] {
        self.inner.as_mut_texels(texel)
    }

    /// Get the matrix-like channel descriptor if the channels are `u8`.
    ///
    /// Returns `Some` only when texel is some multiple of `u8`.
    pub fn channels_u8(&self) -> Option<ChannelsRef<u8>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane
            .as_channel_bytes()?
            .is_compatible(<u8 as canvas::AsTexel>::texel())?;
        Some(ChannelsRef {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    /// Get the matrix-like channel descriptor if the channels are `u16`.
    ///
    /// Returns `Some` only when texel is some multiple of `u16`.
    pub fn channels_u16(&self) -> Option<ChannelsRef<u16>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane
            .as_channel_bytes()?
            .is_compatible(<u16 as canvas::AsTexel>::texel())?;
        Some(ChannelsRef {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    /// Get the matrix-like channel descriptor if the channels are `f32`.
    ///
    /// Returns `Some` only when texel is some multiple of `f32`.
    pub fn channels_f32(&self) -> Option<ChannelsRef<f32>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane
            .as_channel_bytes()?
            .is_compatible(<f32 as canvas::AsTexel>::texel())?;
        Some(ChannelsRef {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    /// Get the matrix-like channel descriptor if the channels are `u8`.
    ///
    /// Returns `Some` only when texel is some multiple of `u8`.
    pub fn channels_u8_mut(&mut self) -> Option<ChannelsMut<u8>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane
            .as_channel_bytes()?
            .is_compatible(<u8 as canvas::AsTexel>::texel())?;
        Some(ChannelsMut {
            inner: self.inner.as_mut().with_layout(layout)?,
        })
    }

    /// Get the matrix-like channel descriptor if the channels are `u16`.
    ///
    /// Returns `Some` only when texel is some multiple of `u16`.
    pub fn channels_u16_mut(&mut self) -> Option<ChannelsMut<u16>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane
            .as_channel_bytes()?
            .is_compatible(<u16 as canvas::AsTexel>::texel())?;
        Some(ChannelsMut {
            inner: self.inner.as_mut().with_layout(layout)?,
        })
    }

    /// Get the matrix-like channel descriptor if the channels are `f32`.
    ///
    /// Returns `Some` only when texel is some multiple of `f32`.
    pub fn channels_f32_mut(&mut self) -> Option<ChannelsMut<f32>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane
            .as_channel_bytes()?
            .is_compatible(<f32 as canvas::AsTexel>::texel())?;
        Some(ChannelsMut {
            inner: self.inner.as_mut().with_layout(layout)?,
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
    pub fn plane(&self, idx: u8) -> Option<BytePlane<'_>> {
        let layout = self.layout().plane(idx)?;
        Some(BytePlane {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    pub fn plane_mut(&mut self, idx: u8) -> Option<BytePlaneMut<'_>> {
        let layout = self.layout().plane(idx)?;
        Some(BytePlaneMut {
            inner: self.inner.as_mut().with_layout(layout)?,
        })
    }

    /// Set the color model.
    ///
    /// This never changes the physical layout of data and preserves all its bits. It returns an
    /// error if the new color is not compatible with the texel's color channels.
    pub fn set_color(&mut self, color: Color) -> Result<(), LayoutError> {
        self.inner.layout_mut_unguarded().set_color(color)
    }

    /// Write into another frame, converting color representation between.
    pub fn convert(&self, into: &mut Self) {
        Converter::new().run_on(self, into)
    }

    pub(crate) fn as_ref(&self) -> CanvasRef<'_, &'_ CanvasLayout> {
        self.inner.as_ref()
    }

    pub(crate) fn as_mut(&mut self) -> CanvasMut<'_, &'_ mut CanvasLayout> {
        self.inner.as_mut()
    }
}

impl<'data> BytePlane<'data> {
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

impl<'data> BytePlaneMut<'data> {
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

impl<'data, T> From<PlaneRef<'data, T>> for BytePlane<'data> {
    fn from(plane: PlaneRef<'data, T>) -> Self {
        BytePlane {
            inner: plane.inner.decay().unwrap(),
        }
    }
}

impl<'data, T> From<PlaneMut<'data, T>> for BytePlaneMut<'data> {
    fn from(plane: PlaneMut<'data, T>) -> Self {
        BytePlaneMut {
            inner: plane.inner.decay().unwrap(),
        }
    }
}
