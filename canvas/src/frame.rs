//! A byte-buffer based image descriptor.

use alloc::borrow::ToOwned;
use alloc::vec::Vec;

use image_texel::image::{ImageMut, ImageRef};
use image_texel::Image;

use crate::color::Color;
use crate::layout::{CanvasLayout, ChannelLayout, LayoutError, PlanarLayout, PlaneBytes};
use crate::shader::Converter;

/// A byte buffer with dynamic color contents.
#[derive(Clone)]
pub struct Canvas {
    inner: Image<CanvasLayout>,
}

/// A byte buffer containing a single plane.
#[derive(Clone)]
pub struct Plane {
    inner: Image<PlaneBytes>,
}

#[expect(dead_code)]
#[doc(hidden)]
#[deprecated = "Use BytePlaneRef"]
pub type BytePlane<'data> = BytePlaneRef<'data>;

/// Represents a single matrix like layer of an image.
///
/// Created from [`Canvas::plane`].
pub struct BytePlaneRef<'data> {
    inner: ImageRef<'data, PlaneBytes>,
}

/// Represents a single matrix like layer of an image.
///
/// Created from [`Canvas::plane_mut`].
pub struct BytePlaneMut<'data> {
    inner: ImageMut<'data, PlaneBytes>,
}

/// Represents a single matrix like layer of an image.
///
/// Created from [`BytePlaneRef::as_texels`].
pub struct PlaneRef<'data, T> {
    inner: ImageRef<'data, PlanarLayout<T>>,
}

/// Represents a single mutable matrix like layer of an image.
///
/// Created from [`BytePlaneMut::as_texels`].
pub struct PlaneMut<'data, T> {
    inner: ImageMut<'data, PlanarLayout<T>>,
}

/// Represent a single matrix with uniform channel type.
///
/// Created from [`Canvas::channels_u8`] and related methods.
pub struct ChannelsRef<'data, C> {
    inner: ImageRef<'data, ChannelLayout<C>>,
}

/// Represent a single matrix with uniform channel type.
///
/// Created from [`Canvas::channels_u8_mut`] and related methods.
pub struct ChannelsMut<'data, C> {
    inner: ImageMut<'data, ChannelLayout<C>>,
}

impl Canvas {
    /// Create a frame by its layout.
    ///
    /// # Usage
    ///
    /// ```
    /// use image_canvas::Canvas;
    /// use image_canvas::layout::{CanvasLayout, SampleParts, Texel};
    ///
    /// // Define what type of color we want to store...
    /// let texel = Texel::new_u8(SampleParts::RgbA);
    /// // and which dimensions to use, chooses a stride for us.
    /// let layout = CanvasLayout::with_texel(&texel, 32, 32)?;
    ///
    /// let frame = Canvas::new(layout);
    /// # use image_canvas::layout::LayoutError;
    /// # Ok::<(), LayoutError>(())
    /// ```
    pub fn new(layout: CanvasLayout) -> Self {
        Canvas {
            inner: Image::new(layout),
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
    pub fn as_texels<T>(&self, texel: image_texel::Texel<T>) -> &[T] {
        self.inner.as_texels(texel)
    }

    /// Return the bytes making up this image as a slice of arbitrary elements.
    pub fn as_texels_mut<T>(&mut self, texel: image_texel::Texel<T>) -> &mut [T] {
        self.inner.as_mut_texels(texel)
    }

    /// Get the matrix-like channel descriptor if the channels are `u8`.
    ///
    /// Returns `Some` only when texel is some multiple of `u8`.
    pub fn channels_u8(&self) -> Option<ChannelsRef<u8>> {
        let plane = self.inner.layout().as_plane()?;
        let layout = plane
            .as_channel_bytes()?
            .is_compatible(<u8 as image_texel::AsTexel>::texel())?;
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
            .is_compatible(<u16 as image_texel::AsTexel>::texel())?;
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
            .is_compatible(<f32 as image_texel::AsTexel>::texel())?;
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
            .is_compatible(<u8 as image_texel::AsTexel>::texel())?;
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
            .is_compatible(<u16 as image_texel::AsTexel>::texel())?;
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
            .is_compatible(<f32 as image_texel::AsTexel>::texel())?;
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
    pub fn plane(&self, idx: u8) -> Option<BytePlaneRef<'_>> {
        let layout = self.layout().plane(idx)?;
        Some(BytePlaneRef {
            inner: self.inner.as_ref().with_layout(layout)?,
        })
    }

    /// Get references to multiple planes at the same time.
    pub fn planes<const N: usize>(&self) -> Option<[BytePlaneRef<'_>; N]> {
        todo!()
    }

    /// Get the untyped, mutable reference to the texel matrix.
    ///
    /// Returns `None` if the image contains data that can not be described as a single texel
    /// plane, e.g. multiple planes or if the plane is not a matrix.
    pub fn plane_mut(&mut self, idx: u8) -> Option<BytePlaneMut<'_>> {
        let layout = self.layout().plane(idx)?;
        Some(BytePlaneMut {
            inner: self.inner.as_mut().with_layout(layout)?,
        })
    }

    /// Get mutable references to multiple planes at the same time.
    ///
    /// This works because planes never overlap. Note that all planes are aligned to the same byte
    /// boundary as the complete canvas bytes.
    pub fn planes_mut<const N: usize>(&mut self) -> Option<[BytePlaneMut<'_>; N]> {
        use image_texel::layout::Bytes;

        let mut layouts = [(); N].map(|()| None);

        for i in 0..N {
            if i > u8::MAX as usize {
                return None;
            }

            layouts[i] = Some(self.layout().plane(i as u8)?);
        }

        let layouts = layouts.map(|layout| layout.unwrap());

        let mut offset = 0;
        // This frame's layout takes 0 bytes, so we can take all contents with split_layout
        let frame: ImageMut<'_, Bytes> = self.as_mut().decay();

        let &Bytes(total_len) = frame.layout();
        let mut frame = frame.with_layout(Bytes(0)).expect("zero-byte layout valid");

        let planes = layouts.map(move |mut layout| {
            layout.sub_offset(offset);
            let mut plane = frame
                .split_layout()
                .with_layout(layout)
                .expect("plane layout within frame");
            let tail = plane.split_layout();
            let &Bytes(tail_len) = tail.layout();
            // Put back all remaining bytes.
            frame = tail.with_layout(Bytes(0)).expect("zero-byte layout valid");
            offset = total_len - tail_len;

            Some(plane)
        });

        if planes.iter().any(|p| p.is_none()) {
            return None;
        }

        Some(planes.map(|p| BytePlaneMut { inner: p.unwrap() }))
    }

    /// Set the color model.
    ///
    /// This never changes the physical layout of data and preserves all its bits. It returns an
    /// error if the new color is not compatible with the texel's color channels.
    pub fn set_color(&mut self, color: Color) -> Result<(), LayoutError> {
        self.inner.layout_mut_unguarded().set_color(color)
    }

    pub(crate) fn as_ref(&self) -> ImageRef<'_, &'_ CanvasLayout> {
        self.inner.as_ref()
    }

    pub(crate) fn as_mut(&mut self) -> ImageMut<'_, &'_ mut CanvasLayout> {
        self.inner.as_mut()
    }
}

/// Conversion related methods.
impl Canvas {
    /// Write into another frame, converting color representation between.
    pub fn convert(&self, into: &mut Self) {
        Converter::new().run_on(self, into)
    }
}

impl<'data> BytePlaneRef<'data> {
    pub fn layout(&self) -> &PlaneBytes {
        self.inner.layout()
    }

    pub fn to_owned(&self) -> Plane {
        let plane = self.inner.layout();
        let bytes = self.inner.as_bytes();

        Plane {
            inner: Image::with_bytes(plane.clone(), bytes),
        }
    }

    pub fn to_canvas(&self) -> Canvas {
        let plane = self.inner.layout();
        let bytes = self.inner.as_bytes();

        Canvas {
            inner: Image::with_bytes(plane.into(), bytes),
        }
    }

    /// Upgrade to a view with strongly typed texel type.
    pub fn as_texels<T>(self, texel: image_texel::Texel<T>) -> Option<PlaneRef<'data, T>> {
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
    pub fn layout(&self) -> &PlaneBytes {
        self.inner.layout()
    }

    pub fn to_owned(&self) -> Plane {
        let plane = self.inner.layout();
        let bytes = self.inner.as_bytes();

        Plane {
            inner: Image::with_bytes(plane.clone(), bytes),
        }
    }

    pub fn to_canvas(&self) -> Canvas {
        let plane = self.inner.layout();
        let bytes = self.inner.as_bytes();

        Canvas {
            inner: Image::with_bytes(plane.into(), bytes),
        }
    }

    /// Upgrade to a view with strongly typed texel type.
    pub fn as_texels<T>(self, texel: image_texel::Texel<T>) -> Option<PlaneRef<'data, T>> {
        if let Some(layout) = self.inner.layout().is_compatible(texel) {
            Some(PlaneRef {
                inner: self.inner.into_ref().with_layout(layout).unwrap(),
            })
        } else {
            None
        }
    }

    /// Upgrade to a mutable view with strongly typed texel type.
    pub fn as_mut_texels<T>(self, texel: image_texel::Texel<T>) -> Option<PlaneMut<'data, T>> {
        if let Some(layout) = self.inner.layout().is_compatible(texel) {
            Some(PlaneMut {
                inner: self.inner.with_layout(layout).unwrap(),
            })
        } else {
            None
        }
    }
}

impl<'data, C> ChannelsRef<'data, C> {
    pub fn layout(&self) -> &ChannelLayout<C> {
        self.inner.layout()
    }

    pub fn as_slice(&self) -> &[C] {
        self.inner.as_slice()
    }

    pub fn into_slice(self) -> &'data [C] {
        self.inner.into_slice()
    }
}

impl<'data, C> ChannelsMut<'data, C> {
    pub fn layout(&self) -> &ChannelLayout<C> {
        self.inner.layout()
    }

    pub fn as_slice(&self) -> &[C] {
        self.inner.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [C] {
        self.inner.as_mut_slice()
    }

    pub fn into_slice(self) -> &'data [C] {
        self.inner.into_slice()
    }

    pub fn into_mut_slice(self) -> &'data mut [C] {
        self.inner.into_mut_slice()
    }
}

impl<'data, T> From<PlaneRef<'data, T>> for BytePlaneRef<'data> {
    fn from(plane: PlaneRef<'data, T>) -> Self {
        BytePlaneRef {
            inner: plane.inner.decay(),
        }
    }
}

impl<'data, T> From<PlaneMut<'data, T>> for BytePlaneMut<'data> {
    fn from(plane: PlaneMut<'data, T>) -> Self {
        BytePlaneMut {
            inner: plane.inner.decay(),
        }
    }
}

impl From<Plane> for Canvas {
    fn from(plane: Plane) -> Self {
        Canvas {
            inner: plane.inner.decay(),
        }
    }
}
