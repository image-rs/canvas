use core::ops::{Deref, DerefMut};

use crate::Pixel;
use crate::buf::Buf;
use zerocopy::{AsBytes, FromBytes};

/// A **r**einterpretable v**ec**tor for an array of pixels.
///
/// It allows efficient conversion to other pixel representations, that is effective
/// reinterpretation casts.
pub struct Rec<P: AsBytes + FromBytes> {
    inner: Buf,
    pixel: Pixel<P>,
}

impl<P: AsBytes + FromBytes> Rec<P> {
    /// Allocate a pixel buffer by the pixel count.
    pub fn new(pixel: Pixel<P>, count: usize) -> Self {
        let mem_size = pixel.size().checked_mul(count).unwrap_or_else(
            || panic!("Requested size overflow"));
        Self::new_bytes(pixel, mem_size)
    }

    /// Allocate a pixel buffer by providing the byte count you wish to allocate.
    pub fn new_bytes(pixel: Pixel<P>, mem_size: usize) -> Self {
        Rec {
            inner: Buf::new(mem_size),
            pixel,
        }
    }

    pub fn as_slice(&self) -> &[P] {
        self.inner.as_pixels(self.pixel)
    }

    pub fn as_mut_slice(&mut self) -> &mut [P] {
        self.inner.as_mut_pixels(self.pixel)
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// Reinterpret the buffer for a different type of pixel.
    ///
    /// Note that this may leave some of the underlying pixels unaccessible if the new type is
    /// larger than the old one and the allocation was not a multiple of the new size. Conversely,
    /// some new bytes may become accessible if the memory length was not a multiple of the
    /// previous pixel type's length.
    pub fn reinterpret<Q>(self, pixel: Pixel<Q>) -> Rec<Q>
        where Q: AsBytes + FromBytes
    {
        Rec {
            inner: self.inner,
            pixel,
        }
    }
}

impl<P: AsBytes + FromBytes> Deref for Rec<P> {
    type Target = [P];

    fn deref(&self) -> &[P] {
        self.as_slice()
    }
}

impl<P: AsBytes + FromBytes> DerefMut for Rec<P> {
    fn deref_mut(&mut self) -> &mut [P] {
        self.as_mut_slice()
    }
}
