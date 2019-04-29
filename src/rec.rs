// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::ops::{Deref, DerefMut};

use crate::{AsPixel, Pixel};
use crate::buf::Buffer;
use zerocopy::{AsBytes, FromBytes};

/// A **r**einterpretable v**ec**tor for an array of pixels.
///
/// It allows efficient conversion to other pixel representations, that is effective
/// reinterpretation casts.
pub struct Rec<P: AsBytes + FromBytes> {
    inner: Buffer,
    pixel: Pixel<P>,
}

impl<P: AsBytes + FromBytes> Rec<P> {
    /// Allocate a pixel buffer by the pixel count.
    pub fn new(count: usize) -> Self where P: AsPixel {
        Self::new_for_pixel(P::pixel(), count)
    }

    /// Allocate a pixel buffer by the pixel count.
    ///
    /// Provides the opportunity to construct the pixel argument via other means than the trait,
    /// for example a dynamically checked expression.
    pub fn new_for_pixel(pixel: Pixel<P>, count: usize) -> Self {
        Self::bytes_for_pixel(pixel, mem_size(pixel, count))
    }

    /// Allocate a pixel buffer by providing the byte count you wish to allocate.
    pub fn bytes_for_pixel(pixel: Pixel<P>, mem_size: usize) -> Self {
        Rec {
            inner: Buffer::new(mem_size),
            pixel,
        }
    }

    /// Change the number of pixels.
    ///
    /// This will always reallocate the buffer if the size exceeds the current capacity.
    pub fn resize(&mut self, count: usize) {
        self.resize_bytes(mem_size(self.pixel, count))
    }

    /// Change the size in bytes.
    ///
    /// The length is afterwards equal to `bytes / mem::size_of::<P>()`, i.e. the quotient rounded
    /// down.
    ///
    /// This will always reallocate the buffer if the size exceeds the current capacity.
    pub fn resize_bytes(&mut self, bytes: usize) {
        self.inner.resize(bytes)
    }

    /// Reallocate the slice to contain exactly as many bytes as necessary.
    ///
    /// The number of contained elements is not changed. However, the number of elements
    /// interpreted as a different type may change.
    ///
    /// ```
    /// # use canvas::Rec;
    /// let rec_u8 = Rec::<u8>::new(7);
    /// assert_eq!(rec_u8.len(), 7);
    ///
    /// let mut rec_u32 = rec_u8.reinterpret::<u32>();
    /// assert_eq!(rec_u32.len(), 1);
    /// rec_u32.shrink_to_fit();
    ///
    /// let rec_u8 = rec_u32.reinterpret::<u8>();
    /// assert_eq!(rec_u8.len(), 4);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        let exact_size = mem_size(self.pixel, self.len());
        self.resize_bytes(exact_size);
        self.inner.shrink_to_fit();
    }

    pub fn as_slice(&self) -> &[P] {
        self.inner.as_pixels(self.pixel)
    }

    pub fn as_mut_slice(&mut self) -> &mut [P] {
        self.inner.as_mut_pixels(self.pixel)
    }

    /// The number of accessible elements for the current type.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// The number of elements that can fit without reallocation.
    pub fn capacity(&self) -> usize {
        self.inner.capacity() / self.pixel.size()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// The total number of managed bytes.
    ///
    /// This will not change even through a reinterpretation casts. This corresponds to the
    /// capacity of the storage.
    pub fn byte_len(&self) -> usize {
        self.as_bytes().len()
    }

    /// The total number of managable bytes.
    pub fn byte_capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Reinterpret the buffer for a different type of pixel.
    ///
    /// See `reinterpret_to` for details.
    pub fn reinterpret<Q>(self) -> Rec<Q>
        where Q: AsPixel + AsBytes + FromBytes
    {
        self.reinterpret_to(Q::pixel())
    }

    /// Reinterpret the buffer for a different type of pixel.
    ///
    /// Note that this may leave some of the underlying pixels unaccessible if the new type is
    /// larger than the old one and the allocation was not a multiple of the new size. Conversely,
    /// some new bytes may become accessible if the memory length was not a multiple of the
    /// previous pixel type's length.
    pub fn reinterpret_to<Q>(self, pixel: Pixel<Q>) -> Rec<Q>
        where Q: AsBytes + FromBytes
    {
        Rec {
            inner: self.inner,
            pixel,
        }
    }
}

fn mem_size<P>(pixel: Pixel<P>, count: usize) -> usize {
    pixel.size().checked_mul(count).unwrap_or_else(
        || panic!("Requested count overflows memory size"))
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
