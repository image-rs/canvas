// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::cmp;
use core::fmt;
use core::ops::{Deref, DerefMut};

use bytemuck::Pod;

use crate::buf::{buf, Buffer};
use crate::{AsPixel, Pixel};

/// A **r**einterpretable v**ec**tor for an array of pixels.
///
/// It allows efficient conversion to other pixel representations, that is effective
/// reinterpretation casts.
pub struct Rec<P: Pod> {
    inner: Buffer,
    length: usize,
    pixel: Pixel<P>,
}

/// Error representation for a failed buffer reuse.
///
/// Indicates that the capacity of the underlying buffer is not large enough to perform the
/// operation without a reallocation. This may be either since the allocation is simply not large
/// enough or due to the requested length not having any representation in memory for the chosen
/// pixel type.
///
/// ```
/// # use canvas::Rec;
/// let mut rec = Rec::<u16>::new(16);
///
/// let err = match rec.reuse(rec.capacity() + 1) {
///     Ok(_) => unreachable!("Increasing capacity would require reallocation"),
///     Err(err) => err,
/// };
///
/// let err = match rec.reuse(usize::max_value()) {
///     Ok(_) => unreachable!("A slice of u16 can never have usize::MAX elements"),
///     Err(err) => err,
/// };
/// ```
pub struct ReuseError {
    requested: Option<usize>,
    capacity: usize,
}

impl<P: Pod> Rec<P> {
    /// Allocate a pixel buffer by the pixel count.
    ///
    /// # Panics
    ///
    /// This function will panic when the byte-length of the slice with the provided count would
    /// exceed the possible `usize` values. To avoid this, use `bytes_for_pixel` with manual
    /// calculation of the byte length instead.
    ///
    /// This function will also panic if the allocation fails.
    pub fn new(count: usize) -> Self
    where
        P: AsPixel,
    {
        Self::new_for_pixel(P::pixel(), count)
    }

    /// Allocate a pixel buffer by the pixel count.
    ///
    /// Provides the opportunity to construct the pixel argument via other means than the trait,
    /// for example a dynamically checked expression.
    ///
    /// # Panics
    ///
    /// This function will panic when the byte-length of the slice with the provided count would
    /// exceed the possible `usize` values. To avoid this, use `bytes_for_pixel` with manual
    /// calculation of the byte length instead.
    ///
    /// This function will also panic if the allocation fails.
    pub fn new_for_pixel(pixel: Pixel<P>, count: usize) -> Self {
        Self::bytes_for_pixel(pixel, mem_size(pixel, count))
    }

    /// Allocate a pixel buffer by providing the byte count you wish to allocate.
    ///
    /// # Panics
    ///
    /// This function will panic if the allocation fails.
    pub fn bytes_for_pixel(pixel: Pixel<P>, mem_size: usize) -> Self {
        Rec {
            inner: Buffer::new(mem_size),
            length: mem_size,
            pixel,
        }
    }

    /// Change the number of pixels.
    ///
    /// This will always reallocate the buffer if the size exceeds the current capacity.
    ///
    /// # Panics
    ///
    /// This function will panic when the byte-length of the slice with the provided count would
    /// exceed the possible `usize` values. To avoid this, use `resize_bytes` with manual
    /// calculation of the byte length instead.
    ///
    /// This function will also panic if an allocation is necessary but fails.
    pub fn resize(&mut self, count: usize) {
        self.resize_bytes(mem_size(self.pixel, count))
    }

    /// Change the size in bytes.
    ///
    /// The length is afterwards equal to `bytes / mem::size_of::<P>()`, i.e. the quotient rounded
    /// down.
    ///
    /// This will always reallocate the buffer if the size exceeds the current capacity.
    ///
    /// # Panics
    ///
    /// This function will panic if an allocation is necessary but fails.
    pub fn resize_bytes(&mut self, bytes: usize) {
        self.inner.grow_to(bytes);
        self.length = bytes;
    }

    /// Change the number of pixels without reallocation.
    ///
    /// Returns `Ok` when the resizing was successfully completed to the requested size and returns
    /// `Err` if this could not have been performed without a reallocation. This function will also
    /// never deallocate memory.
    ///
    /// ```
    /// # use canvas::Rec;
    /// // Initial allocation may panic due to allocation error for now.
    /// let mut buffer: Rec<u16> = Rec::new(100);
    /// buffer.reuse(0)
    ///     .expect("Requested size smaller than allocation");
    /// buffer.reuse(100)
    ///     .expect("The buffer didn't shrink from previous reuse");
    ///
    /// // Capacity may be larger than requested size at initialization.
    /// let capacity = buffer.capacity();
    /// buffer.reuse(capacity)
    ///     .expect("Set to full underlying allocation size.");
    /// ```
    pub fn reuse(&mut self, count: usize) -> Result<(), ReuseError> {
        let bytes = count
            .checked_mul(self.pixel.size())
            .ok_or_else(|| ReuseError {
                requested: None,
                capacity: self.byte_capacity(),
            })?;
        self.reuse_bytes(bytes)
    }

    /// Change the number of bytes without reallocation.
    ///
    /// Returns `Ok` when the resizing was successfully completed to the requested size and returns
    /// `Err` with the new byte size otherwise.
    pub fn reuse_bytes(&mut self, bytes: usize) -> Result<(), ReuseError> {
        if bytes > self.byte_capacity() {
            return Err(ReuseError {
                requested: Some(bytes),
                capacity: self.capacity(),
            });
        }

        // Resize within capacity will not reallocate, thus not panic.
        Ok(self.resize_bytes(bytes))
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
    ///
    /// # Panics
    ///
    /// This function will panic if the allocation fails.
    pub fn shrink_to_fit(&mut self) {
        let exact_size = mem_size(self.pixel, self.len());
        self.inner.resize_to(exact_size);
        self.length = exact_size;
    }

    pub fn as_slice(&self) -> &[P] {
        self.buf().as_pixels(self.pixel)
    }

    pub fn as_mut_slice(&mut self) -> &mut [P] {
        let pixel = self.pixel;
        self.buf_mut().as_mut_pixels(pixel)
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
        self.buf().as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.buf_mut().as_bytes_mut()
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
    where
        Q: AsPixel + Pod,
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
    where
        Q: Pod,
    {
        Rec {
            inner: self.inner,
            length: self.length,
            pixel,
        }
    }

    /// Map all elements to another value.
    ///
    /// See [`map_to`] for details.
    pub fn map<Q>(self, f: impl Fn(P) -> Q) -> Rec<Q>
    where
        P: Copy,
        Q: AsPixel + Pod,
    {
        self.map_to(f, Q::pixel())
    }

    /// Map elements to another value.
    ///
    /// This will keep the logical length of the `Rec` so that the number of pixels stays constant.
    /// If necessary, it will grow the internal buffer to achieve this.
    ///
    /// # Panics
    ///
    /// This function will panic if the allocation fails or the necessary allocation exceeds the
    /// value range of `usize`.
    pub fn map_to<Q>(mut self, f: impl Fn(P) -> Q, pixel: Pixel<Q>) -> Rec<Q>
    where
        P: Copy,
        Q: Pod,
    {
        // Ensure we have enough memory for both representations.
        let length = self.as_slice().len();
        let new_bytes = mem_size(pixel, length);
        self.inner.grow_to(new_bytes);
        self.inner.map_within(..length, 0, f, self.pixel, pixel);
        Rec {
            inner: self.inner,
            length: new_bytes,
            pixel,
        }
    }

    fn buf(&self) -> &buf {
        &self.inner[..self.length]
    }

    fn buf_mut(&mut self) -> &mut buf {
        &mut self.inner[..self.length]
    }
}

fn mem_size<P>(pixel: Pixel<P>, count: usize) -> usize {
    pixel
        .size()
        .checked_mul(count)
        .unwrap_or_else(|| panic!("Requested count overflows memory size"))
}

impl<P: Pod> Deref for Rec<P> {
    type Target = [P];

    fn deref(&self) -> &[P] {
        self.as_slice()
    }
}

impl<P: Pod> DerefMut for Rec<P> {
    fn deref_mut(&mut self) -> &mut [P] {
        self.as_mut_slice()
    }
}

impl<P: Pod> Clone for Rec<P> {
    fn clone(&self) -> Self {
        Rec {
            inner: self.inner.clone(),
            ..*self
        }
    }
}

impl<P: AsPixel + Pod> Default for Rec<P> {
    fn default() -> Self {
        Rec {
            inner: Buffer::default(),
            length: 0,
            pixel: P::pixel(),
        }
    }
}

impl<P: Pod + cmp::PartialEq> cmp::PartialEq for Rec<P> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<P: Pod + cmp::Eq> cmp::Eq for Rec<P> {}

impl<P: Pod + cmp::PartialOrd> cmp::PartialOrd for Rec<P> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<P: Pod + cmp::Ord> cmp::Ord for Rec<P> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<P: Pod + fmt::Debug> fmt::Debug for Rec<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl fmt::Debug for ReuseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.requested {
            None => write!(f, "Buffer reuse failed: Bytes count can not be expressed"),
            Some(requested) => write!(
                f,
                "Buffer reuse failed: {} bytes requested, only {} available",
                requested, self.capacity
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize() {
        let mut buffer: Rec<u8> = Rec::new(0);
        assert_eq!(buffer.capacity(), 0);
        assert_eq!(buffer.len(), 0);
        buffer.resize(4);
        assert!(buffer.capacity() >= 4);
        assert_eq!(buffer.len(), 4);
        buffer.resize(2);
        assert!(buffer.capacity() >= 2);
        assert_eq!(buffer.len(), 2);
        buffer.resize(0);
        buffer.shrink_to_fit();
        assert_eq!(buffer.capacity(), 0);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn map() {
        let mut buffer: Rec<u8> = Rec::new(8);
        assert_eq!(buffer.len(), 8);
        buffer.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);

        let buffer = buffer.map(u32::from);
        assert_eq!(buffer.len(), 8);
        assert_eq!(buffer.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);

        let buffer = buffer.map(|p| p as u8);
        assert_eq!(buffer.len(), 8);
        assert_eq!(buffer.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
