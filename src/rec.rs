// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::cmp;
use core::fmt;
use core::ops::{Deref, DerefMut};

use crate::buf::{buf, Buffer};
use crate::{AsTexel, Texel};

/// A reinterpretable vector for an array of texels.
///
/// It allows efficient conversion to other texel representations, that is effective
/// reinterpretation casts.
pub struct TexelBuffer<P> {
    inner: Buffer,
    length: usize,
    texel: Texel<P>,
}

/// Error representation for a failed buffer reuse.
///
/// Indicates that the capacity of the underlying buffer is not large enough to perform the
/// operation without a reallocation. This may be either since the allocation is simply not large
/// enough or due to the requested length not having any representation in memory for the chosen
/// texel type.
///
/// ```
/// # use canvas::TexelBuffer;
/// let mut buffer = TexelBuffer::<u16>::new(16);
///
/// let err = match buffer.reuse(buffer.capacity() + 1) {
///     Ok(_) => unreachable!("Increasing capacity would require reallocation"),
///     Err(err) => err,
/// };
///
/// let err = match buffer.reuse(usize::max_value()) {
///     Ok(_) => unreachable!("A slice of u16 can never have usize::MAX elements"),
///     Err(err) => err,
/// };
/// ```
pub struct BufferReuseError {
    pub(crate) requested: Option<usize>,
    pub(crate) capacity: usize,
}

impl<P> TexelBuffer<P> {
    /// Allocate a texel buffer by the texel count.
    ///
    /// # Panics
    ///
    /// This function will panic when the byte-length of the slice with the provided count would
    /// exceed the possible `usize` values. To avoid this, use `bytes_for_texel` with manual
    /// calculation of the byte length instead.
    ///
    /// This function will also panic if the allocation fails.
    pub fn new(count: usize) -> Self
    where
        P: AsTexel,
    {
        Self::new_for_texel(P::texel(), count)
    }

    /// Allocate a texel buffer by the texel count.
    ///
    /// Provides the opportunity to construct the texel argument via other means than the trait,
    /// for example a dynamically checked expression.
    ///
    /// # Panics
    ///
    /// This function will panic when the byte-length of the slice with the provided count would
    /// exceed the possible `usize` values. To avoid this, use `bytes_for_texel` with manual
    /// calculation of the byte length instead.
    ///
    /// This function will also panic if the allocation fails.
    pub fn new_for_texel(texel: Texel<P>, count: usize) -> Self {
        Self::bytes_for_texel(texel, mem_size(texel, count))
    }

    /// Allocate a texel buffer by providing the byte count you wish to allocate.
    ///
    /// # Panics
    ///
    /// This function will panic if the allocation fails.
    pub fn bytes_for_texel(texel: Texel<P>, mem_size: usize) -> Self {
        TexelBuffer {
            inner: Buffer::new(mem_size),
            length: mem_size,
            texel,
        }
    }

    /// Allocate a buffer with initial contents.
    ///
    /// The `TexelBuffer` will have a byte capacity that holds exactly as many elements as the slice
    /// contains. Note that the elements are copied bytewise.
    ///
    /// # Panics
    ///
    /// This function will panic if the allocation fails.
    pub fn with_elements(elements: &[P]) -> Self
    where
        P: AsTexel,
    {
        Self::with_elements_for_texel(P::texel(), elements)
    }

    /// Allocate a buffer with initial contents.
    ///
    /// The `TexelBuffer` will have a byte capacity that holds exactly as many elements as the slice
    /// contains. Note that the elements are copied bytewise.
    ///
    /// # Panics
    ///
    /// This function will panic if the allocation fails.
    pub fn with_elements_for_texel(texel: Texel<P>, elements: &[P]) -> Self {
        let src = texel.cast_bytes(elements);
        let mut buffer = TexelBuffer::from_buffer(Buffer::from(src), texel);
        // Will be treated as empty, so adjust to be filled up to count.
        buffer.length = src.len();
        buffer
    }

    pub(crate) fn from_buffer(inner: Buffer, texel: Texel<P>) -> Self {
        TexelBuffer {
            inner,
            texel,
            length: 0,
        }
    }

    /// Change the number of texel.
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
        self.resize_bytes(mem_size(self.texel, count))
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

    /// Change the number of texel without reallocation.
    ///
    /// Returns `Ok` when the resizing was successfully completed to the requested size and returns
    /// `Err` if this could not have been performed without a reallocation. This function will also
    /// never deallocate memory.
    ///
    /// ```
    /// # use canvas::TexelBuffer;
    /// // Initial allocation may panic due to allocation error for now.
    /// let mut buffer: TexelBuffer<u16> = TexelBuffer::new(100);
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
    pub fn reuse(&mut self, count: usize) -> Result<(), BufferReuseError> {
        let bytes = count
            .checked_mul(self.texel.size())
            .ok_or_else(|| BufferReuseError {
                requested: None,
                capacity: self.byte_capacity(),
            })?;
        self.reuse_bytes(bytes)
    }

    /// Change the number of bytes without reallocation.
    ///
    /// Returns `Ok` when the resizing was successfully completed to the requested size and returns
    /// `Err` with the new byte size otherwise.
    pub fn reuse_bytes(&mut self, bytes: usize) -> Result<(), BufferReuseError> {
        if bytes > self.byte_capacity() {
            return Err(BufferReuseError {
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
    /// # use canvas::TexelBuffer;
    /// let buf_u8 = TexelBuffer::<u8>::new(7);
    /// assert_eq!(buf_u8.len(), 7);
    ///
    /// let mut buf_u32 = buf_u8.reinterpret::<u32>();
    /// assert_eq!(buf_u32.len(), 1);
    /// buf_u32.shrink_to_fit();
    ///
    /// let buf_u8 = buf_u32.reinterpret::<u8>();
    /// assert_eq!(buf_u8.len(), 4);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the allocation fails.
    pub fn shrink_to_fit(&mut self) {
        let exact_size = mem_size(self.texel, self.len());
        self.inner.resize_to(exact_size);
        self.length = exact_size;
    }

    pub fn as_slice(&self) -> &[P] {
        self.buf().as_texels(self.texel)
    }

    pub fn as_mut_slice(&mut self) -> &mut [P] {
        let texel = self.texel;
        self.buf_mut().as_mut_texels(texel)
    }

    /// The number of accessible elements for the current type.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// The number of elements that can fit without reallocation.
    pub fn capacity(&self) -> usize {
        self.inner.capacity() / self.texel.size()
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

    /// Reinterpret the buffer for a different type of texel.
    ///
    /// See `reinterpret_to` for details.
    pub fn reinterpret<Q>(self) -> TexelBuffer<Q>
    where
        Q: AsTexel,
    {
        self.reinterpret_to(Q::texel())
    }

    /// Reinterpret the buffer for a different type of texel.
    ///
    /// Note that this may leave some of the underlying texels unaccessible if the new type is
    /// larger than the old one and the allocation was not a multiple of the new size. Conversely,
    /// some new bytes may become accessible if the memory length was not a multiple of the
    /// previous texel type's length.
    pub fn reinterpret_to<Q>(self, texel: Texel<Q>) -> TexelBuffer<Q> {
        TexelBuffer {
            inner: self.inner,
            length: self.length,
            texel,
        }
    }

    /// Map all elements to another value.
    ///
    /// See [`map_to`] for details.
    pub fn map<Q>(self, f: impl Fn(P) -> Q) -> TexelBuffer<Q>
    where
        Q: AsTexel,
    {
        self.map_to(f, Q::texel())
    }

    /// Map elements to another value.
    ///
    /// This will keep the logical length of the `TexelBuffer` so that the number of texels stays constant.
    /// If necessary, it will grow the internal buffer to achieve this.
    ///
    /// # Panics
    ///
    /// This function will panic if the allocation fails or the necessary allocation exceeds the
    /// value range of `usize`.
    pub fn map_to<Q>(mut self, f: impl Fn(P) -> Q, texel: Texel<Q>) -> TexelBuffer<Q> {
        // Ensure we have enough memory for both representations.
        let length = self.as_slice().len();
        let new_bytes = mem_size(texel, length);
        self.inner.grow_to(new_bytes);
        self.inner.map_within(..length, 0, f, self.texel, texel);
        TexelBuffer {
            inner: self.inner,
            length: new_bytes,
            texel,
        }
    }

    fn buf(&self) -> &buf {
        &self.inner[..self.length]
    }

    fn buf_mut(&mut self) -> &mut buf {
        &mut self.inner[..self.length]
    }

    pub(crate) fn into_inner(self) -> Buffer {
        self.inner
    }
}

fn mem_size<P>(texel: Texel<P>, count: usize) -> usize {
    texel
        .size()
        .checked_mul(count)
        .unwrap_or_else(|| panic!("Requested count overflows memory size"))
}

impl<P> Deref for TexelBuffer<P> {
    type Target = [P];

    fn deref(&self) -> &[P] {
        self.as_slice()
    }
}

impl<P> DerefMut for TexelBuffer<P> {
    fn deref_mut(&mut self) -> &mut [P] {
        self.as_mut_slice()
    }
}

impl<P> Clone for TexelBuffer<P> {
    fn clone(&self) -> Self {
        TexelBuffer {
            inner: self.inner.clone(),
            ..*self
        }
    }
}

impl<P: AsTexel> Default for TexelBuffer<P> {
    fn default() -> Self {
        TexelBuffer {
            inner: Buffer::default(),
            length: 0,
            texel: P::texel(),
        }
    }
}

impl<P: AsTexel + Clone> From<&'_ [P]> for TexelBuffer<P> {
    fn from(elements: &'_ [P]) -> Self {
        TexelBuffer::with_elements(elements)
    }
}

impl<P: cmp::PartialEq> cmp::PartialEq for TexelBuffer<P> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<P: cmp::Eq> cmp::Eq for TexelBuffer<P> {}

impl<P: cmp::PartialOrd> cmp::PartialOrd for TexelBuffer<P> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<P: cmp::Ord> cmp::Ord for TexelBuffer<P> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<P: fmt::Debug> fmt::Debug for TexelBuffer<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl fmt::Debug for BufferReuseError {
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
        let mut buffer: TexelBuffer<u8> = TexelBuffer::new(0);
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
        let mut buffer: TexelBuffer<u8> = TexelBuffer::new(8);
        assert_eq!(buffer.len(), 8);
        buffer.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);

        let buffer = buffer.map(u32::from);
        assert_eq!(buffer.len(), 8);
        assert_eq!(buffer.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);

        let buffer = buffer.map(|p| p as u8);
        assert_eq!(buffer.len(), 8);
        assert_eq!(buffer.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn with_elements() {
        const HELLO_WORLD: &[u8] = b"Hello, World!";
        let buffer = TexelBuffer::with_elements(HELLO_WORLD);
        assert_eq!(buffer.as_slice(), HELLO_WORLD);
        assert_eq!(buffer.byte_len(), HELLO_WORLD.len());

        let from_buffer = TexelBuffer::from(HELLO_WORLD);
        assert_eq!(buffer, from_buffer);
    }
}
