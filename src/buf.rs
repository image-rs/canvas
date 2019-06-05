// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::mem;
use core::ops;

use crate::pixels::{MaxAligned, MAX_ALIGN};
use crate::Pixel;
use zerocopy::{AsBytes, ByteSlice, FromBytes, LayoutVerified};

/// Allocates and manages raw bytes.
///
/// Provides a utility to allocate a slice of bytes aligned to the maximally required alignment.
/// Since the elements are much larger than single bytes the inner storage will **not** have exact
/// sizes as one would be used from by using a `Vec` as an allocator. This is instead more close to
/// a `RawVec` and most operations have the same drawback as `Vec::reserve_exact` in not actually
/// being exact.
///
/// Since exact length and capacity semantics are hard to guarantee for most operations, no effort
/// is made to uphold them. Instead. keeping track of the exact, wanted logical length of the
/// requested byte slice is the obligation of the user *under all circumstances*. As a consequence,
/// there are also no operations which explicitely uncouple length and capacity. All operations
/// simply work on best effort of making some number of bytes available.
#[derive(Clone, Default)]
pub(crate) struct Buffer {
    /// The backing memory.
    inner: Vec<MaxAligned>,
}

/// An aligned slice of memory.
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub(crate) struct buf([u8]);

impl Buffer {
    const ELEMENT: MaxAligned = MaxAligned([0; 16]);

    pub fn as_buf(&self) -> &buf {
        buf::new(self.inner.as_slice())
    }

    pub fn as_buf_mut(&mut self) -> &mut buf {
        buf::new_mut(self.inner.as_mut_slice())
    }

    /// Allocate a new `Buf` with a number of bytes.
    ///
    /// Panics if the length is too long to find a properly aligned subregion.
    pub fn new(length: usize) -> Self {
        let alloc_len = Self::alloc_len(length);
        let inner = vec![Self::ELEMENT; alloc_len];

        Buffer {
            inner,
        }
    }

    /// Retrieve the byte capacity of the allocated storage.
    pub fn capacity(&self) -> usize {
        self.inner.capacity() * mem::size_of::<MaxAligned>()
    }

    /// Ensure to contain a minimum number of bytes.
    ///
    /// Only allocates when the new required size is larger than the previous one. Note that this
    /// does not ensure that the new length is exactly the byte count, it may be longer. If the
    /// current length is already large enough then this will not do anything.
    pub fn grow_to(&mut self, bytes: usize) {
        let new_len = Self::alloc_len(bytes);
        if self.len() < new_len {
            self.inner.resize(new_len, Self::ELEMENT);
        }
    }

    /// Reallocate to fit as closely as possible.
    ///
    /// The size after resizing may still be larger than requested.
    pub fn resize_to(&mut self, bytes: usize) {
        let new_len = Self::alloc_len(bytes);
        self.inner.resize(new_len, Self::ELEMENT);
        self.inner.shrink_to_fit()
    }

    /// Calculates the number of elements to have a byte buffer of requested length.
    fn alloc_len(length: usize) -> usize {
        const CHUNK_SIZE: usize = mem::size_of::<MaxAligned>();
        assert!(CHUNK_SIZE > 1);

        // We allocated enough chunks for at least the length. This can never overflow.
        length/CHUNK_SIZE + (length % CHUNK_SIZE != 0) as usize
    }
}

impl buf {
    pub const ALIGNMENT: usize = MAX_ALIGN;

    /// Wraps an aligned buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new<T>(data: &T) -> &Self
        where T: AsRef<[MaxAligned]> + ?Sized
    {
        Self::from_bytes(data.as_ref().as_bytes())
            .unwrap()
    }

    /// Wraps an aligned mutable buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new_mut<T>(data: &mut T) -> &mut Self
        where T: AsMut<[MaxAligned]> + ?Sized
    {
        Self::from_bytes_mut(data.as_mut().as_bytes_mut())
            .unwrap()
    }

    /// Wrap bytes in a `buf`.
    ///
    /// The bytes need to be aligned to `ALIGNMENT`.
    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.as_ptr() as usize % Self::ALIGNMENT == 0 {
            Some(unsafe { &*(bytes as *const [u8] as *const Self) })
        } else {
            None
        }
    }

    /// Wrap bytes in a `buf`.
    ///
    /// The bytes need to be aligned to `ALIGNMENT`.
    pub fn from_bytes_mut(bytes: &mut [u8]) -> Option<&mut Self> {
        if bytes.as_ptr() as usize % Self::ALIGNMENT == 0 {
            Some(unsafe { &mut *(bytes as *mut [u8] as *mut Self) })
        } else {
            None
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    /// Reinterpret the buffer for the specific pixel type.
    ///
    /// The alignment of `P` is already checked to be smaller than `MAX_ALIGN` through the
    /// constructor of `Pixel`. The slice will have the maximum length possible but may leave
    /// unused bytes in the end.
    pub fn as_pixels<P>(&self, _: Pixel<P>) -> &[P]
        where P: FromBytes
    {
        let (bytes, _) = prefix_slice::<_, P>(self.as_bytes());
        LayoutVerified::<_, [P]>::new_slice(bytes).unwrap_or_else(
            || unreachable!("Verified alignment in Pixel and len dynamically")
        ).into_slice()
    }

    /// Reinterpret the buffer mutable for the specific pixel type.
    ///
    /// The alignment of `P` is already checked to be smaller than `MAX_ALIGN` through the
    /// constructor of `Pixel`.
    pub fn as_mut_pixels<P>(&mut self, _: Pixel<P>) -> &mut [P]
        where P: AsBytes + FromBytes
    {
        let (bytes, _) = prefix_slice::<_, P>(self.as_bytes_mut());
        LayoutVerified::<_, [P]>::new_slice(bytes).unwrap_or_else(
            || unreachable!("Verified alignment in Pixel and len dynamically")
        ).into_mut_slice()
    }
}

fn prefix_slice<B, T>(slice: B) -> (B, B) where B: ByteSlice
{
    let size = mem::size_of::<T>();
    let len = (slice.len() / size) * size;
    slice.split_at(len)
}

impl ops::Deref for Buffer {
    type Target = buf;

    fn deref(&self) -> &buf {
        self.as_buf()
    }
}

impl ops::DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut buf {
        self.as_buf_mut()
    }
}

impl ops::Deref for buf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl ops::DerefMut for buf {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_bytes_mut()
    }
}

impl ops::Index<ops::RangeTo<usize>> for buf {
    type Output = buf;

    fn index(&self, idx: ops::RangeTo<usize>) -> &buf {
        Self::from_bytes(&self.0[idx]).unwrap()
    }
}

impl ops::IndexMut<ops::RangeTo<usize>> for buf {
    fn index_mut(&mut self, idx: ops::RangeTo<usize>) -> &mut buf {
        Self::from_bytes_mut(&mut self.0[idx]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixels::{U8, U16, U32, MAX};

    #[test]
    fn single_max_element() {
        let mut buffer = Buffer::new(mem::size_of::<MaxAligned>());
        let slice = buffer.as_mut_pixels(MAX);
        assert!(slice.len() == 1);
    }

    #[test]
    fn reinterpret() {
        let mut buffer = Buffer::new(mem::size_of::<u32>());
        assert!(buffer.as_mut_pixels(U32).len() >= 1);
        buffer.as_mut_pixels(U16).iter_mut().for_each(
            |p| *p = 0x0f0f);
        buffer.as_pixels(U32).iter().for_each(
            |p| assert_eq!(*p, 0x0f0f0f0f));
        buffer.as_pixels(U8).iter().for_each(
            |p| assert_eq!(*p, 0x0f));

        buffer.as_mut_pixels(U8)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, p)| *p = idx as u8);
        assert_eq!(u32::from_be(buffer.as_pixels(U32)[0]), 0x00010203);
    }
}
