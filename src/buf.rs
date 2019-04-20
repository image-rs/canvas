// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::mem;

use crate::pixels::MaxAligned;
use crate::Pixel;
use zerocopy::{AsBytes, ByteSlice, FromBytes, LayoutVerified};

/// Allocates and manages the raw bytes.
///
/// The inner invariants are:
/// * `ptr` points to region with `layout`
/// * `layout` is aligned to at least `MAX_ALIGN`
pub struct Buf {
    /// The backing memory.
    inner: Vec<MaxAligned>,
}

impl Buf {
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// Allocate a new `Buf` with a number of bytes.
    ///
    /// Panics if the length is too long to find a properly aligned subregion.
    pub fn new(length: usize) -> Self {
        const CHUNK_SIZE: usize = mem::size_of::<MaxAligned>();
        // We allocate one alignment more, so that we always find a correctly aligned subslice in
        // the allocated region.
        let alloc_len = length/CHUNK_SIZE + (length % CHUNK_SIZE != 0) as usize;
        let inner = vec![MaxAligned([0; 16]); alloc_len];

        Buf {
            inner,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixels::MAX;

    #[test]
    fn single_max_element() {
        let mut buf = Buf::new(mem::size_of::<MaxAligned>());
        let slice = buf.as_mut_pixels(MAX);
        assert!(slice.len() == 1);
    }
}
