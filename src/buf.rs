// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::mem;
use core::ptr::NonNull;

use crate::pixels::{MAX_ALIGN, MaxAligned};
use crate::Pixel;
use zerocopy::{AsBytes, ByteSlice, FromBytes, LayoutVerified};

/// Allocates and manages the raw bytes.
///
/// The inner invariants are:
/// * `ptr` points to region with `layout`
/// * `layout` is aligned to at least `MAX_ALIGN`
pub struct Buf {
    /// The backing memory.
    ///
    /// We never access the memory this way though, this exists merely for the `Drop` semantics.
    _inner: Box<[u8]>,

    /// Pointer to the beginning of the aligned, valid region.
    ///
    /// Note that this may become redundant once `alloc` is fully stable. Then, we'd rather
    /// allocate the byte region with the correct alignment.
    ptr: NonNull<[u8]>,
}

impl Buf {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { self.ptr.as_ref() }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { self.ptr.as_mut() }
    }

    /// Allocate a new `Buf` with a number of bytes.
    ///
    /// Panics if the length is too long to find a properly aligned subregion.
    pub fn new(length: usize) -> Self {
        // We allocate one alignment more, so that we always find a correctly aligned subslice in
        // the allocated region.
        let alloc_len = length
            .checked_add(MAX_ALIGN)
            .unwrap_or_else(|| panic!("Requested region to large to allocate"));

        let boxed = vec![0 as u8; alloc_len].into_boxed_slice();
        let aligned_ptr = {
            let (_, verified, _) = align_to::<_, MaxAligned>(&*boxed);
            let used = &verified.bytes()[..length];
            used.into()
        };

        Buf {
            _inner: boxed,
            ptr: aligned_ptr,
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

/// Align the byte slice to `T`.
///
/// Compared to `slice::align_to` this method is safe. Panics if the size of the slice is not large
/// enough to contain any correctly aligned address.
fn align_to<B, T>(slice: B) -> (B, LayoutVerified<B, [T]>, B)
    where B: ByteSlice, T: FromBytes
{
    let align = mem::align_of::<T>();
    let addr = slice.as_ptr() as usize;

    // Calculate the next aligned address. We don't particularly care about overflows as they do
    // not invalidate modular arithmetic over the field with size power of two, and any actual
    // overflows in the pointers should be handled properly by the split implementation of `slice`.
    let next = addr.wrapping_add(align).wrapping_sub(1) & !(align - 1);
    let padding = next.wrapping_sub(addr);
    let (pre, slice) = slice.split_at(padding);

    // Now calculate the remaining length for the slice.
    let (slice, post) = prefix_slice::<_, T>(slice);
    let layout = LayoutVerified::new_slice(slice).unwrap_or_else(
        || unreachable!("Verified len and align"));
    (pre, layout, post)
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
