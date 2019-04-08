use core::mem;
use core::alloc::{Layout, LayoutErr};
use core::ptr::NonNull;

use crate::pixels::{MAX_ALIGN, MaxAligned};
use zerocopy::{ByteSlice, FromBytes, LayoutVerified};

/// Allocates and manages the raw bytes.
///
/// The inner invariants are:
/// * `ptr` points to region with `layout`
/// * `layout` is aligned to at least `MAX_ALIGN`
struct Buf {
    /// The backing memory.
    inner: Box<[u8]>,

    /// Pointer to the beginning of the aligned, valid region.
    ///
    /// Note that this may become redundant once `alloc` is fully stable. Then, we'd rather
    /// allocate the byte region with the correct alignment.
    ptr: Option<NonNull<[u8]>>,
}

impl Buf {
    pub fn as_bytes(&self) -> &[u8] {
        match &self.ptr {
            Some(ptr) => unsafe { ptr.as_ref() },
            None => &[],
        }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.ptr {
            Some(ptr) => unsafe { ptr.as_mut() },
            None => &mut [],
        }
    }

    fn layout(bytes: usize) -> Layout {
        Layout::from_size_align(bytes, MAX_ALIGN)
            .unwrap_or_else(layout_panic)
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
            inner: boxed,
            ptr: Some(aligned_ptr),
        }
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
    let size = mem::size_of::<T>();
    let addr = slice.as_ptr() as usize;

    // Calculate the next aligned address. We don't particularly care about overflows as they do
    // not invalidate modular arithmetic over the field with size power of two, and any actual
    // overflows in the pointers should be handled properly by the split implementation of `slice`.
    let next = addr.wrapping_add(align).wrapping_sub(1) & !(align - 1);
    let padding = next.wrapping_sub(addr);
    let (pre, slice) = slice.split_at(padding);

    // Now calculate the remaining length for the slice.
    let len = (slice.len() / size) * size;
    let (slice, post) = slice.split_at(len);
    let layout = LayoutVerified::new_slice(slice).unwrap_or_else(
        || unreachable!("Verified len and align"));
    (pre, layout, post)
}

#[inline(always)]
fn layout_panic<T>(err: LayoutErr) -> T {
    layout_panic_impl(err)
}

#[inline(never)]
#[cold]
fn layout_panic_impl(err: LayoutErr) -> ! {
    panic!("Could not allocate memory with the required layout {}", err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_max_element() {
        let mut buf = Buf::new(mem::size_of::<MaxAligned>());
        let layout = LayoutVerified::<_, [MaxAligned]>::new_slice(buf.as_bytes_mut())
            .expect("Buffer is aligned");
        let slice = layout.into_mut_slice();
        assert!(slice.len() == 1);
    }
}
