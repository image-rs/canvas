// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::mem;
use core::ops;

use crate::pixel::{MaxAligned, Pixel, MAX_ALIGN};
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

        Buffer { inner }
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
        if self.inner.len() < new_len {
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
        length / CHUNK_SIZE + usize::from(length % CHUNK_SIZE != 0)
    }
}

impl buf {
    pub const ALIGNMENT: usize = MAX_ALIGN;

    /// Wraps an aligned buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new<T>(data: &T) -> &Self
    where
        T: AsRef<[MaxAligned]> + ?Sized,
    {
        Self::from_bytes(data.as_ref().as_bytes()).unwrap()
    }

    /// Wraps an aligned mutable buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new_mut<T>(data: &mut T) -> &mut Self
    where
        T: AsMut<[MaxAligned]> + ?Sized,
    {
        Self::from_bytes_mut(data.as_mut().as_bytes_mut()).unwrap()
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
    where
        P: FromBytes,
    {
        let (bytes, _) = prefix_slice::<_, P>(self.as_bytes());
        LayoutVerified::<_, [P]>::new_slice(bytes)
            .unwrap_or_else(|| unreachable!("Verified alignment in Pixel and len dynamically"))
            .into_slice()
    }

    /// Reinterpret the buffer mutable for the specific pixel type.
    ///
    /// The alignment of `P` is already checked to be smaller than `MAX_ALIGN` through the
    /// constructor of `Pixel`.
    pub fn as_mut_pixels<P>(&mut self, _: Pixel<P>) -> &mut [P]
    where
        P: AsBytes + FromBytes,
    {
        let (bytes, _) = prefix_slice::<_, P>(self.as_bytes_mut());
        LayoutVerified::<_, [P]>::new_slice(bytes)
            .unwrap_or_else(|| unreachable!("Verified alignment in Pixel and len dynamically"))
            .into_mut_slice()
    }

    /// Apply a mapping function to some elements.
    ///
    /// The indices `src` and `dest` are indices as if the slice were interpreted as `[P]` or `[Q]`
    /// respectively.
    ///
    /// The types may differ which allows the use of this function to prepare a reinterpretation
    /// cast of a typed buffer. This function chooses the order of function applications such that
    /// values are not overwritten before they are used, i.e. the function arguments are exactly
    /// the previously visible values. This is even less trivial than for copy if the parameter
    /// types differ in size.
    ///
    /// # Panics
    ///
    /// This function panics if `src` or the implied range of `dest` are out of bounds.
    pub fn map_within<P, Q>(
        &mut self,
        src: impl ops::RangeBounds<usize>,
        dest: usize,
        f: impl Fn(P) -> Q,
        p: Pixel<P>,
        q: Pixel<Q>,
    ) where
        P: FromBytes + Copy,
        Q: AsBytes + FromBytes + Copy,
    {
        // By symmetry, a write sequence that map `src` to `dest` without clobbering any values
        // that need to be read later can be applied in reverse to map `dest` to `src` instead.
        // Indeed, one explicit formulation of the clobber condition is: for all writes, the bytes
        // of a write do not overlap with the bytes of any later read. It follows that for all reads
        // the bytes of the read do not overlap with the bytes of any earlier write. Swapping reads
        // and writes and the sequence thus performs a dualisation.
        //
        // W.l.o.g. we concern ourselves only with `size_of::<P>() >= size_of::<Q>()`. Name the
        // byte regions (half open intervals) of the sequences of elements (p)_n, (q)_n and name
        // the indices in the respective indexing space of elements (pi)_n and (qi)_n, let N be the
        // number of elements to map.
        //
        // Let I be the set of indices such that `inf p_I < inf q_I`. We can map p_I to q_I without
        // clobbering by scheduling the indices I from highest to lowest. Let i be an index from I
        // during that sequence, and j any other index not yet scheduled.
        //
        // Example:
        //
        // ```
        // |2   | 1  |3   |
        //     |2 |1 |3 |
        // ```
        //
        // If 0 < j < i then j is in I as well, as implied by |P| >= |Q|. It is simply 
        //  inf p_j = inf p_i - |P|(i-j) <= inf p_i - |Q|(i-j) < inf q_i - |Q|(i-j) = inf q_j
        // By definition, inf q_i > inf p_i >= sup p_j and thus the ranges of the write does not
        // overlap those later reads.
        //
        // If however i < j then j can not be in I, thus inf q_j <= inf p_j. Since we also have
        // from the relation j < i; sup q_i <= inf q_j, the write to q_i can not overlap p_j.
        //
        // Then, we sechdule the remaining indices in forwards direction. To actually perform this
        // scheduling, we must find (as we now know range) I.
        //  inf p_n = |P|*pi_n = |P|(p_start + n)
        //  inf q_n = |Q|*qi_n = |Q|(q_start + n)
        //
        //  inf p_n        < inf q_n       <=>
        //  |P|(p_start+n) < |Q|(q_start+n)<=>
        //  |Q|q_start     > |P|p_start + (|P| - |Q|)n<=>
        //  |Q|q_start - |P|p_start > (|P| - |Q|)n
        //
        // if (|P| - |Q|) != 0 then
        //  n < (|Q|q_start - |P|p_start)/(|P| - |Q|)<=>
        //  n < ceil((|Q|q_start - |P|p_start)/(|P| - |Q|))

        // Returns the 
        fn backwards_past_the_end(
            start_byte_diff: isize,
            size_diff: isize,
        ) -> Option<usize> {
            assert!(size_diff >= 0);
            if size_diff == 0 {
                if start_byte_diff > 0 {
                    Some(0)
                } else {
                    None
                }
            } else {
                if start_byte_diff < 0 {
                    Some(0)
                } else {
                    let floor = start_byte_diff/size_diff;
                    let ceil = (floor as usize) 
                        + usize::from(start_byte_diff % size_diff != 0);
                    Some(ceil)
                }
            }
        }

        let p_start = match src.start_bound() {
            ops::Bound::Included(&bound) => bound,
            ops::Bound::Excluded(&bound) => bound
                .checked_add(1)
                .expect("Range does not specify a valid bound start"),
            ops::Bound::Unbounded => 0,
        };

        let p_end = match src.end_bound() {
            ops::Bound::Excluded(&bound) => bound,
            ops::Bound::Included(&bound) => bound
                .checked_add(1)
                .expect("Range does not specify a valid bound end"),
            ops::Bound::Unbounded => self.as_pixels(p).len(),
        };

        let len = p_end
            .checked_sub(p_start)
            .expect("Bound violates order");

        let q_start = dest;

        let _ = self
            .as_pixels(p)
            .get(p_start..)
            .and_then(|slice| slice.get(..len))
            .expect("Source out of bounds");

        let _ = self
            .as_pixels(q)
            .get(q_start..)
            .and_then(|slice| slice.get(..len))
            .expect("Destination out of bounds");

        assert!(p.size() as isize > 0);
        assert!(q.size() as isize > 0);

        if p.size() >= q.size() {
            let start_diff = (q.size()*q_start).wrapping_sub(p.size()*p_start) as isize;
            let size_diff = p.size() as isize - q.size() as isize;

            let backwards_end = backwards_past_the_end(start_diff, size_diff)
                .unwrap_or(len)
                .min(len);

            self.map_backward(
                p_start,
                q_start,
                backwards_end,
                &f,
                p,
                q);
            self.map_forward(
                p_start + backwards_end,
                q_start + backwards_end,
                len - backwards_end,
                &f,
                p,
                q);
        } else {
            let start_diff = (p.size()*p_start).wrapping_sub(q.size()*q_start) as isize;
            let size_diff = q.size() as isize - p.size() as isize;

            let backwards_end = backwards_past_the_end(start_diff, size_diff)
                .unwrap_or(len)
                .min(len);

            self.map_backward(
                p_start + backwards_end,
                q_start + backwards_end,
                len - backwards_end,
                &f,
                p,
                q);
            self.map_forward(
                p_start,
                q_start,
                backwards_end,
                &f,
                p,
                q);
        }
    }

    /// Internally mapping function when the mapping can be done forwards.
    fn map_forward<P, Q>(
        &mut self,
        src: usize,
        dest: usize,
        len: usize,
        f: impl Fn(P) -> Q,
        p: Pixel<P>,
        q: Pixel<Q>,
    ) where
        P: FromBytes + Copy,
        Q: AsBytes + FromBytes + Copy,
    {
        for idx in 0..len {
            let source_idx = idx + src;
            let target_idx = idx + dest;
            let source = self.as_pixels(p)[source_idx];
            let target = f(source);
            self.as_mut_pixels(q)[target_idx] = target;
        }
    }

    /// Internally mapping function when the mapping can be done backwards.
    fn map_backward<P, Q>(
        &mut self,
        src: usize,
        dest: usize,
        len: usize,
        f: impl Fn(P) -> Q,
        p: Pixel<P>,
        q: Pixel<Q>,
    ) where
        P: FromBytes + Copy,
        Q: AsBytes + FromBytes + Copy,
    {
        for idx in (0..len).rev() {
            let source_idx = idx + src;
            let target_idx = idx + dest;
            let source = self.as_pixels(p)[source_idx];
            let target = f(source);
            self.as_mut_pixels(q)[target_idx] = target;
        }
    }
}

fn prefix_slice<B, T>(slice: B) -> (B, B)
where
    B: ByteSlice,
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
    use crate::pixels::{MAX, U16, U32, U8};

    #[test]
    fn single_max_element() {
        let mut buffer = Buffer::new(mem::size_of::<MaxAligned>());
        let slice = buffer.as_mut_pixels(MAX);
        assert!(slice.len() == 1);
    }

    #[test]
    fn growing() {
        let mut buffer = Buffer::new(0);
        assert_eq!(buffer.capacity(), 0);
        buffer.grow_to(mem::size_of::<MaxAligned>());
        let capacity = buffer.capacity();
        assert!(buffer.capacity() > 0);
        buffer.grow_to(capacity);
        assert_eq!(buffer.capacity(), capacity);
        buffer.grow_to(0);
        assert_eq!(buffer.capacity(), capacity);
        buffer.grow_to(capacity + 1);
        assert!(buffer.capacity() > capacity);
    }

    #[test]
    fn reinterpret() {
        let mut buffer = Buffer::new(mem::size_of::<u32>());
        assert!(buffer.as_mut_pixels(U32).len() >= 1);
        buffer
            .as_mut_pixels(U16)
            .iter_mut()
            .for_each(|p| *p = 0x0f0f);
        buffer
            .as_pixels(U32)
            .iter()
            .for_each(|p| assert_eq!(*p, 0x0f0f0f0f));
        buffer
            .as_pixels(U8)
            .iter()
            .for_each(|p| assert_eq!(*p, 0x0f));

        buffer
            .as_mut_pixels(U8)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, p)| *p = idx as u8);
        assert_eq!(u32::from_be(buffer.as_pixels(U32)[0]), 0x00010203);
    }

    #[test]
    fn mapping_great_to_small() {
        const LEN: usize = 10;
        let mut buffer = Buffer::new(LEN*mem::size_of::<u32>());
        buffer
            .as_mut_pixels(U32)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, p)| *p = idx as u32);

        // Map those numbers in-place.
        buffer.map_within(..LEN, 0, |n: u32| n as u8, U32, U8);
        buffer.map_within(..LEN, 0, |n: u8| n as u32, U8, U32);

        // Back to where we started.
        assert_eq!(
            buffer.as_pixels(U32)[..LEN].to_vec(),
            (0..LEN as u32).collect::<Vec<_>>());

        // This should work even if we don't map to index 0.
        buffer.map_within(0..LEN, 3*LEN, |n: u32| n as u8, U32, U8);
        buffer.map_within(3*LEN..4*LEN, 0, |n: u8| n as u32, U8, U32);

        assert_eq!(
            buffer.as_pixels(U32)[..LEN].to_vec(),
            (0..LEN as u32).collect::<Vec<_>>());
    }
}
