// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::{borrow, cell, cmp, mem, ops, sync::atomic};

use alloc::borrow::ToOwned;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::texel::{constants::MAX, AtomicPart, MaxAligned, MaxAtomic, MaxCell, Texel, MAX_ALIGN};

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
pub struct Buffer {
    /// The backing memory.
    inner: Vec<MaxAligned>,
}

/// Allocates and manages atomically shared bytes.
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
#[derive(Clone)]
pub struct AtomicBuffer {
    /// The backing memory.
    inner: Arc<[MaxAtomic]>,
}

/// Allocates and manages unsynchronized shared bytes.
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
#[derive(Clone)]
pub struct CellBuffer {
    /// The backing memory, aligned by allocating it with the proper type.
    inner: Arc<[MaxCell]>,
}

/// An aligned slice of memory.
///
/// This is a wrapper around a byte slice that additionally requires the slice to be highly
/// aligned.
///
/// See `pixel.rs` for the only constructors.
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct buf([u8]);

/// An aligned slice of atomic memory.
///
/// In contrast to other types, this can not be slice at arbitrary byte ends since we must
/// still utilize potentially full atomic instructions for the underlying interaction! Until we get
/// custom metadata, we have our own 'reference type' here.
///
/// This type is relatively useless in the public interface, this makes interfaces slightly less
/// convenient but it is internal to the library anyways.
///
/// Note: Contrary to `buf`, this type __can not__ be sliced at arbitrary locations. Use the
/// conversion to `atomic_ref` for this.
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct atomic_buf([AtomicPart]);

/// An aligned slice of shared-access memory.
///
/// This is a wrapper around a cell of a byte slice that additionally requires the slice to be
/// highly aligned.
///
/// See `pixel.rs` for the only constructors.
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct cell_buf(cell::Cell<[u8]>);

/// A logical reference to a byte slice from some atomic memory.
///
/// This is a wrapper around a slice of the underlying atomics. From this buffer, in contrast to
/// the shared and the unsynchronized slices, you can _not_ retrieve slices to typed memory.
///
/// See `pixel.rs` for the only constructors.
#[derive(Clone, Copy)]
pub(crate) struct AtomicRef<'lt> {
    buf: &'lt atomic_buf,
    /// The first byte referred to by this slice.
    ///
    /// Not using `core::ops::Range` since we want to be Copy!
    start: usize,
    /// The past-the-end byte referred to by this slice.
    end: usize,
}

/// A copy-on-grow version of a buffer.
pub(crate) enum Cog<'buf> {
    Owned(Buffer),
    // May be used later..
    #[allow(dead_code)]
    Borrowed(&'buf mut buf),
}

impl Buffer {
    const ELEMENT: MaxAligned = MaxAligned([0; MAX_ALIGN]);

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
        let inner = alloc::vec![Self::ELEMENT; alloc_len];

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

impl CellBuffer {
    const ELEMENT: MaxCell = MaxCell::zero();

    /// Allocate a new [`CellBuffer`] with a number of bytes.
    ///
    /// Panics if the length is too long to find a properly aligned subregion.
    pub fn new(length: usize) -> Self {
        let alloc_len = Buffer::alloc_len(length);
        let inner: Vec<_> = (0..alloc_len).map(|_| Self::ELEMENT).collect();

        CellBuffer {
            inner: inner.into(),
        }
    }

    /// Share an existing buffer.
    ///
    /// The library will try, to an extent, to avoid an allocation here. However, it can only do so
    /// if the capacity of the underlying buffer is the same as the logical length of the shared
    /// buffer. Ultimately we rely on the standard libraries guarantees for constructing a
    /// reference counted allocation from an owned vector.
    pub fn with_buffer(buffer: Buffer) -> Self {
        let inner: Vec<_> = buffer.inner.into_iter().map(MaxCell::new).collect();

        CellBuffer {
            inner: inner.into(),
        }
    }

    /// Retrieve the byte capacity of the allocated storage.
    pub fn capacity(&self) -> usize {
        core::mem::size_of_val(&*self.inner)
    }

    /// Copy the data into an owned buffer.
    pub fn to_owned(&self) -> Buffer {
        let inner = self.inner.iter().map(|cell| cell.get()).collect();

        Buffer { inner }
    }
}

impl AtomicBuffer {
    const ELEMENT: MaxAtomic = MaxAtomic::zero();

    /// Allocate a new [`AtomicBuffer`] with a number of bytes.
    ///
    /// Panics if the length is too long to find a properly aligned subregion.
    pub fn new(length: usize) -> Self {
        let alloc_len = Buffer::alloc_len(length);
        let inner: Vec<_> = (0..alloc_len).map(|_| Self::ELEMENT).collect();

        AtomicBuffer {
            inner: inner.into(),
        }
    }

    /// Share an existing buffer.
    ///
    /// The library will try, to an extent, to avoid an allocation here. However, it can only do so
    /// if the capacity of the underlying buffer is the same as the logical length of the shared
    /// buffer. Ultimately we rely on the standard libraries guarantees for constructing a
    /// reference counted allocation from an owned vector.
    pub fn with_buffer(buffer: Buffer) -> Self {
        let inner: Vec<_> = buffer.inner.into_iter().map(MaxAtomic::new).collect();

        AtomicBuffer {
            inner: inner.into(),
        }
    }

    /// Retrieve the byte capacity of the allocated storage.
    pub fn capacity(&self) -> usize {
        core::mem::size_of_val(&*self.inner)
    }

    /// Copy the data into an owned buffer.
    ///
    /// The load will always be relaxed. If more guarantees are required, insert your owned memory
    /// barrier instructions before or after the access or otherwise synchronize the call to this
    /// function.
    pub fn to_owned(&self) -> Buffer {
        let inner = self
            .inner
            .iter()
            .map(|cell| cell.load(atomic::Ordering::Relaxed))
            .collect();

        Buffer { inner }
    }
}

impl Cog<'_> {
    pub(crate) fn to_owned(this: &mut Self) -> &'_ mut Buffer {
        match this {
            Cog::Owned(buffer) => buffer,
            Cog::Borrowed(buffer) => {
                let buffer = buffer.to_owned();
                *this = Cog::Owned(buffer);
                Cog::to_owned(this)
            }
        }
    }

    pub(crate) fn into_owned(this: Self) -> Buffer {
        match this {
            Cog::Owned(buffer) => buffer,
            Cog::Borrowed(buffer) => buffer.to_owned(),
        }
    }

    pub(crate) fn grow_to(this: &mut Self, bytes: usize) -> &mut buf {
        if this.len() < bytes {
            Cog::to_owned(this).grow_to(bytes);
        }

        &mut **this
    }
}

impl buf {
    /// Wraps an aligned buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new<T>(data: &T) -> &Self
    where
        T: AsRef<[MaxAligned]> + ?Sized,
    {
        let bytes = MAX.to_bytes(data.as_ref());
        Self::from_bytes(bytes).unwrap()
    }

    /// Wraps an aligned mutable buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new_mut<T>(data: &mut T) -> &mut Self
    where
        T: AsMut<[MaxAligned]> + ?Sized,
    {
        let bytes = MAX.to_mut_bytes(data.as_mut());
        Self::from_bytes_mut(bytes).unwrap()
    }

    pub fn truncate(&self, at: usize) -> &Self {
        // TODO: worth it to use unsafe for avoiding unwrap checks?
        Self::from_bytes(&self.as_bytes()[..at]).unwrap()
    }

    pub fn truncate_mut(&mut self, at: usize) -> &mut Self {
        // TODO: worth it to use unsafe for avoiding unwrap checks?
        Self::from_bytes_mut(&mut self.as_bytes_mut()[..at]).unwrap()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    pub fn split_at(&self, at: usize) -> (&Self, &Self) {
        assert!(at % MAX_ALIGN == 0);
        let (a, b) = self.0.split_at(at);
        let a = MAX.try_to_slice(a).expect("was previously aligned");
        let b = MAX.try_to_slice(b).expect("asserted to be aligned");
        (Self::new(a), Self::new(b))
    }

    pub fn split_at_mut(&mut self, at: usize) -> (&mut Self, &mut Self) {
        assert!(at % MAX_ALIGN == 0);
        let (a, b) = self.0.split_at_mut(at);
        let a = MAX.try_to_slice_mut(a).expect("was previously aligned");
        let b = MAX.try_to_slice_mut(b).expect("asserted to be aligned");
        (Self::new_mut(a), Self::new_mut(b))
    }

    /// Reinterpret the buffer for the specific texel type.
    ///
    /// The alignment of `P` is already checked to be smaller than `MAX_ALIGN` through the
    /// constructor of `Texel`. The slice will have the maximum length possible but may leave
    /// unused bytes in the end.
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> &[P] {
        pixel.cast_buf(self)
    }

    /// Reinterpret the buffer mutable for the specific texel type.
    ///
    /// The alignment of `P` is already checked to be smaller than `MAX_ALIGN` through the
    /// constructor of `Texel`.
    // FIXME: decide to use naming scheme of `as_bytes_mut` or `as_mut_slice`.
    pub fn as_mut_texels<P>(&mut self, pixel: Texel<P>) -> &mut [P] {
        pixel.cast_mut_buf(self)
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
        p: Texel<P>,
        q: Texel<Q>,
    ) {
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
        fn backwards_past_the_end(start_byte_diff: isize, size_diff: isize) -> Option<usize> {
            assert!(size_diff >= 0);
            if size_diff == 0 {
                if start_byte_diff > 0 {
                    Some(0)
                } else {
                    None
                }
            } else if start_byte_diff < 0 {
                Some(0)
            } else {
                let floor = start_byte_diff / size_diff;
                let ceil = (floor as usize) + usize::from(start_byte_diff % size_diff != 0);
                Some(ceil)
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
            ops::Bound::Unbounded => self.as_texels(p).len(),
        };

        let len = p_end.checked_sub(p_start).expect("Bound violates order");

        let q_start = dest;

        let _ = self
            .as_texels(p)
            .get(p_start..)
            .and_then(|slice| slice.get(..len))
            .expect("Source out of bounds");

        let _ = self
            .as_texels(q)
            .get(q_start..)
            .and_then(|slice| slice.get(..len))
            .expect("Destination out of bounds");

        // Due to both being Texels.
        assert!(p.size() as isize > 0);
        assert!(q.size() as isize > 0);

        if p.size() >= q.size() {
            let start_diff = (q.size() * q_start).wrapping_sub(p.size() * p_start) as isize;
            let size_diff = p.size() as isize - q.size() as isize;

            let backwards_end = backwards_past_the_end(start_diff, size_diff)
                .unwrap_or(len)
                .min(len);

            self.map_backward(p_start, q_start, backwards_end, &f, p, q);
            self.map_forward(
                p_start + backwards_end,
                q_start + backwards_end,
                len - backwards_end,
                &f,
                p,
                q,
            );
        } else {
            let start_diff = (p.size() * p_start).wrapping_sub(q.size() * q_start) as isize;
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
                q,
            );
            self.map_forward(p_start, q_start, backwards_end, &f, p, q);
        }
    }

    /// Internally mapping function when the mapping can be done forwards.
    fn map_forward<P, Q>(
        &mut self,
        src: usize,
        dest: usize,
        len: usize,
        f: impl Fn(P) -> Q,
        p: Texel<P>,
        q: Texel<Q>,
    ) {
        for idx in 0..len {
            let source_idx = idx + src;
            let target_idx = idx + dest;
            let source = p.copy_val(&self.as_texels(p)[source_idx]);
            let target = f(source);
            self.as_mut_texels(q)[target_idx] = target;
        }
    }

    /// Internally mapping function when the mapping can be done backwards.
    fn map_backward<P, Q>(
        &mut self,
        src: usize,
        dest: usize,
        len: usize,
        f: impl Fn(P) -> Q,
        p: Texel<P>,
        q: Texel<Q>,
    ) {
        for idx in (0..len).rev() {
            let source_idx = idx + src;
            let target_idx = idx + dest;
            let source = p.copy_val(&self.as_texels(p)[source_idx]);
            let target = f(source);
            self.as_mut_texels(q)[target_idx] = target;
        }
    }
}

trait ByteSlice: Sized {
    fn len(&self) -> usize;
    fn split_at(self, at: usize) -> (Self, Self);
}

impl<'a> ByteSlice for &'a [u8] {
    fn len(&self) -> usize {
        (**self).len()
    }

    fn split_at(self, at: usize) -> (Self, Self) {
        self.split_at(at)
    }
}

impl<'a> ByteSlice for &'a mut [u8] {
    fn len(&self) -> usize {
        (**self).len()
    }

    fn split_at(self, at: usize) -> (Self, Self) {
        self.split_at_mut(at)
    }
}

impl From<&'_ [u8]> for Buffer {
    fn from(content: &'_ [u8]) -> Self {
        // TODO: can this be optimized to avoid initialization before copy?
        let mut buffer = Buffer::new(content.len());
        buffer[..content.len()].copy_from_slice(content);
        buffer
    }
}

impl From<&'_ [u8]> for AtomicBuffer {
    fn from(values: &'_ [u8]) -> Self {
        let chunks = values.chunks_exact(MAX_ALIGN);
        let remainder = chunks.remainder();

        let capacity = Buffer::alloc_len(values.len());
        let mut buffer = Vec::with_capacity(capacity);

        buffer.extend(chunks.map(|arr| {
            let mut data = MaxAligned([0; MAX_ALIGN]);
            data.0.copy_from_slice(arr);
            MaxAtomic::new(data)
        }));

        if !remainder.is_empty() {
            let mut data = MaxAligned([0; MAX_ALIGN]);
            data.0[..remainder.len()].copy_from_slice(remainder);
            buffer.push(MaxAtomic::new(data));
        }

        AtomicBuffer {
            inner: buffer.into(),
        }
    }
}

impl From<&'_ [u8]> for CellBuffer {
    fn from(values: &'_ [u8]) -> Self {
        let chunks = values.chunks_exact(MAX_ALIGN);
        let remainder = chunks.remainder();

        let capacity = Buffer::alloc_len(values.len());
        let mut buffer = Vec::with_capacity(capacity);

        buffer.extend(chunks.map(|arr| {
            let mut data = [0; MAX_ALIGN];
            data.copy_from_slice(arr);
            MaxCell(cell::Cell::new(data))
        }));

        if !remainder.is_empty() {
            let mut data = [0; MAX_ALIGN];
            data[..remainder.len()].copy_from_slice(remainder);
            buffer.push(MaxCell(cell::Cell::new(data)));
        }

        CellBuffer {
            inner: buffer.into(),
        }
    }
}

impl From<&'_ buf> for Buffer {
    fn from(content: &'_ buf) -> Self {
        content.to_owned()
    }
}

impl Default for &'_ buf {
    fn default() -> Self {
        buf::new(&mut [])
    }
}

impl Default for &'_ mut buf {
    fn default() -> Self {
        buf::new_mut(&mut [])
    }
}

impl borrow::Borrow<buf> for Buffer {
    fn borrow(&self) -> &buf {
        &**self
    }
}

impl borrow::BorrowMut<buf> for Buffer {
    fn borrow_mut(&mut self) -> &mut buf {
        &mut **self
    }
}

impl alloc::borrow::ToOwned for buf {
    type Owned = Buffer;
    fn to_owned(&self) -> Buffer {
        let mut buffer = Buffer::new(self.len());
        buffer.as_bytes_mut().copy_from_slice(self);
        buffer
    }
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

impl ops::Deref for AtomicBuffer {
    type Target = atomic_buf;

    fn deref(&self) -> &atomic_buf {
        atomic_buf::from_slice(&self.inner)
    }
}

impl ops::Deref for CellBuffer {
    type Target = cell_buf;

    fn deref(&self) -> &cell_buf {
        cell_buf::from_slice(&self.inner)
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

impl ops::Deref for Cog<'_> {
    type Target = buf;

    fn deref(&self) -> &buf {
        match self {
            Cog::Owned(buffer) => buffer,
            Cog::Borrowed(buffer) => buffer,
        }
    }
}

impl ops::DerefMut for Cog<'_> {
    fn deref_mut(&mut self) -> &mut buf {
        match self {
            Cog::Owned(buffer) => buffer,
            Cog::Borrowed(buffer) => buffer,
        }
    }
}

impl borrow::Borrow<buf> for Cog<'_> {
    fn borrow(&self) -> &buf {
        &**self
    }
}

impl borrow::BorrowMut<buf> for Cog<'_> {
    fn borrow_mut(&mut self) -> &mut buf {
        &mut **self
    }
}

impl cmp::PartialEq<Cog<'_>> for Cog<'_> {
    fn eq(&self, other: &Cog<'_>) -> bool {
        **self == **other
    }
}

impl cmp::Eq for Cog<'_> {}

impl cmp::PartialEq for buf {
    fn eq(&self, other: &buf) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl cmp::Eq for buf {}

impl cmp::PartialEq for Buffer {
    fn eq(&self, other: &Buffer) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl cmp::Eq for Buffer {}

impl ops::Index<ops::RangeTo<usize>> for buf {
    type Output = buf;

    fn index(&self, idx: ops::RangeTo<usize>) -> &buf {
        self.truncate(idx.end)
    }
}

impl ops::IndexMut<ops::RangeTo<usize>> for buf {
    fn index_mut(&mut self, idx: ops::RangeTo<usize>) -> &mut buf {
        self.truncate_mut(idx.end)
    }
}

#[allow(dead_code)]
impl AtomicRef<'_> {
    /// Overwrite bytes within the vector with new data.
    fn copy_within(&self, _from: core::ops::Range<usize>, _to: usize) {
        todo!()
    }

    /// Overwrite the whole vector with new data.
    fn copy_from(&self, _from: core::ops::Range<usize>, _source: &[MaxAtomic], _to: usize) {
        todo!()
    }
}

impl cell_buf {
    /// Wraps an aligned buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new<T>(_data: &T) -> &Self
    where
        T: AsRef<[MaxCell]> + ?Sized,
    {
        // We can't use `bytemuck` here.
        todo!()
    }

    pub fn truncate(&self, at: usize) -> &Self {
        // We promise this does not panic since the buffer is in fact aligned.
        Self::from_bytes(&self.0.as_slice_of_cells()[..at]).unwrap()
    }

    pub fn split_at(&self, at: usize) -> (&Self, &Self) {
        assert!(at % MAX_ALIGN == 0);
        let (a, b) = self.0.as_slice_of_cells().split_at(at);
        let a = Self::from_bytes(a).expect("was previously aligned");
        let b = Self::from_bytes(b).expect("asserted to be aligned");
        (a, b)
    }

    /// Reinterpret the buffer for the specific texel type.
    ///
    /// The alignment of `P` is already checked to be smaller than `MAX_ALIGN` through the
    /// constructor of `Texel`. The slice will have the maximum length possible but may leave
    /// unused bytes in the end.
    pub fn as_texels<P>(&self, _pixel: Texel<P>) -> &cell::Cell<[P]> {
        todo!()
    }

    pub fn map_within<P, Q>(
        &mut self,
        _src: impl ops::RangeBounds<usize>,
        _dest: usize,
        _f: impl Fn(P) -> Q,
        _p: Texel<P>,
        _q: Texel<Q>,
    ) {
        todo!()
    }
}

impl<'lt> From<&'lt atomic_buf> for AtomicRef<'lt> {
    fn from(value: &'lt atomic_buf) -> AtomicRef<'lt> {
        let end = core::mem::size_of_val(value);

        AtomicRef {
            buf: value,
            start: 0,
            end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texels::{MAX, U16, U32, U8};

    #[test]
    fn single_max_element() {
        let mut buffer = Buffer::new(mem::size_of::<MaxAligned>());
        let slice = buffer.as_mut_texels(MAX);
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
        assert!(buffer.as_mut_texels(U32).len() >= 1);
        buffer
            .as_mut_texels(U16)
            .iter_mut()
            .for_each(|p| *p = 0x0f0f);
        buffer
            .as_texels(U32)
            .iter()
            .for_each(|p| assert_eq!(*p, 0x0f0f0f0f));
        buffer
            .as_texels(U8)
            .iter()
            .for_each(|p| assert_eq!(*p, 0x0f));

        buffer
            .as_mut_texels(U8)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, p)| *p = idx as u8);
        assert_eq!(u32::from_be(buffer.as_texels(U32)[0]), 0x00010203);
    }

    #[test]
    fn mapping_great_to_small() {
        const LEN: usize = 10;
        let mut buffer = Buffer::new(LEN * mem::size_of::<u32>());
        buffer
            .as_mut_texels(U32)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, p)| *p = idx as u32);

        // Map those numbers in-place.
        buffer.map_within(..LEN, 0, |n: u32| n as u8, U32, U8);
        buffer.map_within(..LEN, 0, |n: u8| n as u32, U8, U32);

        // Back to where we started.
        assert_eq!(
            buffer.as_texels(U32)[..LEN].to_vec(),
            (0..LEN as u32).collect::<Vec<_>>()
        );

        // This should work even if we don't map to index 0.
        buffer.map_within(0..LEN, 3 * LEN, |n: u32| n as u8, U32, U8);
        buffer.map_within(3 * LEN..4 * LEN, 0, |n: u8| n as u32, U8, U32);

        assert_eq!(
            buffer.as_texels(U32)[..LEN].to_vec(),
            (0..LEN as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cell_buffer() {
        let data = [0, 0, 255, 0, 255, 0, 255, 0, 0];
        let buffer = CellBuffer::from(&data[..]);
        // Gets rounded up to the next alignment.
        assert_eq!(buffer.capacity(), Buffer::alloc_len(data.len()) * MAX_ALIGN);

        let alternative = CellBuffer::with_buffer(buffer.to_owned());
        assert_eq!(buffer.capacity(), alternative.capacity());
    }

    #[test]
    fn atomic_buffer() {
        let data = [0, 0, 255, 0, 255, 0, 255, 0, 0];
        let buffer = AtomicBuffer::from(&data[..]);
        // Gets rounded up to the next alignment.
        assert_eq!(buffer.capacity(), Buffer::alloc_len(data.len()) * MAX_ALIGN);

        let alternative = CellBuffer::with_buffer(buffer.to_owned());
        assert_eq!(buffer.capacity(), alternative.capacity());
    }
}
