// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::{borrow, cell, cmp, mem, ops, sync::atomic};

use alloc::borrow::ToOwned;
use alloc::rc::Rc;
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
#[derive(Clone, Default)]
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
#[derive(Clone, Default)]
pub struct CellBuffer {
    /// The backing memory, aligned by allocating it with the proper type.
    inner: Rc<[MaxCell]>,
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
/// custom metadata, we have our own 'reference type' here with [`AtomicSliceRef`]. The slice
/// reference always extends over a slice of the underlying [`MaxAtomic`] type and only stores
/// offsets into this.
///
/// This type is relatively useless in the public interface, this makes interfaces slightly less
/// convenient but it is internal to the library anyways.
///
/// Note: Contrary to `buf`, this type __can not__ be sliced at arbitrary locations. Use the
/// conversion to `atomic_ref` for this.
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct atomic_buf(pub(crate) [AtomicPart]);

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
/// The analogue of this is `&[P]` or `&[Cell<P>]` respectively. This is a wrapper around a slice
/// of the underlying atomics. However, note we promise soundness but _not_ absence of tears in the
/// logical data type if the data straddles different underlying atomic representation types. We
/// simply can not promise this. Of course, an external synchronization might be used enforce this
/// additional guarantee.
///
/// For consistency with slices, casting of this type is done via an instance of [`Texel`].
///
/// TODO: We could probably make this type smaller. We store the underlying aligned buffer region
/// but that memory extent can be recreated from an *unaligned* pointer and a length to our actual
/// data. (Due to alignment and size of [`MaxAligned`] being the same, just downwards align the
/// pointer for the base and extend to the next alignment boundary upwards). This requires us to
/// use raw pointers so that the original provenance is retained. Also we must avoid offering
/// methods that would refer to the `buf` attribute's memory outside that minimal region.
pub struct AtomicSliceRef<'lt, P = u8> {
    /// This must be aligned to `MAX_ALIGN`. We could relax it to `AtomicPart` but that would be
    /// dependent on system configuration. Since this invisible state is nevertheless hugely
    /// important for the region considered aliased, let's avoid exposing that varying behavior as
    /// much as possible. Crate-internal operations may use the more granular unit internally.
    pub(crate) buf: &'lt atomic_buf,
    /// The underlying logical texel type this is bound to.
    pub(crate) texel: Texel<P>,
    /// The first byte referred to by this slice.
    ///
    /// Not using `core::ops::Range` since we want to be Copy!
    pub(crate) start: usize,
    /// The past-the-end byte referred to by this slice.
    pub(crate) end: usize,
}

/// A logical reference to a typed element from some atomic memory.
///
/// The analogue of this is `&P` or `&Cell<P>` respectively. Note we promise soundness but _not_
/// absence of tears in the logical data type if the data straddles different underlying atomic
/// representation types. We simply can not promise this. Of course, an external synchronization
/// might be used enforce this additional guarantee.
pub struct AtomicRef<'lt, P = u8> {
    pub(crate) buf: &'lt atomic_buf,
    /// The underlying logical texel type this is bound to.
    pub(crate) texel: Texel<P>,
    /// The first byte referred to by this slice.
    pub(crate) start: usize,
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

    /// Query if two buffers share the same memory region.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// Retrieve the byte capacity of the allocated storage.
    pub fn capacity(&self) -> usize {
        core::mem::size_of_val(&*self.inner)
    }

    /// Get this buffer if there are now copies.
    ///
    /// ```
    /// use image_texel::texels::{AtomicBuffer, U8};
    ///
    /// let mut buffer = AtomicBuffer::new(4);
    /// assert!(buffer.get_mut().is_some());
    /// let alias = buffer.clone();
    /// assert!(buffer.get_mut().is_none());
    /// ```
    pub fn get_mut(&mut self) -> Option<&mut cell_buf> {
        Rc::get_mut(&mut self.inner).map(cell_buf::from_slice_mut)
    }

    /// Ensure this buffer is its own copy.
    ///
    /// ```
    /// use image_texel::texels::{AtomicBuffer, U8};
    ///
    /// let mut buffer = AtomicBuffer::new(4);
    /// let mut alias = buffer.clone();
    ///
    /// U8.store_atomic(buffer.as_texels(U8).index_one(0), 1);
    /// let unshared = buffer.make_mut().as_buf_mut();
    /// let alias = alias.get_mut().expect("Just unaliased");
    ///
    /// unshared.as_mut_texels(U8)[0] = 2;
    /// assert_eq!(alias.as_buf_mut().as_mut_texels(U8)[0], 1);
    /// ```
    pub fn make_mut(&mut self) -> &mut cell_buf {
        if Rc::get_mut(&mut self.inner).is_none() {
            *self = self.to_owned().into();
        }

        Rc::get_mut(&mut self.inner)
            .map(cell_buf::from_slice_mut)
            .expect("we just made a mutable copy")
    }

    /// Copy the data into an owned buffer.
    pub fn to_owned(&self) -> Buffer {
        let inner = self.inner.iter().map(|cell| cell.get()).collect();

        Buffer { inner }
    }

    /// Create an independent copy of the buffer, with a new length.
    ///
    /// The prefix contents of the new buffer will be the same as the current buffer. The new
    /// buffer will _never_ share memory with the current buffer.
    pub fn to_resized(&self, bytes: usize) -> Self {
        let mut working_copy = self.to_owned();
        working_copy.resize_to(bytes);
        Self::with_buffer(working_copy)
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

    /// Query if two buffers share the same memory region.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Retrieve the byte capacity of the allocated storage.
    pub fn capacity(&self) -> usize {
        core::mem::size_of_val(&*self.inner)
    }

    /// Get this buffer if there are now copies.
    ///
    /// ```
    /// use image_texel::texels::{AtomicBuffer, U8};
    ///
    /// let mut buffer = AtomicBuffer::new(4);
    /// assert!(buffer.get_mut().is_some());
    /// let alias = buffer.clone();
    /// assert!(buffer.get_mut().is_none());
    /// ```
    pub fn get_mut(&mut self) -> Option<&mut atomic_buf> {
        Arc::get_mut(&mut self.inner).map(atomic_buf::from_slice_mut)
    }

    /// Ensure this buffer is its own copy.
    ///
    /// ```
    /// use image_texel::texels::{AtomicBuffer, U8};
    ///
    /// let mut buffer = AtomicBuffer::new(4);
    /// let mut alias = buffer.clone();
    ///
    /// U8.store_atomic(buffer.as_texels(U8).index_one(0), 1);
    /// let unshared = buffer.make_mut().as_buf_mut();
    /// let alias = alias.get_mut().expect("Just unaliased");
    ///
    /// unshared.as_mut_texels(U8)[0] = 2;
    /// assert_eq!(alias.as_buf_mut().as_mut_texels(U8)[0], 1);
    /// ```
    pub fn make_mut(&mut self) -> &mut atomic_buf {
        if Arc::get_mut(&mut self.inner).is_none() {
            *self = self.to_owned().into();
        }

        Arc::get_mut(&mut self.inner)
            .map(atomic_buf::from_slice_mut)
            .expect("we just made a mutable copy")
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

    /// Create an independent copy of the buffer, with a new length.
    ///
    /// The prefix contents of the new buffer will be the same as the current buffer. The new
    /// buffer will _never_ share memory with the current buffer.
    pub fn to_resized(&self, bytes: usize) -> Self {
        let mut working_copy = self.to_owned();
        working_copy.resize_to(bytes);
        Self::with_buffer(working_copy)
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

    /// Reduce the number of bytes covered by this buffer slice.
    #[must_use = "Does not mutate self"]
    #[track_caller]
    pub fn truncate(&self, at: usize) -> &Self {
        Self::from_bytes(&self.as_bytes()[..at]).unwrap()
    }

    /// Reduce the number of bytes covered by this mutable buffer slice.
    #[must_use = "Does not mutate self"]
    #[track_caller]
    pub fn truncate_mut(&mut self, at: usize) -> &mut Self {
        Self::from_bytes_mut(&mut self.as_bytes_mut()[..at]).unwrap()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    /// Split at an aligned byte offset.
    #[track_caller]
    pub fn split_at(&self, at: usize) -> (&Self, &Self) {
        assert!(at % MAX_ALIGN == 0);
        let (a, b) = self.0.split_at(at);
        let a = MAX.try_to_slice(a).expect("was previously aligned");
        let b = MAX.try_to_slice(b).expect("asserted to be aligned");
        (Self::new(a), Self::new(b))
    }

    /// Remove everything past the given point, return the tail we removed.
    pub(crate) fn take_at_mut<'a>(this: &mut &'a mut Self, at: usize) -> &'a mut Self {
        let (pre, post) = buf::split_at_mut(core::mem::take(this), at);
        *this = pre;
        post
    }

    /// Mutably split at an aligned byte offset.
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
        TexelMappingBuffer::map_within(self, src, dest, f, p, q)
    }
}

impl TexelMappingBuffer for buf {
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

    fn texel_len<P>(&self, texel: Texel<P>) -> usize {
        self.as_texels(texel).len()
    }
}

/// A buffer in which we can copy, apply a transform, and write back.
trait TexelMappingBuffer {
    fn map_forward<P, Q>(
        &mut self,
        src: usize,
        dest: usize,
        len: usize,
        f: impl Fn(P) -> Q,
        p: Texel<P>,
        q: Texel<Q>,
    );

    fn map_backward<P, Q>(
        &mut self,
        src: usize,
        dest: usize,
        len: usize,
        f: impl Fn(P) -> Q,
        p: Texel<P>,
        q: Texel<Q>,
    );

    fn texel_len<P>(&self, texel: Texel<P>) -> usize;

    fn map_within<P, Q>(
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
            ops::Bound::Unbounded => self.texel_len(p),
        };

        let len = p_end.checked_sub(p_start).expect("Bound violates order");

        let q_start = dest;

        let _ = self
            .texel_len(p)
            .checked_sub(p_start)
            .and_then(|slice| slice.checked_sub(len))
            .expect("Source out of bounds");

        let _ = self
            .texel_len(q)
            .checked_sub(q_start)
            .and_then(|slice| slice.checked_sub(len))
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

impl From<Buffer> for AtomicBuffer {
    fn from(values: Buffer) -> Self {
        // TODO: can this be optimized to avoid the byte-for-byte allocation-copy?
        Self::from(values.as_bytes())
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

impl From<Buffer> for CellBuffer {
    fn from(values: Buffer) -> Self {
        // TODO: can this be optimized to avoid the byte-for-byte allocation-copy?
        Self::from(values.as_bytes())
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

impl cell_buf {
    /// Wraps an aligned buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new<T>(data: &T) -> &Self
    where
        T: AsRef<[MaxCell]> + ?Sized,
    {
        cell_buf::from_slice(data.as_ref())
    }

    /// Get the length of available memory in bytes.
    pub fn len(&self) -> usize {
        self.0.as_slice_of_cells().len()
    }

    /// Reduce the number of bytes covered by this slice.
    #[must_use = "Does not mutate self"]
    #[track_caller]
    pub fn truncate(&self, at: usize) -> &Self {
        // We promise this does not panic since the buffer is in fact aligned.
        Self::from_bytes(&self.0.as_slice_of_cells()[..at]).unwrap()
    }

    /// Split into two aligned buffers.
    ///
    /// # Panics
    ///
    /// This panics if the byte offset given by `at` is not aligned according to max alignment or
    /// if the index is out-of-bounds.
    #[track_caller]
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
    pub fn as_texels<P>(&self, texel: Texel<P>) -> &cell::Cell<[P]> {
        let slice = self.0.as_slice_of_cells();
        texel
            .try_to_cell(slice)
            .expect("A cell_buf is always aligned")
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
        &self,
        src: impl ops::RangeBounds<usize>,
        dest: usize,
        f: impl Fn(P) -> Q,
        p: Texel<P>,
        q: Texel<Q>,
    ) {
        let mut that = self;
        TexelMappingBuffer::map_within(&mut that, src, dest, f, p, q)
    }
}

impl cmp::PartialEq for cell_buf {
    fn eq(&self, other: &Self) -> bool {
        // Doing this comparison discards alignment information that is probably checked for in the
        // kernel of memcmp. If the compiler inlines it, it may be able to remove that. Or not.
        // Should not matter too much but if it does in your benchmarks (be sure to do multiple
        // platforms and check with assembly throughput) then let me know.
        crate::texels::U8.cell_memory_eq(self.0.as_slice_of_cells(), other.0.as_slice_of_cells())
    }
}

impl cmp::PartialEq<[u8]> for cell_buf {
    fn eq(&self, other: &[u8]) -> bool {
        crate::texels::U8.cell_bytes_eq(self.0.as_slice_of_cells(), other)
    }
}

impl cmp::Eq for cell_buf {}

impl cmp::PartialEq for CellBuffer {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl cmp::Eq for CellBuffer {}

impl TexelMappingBuffer for &'_ cell_buf {
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
        let src_buffer = self.as_texels(p).as_slice_of_cells();
        let target_buffer = self.as_texels(q).as_slice_of_cells();

        for idx in 0..len {
            let source_idx = idx + src;
            let target_idx = idx + dest;
            let source = p.copy_cell(&src_buffer[source_idx]);
            let target = f(source);
            target_buffer[target_idx].set(target);
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
        let src_buffer = self.as_texels(p).as_slice_of_cells();
        let target_buffer = self.as_texels(q).as_slice_of_cells();

        for idx in (0..len).rev() {
            let source_idx = idx + src;
            let target_idx = idx + dest;
            let source = p.copy_cell(&src_buffer[source_idx]);
            let target = f(source);
            target_buffer[target_idx].set(target);
        }
    }

    fn texel_len<P>(&self, texel: Texel<P>) -> usize {
        self.as_texels(texel).as_slice_of_cells().len()
    }
}

impl atomic_buf {
    /// Wraps an aligned buffer into `buf`.
    ///
    /// This method will never panic, as the alignment of the data is guaranteed.
    pub fn new<T>(data: &T) -> &Self
    where
        T: AsRef<[MaxAtomic]> + ?Sized,
    {
        atomic_buf::from_slice(data.as_ref())
    }

    /// Get the length of available memory in bytes.
    pub fn len(&self) -> usize {
        core::mem::size_of_val(self)
    }

    pub fn as_buf_mut(&mut self) -> &mut buf {
        buf::from_bytes_mut(atomic_buf::part_mut_slice(&mut self.0)).unwrap()
    }

    /// Split into two aligned buffers.
    ///
    /// # Panics
    ///
    /// This panics if the byte offset given by `at` is not aligned according to max alignment or
    /// if the index is out-of-bounds.
    #[track_caller]
    pub fn split_at(&self, at: usize) -> (&Self, &Self) {
        use crate::texels::U8;

        assert!(at % MAX_ALIGN == 0);
        let slice = self.as_texels(U8);
        let (a, b) = slice.split_at(at);
        let left = atomic_buf::from_bytes(a).expect("was previously aligned");
        let right = atomic_buf::from_bytes(b).expect("was previously aligned");

        (left, right)
    }

    /// Reinterpret the buffer for the specific texel type.
    ///
    /// The alignment of `P` is already checked to be smaller than `MAX_ALIGN` through the
    /// constructor of `Texel`. The slice will have the maximum length possible but may leave
    /// unused bytes in the end.
    pub fn as_texels<P>(&self, texel: Texel<P>) -> AtomicSliceRef<P> {
        use crate::texels::U8;

        let buffer = AtomicSliceRef {
            buf: self,
            start: 0,
            end: core::mem::size_of_val(self),
            texel: U8,
        };

        texel
            .try_to_atomic(buffer)
            .expect("An atomic_buf is always aligned")
    }

    /// Index into this buffer at a generalized, potentially skewed, typed index.
    ///
    /// # Panics
    ///
    /// This method panics if the index is out-of-range.
    pub fn index<T>(&self, index: TexelRange<T>) -> AtomicSliceRef<'_, T> {
        let scale = index.texel.align();

        AtomicSliceRef {
            buf: self,
            start: scale * index.start_per_align,
            end: scale * index.end_per_align,
            texel: index.texel,
        }
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
        &self,
        src: impl ops::RangeBounds<usize>,
        dest: usize,
        f: impl Fn(P) -> Q,
        p: Texel<P>,
        q: Texel<Q>,
    ) {
        let mut that = self;
        TexelMappingBuffer::map_within(&mut that, src, dest, f, p, q)
    }
}

impl cmp::PartialEq for atomic_buf {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        // If they have the same length, they cover the same memory. Do not iterate.
        if (self as *const atomic_buf).addr() == (other as *const atomic_buf).addr() {
            return true;
        }

        // We can iterate these slices in `AtomicPart` at a time. Note that this is not as complex
        // as the `cell_buf` case since it can not cover a partial unit. That complexity only comes
        // with `AtomicSliceRef`.
        let lhs = self.0.iter();
        let rhs = other.0.iter();

        lhs.zip(rhs)
            .all(|(a, b)| a.load(atomic::Ordering::Relaxed) == b.load(atomic::Ordering::Relaxed))
    }
}

impl cmp::PartialEq<[u8]> for atomic_buf {
    fn eq(&self, other: &[u8]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        // We can iterate these slices in `AtomicPart` at a time. Note that this is not as complex
        // as the `cell_buf` case since it can not cover a partial unit. That complexity only comes
        // with `AtomicSliceRef`.
        let lhs = self.0.iter();
        let rhs = other.chunks_exact(mem::size_of::<AtomicPart>());

        lhs.zip(rhs)
            // Let the compiler deal with the potentially unaligned load. However it may run better
            // if we also had an aligned other buffer as a (semi-common) special case? Note how the
            // value loaded from the atomic varies by platform but all integers have that
            // `to_ne_bytes´ method and we iterate the slice by that type's size chunks. Should get
            // optimized away as a compile time constant but we could switch to `array_chunks` in
            // due time.
            .all(|(a, b)| a.load(atomic::Ordering::Relaxed).to_ne_bytes() == *b)
    }
}

impl cmp::Eq for atomic_buf {}

impl cmp::PartialEq for AtomicBuffer {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl cmp::Eq for AtomicBuffer {}

impl TexelMappingBuffer for &'_ atomic_buf {
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
        let src_buffer = self.as_texels(p);
        let target_buffer = self.as_texels(q);

        // FIXME: isn't it particularly inefficient to load values one-by-one? But we offer that
        // primitive. A stack buffer for a statically sized burst of values would be better though.

        for idx in 0..len {
            let source_idx = idx + src;
            let target_idx = idx + dest;
            let source = p.load_atomic(src_buffer.index_one(source_idx));
            let target = f(source);
            q.store_atomic(target_buffer.index_one(target_idx), target);
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
        let src_buffer = self.as_texels(p);
        let target_buffer = self.as_texels(q);

        for idx in (0..len).rev() {
            let source_idx = idx + src;
            let target_idx = idx + dest;
            let source = p.load_atomic(src_buffer.index_one(source_idx));
            let target = f(source);
            q.store_atomic(target_buffer.index_one(target_idx), target);
        }
    }

    fn texel_len<P>(&self, texel: Texel<P>) -> usize {
        self.as_texels(texel).len()
    }
}

impl<'lt, P> AtomicSliceRef<'lt, P> {
    /// Grab a single element.
    ///
    /// Not `get` since it does not return a reference, and we can not use the standard SliceIndex
    /// trait anyways. Also we do not implement the assertion outside of debug for now, it is also
    /// not used for unsafe code.
    #[track_caller]
    pub fn index_one(self, idx: usize) -> AtomicRef<'lt, P> {
        assert!(idx < self.len());

        AtomicRef {
            buf: self.buf,
            start: self.start + idx * self.texel.size(),
            texel: self.texel,
        }
    }

    /// Get a subslice with the specified tuple of bounds.
    ///
    /// Returns `None` if the bounds are out-of-range or if the bounds are otherwise invalid.
    pub fn get_bounds(self, bounds: (ops::Bound<usize>, ops::Bound<usize>)) -> Option<Self> {
        let (start, end) = bounds;
        let len = self.len();

        let start = match start {
            ops::Bound::Included(start) => start,
            ops::Bound::Excluded(start) => start.checked_add(1)?,
            ops::Bound::Unbounded => 0,
        };

        let end = match end {
            ops::Bound::Included(end) => end.checked_add(1)?,
            ops::Bound::Excluded(end) => end,
            ops::Bound::Unbounded => len,
        };

        if start > end || end > len {
            None
        } else {
            Some(AtomicSliceRef {
                buf: self.buf,
                start: self.start + start * self.texel.size(),
                end: self.start + end * self.texel.size(),
                texel: self.texel,
            })
        }
    }

    /// See [`Self::get_bounds`] but generic over the bound type.
    pub fn get(self, bounds: impl core::ops::RangeBounds<usize>) -> Option<Self> {
        let start = bounds.start_bound().cloned();
        let end = bounds.end_bound().cloned();
        self.get_bounds((start, end))
    }

    /// See [`Self::get_bounds`] and panics appropriately.
    #[track_caller]
    pub fn index(self, bounds: impl core::ops::RangeBounds<usize>) -> Self {
        #[cold]
        fn panic_on_bounds() -> ! {
            panic!("Bounds are out of range");
        }

        match self.get(bounds) {
            Some(some) => some,
            None => panic_on_bounds(),
        }
    }

    /// Fill this slice with data from a shared read buffer.
    #[track_caller]
    pub fn read_from_slice(&self, data: &[P]) {
        self.texel.store_atomic_slice(*self, data);
    }

    /// Read from this slice with data from a shared read buffer.
    ///
    /// Note that this reads every single unit as if relaxed.
    #[track_caller]
    pub fn write_to_slice(&self, data: &mut [P]) {
        self.texel.load_atomic_slice(*self, data);
    }

    #[track_caller]
    pub fn split_at(self, at: usize) -> (Self, Self) {
        let left = self.index(..at);
        let right = self.index(at..);
        (left, right)
    }

    /// Reduce the number of bytes covered by this slice.
    #[must_use = "Does not mutate self"]
    #[track_caller]
    pub fn truncate_bytes(self, at: usize) -> Self {
        let len = (self.end - self.start).min(at);
        AtomicSliceRef {
            end: self.start + len,
            ..self
        }
    }

    pub(crate) fn as_ptr_range(self) -> core::ops::Range<*mut P> {
        let base = self.buf.0.as_ptr_range();
        ((base.start as *mut u8).wrapping_add(self.start) as *mut P)
            ..((base.start as *mut u8).wrapping_add(self.end) as *mut P)
    }

    /// Equivalent of [`core::slice::from_ref`] but we have no mutable analogue.
    pub(crate) fn from_ref(value: AtomicRef<'lt, P>) -> Self {
        AtomicSliceRef {
            buf: value.buf,
            start: value.start,
            end: value.start + value.texel.size(),
            texel: value.texel,
        }
    }

    /// Get the number of elements referenced by this slice.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start) / self.texel.size()
    }
}

impl<P> Clone for AtomicSliceRef<'_, P> {
    fn clone(&self) -> Self {
        AtomicSliceRef { ..*self }
    }
}

impl<P> Copy for AtomicSliceRef<'_, P> {}

impl<P> Clone for AtomicRef<'_, P> {
    fn clone(&self) -> Self {
        AtomicRef { ..*self }
    }
}

impl<P> Copy for AtomicRef<'_, P> {}

/// A range representation that casts bytes to a specific texel type.
///
/// Note this type also has the invariant that the identified range fits into memory for the given
/// texel type.
#[derive(Clone, Copy, Debug)]
pub struct TexelRange<T> {
    texel: Texel<T>,
    start_per_align: usize,
    end_per_align: usize,
}

impl<T> TexelRange<T> {
    /// Create a new range from a texel type and a range (in units of `T`).
    pub fn new(texel: Texel<T>, range: ops::Range<usize>) -> Option<Self> {
        let end_byte = range
            .end
            .checked_mul(texel.size())
            .filter(|&n| n <= isize::MAX as usize)?;
        let start_byte = (range.start.min(range.end))
            .checked_mul(texel.size())
            .filter(|&n| n <= isize::MAX as usize)?;

        debug_assert!(
            end_byte % texel.align() == 0,
            "Texel must be valid for its type layout"
        );

        debug_assert!(
            start_byte % texel.align() == 0,
            "Texel must be valid for its type layout"
        );

        Some(TexelRange {
            texel,
            start_per_align: start_byte / texel.align(),
            end_per_align: end_byte / texel.align(),
        })
    }

    /// Construct from a range of bytes.
    ///
    /// The range must be aligned to the type `T` and the length of the range must be a multiple of
    /// the size. However, in contrast to [`Self::new`] it may be skewed with regards to the size
    /// of the type. For instance, a slice `[u8; 3]` may begin one byte into the underlying buffer.
    ///
    /// Note that a range with its end before the start is interpreted as an empty range and only
    /// has to fulfill the alignment requirement for its start byte.
    ///
    /// # Examples
    ///
    /// ```
    /// use image_texel::texels::{U16, TexelRange};
    ///
    /// assert!(TexelRange::from_byte_range(U16, 0..4).is_some());
    /// // Misaligned.
    /// assert!(TexelRange::from_byte_range(U16, 1..5).is_none());
    /// // Okay.
    /// assert!(TexelRange::from_byte_range(U16.array::<4>(), 2..10).is_some());
    /// // Okay but empty.
    /// assert!(TexelRange::from_byte_range(U16.array::<4>(), 2..0).is_some());
    /// ```
    pub fn from_byte_range(texel: Texel<T>, range: ops::Range<usize>) -> Option<Self> {
        let start_byte = range.start;
        let end_byte = range.end.max(start_byte);

        if start_byte % texel.align() != 0
            || end_byte % texel.align() != 0
            || (end_byte - start_byte) % texel.size() != 0
        {
            return None;
        }

        Some(TexelRange {
            texel,
            start_per_align: start_byte / texel.align(),
            end_per_align: end_byte / texel.align(),
        })
    }
}

impl<T> core::ops::Index<TexelRange<T>> for buf {
    type Output = [T];

    fn index(&self, index: TexelRange<T>) -> &Self::Output {
        let scale = index.texel.align();
        let bytes = &self.0[scale * index.start_per_align..scale * index.end_per_align];
        let slice = index.texel.try_to_slice(bytes);
        // We just multiplied the indices by the alignment..
        slice.expect("byte indices validly aligned")
    }
}

impl<T> core::ops::IndexMut<TexelRange<T>> for buf {
    fn index_mut(&mut self, index: TexelRange<T>) -> &mut Self::Output {
        let scale = index.texel.align();
        let bytes = &mut self.0[scale * index.start_per_align..scale * index.end_per_align];
        let slice = index.texel.try_to_slice_mut(bytes);
        // We just multiplied the indices by the alignment..
        slice.expect("byte indices validly aligned")
    }
}

impl Default for &'_ cell_buf {
    fn default() -> Self {
        cell_buf::new(&mut [])
    }
}

impl Default for &'_ atomic_buf {
    fn default() -> Self {
        atomic_buf::new(&mut [])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texels::{MAX, U16, U32, U8};

    // When it's all over.
    struct AlignMeUp<N>([MaxAligned; 0], N);

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

        let contents: &cell_buf = &*buffer;
        let slice: &[cell::Cell<u8>] = contents.as_texels(U8).as_slice_of_cells();
        assert!(cell_buf::from_bytes(slice).is_some());
    }

    #[test]
    fn atomic_buffer() {
        let data = [0, 0, 255, 0, 255, 0, 255, 0, 0];
        let buffer = AtomicBuffer::from(&data[..]);
        // Gets rounded up to the next alignment.
        assert_eq!(buffer.capacity(), Buffer::alloc_len(data.len()) * MAX_ALIGN);

        let alternative = CellBuffer::with_buffer(buffer.to_owned());
        assert_eq!(buffer.capacity(), alternative.capacity());

        let contents: &atomic_buf = &*buffer;
        let slice: AtomicSliceRef<u8> = contents.as_texels(U8);
        assert!(atomic_buf::from_bytes(slice).is_some());
    }

    #[test]
    fn mapping_cells() {
        const LEN: usize = 10;
        // Look, we can actually map over this buffer while it is *not* mutable.
        let buffer = CellBuffer::new(LEN * mem::size_of::<u32>());
        // And receive all the results in this shared copy of our buffer.
        let output_tap = buffer.clone();
        assert!(buffer.ptr_eq(&output_tap));

        buffer
            .as_texels(U32)
            .as_slice_of_cells()
            .iter()
            .enumerate()
            .for_each(|(idx, p)| p.set(idx as u32));

        // Map those numbers in-place.
        buffer.map_within(..LEN, 0, |n: u32| n as u8, U32, U8);
        buffer.map_within(..LEN, 0, |n: u8| n as u32, U8, U32);

        // Back to where we started.
        assert_eq!(
            output_tap.as_texels(U32).as_slice_of_cells()[..LEN]
                .iter()
                .map(cell::Cell::get)
                .collect::<Vec<_>>(),
            (0..LEN as u32).collect::<Vec<_>>()
        );

        // This should work even if we don't map to index 0.
        buffer.map_within(0..LEN, 3 * LEN, |n: u32| n as u8, U32, U8);
        buffer.map_within(3 * LEN..4 * LEN, 0, |n: u8| n as u32, U8, U32);

        assert_eq!(
            output_tap.as_texels(U32).as_slice_of_cells()[..LEN]
                .iter()
                .map(cell::Cell::get)
                .collect::<Vec<_>>(),
            (0..LEN as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn mapping_atomics() {
        const LEN: usize = 10;
        let mut initial_state = Buffer::new(LEN * mem::size_of::<u32>());

        initial_state
            .as_mut_texels(U32)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, p)| *p = idx as u32);

        // Look, we can actually map over this buffer while it is *not* mutable.
        let buffer = AtomicBuffer::with_buffer(initial_state);
        // And receive all the results in this shared copy of our buffer.
        let output_tap = buffer.clone();
        // assert!(buffer.ptr_eq(&output_tap));

        // Map those numbers in-place.
        buffer.map_within(..LEN, 0, |n: u32| n as u8, U32, U8);
        buffer.map_within(..LEN, 0, |n: u8| n as u32, U8, U32);

        // Back to where we started.
        assert_eq!(
            output_tap.to_owned().as_texels(U32)[..LEN].to_vec(),
            (0..LEN as u32).collect::<Vec<_>>()
        );

        // This should work even if we don't map to index 0.
        buffer.map_within(0..LEN, 3 * LEN, |n: u32| n as u8, U32, U8);
        buffer.map_within(3 * LEN..4 * LEN, 0, |n: u8| n as u32, U8, U32);

        assert_eq!(
            output_tap.to_owned().as_texels(U32)[..LEN].to_vec(),
            (0..LEN as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cell_construction() {
        let data = [const { MaxCell::zero() }; 10];
        let _empty = cell_buf::new(&data[..0]);
        let cell = cell_buf::new(&data);

        let (first, tail) = cell.split_at(MAX_ALIGN);
        let another_first = cell_buf::new(&data[..1]);

        let data: Vec<_> = (0u8..).take(MAX_ALIGN).collect();
        U8.store_cell_slice(first.as_texels(U8).as_slice_of_cells(), &data);
        let mut alternative: Vec<_> = (1u8..).take(MAX_ALIGN).collect();
        U8.load_cell_slice(
            another_first.as_texels(U8).as_slice_of_cells(),
            &mut alternative,
        );

        // These two alias, so the read must have worked. Alternative must now be changed.
        assert_eq!(data, alternative);

        U8.load_cell_slice(
            tail.truncate(MAX_ALIGN).as_texels(U8).as_slice_of_cells(),
            &mut alternative,
        );
        assert_ne!(data, alternative);
    }

    #[test]
    #[should_panic]
    fn cell_unaligned_split() {
        let data = [const { MaxCell::zero() }; 10];
        // 1 is not an aligned index.
        cell_buf::new(&data).split_at(1);
    }

    #[test]
    #[should_panic]
    fn cell_oob_split() {
        let data = [const { MaxCell::zero() }; 1];
        // this is out of bounds.
        cell_buf::new(&data).split_at(MAX_ALIGN + 1);
    }

    #[test]
    fn cell_empty() {
        let empty = cell_buf::new(&[]);
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn cell_from_bytes() {
        const SIZE: usize = 16;

        let data = [0u8; SIZE].map(cell::Cell::new);
        let data: AlignMeUp<[_; SIZE]> = AlignMeUp([], data);

        let empty = cell_buf::from_bytes(&data.1[..]).expect("this was properly aligned");
        assert_eq!(empty.len(), SIZE);
    }

    #[test]
    fn cell_unaligned_from_bytes() {
        let data = [const { MaxCell::zero() }; 1];
        let unaligned = &cell_buf::new(&data).as_texels(U8).as_slice_of_cells()[1..];
        assert!(cell_buf::from_bytes(unaligned).is_none());
    }

    #[test]
    fn cell_from_mut_bytes() {
        const SIZE: usize = 16;
        let mut data: AlignMeUp<[_; SIZE]> = AlignMeUp([], [0u8; SIZE]);

        let empty = cell_buf::from_bytes_mut(&mut data.1[..]).expect("this was properly aligned");
        assert_eq!(empty.len(), SIZE);
    }

    #[test]
    fn cell_unaligned_from_mut_bytes() {
        const SIZE: usize = 16;
        let mut data: AlignMeUp<[_; SIZE]> = AlignMeUp([], [0; SIZE]);

        let unaligned = &mut data.1[1..];
        // Should fail since we must not be able to construct a buffer from unaligned bytes.
        assert!(cell_buf::from_bytes_mut(unaligned).is_none());
    }

    #[test]
    fn cell_equality() {
        let data = [const { MaxCell::zero() }; 3];
        let lhs = cell_buf::new(&data[0..1]);
        let rhs = cell_buf::new(&data[1..2]);

        let uneq = cell_buf::new(&data[2..3]);
        uneq.as_texels(U8).as_slice_of_cells()[0].set(1);

        // No `Debug` hence.
        assert!(lhs == lhs, "Must be equal with itself");
        assert!(lhs == rhs, "Must be equal with same data");
        assert!(lhs != uneq, "Must only be equal with same data");

        let mut buffer = [0x42; mem::size_of::<MaxCell>()];
        assert!(*lhs != buffer[..], "Must only be equal with its data");

        U8.load_cell_slice(lhs.as_texels(U8).as_slice_of_cells(), &mut buffer);
        assert!(*lhs == buffer[..], "Must be equal with its data");
    }

    #[test]
    fn atomic_empty() {
        let empty = atomic_buf::new(&[]);
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn atomic_construction() {
        let data = [const { MaxAtomic::zero() }; 10];
        let cell = atomic_buf::new(&data);

        let (first, tail) = cell.split_at(MAX_ALIGN);
        let another_first = atomic_buf::new(&data[..1]);
        assert_eq!(another_first.as_texels(U8).len(), MAX_ALIGN);
        assert_eq!(first.as_texels(U8).len(), MAX_ALIGN);

        let data: Vec<_> = (0u8..).take(MAX_ALIGN).collect();
        first.as_texels(U8).read_from_slice(&data);
        let mut alternative: Vec<_> = (1u8..).take(MAX_ALIGN).collect();
        another_first.as_texels(U8).write_to_slice(&mut alternative);

        // These two alias, so the read must have worked. Alternative must now be changed.
        assert_eq!(data, alternative);

        // And the tail does not alias, so we reset alternative back to zero.
        tail.as_texels(U8)
            .index(..MAX_ALIGN)
            .write_to_slice(&mut alternative);
        assert_ne!(data, alternative);

        let another_first = atomic_buf::from_bytes(first.as_texels(U8))
            .expect("the whole buffer is always aligned");
        another_first.as_texels(U8).write_to_slice(&mut alternative);
        assert_eq!(data, alternative);
    }

    #[test]
    fn atomic_from_bytes() {
        let data = [const { MaxAtomic::zero() }; 1];
        let cell = atomic_buf::new(&data);

        // Best way to get a buffer is to get it from an existing one..
        let data = cell.as_texels(U8);
        let new_buf = atomic_buf::from_bytes(data).expect("this was properly aligned");
        assert_eq!(new_buf.len(), MAX_ALIGN);
    }

    #[test]
    fn atomic_unaligned_from_bytes() {
        let data = [const { MaxAtomic::zero() }; 1];
        let cell = atomic_buf::new(&data);

        let unaligned = cell.as_texels(U8).index(1..);
        assert!(atomic_buf::from_bytes(unaligned).is_none());
    }

    #[test]
    fn atomic_from_mut_bytes() {
        const SIZE: usize = MAX_ALIGN * 2;
        let mut data: AlignMeUp<[_; SIZE]> = AlignMeUp([], [0u8; SIZE]);

        let empty = atomic_buf::from_bytes_mut(&mut data.1[..]).expect("this was properly aligned");
        assert_eq!(empty.len(), SIZE);
    }

    #[test]
    fn atomic_too_small_from_mut_bytes() {
        const SIZE: usize = MAX_ALIGN / 2;
        let mut data: AlignMeUp<[_; SIZE]> = AlignMeUp([], [0; SIZE]);

        let unaligned = &mut data.1[1..];
        // Should fail since we must not be able to construct a buffer out of smaller units to
        // avoid differing type behavior.
        assert!(atomic_buf::from_bytes_mut(unaligned).is_none());
    }

    #[test]
    fn atomic_unaligned_from_mut_bytes() {
        const SIZE: usize = 16;
        let mut data: AlignMeUp<[_; SIZE]> = AlignMeUp([], [0; SIZE]);

        let unaligned = &mut data.1[1..];
        // Should fail since we must not be able to construct a buffer from unaligned bytes.
        assert!(atomic_buf::from_bytes_mut(unaligned).is_none());
    }

    #[test]
    fn atomic_equality() {
        let data = [const { MaxAtomic::zero() }; 3];
        let lhs = atomic_buf::new(&data[0..1]);
        let rhs = atomic_buf::new(&data[1..2]);

        let uneq = atomic_buf::new(&data[2..3]);
        U8.store_atomic(uneq.as_texels(U8).index_one(0), 1);

        // No `Debug` hence.
        assert!(lhs == lhs, "Must be equal with itself");
        assert!(lhs == rhs, "Must be equal with same data");
        assert!(lhs != uneq, "Must only be equal with same data");

        let mut buffer = [0x42; mem::size_of::<MaxCell>()];
        assert!(*lhs != buffer[..], "Must only be equal with its data");

        U8.load_atomic_slice(lhs.as_texels(U8), &mut buffer);
        assert!(*lhs == buffer[..], "Must be equal with its data");
    }

    #[test]
    fn atomic_with_u8() {
        // Check that writing and reading works at different offsets.
        for offset in 0..MAX_ALIGN {
            let slice = [const { MaxAtomic::zero() }; 4];
            let atomic = atomic_buf::new(&slice[..]);

            let mut iota = 0;
            let data = [(); 3 * MAX_ALIGN].map(move |_| {
                let n = iota;
                iota += 1;
                n
            });

            let target = atomic.as_texels(U8).index(offset..).index(..3 * MAX_ALIGN);
            U8.store_atomic_slice(target, &data[..]);

            let mut check = [0; 3 * MAX_ALIGN];
            U8.load_atomic_slice(target, &mut check[..]);

            let cells = [const { core::cell::Cell::new(0) }; 3 * MAX_ALIGN];
            U8.load_atomic_to_cells(target, &cells[..]);

            assert_eq!(data, check);
            assert_eq!(data, cells.map(|x| x.into_inner()));

            let mut check = [0; 4 * MAX_ALIGN];
            U8.load_atomic_slice(atomic.as_texels(U8), &mut check[..]);

            assert_eq!(data, check[offset..][..3 * MAX_ALIGN], "offset {offset}");
        }
    }

    #[test]
    fn atomic_with_u16() {
        use crate::texels::U16;

        // Check that writing and reading works at different offsets.
        for offset in 0..MAX_ALIGN / 2 {
            let slice = [const { MaxAtomic::zero() }; 4];
            let atomic = atomic_buf::new(&slice[..]);

            let mut iota = 0;
            let data = [(); 3 * MAX_ALIGN / 2].map(move |_| {
                let n = iota;
                iota += 1;
                n
            });

            let target = atomic.as_texels(U16).index(offset..).index(..3 * MAX_ALIGN / 2);
            U16.store_atomic_slice(target, &data[..]);

            let mut check = [0; 3 * MAX_ALIGN / 2];
            U16.load_atomic_slice(target, &mut check[..]);

            let cells = [const { core::cell::Cell::new(0) }; 3 * MAX_ALIGN / 2];
            U16.load_atomic_to_cells(target, &cells[..]);

            assert_eq!(data, check);
            assert_eq!(data, cells.map(|x| x.into_inner()));
        }
    }

    #[test]
    fn atomic_from_cells() {
        for offset in 0..4 {
            let data = [const { MaxAtomic::zero() }; 1];
            let lhs = atomic_buf::new(&data[0..1]);

            let data = [const { MaxCell::zero() }; 1];
            let rhs = cell_buf::new(&data[0..1]);

            // Create a value that checks we write to the correct bytes.
            let source = rhs.as_texels(U8).as_slice_of_cells();
            U8.store_cell_slice(&source[4..8], &[0x84; 4]);
            U8.store_cell_slice(&source[2..4], &[1, 2]);
            let source = &source[..8 - offset];
            // Initialize the first 8 bytes of the atomic.
            U8.store_atomic_from_cells(lhs.as_texels(U8).index(offset..8), source);

            let mut buffer = [0x42; mem::size_of::<MaxCell>()];
            U8.load_atomic_slice(lhs.as_texels(U8), &mut buffer);

            assert!(
                buffer[..offset].iter().all(|&x| x == 0),
                "Must still be unset",
            );

            assert!(
                buffer[offset..][..4] == [0, 0, 1, 2],
                "Must contain the data",
            );

            assert!(
                buffer[offset..8][4..].iter().all(|&x| x == 0x84),
                "Must be initialized by tail {:?}",
                &buffer[offset..][4..],
            );
        }
    }

    #[test]
    fn atomic_to_cells() {
        for offset in 0..4 {
            let data = [const { MaxAtomic::zero() }; 1];
            let lhs = atomic_buf::new(&data[0..1]);

            let data = [const { MaxCell::zero() }; 1];
            let rhs = cell_buf::new(&data[0..1]);

            U8.store_atomic_slice(lhs.as_texels(U8).index(4..8), &[0x84; 4]);
            U8.store_atomic_slice(lhs.as_texels(U8).index(offset..).index(..4), &[0, 0, 1, 2]);

            // Create a value that checks we write to the correct bytes.
            let target = rhs.as_texels(U8).as_slice_of_cells();
            // Initialize the first 8 bytes of the atomic.
            U8.load_atomic_to_cells(lhs.as_texels(U8).index(offset..8), &target[..8 - offset]);

            let mut buffer = [0x42; mem::size_of::<MaxCell>()];
            U8.load_cell_slice(target, &mut buffer);

            assert!(
                buffer[..4] == [0, 0, 1, 2],
                "Must contain the data {:?}",
                &buffer[..4],
            );

            assert!(
                buffer[..8 - offset][4..].iter().all(|&x| x == 0x84),
                "Must be initialized by tail {:?}",
                &buffer[..8 - offset][4..],
            );
        }
    }
}
