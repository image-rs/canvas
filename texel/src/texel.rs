// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
#![allow(unsafe_code)]

use core::cell::Cell;
use core::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use core::marker::PhantomData;
use core::{fmt, hash, mem, num, ops, ptr, slice, sync::atomic};

use crate::buf::{atomic_buf, buf, cell_buf, AtomicRef, AtomicSliceRef, TexelRange};

/// Marker struct to denote a texel type.
///
/// Can be constructed only for types that have expected alignment and no byte invariants. It
/// always implements `Copy` and `Clone`, regardless of the underlying type and is zero-sized.
///
/// This is the central encapsulation of unsafety in this crate. It utilizes `bytemuck` for a safe
/// interface but permits other types with an unsafe interface, and offers the cast operations
/// without a bound on the `Pod` trait. Note that `Pod` is a pure marker trait; its properties must
/// hold even if it is not explicitly mentioned. If all constructors (safely or unsafely) ensure
/// that its properties hold we can use `Texel` as a witness type for the bound and subsequently
/// write interfaces to take an instance instead of having a static type bound. This achieves two
/// effects:
/// * Firstly, it makes the interface independent of the chosen transmutation crate. Potentially we
///   will have a method to construct the `Texel` via a `core` trait.
/// * Secondly, it allows creating texel of third-party types for which the bound can not be
///   implemented. Crucially, this includes SIMD representations that would be a burden to support
///   directly. And conversely you can also deal with arbitrary existing texel without a bound in
///   your own interfaces!
pub struct Texel<P: ?Sized>(PhantomData<P>);

/// Marker struct to denote that P is transparently wrapped in O.
///
/// The only way to construct it is by accessing its associated constant which only exists when the
/// bound `bytemuck::TransparentWrapper` holds as required. This encodes a type-level set and is
/// a workaround for such bounds not yet being allowed in `const fn`. Expect this type to be
/// deprecated sooner or later.
pub struct IsTransparentWrapper<P, O>(PhantomData<(P, O)>);

/// Describes a type which can represent a `Texel` and for which this is statically known.
pub trait AsTexel {
    /// Get the texel struct for this type.
    ///
    /// The naive implementation of merely unwrapping the result of `Texel::for_type` **panics** on
    /// any invalid type. This trait should only be implemented when you know for sure that the
    /// type is correct.
    fn texel() -> Texel<Self>;
}

macro_rules! def_max_align {
    (
        match cfg(target) {
            $($($arch:literal)|* => $num:literal),*,
        }

        $(#[$common_attr:meta])*
        struct MaxAligned(..);

        $(#[$atomic_attr:meta])*
        struct MaxAtomic(..);

        $(#[$cell_attr:meta])*
        struct MaxCell(..);
    ) => {
        /// A byte-like-type that is aligned to the required max alignment.
        ///
        /// This type does not contain padding and implements `Pod`. Generally, the alignment and size
        /// requirement is kept small to avoid overhead.
        $(#[$common_attr])*
        $(
            #[cfg_attr(
                any($(target_arch = $arch),*),
                repr(align($num))
            )]
        )*
        pub struct MaxAligned(pub(crate) [u8; MAX_ALIGN]);

        /* Note: We need to be really careful to avoid peril for several reasons.
         *
         * Firstly, the Rust atomic model forbids us from doing unsynchronized access (stores _or_
         * loads) with differing sizes to the same memory location. For now, and for the
         * foreseeable future. Since we do not synchronize the access to the buffer, we must use
         * the same size everywhere.
         *
         * Secondly, using any type other than `AtomicU8` for these makes it hard for us to slice
         * the buffer at arbitrary points. For true references we might work around this by custom
         * metadata, yet this is not stable. Hence, we _must_ use a non-reference type wrapper for
         * the kind of access we need. Or rather, the initial buffer allocation can deref into a
         * reference to a slice of atomics but to slice it we must use our own type. And all
         * operations are implemented to work on full units of this atomic type.
         *
         * At least for relaxed operations, the larger unit is somewhat equivalent. It's certainly
         * at bit of a balance. Larger units might be more costly from destructive interference
         * between different accesses, but small units are costly due to added instructions.
         *
         * View the selection below as a 'best-effort' really.
         **/
        #[cfg(all(
            not(target_has_atomic = "8"),
            not(target_has_atomic = "16"),
            not(target_has_atomic = "32"),
            not(target_has_atomic = "64"),
        ))]
        compile_error!("Synchronous buffer API requires one atomic unsigned type");

        #[cfg(all(
            target_has_atomic = "8",
            not(target_has_atomic = "16"),
            not(target_has_atomic = "32"),
            not(target_has_atomic = "64"),
        ))]
        pub(crate) type AtomicPart = core::sync::atomic::AtomicU8;
        #[cfg(all(
            target_has_atomic = "16",
            not(target_has_atomic = "32"),
            not(target_has_atomic = "64"),
        ))]
        pub(crate) type AtomicPart = core::sync::atomic::AtomicU16;
        #[cfg(all(
            target_has_atomic = "32",
            not(target_has_atomic = "64"),
        ))]
        pub(crate) type AtomicPart = core::sync::atomic::AtomicU32;
        #[cfg(all(
            target_has_atomic = "64",
        ))]
        pub(crate) type AtomicPart = core::sync::atomic::AtomicU64;

        const ATOMIC_PARTS: usize = MAX_ALIGN / core::mem::size_of::<AtomicPart>();

        $(
            #[cfg_attr(
                any($(target_arch = $arch),*),
                repr(align($num))
            )]
        )*
        $(#[$atomic_attr])*
        pub struct MaxAtomic(pub(crate) [AtomicPart; ATOMIC_PARTS]);

        $(
            #[cfg_attr(
                any($(target_arch = $arch),*),
                repr(align($num))
            )]
        )*
        $(#[$cell_attr])*
        pub struct MaxCell(pub(crate) Cell<[u8; MAX_ALIGN]>);

        $(
            #[cfg(
                any($(target_arch = $arch),*),
            )]
            pub(crate) const MAX_ALIGN: usize = $num;
        )*

        #[cfg(
            not(any(
                $(any($(target_arch = $arch),*)),*
            )),
        )]
        pub(crate) const MAX_ALIGN: usize = 8;
    }
}

def_max_align! {
    match cfg(target) {
        "x86" | "x86_64" => 32,
        "arm" => 16,
        "aarch64" => 16,
        "wasm32" => 16,
    }

    /// A byte-like-type that is aligned to the required max alignment.
    ///
    /// This type does not contain padding and implements `Pod`. Generally, the alignment and size
    /// requirement is kept small to avoid overhead.
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct MaxAligned(..);

    /// Atomic equivalence of [`MaxAligned`].
    ///
    /// This contains some instance of [`core::sync::atomic::AtomicU8`].
    struct MaxAtomic(..);

    /// A cell of a byte array equivalent to [`MaxAligned`].
    struct MaxCell(..);
}

unsafe impl bytemuck::Zeroable for MaxAligned {}
unsafe impl bytemuck::Pod for MaxAligned {}

macro_rules! builtin_texel {
    ( $name:ty ) => {
        impl AsTexel for $name {
            fn texel() -> Texel<Self> {
                const _: () = {
                    assert!(Texel::<$name>::check_invariants());
                };

                unsafe { Texel::new_unchecked() }
            }
        }
    };
}

pub(crate) mod constants {
    use super::{AsTexel, MaxAligned, Texel};

    macro_rules! constant_texel {
        ($(($name:ident, $type:ty)),*) => {
            $(pub const $name: Texel<$type> = Texel(core::marker::PhantomData) ;
              impl AsTexel for $type {
                  fn texel() -> Texel<Self> {
                      const _: () = {
                          assert!(Texel::<$type>::check_invariants());
                      };

                      $name
                  }
              }
              )*
        }
    }

    constant_texel!(
        (I8, i8),
        (U8, u8),
        (I16, i16),
        (U16, u16),
        (I32, i32),
        (U32, u32),
        (F32, f32),
        (I64, i64),
        (U64, u64),
        (F64, f64),
        (USIZE, usize),
        (ISIZE, isize),
        (MAX, MaxAligned)
    );

    impl<T: AsTexel> AsTexel for [T; 1] {
        fn texel() -> Texel<[T; 1]> {
            T::texel().array::<1>()
        }
    }

    impl<T: AsTexel> AsTexel for [T; 2] {
        fn texel() -> Texel<[T; 2]> {
            T::texel().array::<2>()
        }
    }

    impl<T: AsTexel> AsTexel for [T; 3] {
        fn texel() -> Texel<[T; 3]> {
            T::texel().array::<3>()
        }
    }

    impl<T: AsTexel> AsTexel for [T; 4] {
        fn texel() -> Texel<[T; 4]> {
            T::texel().array::<4>()
        }
    }

    impl<T: AsTexel> AsTexel for [T; 5] {
        fn texel() -> Texel<[T; 5]> {
            T::texel().array::<5>()
        }
    }

    impl<T: AsTexel> AsTexel for [T; 6] {
        fn texel() -> Texel<[T; 6]> {
            T::texel().array::<6>()
        }
    }

    impl<T: AsTexel> AsTexel for [T; 7] {
        fn texel() -> Texel<[T; 7]> {
            T::texel().array::<7>()
        }
    }

    impl<T: AsTexel> AsTexel for [T; 8] {
        fn texel() -> Texel<[T; 8]> {
            T::texel().array::<8>()
        }
    }

    impl<T: AsTexel> AsTexel for ::core::num::Wrapping<T> {
        fn texel() -> Texel<::core::num::Wrapping<T>> {
            T::texel().num_wrapping()
        }
    }
}

#[cfg(target_arch = "x86")]
mod x64 {
    use super::{AsTexel, Texel};
    use core::arch::x86;

    builtin_texel!(x86::__m128);

    builtin_texel!(x86::__m128);
    builtin_texel!(x86::__m128d);
    builtin_texel!(x86::__m128i);
    builtin_texel!(x86::__m256);
    builtin_texel!(x86::__m256d);
    builtin_texel!(x86::__m256i);
}

#[cfg(target_arch = "x86_64")]
mod x64_64 {
    use super::{AsTexel, Texel};
    use core::arch::x86_64;

    builtin_texel!(x86_64::__m128);
    builtin_texel!(x86_64::__m128d);
    builtin_texel!(x86_64::__m128i);
    builtin_texel!(x86_64::__m256);
    builtin_texel!(x86_64::__m256d);
    builtin_texel!(x86_64::__m256i);
}

#[cfg(target_arch = "arm")]
mod arm { /* all types unstable */
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{AsTexel, Texel};
    use core::arch::aarch64;

    builtin_texel!(aarch64::float64x1_t);
    builtin_texel!(aarch64::float64x1x2_t);
    builtin_texel!(aarch64::float64x1x3_t);
    builtin_texel!(aarch64::float64x1x4_t);
    builtin_texel!(aarch64::float64x2_t);
    builtin_texel!(aarch64::float64x2x2_t);
    builtin_texel!(aarch64::float64x2x3_t);
    builtin_texel!(aarch64::float64x2x4_t);
}

#[cfg(target_arch = "wasm32")]
mod arm {
    use super::{AsTexel, Texel};
    use core::arch::wasm32;

    builtin_texel!(wasm32::v128);
}

impl<P: bytemuck::Pod> Texel<P> {
    /// Try to construct an instance of the marker.
    ///
    /// If successful, you can freely use it to access the image buffers. This requires:
    /// - The type must have an alignment of *at most* `MAX_ALIGN`.
    /// - The type must *not* be a ZST.
    /// - The type must *not* have any Drop-glue (no drop, any contain not part that is Drop).
    pub const fn for_type() -> Option<Self> {
        if Texel::<P>::check_invariants() {
            Some(Texel(PhantomData))
        } else {
            None
        }
    }
}

impl<P, O: bytemuck::TransparentWrapper<P>> IsTransparentWrapper<P, O> {
    pub const CONST: Self = IsTransparentWrapper(PhantomData);
}

/// The **only** ways to construct a `buf`, protecting the alignment invariant.
/// Hint: This is an unsized type so there is no safe way of constructing it.
impl buf {
    pub const ALIGNMENT: usize = MAX_ALIGN;

    /// Wrap bytes in a `buf`.
    ///
    /// The bytes need to be aligned to `ALIGNMENT`.
    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.as_ptr() as usize % Self::ALIGNMENT == 0 {
            // SAFETY: this is an almost trivial cast of unsized references. Additionally, we still
            // guarantee that this is at least aligned to `MAX_ALIGN`.
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
            // SAFETY: this is an almost trivial cast of unsized references. Additionally, we still
            // guarantee that this is at least aligned to `MAX_ALIGN`.
            Some(unsafe { &mut *(bytes as *mut [u8] as *mut Self) })
        } else {
            None
        }
    }
}

impl atomic_buf {
    pub const ALIGNMENT: usize = MAX_ALIGN;

    pub fn from_slice(values: &[MaxAtomic]) -> &Self {
        debug_assert_eq!(values.as_ptr() as usize % Self::ALIGNMENT, 0);
        let ptr = values.as_ptr() as *const AtomicPart;
        let count = values.len() * ATOMIC_PARTS;
        // Safety: these types are binary compatible, they wrap atomics of the same size,  and
        // starting at the same address, with a pointer of the same provenance which will be valid
        // for the whole lifetime.
        //
        // This case relaxes the alignment requirements from `MaxAtomic` to that of the underlying
        // atomic, which allows us to go beyond the public interface.
        //
        // The new size covered by the slice is the same as the input slice, since there are
        // `ATOMIC_PARTS` units within each `MaxAtomic`. The memory invariants of the new type are
        // the same as the old type, which is that we access only with atomics instructions of the
        // size of the `AtomicPart` type.
        let atomics = core::ptr::slice_from_raw_parts::<AtomicPart>(ptr, count);
        // Safety: `atomic_buf` has the same layout as a `[MaxAtomic]` and wraps it transparently.
        unsafe { &*(atomics as *const Self) }
    }

    /// Wrap a sub-slice of bytes from an atomic buffer into a new `atomic_buf`.
    ///
    /// The bytes need to be aligned to `ALIGNMENT`. Returns `None` if these checks fail and return
    /// the newly wrapped buffer in `Some` otherwise.
    pub fn from_bytes(bytes: AtomicSliceRef<u8>) -> Option<&Self> {
        if bytes.start % Self::ALIGNMENT == 0 {
            let offset = bytes.start / core::mem::size_of::<AtomicPart>();
            let len = bytes.len().div_ceil(core::mem::size_of::<AtomicPart>());
            let buffer = &bytes.buf.0[offset..][..len];
            // Safety: these types are binary compatible. The metadata is also the same, as both
            // types encapsulate a slice of `AtomicPart`-sized types.
            Some(unsafe { &*(buffer as *const _ as *const Self) })
        } else {
            None
        }
    }

    /// Wrap bytes in an atomic `buf`.
    ///
    /// The bytes need to be aligned to `ALIGNMENT`. Additionally the length must be a multiple of
    /// the `MaxAtomic` size's units. Returns `None` if these checks fail and return the newly
    /// wrapped buffer in `Some` otherwise.
    pub fn from_bytes_mut(bytes: &mut [u8]) -> Option<&Self> {
        if bytes.as_ptr() as usize % Self::ALIGNMENT != 0 {
            None
        } else if bytes.len() % core::mem::size_of::<MaxAtomic>() != 0 {
            None
        } else {
            let len = bytes.len() / core::mem::size_of::<MaxAtomic>();
            let ptr = bytes.as_mut_ptr() as *mut MaxAtomic;
            // SAFETY: We fulfill the alignment and length requirements for this cast, i.e. there
            // are enough bytes available in this slice. Additionally, we still guarantee that this
            // is at least aligned to `MAX_ALIGN`. We also have the shared read-write provenance on
            // our pointer that a shared reference to atomic requires.
            let slice = unsafe { &*ptr::slice_from_raw_parts(ptr, len) };
            Some(atomic_buf::from_slice(slice))
        }
    }
}

impl cell_buf {
    pub const ALIGNMENT: usize = MAX_ALIGN;

    pub fn from_slice(values: &[MaxCell]) -> &Self {
        debug_assert_eq!(values.as_ptr() as usize % Self::ALIGNMENT, 0);
        let ptr = values.as_ptr() as *const Cell<u8>;
        let count = core::mem::size_of_val(values);
        // Safety: constructs a pointer to a slice validly covering exactly the values in the
        // input. The byte length is determined by `size_of_val` and starting at the same address,
        // with a pointer of the same provenance which will be valid for the whole lifetime. The
        // memory invariants of the new type are the same as the old type, which is that we access
        // only with atomics instructions of the size of the `AtomicPart` type.
        let memory = core::ptr::slice_from_raw_parts::<Cell<u8>>(ptr, count);
        // Safety: these types are binary compatible, they wrap memory of the same size.
        // This case relaxes the alignment requirements from `MaxAtomic` to that of the underlying
        // atomic, which allows us to go beyond the public interface.
        unsafe { &*(memory as *const Self) }
    }

    /// Interpret a slice of bytes in an unsynchronized shared `cell_buf`.
    ///
    /// The bytes need to be aligned to `ALIGNMENT`.
    pub fn from_bytes(bytes: &[Cell<u8>]) -> Option<&Self> {
        if bytes.as_ptr() as usize % Self::ALIGNMENT == 0 {
            // Safety: these types are binary compatible. The metadata is also the same, as both
            // types encapsulate a slice of `u8`-sized types.
            Some(unsafe { &*(bytes as *const [_] as *const Cell<[u8]> as *const cell_buf) })
        } else {
            None
        }
    }

    /// Wrap bytes in an unsynchronized shared `cell_buf`.
    ///
    /// The bytes need to be aligned to `ALIGNMENT`.
    pub fn from_bytes_mut(bytes: &mut [u8]) -> Option<&Self> {
        let slice = Cell::from_mut(bytes).as_slice_of_cells();
        Self::from_bytes(slice)
    }
}

impl<P> Texel<P> {
    /// Create a witness certifying `P` as a texel without checks.
    ///
    /// # Safety
    ///
    /// The type `P` must __not__:
    /// * have any validity invariants, i.e. is mustn't contain any padding.
    /// * have any safety invariants. This implies it can be copied.
    /// * have an alignment larger than [`MaxAligned`].
    /// * be a zero-size type.
    ///
    /// Furthermore, tentatively, the type must not have any drop glue. That is its members are all
    /// simple types without Drop implementations. This requirement exists mainly to avoid code
    /// accidentally leaking instances, and ensures that copies created from their byte
    /// representation—which is safe according to the other invairants— do not cause unexpected
    /// effects.
    ///
    /// Note that the alignment requirement with regards to `MaxAligned` is __architecture
    /// dependent__ as the exact bound varies across the `target_arch` feature. Where possible, add
    /// static assertions to each call site of this function.
    ///
    /// [`MaxAligned`]: struct.MaxAligned.html
    pub const unsafe fn new_unchecked() -> Self {
        debug_assert!(Self::check_invariants());
        Texel(PhantomData)
    }

    /// Note this isn't exhaustive. Indeed, we have no way to check for padding.
    pub(crate) const fn check_invariants() -> bool {
        mem::align_of::<P>() <= MAX_ALIGN && mem::size_of::<P>() > 0 && !mem::needs_drop::<P>()
    }

    /// Proxy of `core::mem::align_of`.
    pub const fn align(self) -> usize {
        mem::align_of::<P>()
    }

    /// Proxy of `core::mem::size_of`.
    pub const fn size(self) -> usize {
        mem::size_of::<P>()
    }

    /// Publicly visible function to use the guarantee of non-ZST.
    pub const fn size_nz(self) -> core::num::NonZeroUsize {
        match core::num::NonZeroUsize::new(self.size()) {
            None => panic!(""),
            Some(num) => num,
        }
    }

    // A number of constructors that are technically unsafe. Note that we could write them as safe
    // code here to pad our stats but they are not checked by the type system so it's risky. Better
    // explain their safety in the code as comments.

    /// Construct a texel as an array of no elements.
    ///
    /// # Panics
    ///
    /// This function panics when called with `N` equal to 0.
    pub const fn array<const N: usize>(self) -> Texel<[P; N]> {
        if N == 0 {
            panic!()
        }

        // Safety:
        // * has no validity/safety invariants
        // * has the same alignment as P which is not larger then MaxAligned
        unsafe { Texel::new_unchecked() }
    }

    /// Construct a texel by wrapping into a transparent wrapper.
    ///
    /// TODO: a constructor for `Texel<O>` based on proof of transmutation from &mut P to &mut O,
    /// based on the standard transmutation RFC. This is more flexible than bytemuck's
    /// TransparentWrapper trait.
    pub const fn transparent_wrap<O>(self, _: IsTransparentWrapper<P, O>) -> Texel<O> {
        // Safety:
        // * P and O must have the same invariants, none
        // * P and O have the same alignment
        unsafe { Texel::new_unchecked() }
    }

    /// Construct a texel by unwrapping a transparent wrapper.
    pub const fn transparent_unwrap<O>(self, _: IsTransparentWrapper<O, P>) -> Texel<O> {
        // Safety:
        // * P and O must have the same invariants, none
        // * P and O have the same alignment
        unsafe { Texel::new_unchecked() }
    }

    /// Construct a texel that contains a number in the standard `Wrapping` type.
    pub const fn num_wrapping(self) -> Texel<num::Wrapping<P>> {
        // * Texel<P> = Self certifies the byte properties.
        // * `core::num::Wrapping` is `repr(transparent)
        unsafe { Texel::new_unchecked() }
    }
}

impl<T, const N: usize> Texel<[T; N]> {
    /// Construct a texel, from an array of elements.
    pub const fn array_element(self) -> Texel<T> {
        // Safety:
        // We'll see that all properties are implied by _any_ suitable array.
        // - The type must have an alignment of *at most* `MAX_ALIGN`. Array and inner type have
        //   the same alignment.
        // - The type must *not* be a ZST. The array would otherwise be a ZST.
        // - The type must *not* have any Drop-glue (no drop, any contain not part that is Drop).
        //   The array would otherwise have Drop-glue.
        unsafe { Texel::new_unchecked() }
    }
}

/// Operations that can be performed based on the evidence of Texel.
impl<P> Texel<P> {
    /// Copy a texel.
    ///
    /// Note that this does not require `Copy` because that requirement was part of the
    /// requirements of constructing this `Texel` witness.
    pub fn copy_val(self, val: &P) -> P {
        // SAFETY: by the constructor, this type can be copied byte-by-byte.
        unsafe { ptr::read(val) }
    }

    pub fn copy_cell(self, val: &Cell<P>) -> P {
        // SAFETY: by the constructor, this inner type can be copied byte-by-byte. And `Cell` is a
        // transparent wrapper so it can be read byte-by-byte as well.
        unsafe { ptr::read(val) }.into_inner()
    }

    /// Efficiently store a slice of shared read values to cells.
    ///
    /// We choose an outer slice for the parameter only since the standard library offers the
    /// transposition out of the type parameter, but not its inverse yet. Call
    /// [`Cell::as_slice_of_cells`] as needed.
    #[track_caller]
    pub fn store_cell_slice(self, val: &[Cell<P>], from: &[P]) {
        assert_eq!(from.len(), val.len());
        // SAFETY: by the constructor, this inner type can be copied byte-by-byte. And `Cell` is a
        // transparent wrapper. By our assertion the slices are of the same length. Note we do not
        // assert these slices to be non-overlapping! We could have `P = Cell<X>` and then it's
        // unclear if Rust allows these to overlap or not. I guess we currently have that `Cell<X>`
        // is never `Copy` so we couldn't have such a `Texel` but alas that negative impl is not
        // guaranteed by any logic I came across.
        unsafe {
            ptr::copy(
                from.as_ptr(),
                // SAFETY: the slice of `Cell`s is all `UnsafeCell`.
                //
                // <https://github.com/rust-lang/rust/issues/88248#issuecomment-2397394716>
                (val as *const [Cell<P>] as *mut [Cell<P>]).cast(),
                from.len(),
            )
        }
    }

    /// Efficiently copy a slice of values from cells to an owned buffer.
    ///
    /// We choose an outer slice for the parameter only since the standard library offers the
    /// transposition out of the type parameter, but not its inverse yet. Call
    /// [`Cell::as_slice_of_cells`] as needed.
    #[track_caller]
    pub fn load_cell_slice(self, val: &[Cell<P>], into: &mut [P]) {
        assert_eq!(into.len(), val.len());
        // SAFETY: see `store_cell_slice` but since we have a mutable reference to the target we
        // can assume it does not overlap.
        unsafe {
            ptr::copy_nonoverlapping(
                (val as *const [Cell<P>]).cast(),
                into.as_mut_ptr(),
                into.len(),
            )
        }
    }

    /// Load a value from an atomic slice.
    ///
    /// The results is only correct if no concurrent modification occurs. The library promises
    /// *basic soundness* but no particular defined behaviour under parallel modifications to the
    /// memory bytes which describe the value to be loaded.
    ///
    /// Each atomic unit is read at most once.
    pub fn load_atomic(self, val: AtomicRef<P>) -> P {
        // SAFETY: by `Texel` being a POD this is a valid representation.
        let mut value = unsafe { core::mem::zeroed::<P>() };
        let slice = AtomicSliceRef::from_ref(val);
        self.load_atomic_slice_unchecked(slice, core::slice::from_mut(&mut value));
        value
    }

    /// Load values from an atomic slice.
    ///
    /// The results is only correct if no concurrent modification occurs. The library promises
    /// *basic soundness* but no particular defined behaviour under parallel modifications to the
    /// memory bytes which describe the value to be loaded.
    ///
    /// Each atomic unit is read at most once.
    ///
    /// # Panics
    ///
    /// This method panics if the slice and the target buffer do not have the same logical length.
    #[track_caller]
    pub fn load_atomic_slice(self, val: AtomicSliceRef<P>, into: &mut [P]) {
        assert_eq!(val.len(), into.len());
        self.load_atomic_slice_unchecked(val, into);
    }

    fn load_atomic_slice_unchecked(self, val: AtomicSliceRef<P>, into: &mut [P]) {
        let offset = val.start / core::mem::size_of::<AtomicPart>();
        let mut initial_skip = val.start % core::mem::size_of::<AtomicPart>();
        let mut target = self.to_mut_bytes(into);

        let mut buffer = val.buf.0[offset..].iter();
        // By the invariants of `AtomicRef`, that number of bytes is in-bounds.
        let mut load = buffer.next().unwrap().load(atomic::Ordering::Relaxed);

        loop {
            let input = &bytemuck::bytes_of(&load)[initial_skip..];
            let copy_len = input.len().min(target.len());
            target[..copy_len].copy_from_slice(&input[..copy_len]);
            target = &mut target[copy_len..];

            if target.is_empty() {
                break;
            }

            load = buffer.next().unwrap().load(atomic::Ordering::Relaxed);
            initial_skip = 0;
        }
    }

    /// Store a value to an atomic slice.
    ///
    /// The results is only correct if no concurrent modification occurs. The library promises
    /// *basic soundness* but no particular defined behaviour under parallel modifications to the
    /// memory bytes which describe the value to be store.
    ///
    /// Provides the same wait-freeness as the underlying platform for `fetch_*` instructions, that
    /// is this does not use `compare_exchange_weak`. This implies that concurrent modifications to
    /// bytes *not* covered by this particular representation will not inherently block progress.
    pub fn store_atomic(self, val: AtomicRef<P>, value: P) {
        let slice = AtomicSliceRef::from_ref(val);
        self.store_atomic_slice_unchecked(slice, core::slice::from_ref(&value));
    }

    /// Store values to an atomic slice.
    ///
    /// The results is only correct if no concurrent modification occurs. The library promises
    /// *basic soundness* but no particular defined behaviour under parallel modifications to the
    /// memory bytes which describe the value to be store.
    ///
    /// Provides the same wait-freeness as the underlying platform for `fetch_*` instructions, that
    /// is this does not use `compare_exchange_weak`. This implies that concurrent modifications to
    /// bytes *not* covered by this particular representation will not inherently block progress.
    ///
    /// # Panics
    ///
    /// This method panics if the slice and the source buffer do not have the same logical length.
    #[track_caller]
    pub fn store_atomic_slice(self, val: AtomicSliceRef<P>, source: &[P]) {
        assert_eq!(val.len(), source.len());
        self.store_atomic_slice_unchecked(val, source);
    }

    fn store_atomic_slice_unchecked(self, val: AtomicSliceRef<P>, from: &[P]) {
        let offset = val.start / core::mem::size_of::<AtomicPart>();
        let mut initial_skip = val.start % core::mem::size_of::<AtomicPart>();

        let mut source = self.to_bytes(from);
        let mut buffer = val.buf.0[offset..].iter();

        loop {
            let into = buffer.next().unwrap();
            let original = into.load(atomic::Ordering::Relaxed);
            let mut value = original;

            let target = &mut bytemuck::bytes_of_mut(&mut value)[initial_skip..];
            let copy_len = target.len().min(source.len());
            target[..copy_len].copy_from_slice(&source[..copy_len]);
            source = &source[copy_len..];

            // Any bits we did not modify, including those outside our own range, will not get
            // modified by this instruction. This provides the basic conflict guarantee.
            into.fetch_xor(original ^ value, atomic::Ordering::Relaxed);

            if source.is_empty() {
                break;
            }

            initial_skip = 0;
        }
    }

    /// Reinterpret a slice of aligned bytes as a slice of the texel.
    ///
    /// Note that the size (in bytes) of the slice will be shortened if the size of `P` is not a
    /// divisor of the input slice's size.
    pub fn to_slice<'buf>(self, buffer: &'buf [MaxAligned]) -> &'buf [P] {
        self.cast_buf(buf::new(buffer))
    }

    /// Reinterpret a slice of aligned bytes as a mutable slice of the texel.
    ///
    /// Note that the size (in bytes) of the slice will be shortened if the size of `P` is not a
    /// divisor of the input slice's size.
    pub fn to_mut_slice<'buf>(self, buffer: &'buf mut [MaxAligned]) -> &'buf mut [P] {
        self.cast_mut_buf(buf::new_mut(buffer))
    }

    /// Try to reinterpret a slice of bytes as a slice of the texel.
    ///
    /// This returns `Some` if the buffer is suitably aligned, and `None` otherwise.
    pub fn try_to_slice<'buf>(self, bytes: &'buf [u8]) -> Option<&'buf [P]> {
        if bytes.as_ptr() as usize % mem::align_of::<P>() == 0 {
            // SAFETY:
            // - The `pod`-ness is certified by `self`, which makes the bytes a valid
            //   representation of P.
            // - The total size is at most `bytes` by construction.
            let len = bytes.len() / mem::size_of::<P>();
            Some(unsafe { &*ptr::slice_from_raw_parts(bytes.as_ptr() as *const P, len) })
        } else {
            None
        }
    }

    /// Try to reinterpret a slice of bytes as a slice of the texel.
    ///
    /// This returns `Some` if the buffer is suitably aligned, and `None` otherwise.
    pub fn try_to_slice_mut<'buf>(self, bytes: &'buf mut [u8]) -> Option<&'buf mut [P]> {
        if let Some(slice) = self.try_to_slice(bytes) {
            // SAFETY:
            // - The `pod`-ness is certified by `self`, which makes the bytes a valid
            //   representation of P. Conversely, it makes any P valid as bytes.
            let len = slice.len();
            Some(unsafe { &mut *ptr::slice_from_raw_parts_mut(bytes.as_mut_ptr() as *mut P, len) })
        } else {
            None
        }
    }

    /// Reinterpret a shared slice as a some particular type.
    ///
    /// Note that the size (in bytes) of the slice will be shortened if the size of `P` is not a
    /// divisor of the input slice's size.
    pub fn to_cell<'buf>(self, buffer: &'buf [MaxCell]) -> &'buf Cell<[P]> {
        cell_buf::from_slice(buffer).as_texels(self)
    }

    /// Reinterpret a slice of texel as memory.
    ///
    /// Note that you can convert a reference to a single value by [`core::slice::from_ref`].
    pub fn try_to_cell<'buf>(self, bytes: &'buf [Cell<u8>]) -> Option<&'buf Cell<[P]>> {
        // Safety:
        // - The `pod`-ness certified by `self` ensures the cast of the contents of the memory is
        //   valid. All representations are a valid P and conversely and P is valid as bytes. Since
        //   Cell is a transparent wrapper the types are compatible.
        // - We uphold the share invariants of `Cell`, which are trivial (less than those required
        //   and provided by a shared reference).
        if bytes.as_ptr() as usize % mem::align_of::<P>() == 0 {
            let len = bytes.len() / mem::size_of::<P>();
            let ptr = ptr::slice_from_raw_parts(bytes.as_ptr() as *const P, len);
            Some(unsafe { &*(ptr as *const Cell<[P]>) })
        } else {
            None
        }
    }

    /// Reinterpret a slice of atomically access memory with a type annotation.
    pub fn try_to_atomic<'buf>(
        self,
        bytes: AtomicSliceRef<'buf, u8>,
    ) -> Option<AtomicSliceRef<'buf, P>> {
        if bytes.start % mem::align_of::<P>() == 0 {
            let end = bytes.end - bytes.end % mem::align_of::<P>();
            Some(AtomicSliceRef {
                buf: bytes.buf,
                start: bytes.start,
                end,
                texel: self,
            })
        } else {
            None
        }
    }

    /// Reinterpret a slice of texel as memory.
    ///
    /// Note that you can convert a reference to a single value by [`core::slice::from_ref`].
    pub fn to_bytes<'buf>(self, texel: &'buf [P]) -> &'buf [u8] {
        // Safety:
        // * lifetime is not changed
        // * keeps the exact same size
        // * validity for byte reading checked by Texel constructor
        unsafe { slice::from_raw_parts(texel.as_ptr() as *const u8, mem::size_of_val(texel)) }
    }

    /// Reinterpret a mutable slice of texel as memory.
    ///
    /// Note that you can convert a reference to a single value by [`core::slice::from_mut`].
    pub fn to_mut_bytes<'buf>(self, texel: &'buf mut [P]) -> &'buf mut [u8] {
        // Safety:
        // * lifetime is not changed
        // * keeps the exact same size
        // * validity as bytes checked by Texel constructor
        unsafe { slice::from_raw_parts_mut(texel.as_mut_ptr() as *mut u8, mem::size_of_val(texel)) }
    }

    /// Reinterpret a slice of texel as memory.
    ///
    /// Note that you can convert a reference to a single value by [`core::slice::from_ref`].
    pub fn cell_bytes<'buf>(self, texel: &'buf [Cell<P>]) -> &'buf Cell<[u8]> {
        let ptr: *const [u8] =
            { ptr::slice_from_raw_parts(texel.as_ptr() as *const u8, mem::size_of_val(texel)) };

        // Safety:
        // * lifetime is not changed
        // * kept the exact same size
        // * validity for byte representations both ways checked by Texel constructor
        unsafe { &*(ptr as *const Cell<[u8]>) }
    }

    /// Reinterpret a slice of atomically modified texels as atomic bytes.
    pub fn atomic_bytes<'buf>(self, texel: AtomicSliceRef<'buf, P>) -> AtomicSliceRef<'buf, u8> {
        AtomicSliceRef {
            buf: texel.buf,
            start: texel.start,
            end: texel.end,
            texel: constants::U8,
        }
    }

    /// Compare two cell slices by memory, not by any content equality.
    ///
    /// TODO: expose this, but under what name?
    pub(crate) fn cell_memory_eq<'a, 'b>(self, a: &'a [Cell<P>], b: &'b [Cell<P>]) -> bool {
        let len = mem::size_of_val(a);

        if len != mem::size_of_val(b) {
            return false;
        }

        // SAFETY: the same reasoning applies for both.
        // - this covers the exact memory range as the underlying slice of cells.
        // - the Texel certifies it is initialized memory.
        // - the lifetime is the same.
        // - the memory in the slice is not mutated. This is a little more subtle but `Cell` is not
        //   `Sync` so this thread is the only that could modify those contents currently as we
        //   have a reference to those contents. But also in this thread this function _is
        //   currently running_ and so it suffices that it does not to modify the contents. It does
        //   not access the slice through the cell in any way.
        // - the total size is at most `isize::MAX` since it was already a reference to it.
        let lhs: &'a [u8] = unsafe { slice::from_raw_parts(a.as_ptr() as *const u8, len) };
        let rhs: &'b [u8] = unsafe { slice::from_raw_parts(b.as_ptr() as *const u8, len) };

        lhs == rhs
    }

    /// Compare a slices with untyped memory.
    ///
    /// TODO: expose this, but under what name?
    pub(crate) fn cell_bytes_eq<'a, 'b>(self, a: &'a [Cell<P>], rhs: &[u8]) -> bool {
        let len = mem::size_of_val(a);

        if len != mem::size_of_val(rhs) {
            return false;
        }

        // SAFETY: see `cell_memory_eq`.
        let lhs: &'a [u8] = unsafe { slice::from_raw_parts(a.as_ptr() as *const u8, len) };

        // Really these two should not be overlapping! If the compiler knew, maybe a better memory
        // compare that is more aware of the cache effects of loading? But to be honest it should
        // not matter much.
        debug_assert!({
            let a_range = lhs.as_ptr_range();
            let b_range = rhs.as_ptr_range();

            a_range.end <= b_range.start || b_range.end <= a_range.start
        });

        lhs == rhs
    }

    pub(crate) fn cast_buf<'buf>(self, buffer: &'buf buf) -> &'buf [P] {
        debug_assert_eq!(buffer.as_ptr() as usize % mem::align_of::<MaxAligned>(), 0);
        debug_assert_eq!(buffer.as_ptr() as usize % mem::align_of::<P>(), 0);
        // Safety:
        // * data is valid for reads as memory size is not enlarged
        // * lifetime is not changed
        // * validity for arbitrary data as required by Texel constructor
        // * alignment checked by Texel constructor
        // * the size fits in an allocation, see first bullet point.
        unsafe {
            slice::from_raw_parts(
                buffer.as_ptr() as *const P,
                buffer.len() / self.size_nz().get(),
            )
        }
    }

    pub(crate) fn cast_mut_buf<'buf>(self, buffer: &'buf mut buf) -> &'buf mut [P] {
        debug_assert_eq!(buffer.as_ptr() as usize % mem::align_of::<MaxAligned>(), 0);
        debug_assert_eq!(buffer.as_ptr() as usize % mem::align_of::<P>(), 0);
        // Safety:
        // * data is valid for reads and writes as memory size is not enlarged
        // * lifetime is not changed
        // * validity for arbitrary data as required by Texel constructor
        // * alignment checked by Texel constructor
        // * the size fits in an allocation, see first bullet point.
        unsafe {
            slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut P,
                buffer.len() / self.size_nz().get(),
            )
        }
    }

    /// Construct a range indexing to a slice of this texel.
    ///
    /// See [`TexelRange::new`] as this is just a proxy.
    ///
    /// ```
    /// use image_texel::{texels::{U16, buf}, TexelBuffer};
    ///
    /// let buffer = TexelBuffer::with_elements(&[1u32, 2, 3, 4]);
    /// let range = U16.to_range(4..8).unwrap();
    /// let u16_view = &buffer.as_buf()[range];
    ///
    /// assert_eq!(u16_view.len(), 4);
    /// // This view extends over the `3u32` and `4u32` elements.
    /// // Results depend on native endianess of the `u32` type.
    /// assert!(u16_view[0] == 3 || u16_view[1] == 3);
    /// assert!(u16_view[2] == 4 || u16_view[3] == 4);
    /// ```
    pub fn to_range(self, range: ops::Range<usize>) -> Option<TexelRange<P>> {
        TexelRange::new(self, range)
    }
}

const _: () = {
    const fn atomic_is_size_equivalent_of_aligned() {}
    const fn atomic_is_align_equivalent_of_aligned() {}

    [atomic_is_size_equivalent_of_aligned()]
        [!(core::mem::size_of::<MaxAtomic>() == core::mem::size_of::<MaxAligned>()) as usize];

    [atomic_is_align_equivalent_of_aligned()]
        [!(core::mem::align_of::<MaxAtomic>() == core::mem::align_of::<MaxAligned>()) as usize];
};

impl MaxAtomic {
    /// Create a vector of atomic zero-bytes.
    pub const fn zero() -> Self {
        const Z: AtomicPart = AtomicPart::new(0);
        MaxAtomic([Z; ATOMIC_PARTS])
    }

    /// Create a vector from values initialized synchronously.
    pub fn new(contents: MaxAligned) -> Self {
        let mut result = Self::zero();
        let from = bytemuck::bytes_of(&contents);
        let from = from.chunks_exact(core::mem::size_of::<AtomicPart>());

        for (part, src) in result.0.iter_mut().zip(from) {
            let to = bytemuck::bytes_of_mut(AtomicPart::get_mut(part));
            to.copy_from_slice(src);
        }

        result
    }

    /// Unwrap an owned value.
    pub fn into_inner(mut self) -> MaxAligned {
        let mut result = MaxAligned([0; MAX_ALIGN]);
        let from = bytemuck::bytes_of_mut(&mut result);
        let from = from.chunks_exact_mut(core::mem::size_of::<AtomicPart>());

        for (part, to) in self.0.iter_mut().zip(from) {
            let src = bytemuck::bytes_of(AtomicPart::get_mut(part));
            to.copy_from_slice(src);
        }

        result
    }

    /// Load the data into an owned value.
    pub fn load(&self, ordering: atomic::Ordering) -> MaxAligned {
        let mut result = MaxAligned([0; MAX_ALIGN]);
        let from = bytemuck::bytes_of_mut(&mut result);
        let from = from.chunks_exact_mut(core::mem::size_of::<AtomicPart>());

        for (part, to) in self.0.iter().zip(from) {
            let data = part.load(ordering);
            let src = bytemuck::bytes_of(&data);
            to.copy_from_slice(src);
        }

        result
    }
}

impl MaxCell {
    /// Create a vector of atomic zero-bytes.
    pub const fn zero() -> Self {
        MaxCell(Cell::new([0; MAX_ALIGN]))
    }

    /// Create a vector from values initialized synchronously.
    pub fn new(contents: MaxAligned) -> Self {
        MaxCell(Cell::new(contents.0))
    }

    /// Overwrite the contents with new information from another cell.
    pub fn set(&self, newval: &Self) {
        self.0.set(newval.0.get())
    }

    /// Read the current contents from this cell into an owned value.
    pub fn get(&self) -> MaxAligned {
        MaxAligned(self.0.get())
    }

    /// Unwrap an owned value.
    pub fn into_inner(self) -> MaxAligned {
        MaxAligned(self.0.into_inner())
    }
}

/// This is a pure marker type.
impl<P> Clone for Texel<P> {
    fn clone(&self) -> Self {
        Texel(PhantomData)
    }
}

impl<P> PartialEq for Texel<P> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<P> Eq for Texel<P> {}

impl<P> PartialOrd for Texel<P> {
    fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

impl<P> Ord for Texel<P> {
    fn cmp(&self, _: &Self) -> Ordering {
        Ordering::Equal
    }
}

/// This is a pure marker type.
impl<P> Copy for Texel<P> {}

impl<P> fmt::Debug for Texel<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Texel")
            .field("size", &self.size())
            .field("align", &self.align())
            .finish()
    }
}

impl<P> hash::Hash for Texel<P> {
    fn hash<H: hash::Hasher>(&self, _: &mut H) {}
}
