// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
#![allow(unsafe_code)]

use core::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use core::marker::PhantomData;
use core::{fmt, hash, mem, num, ptr, slice};

use crate::buf::buf;

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

pub(crate) const MAX_ALIGN: usize = 16;

/// A byte-like-type that is aligned to the required max alignment.
///
/// This type does not contain padding and implements `Pod`.
#[derive(Clone, Copy)]
#[repr(align(16))]
#[repr(C)]
pub struct MaxAligned(pub(crate) [u8; 16]);

unsafe impl bytemuck::Zeroable for MaxAligned {}
unsafe impl bytemuck::Pod for MaxAligned {}

pub(crate) mod constants {
    use super::{AsTexel, MaxAligned, Texel};

    macro_rules! constant_texel {
        ($(($name:ident, $type:ty)),*) => {
            $(pub const $name: Texel<$type> = Texel(core::marker::PhantomData) ;
              impl AsTexel for $type {
                  fn texel() -> Texel<Self> {
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

    impl<T: AsTexel> AsTexel for ::core::num::Wrapping<T> {
        fn texel() -> Texel<::core::num::Wrapping<T>> {
            T::texel().num_wrapping()
        }
    }
}

impl<P: bytemuck::Pod> Texel<P> {
    /// Try to construct an instance of the marker.
    ///
    /// If successful, you can freely use it to access the image buffers. This requires:
    /// - The type must have an alignment of *at most* `MAX_ALIGN`.
    /// - The type must *not* be a ZST.
    /// - The type must *not* have any Drop-glue (no drop, any contain not part that is Drop).
    pub fn for_type() -> Option<Self> {
        if mem::align_of::<P>() <= MAX_ALIGN && mem::size_of::<P>() > 0 && !mem::needs_drop::<P>() {
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

impl<P> Texel<P> {
    /// Create a witness certifying `P` as a texel without checks.
    ///
    /// # Safety
    ///
    /// The type `P` must not:
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
    /// [`MaxAligned`]: struct.MaxAligned.html
    pub const unsafe fn new_unchecked() -> Self {
        Texel(PhantomData)
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
    /// TODO: a constructor for Texel<O> based on proof of transmutation from &mut P to &mut O,
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
    pub fn try_to_slice_mut<'buf>(self, bytes: &'buf mut [u8]) -> Option<&'buf [P]> {
        if let Some(slice) = self.try_to_slice(bytes) {
            // SAFETY:
            // - The `pod`-ness is certified by `self`, which makes the bytes a valid
            //   representation of P. Conversely, it makes any P valid as bytes.
            let len = slice.len();
            Some(unsafe { &*ptr::slice_from_raw_parts_mut(bytes.as_ptr() as *mut P, len) })
        } else {
            None
        }
    }

    /// Reinterpret a slice of texel as memory.
    pub fn to_bytes<'buf>(self, texel: &'buf [P]) -> &'buf [u8] {
        self.cast_bytes(texel)
    }

    /// Reinterpret a mutable slice of texel as memory.
    pub fn to_mut_bytes<'buf>(self, texel: &'buf mut [P]) -> &'buf mut [u8] {
        self.cast_mut_bytes(texel)
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

    pub(crate) fn cast_bytes<'buf>(self, texel: &'buf [P]) -> &'buf [u8] {
        // Safety:
        // * lifetime is not changed
        // * keeps the exact same size
        // * validity for byte reading checked by Texel constructor
        unsafe { slice::from_raw_parts(texel.as_ptr() as *const u8, mem::size_of_val(texel)) }
    }

    pub(crate) fn cast_mut_bytes<'buf>(self, texel: &'buf mut [P]) -> &'buf mut [u8] {
        // Safety:
        // * lifetime is not changed
        // * keeps the exact same size
        // * validity as bytes checked by Texel constructor
        unsafe { slice::from_raw_parts_mut(texel.as_ptr() as *mut u8, mem::size_of_val(texel)) }
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
