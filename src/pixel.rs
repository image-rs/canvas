// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
#![allow(unsafe_code)]

use core::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use core::marker::PhantomData;
use core::{fmt, hash, mem, ptr, slice};

use crate::buf::buf;

/// Marker struct to denote a pixel type.
///
/// Can be constructed only for types that have expected alignment and no byte invariants. It
/// always implements `Copy` and `Clone`, regardless of the underlying type and is zero-sized.
///
/// This is the central encapsulation of unsafety in this crate. It utilizes `bytemuck` for a safe
/// interface but permits other types with an unsafe interface, and offers the cast operations
/// without a bound on the `Pod` trait. Note that `Pod` is a pure marker trait; its properties must
/// hold even if it is not explicitly mentioned. If all constructors (safely or unsafely) ensure
/// that its properties hold we can use `Pixel` as a witness type for the bound and subsequently
/// write interfaces to take an instance instead of having a static type bound. This achieves two
/// effects:
/// * Firstly, it makes the interface independent of the chosen transmutation crate. Potentially we
///   will have a method to construct the `Pixel` via a `core` trait.
/// * Secondly, it allows creating pixels of third-party types for which the bound can not be
///   implemented. Crucially, this includes SIMD representations that would be a burden to support
///   directly. And conversely you can also deal with arbitrary existing pixels without a bound in
///   your own interfaces!
pub struct Pixel<P: ?Sized>(PhantomData<P>);

/// Describes a type which can represent a `Pixel` and for which this is statically known.
pub trait AsPixel {
    /// Get the pixel struct for this type.
    ///
    /// The naive implementation of merely unwrapping the result of `Pixel::for_type` **panics** on
    /// any invalid type. This trait should only be implemented when you know for sure that the
    /// type is correct.
    fn pixel() -> Pixel<Self>;
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
    use super::{AsPixel, MaxAligned, Pixel};

    macro_rules! constant_pixels {
        ($(($name:ident, $type:ty)),*) => {
            $(pub const $name: Pixel<$type> = Pixel(core::marker::PhantomData) ;
              impl AsPixel for $type {
                  fn pixel() -> Pixel<Self> {
                      $name
                  }
              }
              )*
        }
    }

    constant_pixels!(
        (EMPTY, ()),
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
        (RGB, [u8; 3]),
        (RGBA, [u8; 4]),
        (MAX, MaxAligned)
    );
}

impl<P: bytemuck::Pod> Pixel<P> {
    /// Try to construct an instance of the marker.
    ///
    /// If successful, you can freely use it to access the image buffers.
    pub fn for_type() -> Option<Self> {
        if mem::align_of::<P>() <= MAX_ALIGN && !mem::needs_drop::<P>() {
            Some(Pixel(PhantomData))
        } else {
            None
        }
    }
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

impl<P> Pixel<P> {
    /// Create a witness certifying `P` as a pixel without checks.
    ///
    /// # Safety
    ///
    /// The type `P` must not:
    /// * have any validity invariants, i.e. is mustn't contain any padding.
    /// * have any safety invariants. This implies it can be copied.
    /// * have an alignment larger than [`MaxAligned`].
    ///
    /// [`MaxAligned`]: struct.MaxAligned.html
    pub const unsafe fn new_unchecked() -> Self {
        Pixel(PhantomData)
    }

    /// Proxy of `core::mem::align_of`.
    pub const fn align(self) -> usize {
        mem::align_of::<P>()
    }

    /// Proxy of `core::mem::size_of`.
    pub const fn size(self) -> usize {
        mem::size_of::<P>()
    }

    /// Copy a pixel.
    ///
    /// Note that this does not require `Copy` because that requirement was part of the
    /// requirements of constructing this `Pixel` witness.
    pub fn copy_val(self, val: &P) -> P {
        // SAFETY: by the constructor, this type can be copied byte-by-byte.
        unsafe { ptr::read(val) }
    }

    /// Reinterpret a slice of aligned bytes as a slice of the pixel.
    ///
    /// Note that the size (in bytes) of the slice will be shortened if the size of `P` is not a
    /// divisor of the input slice's size.
    pub fn cast_to_slice<'buf>(self, buffer: &'buf [MaxAligned]) -> &'buf [P] {
        self.cast_buf(buf::new(buffer))
    }

    /// Reinterpret a slice of aligned bytes as a mutable slice of the pixel.
    ///
    /// Note that the size (in bytes) of the slice will be shortened if the size of `P` is not a
    /// divisor of the input slice's size.
    pub fn cast_to_mut_slice<'buf>(self, buffer: &'buf mut [MaxAligned]) -> &'buf mut [P] {
        self.cast_mut_buf(buf::new_mut(buffer))
    }

    /// Reinterpret a slice of pixels as memory.
    pub fn cast_to_bytes<'buf>(self, pixel: &'buf [P]) -> &'buf [u8] {
        self.cast_bytes(pixel)
    }

    /// Reinterpret a mutable slice of pixels as memory.
    pub fn cast_to_mut_bytes<'buf>(self, pixel: &'buf mut [P]) -> &'buf mut [u8] {
        self.cast_mut_bytes(pixel)
    }

    pub(crate) fn cast_buf<'buf>(self, buffer: &'buf buf) -> &'buf [P] {
        debug_assert_eq!(buffer.as_ptr() as usize % mem::align_of::<MaxAligned>(), 0);
        debug_assert_eq!(buffer.as_ptr() as usize % mem::align_of::<P>(), 0);
        // Safety:
        // * data is valid for reads as memory size is not enlarged
        // * lifetime is not changed
        // * validity for arbitrary data as required by Pixel constructor
        // * alignment checked by Pixel constructor
        // * the size fits in an allocation, see first bullet point.
        unsafe {
            if mem::size_of::<P>() == 0 {
                slice::from_raw_parts(buffer.as_ptr() as *const P, usize::MAX)
            } else {
                slice::from_raw_parts(
                    buffer.as_ptr() as *const P,
                    buffer.len() / mem::size_of::<P>(),
                )
            }
        }
    }

    pub(crate) fn cast_mut_buf<'buf>(self, buffer: &'buf mut buf) -> &'buf mut [P] {
        debug_assert_eq!(buffer.as_ptr() as usize % mem::align_of::<MaxAligned>(), 0);
        debug_assert_eq!(buffer.as_ptr() as usize % mem::align_of::<P>(), 0);
        // Safety:
        // * data is valid for reads and writes as memory size is not enlarged
        // * lifetime is not changed
        // * validity for arbitrary data as required by Pixel constructor
        // * alignment checked by Pixel constructor
        // * the size fits in an allocation, see first bullet point.
        unsafe {
            if mem::size_of::<P>() == 0 {
                slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut P, usize::MAX)
            } else {
                slice::from_raw_parts_mut(
                    buffer.as_mut_ptr() as *mut P,
                    buffer.len() / mem::size_of::<P>(),
                )
            }
        }
    }

    pub(crate) fn cast_bytes<'buf>(self, pixel: &'buf [P]) -> &'buf [u8] {
        // Safety:
        // * lifetime is not changed
        // * keeps the exact same size
        // * validity for byte reading checked by Pixel constructor
        unsafe { slice::from_raw_parts(pixel.as_ptr() as *const u8, mem::size_of_val(pixel)) }
    }

    pub(crate) fn cast_mut_bytes<'buf>(self, pixel: &'buf mut [P]) -> &'buf mut [u8] {
        // Safety:
        // * lifetime is not changed
        // * keeps the exact same size
        // * validity as bytes checked by Pixel constructor
        unsafe { slice::from_raw_parts_mut(pixel.as_ptr() as *mut u8, mem::size_of_val(pixel)) }
    }
}

/// This is a pure marker type.
impl<P> Clone for Pixel<P> {
    fn clone(&self) -> Self {
        Pixel(PhantomData)
    }
}

impl<P> PartialEq for Pixel<P> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<P> Eq for Pixel<P> {}

impl<P> PartialOrd for Pixel<P> {
    fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

impl<P> Ord for Pixel<P> {
    fn cmp(&self, _: &Self) -> Ordering {
        Ordering::Equal
    }
}

/// This is a pure marker type.
impl<P> Copy for Pixel<P> {}

impl<P> fmt::Debug for Pixel<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Pixel")
            .field("size", &self.size())
            .field("align", &self.align())
            .finish()
    }
}

impl<P> hash::Hash for Pixel<P> {
    fn hash<H: hash::Hasher>(&self, _: &mut H) {}
}
