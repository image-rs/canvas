// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
use core::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use core::marker::PhantomData;
use core::{fmt, hash, mem};

/// Marker struct to denote a pixel type.
///
/// Can be constructed only for types that have expected alignment and no byte invariants. It
/// always implements `Copy` and `Clone`, regardless of the underlying type and is zero-sized.
pub struct Pixel<P: ?Sized>(PhantomData<P>);

/// Describes a type which can represent a `Pixel`.
pub trait AsPixel {
    /// Get the pixel struct for this type.
    ///
    /// The naive implementation of merely unwraping the result of `Pixel::for_type` **panics** on
    /// any invalid type. This trait should only be implemented when you know for sure that the
    /// type is correct.
    fn pixel() -> Pixel<Self>;
}

pub(crate) const MAX_ALIGN: usize = 16;

/// A byte-like-type that is aligned to the required max alignment.
///
/// This type does not contain padding and implements `FromBytes`.
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

impl<P> Pixel<P> {
    /// Proxy of `core::mem::align_of`.
    pub fn align(self) -> usize {
        mem::align_of::<P>()
    }

    /// Proxy of `core::mem::size_of`.
    pub fn size(self) -> usize {
        mem::size_of::<P>()
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
