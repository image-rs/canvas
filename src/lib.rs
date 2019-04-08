mod buf;
mod canvas;

use core::marker::PhantomData;
use core::mem;

use zerocopy::FromBytes;

pub use self::canvas::Canvas;

/// Marker struct to denote a pixel type.
///
/// Can be constructed only for types that have expected alignment and no byte invariants.
#[derive(Clone, Copy)]
pub struct Pixel<P>(PhantomData<P>);

/// Constants for predefined pixel types.
pub mod pixels {
    use zerocopy::{AsBytes, FromBytes};

    macro_rules! constant_pixels {
        ($(($name:ident, $type:ty)),*) => {
            $(pub const $name: crate::Pixel<$type> = crate::Pixel(::core::marker::PhantomData);)*
        }
    }

    constant_pixels!((I8, i8), (U8, u8), (I16, i16), (U16, u16), (RGB, [u8; 3]), (RGBA, [u8; 4]));
    pub(crate) const MAX_ALIGN: usize = 16;

    /// A byte-like-type that is aligned to the required max alignment.
    ///
    /// This type does not contain padding and implements `FromBytes`.
    #[derive(AsBytes, FromBytes)]
    #[repr(align(16))]
    #[repr(C)]
    pub struct MaxAligned([u8; 16]);
}

impl<P: FromBytes> Pixel<P> {
    /// Try to construct an instance of the marker.
    ///
    /// If successful, you can freely use it to access the image buffers.
    pub fn for_type() -> Option<Self> {
        if mem::align_of::<P>() <= pixels::MAX_ALIGN {
            Some(Pixel(PhantomData))
        } else {
            None
        }
    }

    /// Proxy of `core::mem::align_of`.
    pub fn align(self) -> usize {
        mem::align_of::<P>()
    }

    /// Proxy of `core::mem::size_of`.
    pub fn size(self) -> usize {
        mem::size_of::<P>()
    }
}

