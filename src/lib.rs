// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
//! # Canvas
//!
//! An image canvas compatible with transmuting its byte content.
//!
//! ## Usage
//!
//! ```
//! # fn send_over_network(_: &[u8]) { };
//! use canvas::Canvas;
//! let mut canvas = Canvas::with_width_and_height(400, 400);
//!
//! // Draw a bright red line.
//! for i in 0..400 {
//!     // Assign color as u8-RGBA
//!     canvas[(i, i)] = [0xFF, 0x00, 0x00, 0xFF];
//! }
//!
//! // Encode to network endian.
//! let mut encoded = canvas.transmute::<u32>();
//! encoded 
//!     .as_mut_slice()
//!     .iter_mut()
//!     .for_each(|p| *p = p.to_be());
//!
//! // Send the raw bytes
//! send_over_network(encoded.as_bytes());
//! ```
mod buf;
mod canvas;
mod rec;

use core::cmp::{PartialEq, Eq, PartialOrd, Ord, Ordering};
use core::marker::PhantomData;
use core::mem;

use zerocopy::{AsBytes, FromBytes};

pub use self::rec::{Rec, ReuseError};
pub use self::canvas::{Canvas, CanvasReuseError, Layout};

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

/// Constants for predefined pixel types.
pub mod pixels {
    use zerocopy::{AsBytes, FromBytes};

    pub(crate) const MAX_ALIGN: usize = 16;

    /// A byte-like-type that is aligned to the required max alignment.
    ///
    /// This type does not contain padding and implements `FromBytes`.
    #[derive(Clone, Copy, AsBytes, FromBytes)]
    #[repr(align(16))]
    #[repr(C)]
    pub struct MaxAligned(pub [u8; 16]);

    macro_rules! constant_pixels {
        ($(($name:ident, $type:ty)),*) => {
            $(pub const $name: crate::Pixel<$type> = crate::Pixel(::core::marker::PhantomData);
              impl crate::AsPixel for $type {
                  fn pixel() -> crate::Pixel<Self> {
                      $name
                  }
              }
              )*
        }
    }

    constant_pixels!(
        (I8, i8),
        (U8, u8),
        (I16, i16),
        (U16, u16),
        (I32, i32),
        (U32, u32),
        (RGB, [u8; 3]),
        (RGBA, [u8; 4]),
        (MAX, MaxAligned)
    );
}

impl<P: AsBytes + FromBytes> Pixel<P> {
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

impl<P> Eq for Pixel<P> { }

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
impl<P> Copy for Pixel<P> { }
