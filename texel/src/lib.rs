// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
//! # Matrix
//!
//! An image compatible with transmuting its byte content.
//!
//! This library is strictly `no_std`, and aims to offer utilities to represent and share image
//! buffers between platforms, byte representations, and processing methods. It acknowledges that,
//! in a typical image pipeline, there may exist many valid but competing representations:
//!
//! - A reader that decodes pixel representations into bytes (in network endian).
//! - Some other decoder that returns `Vec<[[u16; 3]]>` of native endian data.
//! - Some library transformation that consumes `&[Rgb<u16>]`.
//! - Some SIMD usage that requires data is passed as `&[Simd<u16, 16>]`.
//! - Some GPU buffer written by a highly-aligned, and line-padded `&[u8]`.
//! - Some GPU buffer containing texels of 4×2 pixels each.
//! - A *shared* buffer that represents pixels as `&[[AtomicU8; 4]]`.
//! - A non-planar layout that splits channels to different pages.
//!
//! This crate offers the language to ensure that as many uses cases as possible can share
//! allocations, or even offer zero-copy conversion.
//!
//! ## Usage
//!
//! ```
//! # fn send_over_network(_: &[u8]) { };
//! use image_texel::Matrix;
//! let mut image = Matrix::<[u8; 4]>::with_width_and_height(400, 400);
//!
//! // Draw a bright red line.
//! for i in 0..400 {
//!     // Assign color as u8-RGBA
//!     image[(i, i)] = [0xFF, 0x00, 0x00, 0xFF];
//! }
//!
//! // Encode to network endian.
//! let mut encoded = image.transmute::<u32>();
//! encoded
//!     .as_mut_slice()
//!     .iter_mut()
//!     .for_each(|p| *p = p.to_be());
//!
//! // Send the raw bytes
//! send_over_network(encoded.as_bytes());
//! ```
// Be std for doctests, avoids a weird warning about missing allocator.
#![cfg_attr(not(doctest), no_std)]
// The only module allowed to be `unsafe` is `texel`. We need it however, as we have a custom
// dynamically sized type with an unsafe alignment invariant.
#![deny(unsafe_code)]
extern crate alloc;

mod buf;
pub mod image;
pub mod layout;
mod matrix;
mod rec;
mod stride;
mod texel;

pub use self::image::Image;
pub use self::matrix::{Matrix, MatrixReuseError};
pub use self::rec::{BufferReuseError, TexelBuffer};
pub use self::texel::{AsTexel, Texel};

/// Constants for predefined texel types.
///
/// Holding an instance of `Texel<T>` certifies that the type `T` is compatible with the texel
/// concept, that is: its alignment requirement is *small* enough, its size is non-zero, it does
/// not contain any padding, and it is a plain old data type without any inner invariants
/// (including sharing predicates). These assertions allow a number of operations such as
/// reinterpreting aligned byte slices, writing to and read from byte buffers, fallible cast to
/// other texels, etc.
///
/// For types that guarantee the property, see [`AsTexel`] and its impls.
///
/// # Extending
///
/// The recommended method of extending this with a custom type is by implementing `bytemuck::Pod`
/// for this type. This applies a number of consistency checks.
///
/// ```rust
/// use bytemuck::{Pod, Zeroable};
/// use image_texel::{AsTexel, Texel};
///
/// #[derive(Clone, Copy, Pod, Zeroable)]
/// #[repr(C)]
/// struct Rgb(pub [u8; 3]);
///
/// impl AsTexel for Rgb {
///     fn texel() -> Texel<Rgb> {
///         Texel::for_type().expect("verified by bytemuck and image_texel")
///     }
/// }
///
/// impl Rgb {
///     const TEXEL: Texel<Self> = match Texel::for_type() {
///         Some(texel) => texel,
///         None => panic!("compilation error"),
///     };
/// }
/// ```
pub mod texels {
    pub use crate::texel::constants::*;
    pub use crate::texel::IsTransparentWrapper;
    pub use crate::texel::MaxAligned;
}
