// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
//! # Matrix
//!
//! An image canvas compatible with transmuting its byte content.
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
//! use canvas::Matrix;
//! let mut canvas = Matrix::<[u8; 4]>::with_width_and_height(400, 400);
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
// Be std for doctests, avoids a weird warning about missing allocator.
#![cfg_attr(not(doctest), no_std)]
// The only module allowed to be `unsafe` is `texel`. We need it however, as we have a custom
// dynamically sized type with an unsafe alignment invariant.
#![deny(unsafe_code)]
extern crate alloc;

mod buf;
pub mod canvas;
pub mod layout;
mod matrix;
mod rec;
pub mod stride;
mod texel;

pub use self::canvas::Canvas;
pub use self::matrix::{Matrix, MatrixReuseError};
pub use self::rec::{BufferReuseError, TexelBuffer};
pub use self::texel::{AsTexel, Texel};

/// Constants for predefined texel types.
pub mod texels {
    pub use crate::texel::constants::*;
    pub use crate::texel::IsTransparentWrapper;
    pub use crate::texel::MaxAligned;
}
