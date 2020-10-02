// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
//! # Matrix
//!
//! An image canvas compatible with transmuting its byte content.
//!
//! ## Usage
//!
//! ```
//! # fn send_over_network(_: &[u8]) { };
//! use canvas::Matrix;
//! let mut canvas = Matrix::with_width_and_height(400, 400);
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
// The only module allowed to be `unsafe` is `pixel`. We need it however, as we have a custom
// dynamically sized type with an unsafe alignment invariant.
#![deny(unsafe_code)]
extern crate alloc;

mod buf;
mod canvas;
mod drm;
pub mod layout;
mod matrix;
mod pixel;
mod rec;

pub use self::canvas::Canvas;
pub use self::matrix::{Layout, Matrix, MatrixReuseError};
pub use self::pixel::{AsPixel, Pixel};
pub use self::rec::{Rec, ReuseError};

/// Constants for predefined pixel types.
pub mod pixels {
    pub use crate::pixel::constants::*;
    pub use crate::pixel::MaxAligned;
}
