//! An opinionated in-memory buffer for image data.
//!
//! # General Usage
//!
//! Let us start by creating a simple RgbA buffer. For this we need to:
//!
//! 1. Specify the texel type with the right depth and channels.
//! 2. Define the layout, a plain matrix with width and height
//! 3. Allocate the frame with the layout
//!
//! [1]: https://crates.io/crates/rgb
//! [2]: https://crates.io/crates/palette
//!
//! ```
//! use image_canvas::Canvas;
//! use image_canvas::layout::{CanvasLayout, SampleParts, Texel};
//!
//! // Define what type of color we want to store...
//! let texel = Texel::new_u8(SampleParts::RgbA);
//! // and which dimensions to use, chooses a stride for us.
//! let layout = CanvasLayout::with_texel(&texel, 32, 32)?;
//!
//! let frame = Canvas::new(layout);
//! # use image_canvas::layout::LayoutError;
//! # Ok::<(), LayoutError>(())
//! ```
//!
//! Converting to a different color is also possible:
//! 1. Explicitly assign a fitting `Color` to source and target
//! 2. Call the conversion method.
//!
//! ```
//! use image_canvas::Canvas;
//! use image_canvas::color::Color;
//! use image_canvas::layout::{CanvasLayout, SampleParts, Texel};
//!
//! let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Lab), 32, 32)?;
//! let mut from = Canvas::new(layout.clone());
//! from.set_color(Color::Oklab)?;
//!
//! let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
//! let mut into = Canvas::new(layout);
//! into.set_color(Color::SRGB)?;
//!
//! // … omitted: some pixel initialization
//! from.convert(&mut into);
//!
//!// Now read the sRGB frame, e.g. to initialize an HTTP canvas
//! into.as_bytes();
//!
//! # use image_canvas::layout::LayoutError;
//! # Ok::<(), LayoutError>(())
//! ```
// Deny, not forbid, unsafe code. In `arch` module we have inherently unsafe code, for the moment.
// Maybe at a future point we gain some possibility to write such code safely.
#![deny(unsafe_code)]
// Be std for doctests, avoids a weird warning about missing allocator.
#![cfg_attr(not(doctest), no_std)]

#[cfg(feature = "runtime-features")]
extern crate std;

#[macro_use]
extern crate alloc;

mod arch;
mod bits;
/// Putting it all together with a buffer type.
pub mod color;
mod color_matrix;
/// The main frame module.
mod frame;
/// The layout implementation, builders, descriptors.
pub mod layout;
/// Conversion operation.
mod shader;

#[cfg(test)]
mod tests;

pub use self::frame::{Canvas, Plane};

pub use self::shader::{Converter, ConverterPlaneHandle, ConverterRun};

pub mod canvas {
    pub use crate::frame::{
        ArcCanvas, BytePlaneAtomics, BytePlaneCells, BytePlaneMut, BytePlaneRef,
        BytePlaneRef as BytePlane, ChannelsMut, ChannelsRef, PlaneMut, PlaneRef, RcCanvas,
    };
}

#[doc(hidden)]
pub use self::canvas::{PlaneMut, PlaneRef};
