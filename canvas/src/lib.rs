//! A color-accurate software frame buffer.
//!
//! Please note, color is a work-in-progress with spotty support. See all locations marked with
//! `FIXME(color):` for progress. See `FIXME(perf)` for known suboptimal performance.
//!
//! Separately, planar support is also a work-in-progress. See `FIXME(planar)` for progress.
//!
//! # Usage
//!
//! Creating a simple RgbA frame buffer is as a easy as:
//! 1. Specifying a texel with the right depth and channels
//! 2. Defining the layout, with width and height
//! 3. Allocating the frame utilizing the layout
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

mod arch;
/// Putting it all together with a buffer type.
pub mod color;
mod color_matrix;
/// The main frame module.
mod frame;
/// The complex layout implementation.
#[path = "layout.rs"]
mod layout_;
/// Conversion operation.
mod shader;

#[cfg(test)]
mod tests;

pub use self::frame::{Canvas, Plane, PlaneMut, PlaneRef};

pub mod layout {
    pub use crate::layout_::{
        Block, CanvasLayout, ChannelLayout, LayoutError, PlanarLayout, PlaneBytes,
        RowLayoutDescription, SampleBits, SampleParts, Texel,
    };
}
