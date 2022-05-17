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
//! use image_canvas::{Frame, FrameLayout, SampleParts, Texel};
//!
//! // Define what type of color we want to store...
//! let texel = Texel::new_u8(SampleParts::RgbA);
//! // and which dimensions to use, chooses a stride for us.
//! let layout = FrameLayout::with_texel(&texel, 32, 32)?;
//!
//! let frame = Frame::new(layout);
//! # use image_canvas::LayoutError;
//! # Ok::<(), LayoutError>(())
//! ```
//!
//! Converting to a different color is also possible:
//! 1. Explicitly assign a fitting `Color` to source and target
//! 2. Call the conversion method.
//!
//! ```
//! use image_canvas::{Color, Frame, FrameLayout, SampleParts, Texel};
//!
//! let layout = FrameLayout::with_texel(&Texel::new_u8(SampleParts::Lab), 32, 32)?;
//! let mut from = Frame::new(layout.clone());
//! from.set_color(Color::Oklab)?;
//!
//! let layout = FrameLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
//! let mut into = Frame::new(layout);
//! into.set_color(Color::SRGB)?;
//!
//! // … omitted: some pixel initialization
//! from.convert(&mut into);
//!
//!// Now read the sRGB frame, e.g. to initialize an HTTP canvas
//! into.as_bytes();
//!
//! # use image_canvas::LayoutError;
//! # Ok::<(), LayoutError>(())
//! ```

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
