//! A color-accurate software frame buffer.
//!
//! Please note, color is a work-in-progress with spotty support. See all locations marked with
//! `FIXME(color):` for progress. See `FIXME(perf)` for known suboptimal performance.
//!
//! # Usage
//!
//! ```
//! use image_framebuf::{Frame, FrameLayout, SampleParts, Texel};
//!
//! // Define what type of color we want to store...
//! let texel = Texel::new_u8(SampleParts::RgbA);
//! // and which dimensions to use, chooses a stride for us.
//! let layout = FrameLayout::with_texel(&texel, 32, 32)?;
//!
//! let frame = Frame::new(layout);
//! # use image_framebuf::LayoutError;
//! # Ok::<(), LayoutError>(())
//! ```

/// Putting it all together with a buffer type.
mod color;
/// The main frame module.
mod frame;
/// The complex layout implementation.
mod layout;
/// Conversion operation.
mod shader;

#[cfg(test)]
mod tests;

pub use self::color::{
    Color, ColorChannel, ColorChannelModel, Luminance, Primaries, Transfer, Whitepoint,
};
pub use self::frame::{Frame, Plane, PlaneMut, PlaneRef};
pub use self::layout::{
    Block, ChannelLayout, FrameLayout, LayoutError, PlanarLayout, PlaneBytes, RowLayoutDescription,
    SampleBits, SampleParts, Texel,
};
