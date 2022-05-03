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
