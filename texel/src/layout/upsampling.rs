use crate::layout::TexelLayout;

/// Planar chroma 2×2 block-wise sub-sampled image.
///
/// FIXME: figure out if this is 'right' to expose in this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Yuv420p {
    channel: TexelLayout,
    width: u32,
    height: u32,
}

impl Yuv420p {
    #[expect(dead_code)]
    pub fn from_width_height(channel: TexelLayout, width: u32, height: u32) -> Option<Self> {
        use core::convert::TryFrom;
        if width % 2 != 0 || height % 2 != 0 {
            return None;
        }

        let mwidth = usize::try_from(width).ok()?;
        let mheight = usize::try_from(height).ok()?;

        let y_count = mwidth.checked_mul(mheight)?;
        let uv_count = y_count / 2;

        let count = y_count.checked_add(uv_count)?;
        let _ = count.checked_mul(channel.size)?;

        Some(Yuv420p {
            channel,
            width,
            height,
        })
    }

    pub const fn byte_len(self) -> usize {
        let ylen = (self.width as usize) * (self.height as usize) * self.channel.size;
        ylen + ylen / 2
    }
}
