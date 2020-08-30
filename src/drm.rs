use crate::layout::DynLayout;

/// A direct rendering manager format info.
///
/// This structure is a cleaned-up version of the one present in the Linux kernel rendering
/// subsystem, with deprecated fields removed. It is not a `Layout` itself since it is not
/// guaranteed to be internally consistent, it does not yet describe width and height or it might
/// otherwise not be supported. Try to convert it to a `DynLayout`.
///
/// See: the Linux kernel header `drm/drm_fourcc.h`.
#[derive(Clone, Copy, Debug, Hash)]
pub struct DrmFormatInfo {
    /// The 4CC format identifier.
    pub format: FourCC,
    /// The number of image color planes (1 to 3).
    pub num_planes: u8,
    /// The number of bytes per block.
    pub char_per_block: [u8; 4],
    /// The width of a block in pixels.
    pub block_w: [u8; 4],
    /// The height of a block in pixels.
    pub block_h: [u8; 4],
    /// The horizontal chroma subsampling factor.
    pub hsub: u8,
    /// The vertical chroma subsampling factor.
    pub vsub: u8,
    /// Does the format embed an alpha component?
    pub has_alpha: bool,
    /// Is it a YUV format?
    pub is_yuv: bool,
}

/// A descriptor for a single frame buffer.
///
/// In Linux, used to request new buffers or reallocation of buffers.
///
/// See: the Linux kernel header `drm/drm_mode.h`.
#[derive(Clone, Copy, Debug, Hash)]
pub struct DrmFramebufferCmd {
    pub width: u32,
    pub height: u32,
    pub fourcc: FourCC,
    pub flags: i32,
    pub pitches: [u32; 4],
    pub offsets: [u32; 4],
    pub modifier: [u64; 4],
}

/// The filled-in info about a frame buffer.
pub(crate) struct DrmFramebuffer {
    pub format: DrmFormatInfo,
    pub pitches: [u32; 4],
    pub offsets: [u32; 4],
    pub modifier: u64,
    pub width: u32,
    pub height: u32,
    pub flags: i32,
}

/// A direct rendering manager format info that is supported as a layout.
pub struct DrmLayout {
    pub(crate) info: DrmFramebuffer,
}

/// An error converting an info into a supported layout.
pub struct BadDrmError {
    _private: (),
}

/// A 4CC format identifier.
///
/// This exist to define the common formats as constants and to typify the conversion and
/// representation of values involved. The code is always stored as little endian.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FourCC(u32);

impl DrmFormatInfo {
    /// Create a layout with particular dimensions.
    ///
    /// This is a partial function to represent that not all descriptors can be convert to a
    /// possible dynamic layouts. No successful conversion will get removed across SemVer
    /// compatible versions.
    pub fn into_layout(self, _: u32, _: u32) -> Option<DynLayout> {
        None
    }
}

impl DrmLayout {
    pub fn new(info: &DrmFramebufferCmd) -> Result<Self, BadDrmError> {
        let info = info.fourcc.info()?;
        Err(BadDrmError { _private: () })
    }
}

impl FourCC {
    /* Relevant formats according to Linux header `uapi/drm/drm_fourcc.h` */
    /// The constant denoting an invalid format, e.g. signalling a missing format.
    pub const INVALID: Self = FourCC(0);
    /// Single 8 bpp grey color.
    pub const C8: Self = FourCC::from(*b"C8  ");

    /* 8 bpp rgb */
    /// 8bpp rgb with 3 bits red, 3 bits green, 2 bits blue.
    pub const RGB332: Self = FourCC::from(*b"RGB8");
    /// 8bpp rgb with 2 bits red, 3 bits green, 3 bits blue.
    pub const BGR332: Self = FourCC::from(*b"BGR8");

    /* 16 bpp rgb */
    /// 16 bpp xrgb with 4 bits each.
    pub const XRGB444: Self = FourCC::from(*b"XR12");
    /// 16 bpp xbgr with 4 bits each.
    pub const XBRG444: Self = FourCC::from(*b"XB12");
    /// 16 bpp rgbx with 4 bits each.
    pub const RGBX444: Self = FourCC::from(*b"RX12");
    /// 16 bpp bgrx with 4 bits each.
    pub const BGRX444: Self = FourCC::from(*b"BX12");

    const fn from(arr: [u8; 4]) -> Self {
        // FourCC(u32::from_be_bytes(arr)); not yet stable as const-fn
        FourCC(arr[0] as u32 | (arr[1] as u32) << 8 | (arr[2] as u32) << 16 | (arr[3] as u32) << 24)
    }

    fn info(self) -> Result<DrmFormatInfo, BadDrmError> {
        Err(BadDrmError { _private: () })
    }
}
