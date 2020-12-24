use crate::{layout, stride};
use core::convert::TryFrom;

/// A direct rendering manager format info.
///
/// The format information describes the byte layout of pixels of that particular type.
///
/// This structure is a cleaned-up version of the one present in the Linux kernel rendering
/// subsystem, with deprecated fields removed. It is not a `Layout` itself since it is not
/// guaranteed to be internally consistent, it does not yet describe width and height or it might
/// otherwise not be supported. Try to convert it to a `DynLayout`.
///
/// See: the Linux kernel header `drm/drm_fourcc.h` and
///
/// https://www.kernel.org/doc/html/latest/gpu/drm-kms.html#c.drm_format_info
#[derive(Clone, Copy, Debug, Hash)]
pub struct DrmFormatInfo {
    /// The 4CC format identifier.
    pub format: FourCC,
    /// The number of image color planes (1 to 3).
    pub num_planes: u8,
    /// Number of bytes per block (per plane).
    ///
    /// Blocks are defined as a rectangle of pixels which are stored next to each other in a byte
    /// aligned memory region. Since we have no backwards compatibility considerations there is no
    /// `cpp` member.
    ///
    /// For formats that are intended to be used only with non-linear modifiers char_per_block must
    /// be 0 in the generic format table.
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

struct PlaneInfo {
    /// The 4CC of the whole buffer format.
    format: FourCC,
    /// Characters per block of this plane.
    char_per_block: u8,
    /// The width of a block in pixels.
    block_w: u8,
    /// The height of a block in pixels.
    block_h: u8,
    /// The horizontal chroma subsampling factor.
    hsub: u8,
    /// The vertical chroma subsampling factor.
    vsub: u8,
    /// Does the format embed an alpha component?
    has_alpha: bool,
    /// Is it a YUV format?
    is_yuv: bool,
}

/// A descriptor for a single frame buffer.
///
/// In Linux, used to request new buffers or reallocation of buffers. Here, we use it similarly as
/// the builder type to fallibly construct a `DrmFramebuffer`, a complete layout descriptor for one
/// sized image.
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
///
/// This is equivalent to `drm_framebuffer`, minus the kernel internal stuff.
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
    /// The frame buffer layout, checked for internal consistency.
    pub(crate) info: DrmFramebuffer,
}

/// The index of a plane in a frame buffer.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlaneIdx {
    First = 1,
    Second = 2,
    Third = 3,
}

/// The layout of one plane of a DRM buffer.
pub struct PlaneLayout {
    format: PlaneInfo,
    pitch: u32,
    offset: u32,
    modifier: u64,
    width: u32,
    height: u32,
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
    /// Default values for an info of a format that has 1×1 blocks, i.e. each pixel is stored
    /// individually. Some info is left out to require it being filled in.
    const PIXEL1_TEMPLATE: Self = DrmFormatInfo {
        format: FourCC::INVALID,
        num_planes: 0,
        char_per_block: [0; 4],
        block_w: [1; 4],
        block_h: [1; 4],
        hsub: 1,
        vsub: 1,
        has_alpha: false,
        is_yuv: false,
    };

    /// Create a layout with particular dimensions.
    ///
    /// This is a partial function to represent that not all descriptors can be convert to a
    /// possible dynamic layouts. No successful conversion will get removed across SemVer
    /// compatible versions.
    pub fn into_layout(self, _: u32, _: u32) -> Option<layout::DynLayout> {
        None
    }

    fn plane_width(self, _: u32, _: PlaneIdx) -> Option<u32> {
        todo!()
    }

    fn plane_height(self, _: u32, _: PlaneIdx) -> Option<u32> {
        todo!()
    }
}

impl PlaneIdx {
    const PLANES: [PlaneIdx; 3] = [PlaneIdx::First, PlaneIdx::Second, PlaneIdx::Third];
}

impl DrmLayout {
    pub fn new(info: &DrmFramebufferCmd) -> Result<Self, BadDrmError> {
        const DEFAULT_ERR: BadDrmError = BadDrmError { _private: () };
        let format_info = info.fourcc.info()?;
        usize::try_from(info.width).map_err(|_| DEFAULT_ERR)?;
        usize::try_from(info.height).map_err(|_| DEFAULT_ERR)?;

        if format_info.num_planes < 1 || format_info.num_planes > 3 {
            return Err(DEFAULT_ERR);
        }

        for &plane in &PlaneIdx::PLANES[..usize::from(format_info.num_planes)] {
            let width = format_info
                .plane_width(info.width, plane)
                .ok_or(DEFAULT_ERR)?;
            let height = format_info
                .plane_height(info.height, plane)
                .ok_or(DEFAULT_ERR)?;
            todo!()
        }

        Err(BadDrmError { _private: () })
    }

    /// Get the layout of the nth plane of this frame buffer.
    pub fn plane(&self, _: PlaneIdx) -> Option<PlaneLayout> {
        todo!()
    }

    fn width(&self) -> usize {
        self.info.width as usize
    }

    fn height(&self) -> usize {
        self.info.height as usize
    }
}

impl PlaneLayout {
    fn width(&self) -> usize {
        self.width as usize
    }

    fn height(&self) -> usize {
        self.height as usize
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
    pub const XBGR444: Self = FourCC::from(*b"XB12");
    /// 16 bpp rgbx with 4 bits each.
    pub const RGBX444: Self = FourCC::from(*b"RX12");
    /// 16 bpp bgrx with 4 bits each.
    pub const BGRX444: Self = FourCC::from(*b"BX12");

    const fn from(arr: [u8; 4]) -> Self {
        // FourCC(u32::from_be_bytes(arr)); not yet stable as const-fn
        FourCC(arr[0] as u32 | (arr[1] as u32) << 8 | (arr[2] as u32) << 16 | (arr[3] as u32) << 24)
    }

    pub fn info(self) -> Result<DrmFormatInfo, BadDrmError> {
        let mut info = match self {
            FourCC::C8 => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [1, 0, 0, 0],
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::RGB332 | FourCC::BGR332 => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [1, 0, 0, 0],
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::XRGB444 | FourCC::XBGR444 | FourCC::RGBX444 | FourCC::BGRX444 => {
                DrmFormatInfo {
                    num_planes: 1,
                    char_per_block: [2, 0, 0, 0],
                    ..DrmFormatInfo::PIXEL1_TEMPLATE
                }
            }
            _ => return Err(BadDrmError { _private: () }),
        };
        info.format = self;
        Ok(info)
    }
}

impl layout::Layout for DrmLayout {
    fn byte_len(&self) -> usize {
        todo!()
    }
}

impl layout::Layout for PlaneLayout {
    fn byte_len(&self) -> usize {
        todo!()
    }
}

impl stride::Strided for PlaneLayout {
    fn strided(&self) -> stride::StrideLayout {
        todo!()
    }
}
