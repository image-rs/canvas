use crate::{layout, pixel, stride};
use core::convert::TryFrom;
use core::fmt;
use core::ops::Range;

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
/// the builder type to fallibly construct a `DrmLayout`, a complete layout descriptor for one
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
/// This is equivalent to `drm_framebuffer`, minus the kernel internal stuff. This does not own any
/// image data of its own, it's just an internally validated descriptor.
pub(crate) struct DrmFramebufferInfo {
    pub format: DrmFormatInfo,
    pub pitches: [u32; 4],
    pub offsets: [u32; 4],
    pub modifier: u64,
    pub width: u32,
    pub height: u32,
    /// A bit mask for which modifiers are actually to be enabled. All 0 for now.
    pub flags: i32,
}

/// A direct rendering manager format info that is supported as a layout.
///
/// You can't edit this format in-place. This ensures that a bunch of pre-computation are always
/// fresh. It might be relaxed later when we find a strategy to ensure this through other means.
pub struct DrmLayout {
    /// The frame buffer layout, checked for internal consistency.
    pub(crate) info: DrmFramebufferInfo,
    pub(crate) total_len: usize,
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
    element: layout::Element,
    pitch: u32,
    offset: u32,
    width: u32,
    height: u32,
}

#[derive(Debug)]
enum BadDrmKind {
    Unknown,
    PlaneCount,
    InconsistentModifier,
    IllegalModifier,
    ZeroBlockSize,
    ZeroBlockWidth,
    ZeroBlockHeight,
    OverlappingPlanes,
    UndescribableElement,
    IllegalPlaneWidth,
    IllegalPlaneHeight,
    LineSize,
    IllegalPlanePitch,
    PlaneSize,
    PlaneEnd,
    BufferSize,
    IllegalSubsampling,
    IllegalHsub,
    IllegalVsub,
}

/// An error converting an info into a supported layout.
pub struct BadDrmError {
    kind: BadDrmKind,
}

fn round_up_div(dimension: u32, div: u8) -> u32 {
    let div = u32::from(div);
    dimension / div + if dimension % div == 0 { 0 } else { 1 }
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
    ///
    /// This only works for formats where all planes can be laid out consecutively in memory.
    pub fn as_layout(&self, width: u32, height: u32) -> Result<DrmLayout, BadDrmError> {
        let mut request = DrmFramebufferCmd {
            width,
            height,
            fourcc: self.format,
            flags: 0,
            pitches: [0; 4],
            offsets: [0; 4],
            modifier: [0; 4],
        };

        let mut plane_offset = 0;

        let planes = &PlaneIdx::PLANES[..usize::from(self.num_planes)];
        for (idx, &plane) in planes.iter().enumerate() {
            let plane_width = self
                .plane_width(width, plane)
                .ok_or(BadDrmError::DEFAULT_ERR)?;
            let plane_height = self
                .plane_height(height, plane)
                .ok_or(BadDrmError::DEFAULT_ERR)?;
            // This can overflow buy later will be checked more strictly in `DrmLayout::new`.
            let line_bytes = u32::from(self.char_per_block[idx]).wrapping_mul(plane_width);
            let bytes = line_bytes.wrapping_mul(plane_height);
            request.offsets[idx] = plane_offset;
            request.pitches[idx] = line_bytes;
            // This can overflow buy later will be checked more strictly in `DrmLayout::new`.
            plane_offset = plane_offset.wrapping_add(bytes);
        }

        DrmLayout::new(&request)
    }

    fn plane_width(self, width: u32, idx: PlaneIdx) -> Option<u32> {
        let idx = idx.to_index();
        // If this one of the subsampled yuv planes.
        let width = round_up_div(width, self.block_w[idx]);
        Some(width)
    }

    fn plane_height(self, height: u32, idx: PlaneIdx) -> Option<u32> {
        // If this one of the subsampled yuv planes.
        let idx = idx.to_index();
        let height = round_up_div(height, self.block_h[idx]);
        Some(height)
    }

    /// The element describing each block (atomic unit) of the described layout.
    ///
    /// If plane is outside the number of planes of this format or if the blocks can not be
    /// described by a single element (e.g. they have extrinsic alignment requirements, their size
    /// is not divisible by their alignment) then None is returned.
    pub fn block_element(self, plane: PlaneIdx) -> Option<layout::Element> {
        if usize::from(self.num_planes) <= plane.to_index() {
            return None;
        }

        Some(match self.format {
            FourCC::C8 | FourCC::RGB332 | FourCC::BGR332 => pixel::constants::U8.into(),
            FourCC::XRGB444
            | FourCC::XBGR444
            | FourCC::RGBX444
            | FourCC::BGRX444
            | FourCC::RGB565
            | FourCC::BGR565
            | FourCC::ARGB444
            | FourCC::ABGR444
            | FourCC::RGBA444
            | FourCC::BGRA444 => pixel::constants::U16.into(),
            FourCC::RGB888 | FourCC::BGR888 => pixel::constants::U8.array3().into(),
            FourCC::XRGB8888
            | FourCC::XBGR8888
            | FourCC::RGBX8888
            | FourCC::BGRX8888
            | FourCC::ARGB8888
            | FourCC::ABGR8888
            | FourCC::RGBA8888
            | FourCC::BGRA8888 => pixel::constants::U8.array4().into(),
            FourCC::XRGB2101010
            | FourCC::XBGR2101010
            | FourCC::RGBX1010102
            | FourCC::BGRX1010102
            | FourCC::ARGB2101010
            | FourCC::ABGR2101010
            | FourCC::RGBA1010102
            | FourCC::BGRA1010102 => pixel::constants::U32.into(),
            FourCC::XRGB16161616F
            | FourCC::XBGR16161616F
            | FourCC::ARGB16161616F
            | FourCC::ABGR16161616F => pixel::constants::U16.array4().into(),
            FourCC::YUYV | FourCC::YVYU | FourCC::AYUV => pixel::constants::U8.array4().into(),
            FourCC::VUY101010 => pixel::constants::U32.into(),
            FourCC::VUY888 => pixel::constants::U8.array3().into(),
            // Actually planar formats.
            FourCC::XRGB888_A8 | FourCC::XBGR888_A8 => {
                if plane == PlaneIdx::First {
                    pixel::constants::U8.array4().into()
                } else {
                    pixel::constants::U8.into()
                }
            }
            FourCC::RGB888_A8 | FourCC::BGR888_A8 => {
                if plane == PlaneIdx::First {
                    pixel::constants::U8.array3().into()
                } else {
                    pixel::constants::U8.into()
                }
            }
            FourCC::RGB565_A8 | FourCC::BGR565_A8 => {
                if plane == PlaneIdx::First {
                    pixel::constants::U16.into()
                } else {
                    pixel::constants::U8.into()
                }
            }
            FourCC::NV12
            | FourCC::NV21
            | FourCC::NV16
            | FourCC::NV61
            | FourCC::NV24
            | FourCC::NV42 => {
                if plane == PlaneIdx::First {
                    pixel::constants::U8.into()
                } else {
                    pixel::constants::U8.array2().into()
                }
            }
            FourCC::YUV410
            | FourCC::YVU410
            | FourCC::YUV411
            | FourCC::YVU411
            | FourCC::YUV420
            | FourCC::YVU420
            | FourCC::YUV422
            | FourCC::YVU422
            | FourCC::YUV444
            | FourCC::YVU444 => pixel::constants::U8.into(),
            // No element that fits (or not implemented?).
            _ => return None,
        })
    }
}

impl PlaneIdx {
    const PLANES: [PlaneIdx; 3] = [PlaneIdx::First, PlaneIdx::Second, PlaneIdx::Third];
    pub fn to_index(self) -> usize {
        self as usize - 1
    }
}

impl DrmLayout {
    /// Try to construct a layout from a filled request.
    ///
    /// Due to limited support we enforce a number of extra conditions:
    /// * Modifier must be `0`, for all planes.
    /// * Only YUV can be sub sampled.
    pub fn new(info: &DrmFramebufferCmd) -> Result<Self, BadDrmError> {
        const DEFAULT_ERR: BadDrmError = BadDrmError::DEFAULT_ERR;
        let format_info = info.fourcc.info()?;
        usize::try_from(info.width).map_err(|_| DEFAULT_ERR)?;
        usize::try_from(info.height).map_err(|_| DEFAULT_ERR)?;

        if format_info.num_planes < 1 || format_info.num_planes > 3 {
            return Err(BadDrmKind::PlaneCount.into());
        }

        let modifier = info.modifier[0];
        if info.modifier.iter().any(|&m| m != modifier) {
            // All modifiers must be the same (and as later enforced 0 since we don't support
            // vendor specific codes at the moment).
            return Err(BadDrmKind::InconsistentModifier.into());
        }

        let mut last_plane_end = 0;
        let planes = PlaneIdx::PLANES[..usize::from(format_info.num_planes)]
            .iter()
            .enumerate();

        for (idx, &plane) in planes {
            if info.modifier[idx] != 0 {
                return Err(BadDrmKind::IllegalModifier.into());
            }

            if format_info.char_per_block[idx] == 0 {
                return Err(BadDrmKind::ZeroBlockSize.into());
            }

            if format_info.block_w[idx] == 0 {
                return Err(BadDrmKind::ZeroBlockWidth.into());
            }

            if format_info.block_h[idx] == 0 {
                return Err(BadDrmKind::ZeroBlockHeight.into());
            }

            if info.offsets[idx] < last_plane_end {
                // Only planes in order supported.
                return Err(BadDrmKind::OverlappingPlanes.into());
            }

            format_info
                .block_element(plane)
                .ok_or(BadDrmKind::UndescribableElement)?;

            let width = format_info
                .plane_width(info.width, plane)
                .ok_or(BadDrmKind::IllegalPlaneWidth)?;
            let height = format_info
                .plane_height(info.height, plane)
                .ok_or(BadDrmKind::IllegalPlaneHeight)?;

            let char_per_line = u32::from(format_info.char_per_block[idx])
                .checked_mul(width)
                .ok_or(BadDrmKind::LineSize)?;

            if info.pitches[idx] < char_per_line {
                return Err(BadDrmKind::IllegalPlanePitch.into());
            }

            let char_for_plane = info.pitches[idx]
                .checked_mul(height)
                .ok_or(BadDrmKind::PlaneSize)?;

            last_plane_end = info.offsets[idx]
                .checked_add(char_for_plane)
                .ok_or(BadDrmKind::PlaneEnd)?;
        }

        // Validates that all indices are valid as planes are ordered.
        let total_len = usize::try_from(last_plane_end).map_err(|_| BadDrmKind::BufferSize)?;

        if !format_info.is_yuv && (format_info.hsub != 1 || format_info.vsub != 1) {
            // subsampling only supported for yuv.
            return Err(BadDrmKind::IllegalSubsampling.into());
        }

        if format_info.hsub > 4 || !format_info.hsub.is_power_of_two() {
            return Err(BadDrmKind::IllegalHsub.into());
        }

        if format_info.vsub > 4 || !format_info.vsub.is_power_of_two() {
            return Err(BadDrmKind::IllegalVsub.into());
        }

        let descriptor = DrmFramebufferInfo {
            format: format_info,
            pitches: info.pitches,
            offsets: info.offsets,
            modifier,
            width: info.width,
            height: info.height,
            flags: 0,
        };

        Ok(DrmLayout {
            info: descriptor,
            total_len,
        })
    }

    /// Get the FourCC of this layout.
    pub fn fourcc(&self) -> FourCC {
        self.info.format.format
    }

    /// Get the layout of the nth plane of this frame buffer.
    pub fn plane(&self, plane_idx: PlaneIdx) -> Option<PlaneLayout> {
        let idx = plane_idx.to_index();

        if self.info.format.char_per_block[idx] == 0
            || self.info.format.block_w[idx] == 0
            || self.info.format.block_h[idx] == 0
        {
            // Not a Plane in the sense we're looking for.
            // TODO: this is not supported (we don't accept it in the constructor) and we might
            // want to make that distinction clear. Good for now though for forward compatible.
            return None;
        }

        Some(PlaneLayout {
            format: PlaneInfo {
                format: self.info.format.format,
                char_per_block: self.info.format.char_per_block[idx],
                block_w: self.info.format.block_w[idx],
                block_h: self.info.format.block_h[idx],
                hsub: self.info.format.hsub,
                vsub: self.info.format.vsub,
                has_alpha: self.info.format.has_alpha,
                is_yuv: self.info.format.is_yuv,
            },
            element: self.info.format.block_element(plane_idx).unwrap(),
            pitch: self.info.pitches[idx],
            offset: self.info.offsets[idx],
            width: self
                .info
                .format
                .plane_width(self.info.width, plane_idx)
                .unwrap(),
            height: self
                .info
                .format
                .plane_height(self.info.height, plane_idx)
                .unwrap(),
        })
    }

    /// The apparent width as a usize, as validated in constructor.
    pub fn width(&self) -> usize {
        self.info.width as usize
    }

    /// The apparent height as a usize, as validated in constructor.
    pub fn height(&self) -> usize {
        self.info.height as usize
    }
}

impl PlaneLayout {
    /// Get the FourCC of this layout.
    pub fn fourcc(&self) -> FourCC {
        self.format.format
    }

    fn byte_range(&self) -> Range<usize> {
        let start = self.offset as usize;
        let len = self.height() * self.pitch as usize;
        start..start + len
    }

    fn element(&self) -> layout::Element {
        self.element
    }

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

    /// 16 bpp argb with 4 bits each.
    pub const ARGB444: Self = FourCC::from(*b"AR12");
    /// 16 bpp abgr with 4 bits each.
    pub const ABGR444: Self = FourCC::from(*b"AB12");
    /// 16 bpp rgba with 4 bits each.
    pub const RGBA444: Self = FourCC::from(*b"RA12");
    /// 16 bpp bgra with 4 bits each.
    pub const BGRA444: Self = FourCC::from(*b"BA12");

    // TODO 1:5:5:5 formats..?

    /// 16 bpp rgb with 5-6-5 bit.
    pub const RGB565: Self = FourCC::from(*b"RG16");
    /// 16 bpp bgr with 5-6-5 bit.
    pub const BGR565: Self = FourCC::from(*b"BG16");

    /* 24 bpp rgb */

    /// 24 bpp rgb with 8 bits each.
    pub const RGB888: Self = FourCC::from(*b"RG24");
    /// 24 bpp bgr with 8 bits each.
    pub const BGR888: Self = FourCC::from(*b"BG24");

    /* 32 bpp rgb */
    pub const XRGB8888: Self = FourCC::from(*b"XR24");
    pub const XBGR8888: Self = FourCC::from(*b"XB24");
    pub const RGBX8888: Self = FourCC::from(*b"RX24");
    pub const BGRX8888: Self = FourCC::from(*b"BX24");

    pub const ARGB8888: Self = FourCC::from(*b"AR24");
    pub const ABGR8888: Self = FourCC::from(*b"AB24");
    pub const RGBA8888: Self = FourCC::from(*b"RA24");
    pub const BGRA8888: Self = FourCC::from(*b"BA24");

    pub const XRGB2101010: Self = FourCC::from(*b"XR30");
    pub const XBGR2101010: Self = FourCC::from(*b"XB30");
    pub const RGBX1010102: Self = FourCC::from(*b"RX30");
    pub const BGRX1010102: Self = FourCC::from(*b"BX30");

    pub const ARGB2101010: Self = FourCC::from(*b"AR30");
    pub const ABGR2101010: Self = FourCC::from(*b"AB30");
    pub const RGBA1010102: Self = FourCC::from(*b"RA30");
    pub const BGRA1010102: Self = FourCC::from(*b"BA30");

    /* 64bpp floating point
     * IEEE 754-2008 binary16 half-precision float
     * [15:0] sign:exponent:mantissa 1:5:10*
     */

    /// Half-precision float xrgb
    pub const XRGB16161616F: Self = FourCC::from(*b"XR4H");
    /// Half-precision float xbgr
    pub const XBGR16161616F: Self = FourCC::from(*b"XB4H");

    /// Half-precision float argb
    pub const ARGB16161616F: Self = FourCC::from(*b"AR4H");
    /// Half-precision float abgr
    pub const ABGR16161616F: Self = FourCC::from(*b"AB4H");

    /* YUV (or more accurately digital YCbCr) formats.
     *
     * The best resource that briefly explains the layout of where samples come from, how many
     * there are any how the codes relate to sampling modes is, surprisingly, a Microsoft article:
     * https://docs.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering
     */

    /// Packed YUYV (YCbYCr) with 8 bits each.
    pub const YUYV: Self = FourCC::from(*b"YUYV");
    /// Packed YVYU (YCrYCb) with 8 bits each.
    pub const YVYU: Self = FourCC::from(*b"YVYU");

    /// YUV with alpha, 8 bits each.
    pub const AYUV: Self = FourCC::from(*b"AYUV");
    /// YUV without alpha, 8 bits each.
    pub const XYUV888: Self = FourCC::from(*b"XYUV");
    /// 24-bit VUY, 8 bits each.
    pub const VUY888: Self = FourCC::from(*b"VU24");
    /// Packed 32-bit VUY, 10 bits each.
    pub const VUY101010: Self = FourCC::from(*b"VU30");

    // TODO: a few more YUV formats.

    /* 2 plane formats.
     * See the non-planar base format for layout.
     * RGB + A, and Y + CbCr
     */
    /// XRGB with 8 bits and padding on first plane, alpha with 8 bits on second.
    pub const XRGB888_A8: Self = FourCC::from(*b"XRA8");
    /// XBGR with 8 bits and padding on first plane, alpha with 8 bits on second.
    pub const XBGR888_A8: Self = FourCC::from(*b"XBA8");
    /// RGB with 8 bits each on first plane, alpha with 8 bits on second.
    pub const RGB888_A8: Self = FourCC::from(*b"R8A8");
    /// BGR with 8 bits each on first plane, alpha with 8 bits on second.
    pub const BGR888_A8: Self = FourCC::from(*b"B8A8");
    /// RGB with 5-6-5 bits on first plane, alpha with 8 bits on second.
    pub const RGB565_A8: Self = FourCC::from(*b"R5A8");
    /// BGR with 5-6-5 bits on first plane, alpha with 8 bits on second.
    pub const BGR565_A8: Self = FourCC::from(*b"B5A8");

    /// Y+CbCr with 8 bits each and 2×2 subsampled on chromatic plane.
    pub const NV12: Self = FourCC::from(*b"NV12");
    /// Y+CrCb with 8 bits each and 2×2 subsampled on chromatic plane.
    pub const NV21: Self = FourCC::from(*b"NV21");
    /// Y+CbCr with 8 bits each and 2×1 subsampled on chromatic plane.
    pub const NV16: Self = FourCC::from(*b"NV16");
    /// Y+CrCb with 8 bits each and 2×1 subsampled on chromatic plane.
    pub const NV61: Self = FourCC::from(*b"NV61");
    /// Y+CbCr with 8 bits each and not subsampled on chromatic plane.
    pub const NV24: Self = FourCC::from(*b"NV24");
    /// Y+CrCb with 8 bits each and not subsampled on chromatic plane.
    pub const NV42: Self = FourCC::from(*b"NV42");

    // TODO: formats P???, Q???

    /// Y+Cb+Cr with 8 bits each and 4×4 subsampled on chromatic planes.
    pub const YUV410: Self = FourCC::from(*b"YUV9");
    /// Y+Cr+Cb with 8 bits each and 4×4 subsampled on chromatic planes.
    pub const YVU410: Self = FourCC::from(*b"YVU9");
    /// Y+Cb+Cr with 8 bits each and 4×1 subsampled on chromatic planes.
    pub const YUV411: Self = FourCC::from(*b"YU11");
    /// Y+Cr+Cb with 8 bits each and 4×1 subsampled on chromatic planes.
    pub const YVU411: Self = FourCC::from(*b"YV11");
    /// Y+Cb+Cr with 8 bits each and 2×2 subsampled on chromatic planes.
    pub const YUV420: Self = FourCC::from(*b"YU12");
    /// Y+Cr+Cb with 8 bits each and 2×2 subsampled on chromatic planes.
    pub const YVU420: Self = FourCC::from(*b"YV12");
    /// Y+Cb+Cr with 8 bits each and 2×1 subsampled on chromatic planes.
    pub const YUV422: Self = FourCC::from(*b"YU16");
    /// Y+Cr+Cb with 8 bits each and 2×1 subsampled on chromatic planes.
    pub const YVU422: Self = FourCC::from(*b"YV16");
    /// Y+Cb+Cr with 8 bits each and not subsampled on chromatic planes.
    pub const YUV444: Self = FourCC::from(*b"YU24");
    /// Y+Cr+Cb with 8 bits each and not subsampled on chromatic planes.
    pub const YVU444: Self = FourCC::from(*b"YV24");

    const fn from(arr: [u8; 4]) -> Self {
        // FourCC(u32::from_le_bytes(arr)); not yet stable as const-fn
        FourCC(arr[0] as u32 | (arr[1] as u32) << 8 | (arr[2] as u32) << 16 | (arr[3] as u32) << 24)
    }

    /// Try to convert the format into an info.
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
            FourCC::XRGB444
            | FourCC::XBGR444
            | FourCC::RGBX444
            | FourCC::BGRX444
            | FourCC::RGB565
            | FourCC::BGR565 => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [2, 0, 0, 0],
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::ARGB444 | FourCC::ABGR444 | FourCC::RGBA444 | FourCC::BGRA444 => {
                DrmFormatInfo {
                    num_planes: 1,
                    char_per_block: [2, 0, 0, 0],
                    has_alpha: true,
                    ..DrmFormatInfo::PIXEL1_TEMPLATE
                }
            }
            FourCC::RGB888 | FourCC::BGR888 => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [3, 0, 0, 0],
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::XRGB8888
            | FourCC::XBGR8888
            | FourCC::RGBX8888
            | FourCC::BGRX8888
            | FourCC::XRGB2101010
            | FourCC::XBGR2101010
            | FourCC::RGBX1010102
            | FourCC::BGRX1010102 => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [4, 0, 0, 0],
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::ARGB8888
            | FourCC::ABGR8888
            | FourCC::RGBA8888
            | FourCC::BGRA8888
            | FourCC::ARGB2101010
            | FourCC::ABGR2101010
            | FourCC::RGBA1010102
            | FourCC::BGRA1010102 => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [4, 0, 0, 0],
                has_alpha: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::XRGB16161616F | FourCC::XBGR16161616F => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [8, 0, 0, 0],
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::ARGB16161616F | FourCC::ABGR16161616F => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [8, 0, 0, 0],
                has_alpha: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::YUYV | FourCC::YVYU => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [4, 0, 0, 0],
                block_w: [2, 0, 0, 0],
                hsub: 2,
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::XYUV888 | FourCC::VUY101010 => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [4, 0, 0, 0],
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::AYUV => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [4, 0, 0, 0],
                is_yuv: true,
                has_alpha: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::VUY888 => DrmFormatInfo {
                num_planes: 1,
                char_per_block: [3, 0, 0, 0],
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::XRGB888_A8 | FourCC::XBGR888_A8 => DrmFormatInfo {
                num_planes: 2,
                char_per_block: [4, 1, 0, 0],
                has_alpha: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::RGB888_A8 | FourCC::BGR888_A8 => DrmFormatInfo {
                num_planes: 2,
                char_per_block: [3, 1, 0, 0],
                has_alpha: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::RGB565_A8 | FourCC::BGR565_A8 => DrmFormatInfo {
                num_planes: 2,
                char_per_block: [2, 1, 0, 0],
                has_alpha: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::NV12 | FourCC::NV21 => DrmFormatInfo {
                num_planes: 2,
                char_per_block: [1, 2, 0, 0],
                block_w: [1, 2, 0, 0],
                block_h: [1, 2, 0, 0],
                vsub: 2,
                hsub: 2,
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::NV16 | FourCC::NV61 => DrmFormatInfo {
                num_planes: 2,
                char_per_block: [1, 2, 0, 0],
                block_w: [1, 2, 0, 0],
                block_h: [1, 1, 0, 0],
                vsub: 2,
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::NV24 | FourCC::NV42 => DrmFormatInfo {
                num_planes: 2,
                char_per_block: [1, 2, 0, 0],
                block_w: [1, 1, 0, 0],
                block_h: [1, 1, 0, 0],
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::YUV410 | FourCC::YVU410 => DrmFormatInfo {
                num_planes: 3,
                char_per_block: [1, 1, 1, 0],
                block_w: [1, 4, 4, 0],
                block_h: [1, 4, 4, 0],
                vsub: 4,
                hsub: 4,
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::YUV411 | FourCC::YVU411 => DrmFormatInfo {
                num_planes: 3,
                char_per_block: [1, 1, 1, 0],
                block_w: [1, 4, 4, 0],
                block_h: [1, 1, 1, 0],
                vsub: 4,
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::YUV420 | FourCC::YVU420 => DrmFormatInfo {
                num_planes: 3,
                char_per_block: [1, 1, 1, 0],
                block_w: [1, 2, 2, 0],
                block_h: [1, 2, 2, 0],
                vsub: 2,
                hsub: 2,
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::YUV422 | FourCC::YVU422 => DrmFormatInfo {
                num_planes: 3,
                char_per_block: [1, 1, 1, 0],
                block_w: [1, 2, 2, 0],
                block_h: [1, 1, 1, 0],
                vsub: 2,
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            FourCC::YUV444 | FourCC::YVU444 => DrmFormatInfo {
                num_planes: 3,
                char_per_block: [1, 1, 1, 0],
                block_w: [1, 1, 1, 0],
                block_h: [1, 1, 1, 0],
                is_yuv: true,
                ..DrmFormatInfo::PIXEL1_TEMPLATE
            },
            _ => return Err(BadDrmError::DEFAULT_ERR),
        };
        info.format = self;
        Ok(info)
    }
}

impl BadDrmError {
    const DEFAULT_ERR: BadDrmError = BadDrmError {
        kind: BadDrmKind::Unknown,
    };
}

impl From<BadDrmKind> for BadDrmError {
    fn from(kind: BadDrmKind) -> Self {
        BadDrmError { kind }
    }
}

impl layout::Layout for DrmLayout {
    fn byte_len(&self) -> usize {
        self.total_len
    }
}

impl layout::Layout for PlaneLayout {
    fn byte_len(&self) -> usize {
        self.byte_range().end
    }
}

impl stride::Strided for PlaneLayout {
    fn strided(&self) -> stride::StrideLayout {
        let element = self.element();
        let width = self.width();
        let height = self.height();
        let matrix = layout::Matrix::from_width_height(element, width, height)
            .expect("Fits in memory because the plane does");
        stride::StrideLayout::with_row_major(matrix)
    }
}

impl fmt::Debug for BadDrmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("BadDrmError").field(&self.kind).finish()
    }
}

#[test]
fn example_layouts() {
    fn assert_4x4_layout_for(cc: FourCC) {
        let info = cc.info().expect("Has info for");
        let layout = info.as_layout(4, 4).expect("Compile to 4x4 layout");
        assert_eq!(layout.width(), 4);
        assert_eq!(layout.height(), 4);
    }

    assert_4x4_layout_for(FourCC::C8);
    assert_4x4_layout_for(FourCC::RGB332);
    assert_4x4_layout_for(FourCC::RGB888);
    assert_4x4_layout_for(FourCC::RGB565);
    assert_4x4_layout_for(FourCC::YUYV);
    assert_4x4_layout_for(FourCC::AYUV);
    assert_4x4_layout_for(FourCC::ARGB16161616F);
    assert_4x4_layout_for(FourCC::RGB565_A8);
    assert_4x4_layout_for(FourCC::NV12);
    assert_4x4_layout_for(FourCC::NV16);
    assert_4x4_layout_for(FourCC::NV24);
    assert_4x4_layout_for(FourCC::YUV410);
    assert_4x4_layout_for(FourCC::YUV411);
    assert_4x4_layout_for(FourCC::YUV420);
    assert_4x4_layout_for(FourCC::YUV422);
    assert_4x4_layout_for(FourCC::YUV444);
}

#[test]
fn simple_planes() {
    use stride::Strided;

    let info = FourCC::RGB565_A8.info().expect("Has info for");
    let layout = info.as_layout(900, 600).expect("Compile to a small layout");
    assert_eq!(layout.width(), 900);
    assert_eq!(layout.height(), 600);

    let first = layout.plane(PlaneIdx::First).unwrap();
    let second = layout.plane(PlaneIdx::Second).unwrap();
    assert!(layout.plane(PlaneIdx::Third).is_none());

    assert_eq!(first.width(), 900);
    assert_eq!(first.height(), 600);
    let first = first.strided().spec();
    assert_eq!(first.width, 900);
    assert_eq!(first.height, 600);
    assert_eq!(first.element.size(), 2);

    assert_eq!(second.width(), 900);
    assert_eq!(second.height(), 600);
    let second = second.strided().spec();
    assert_eq!(second.width, 900);
    assert_eq!(second.height, 600);
    assert_eq!(second.element.size(), 1);
}

#[test]
fn yuv_planes() {
    use stride::Strided;

    let info = FourCC::NV12.info().expect("Has info for");
    let layout = info.as_layout(900, 600).expect("Compile to a small layout");
    assert_eq!(layout.width(), 900);
    assert_eq!(layout.height(), 600);

    let first = layout.plane(PlaneIdx::First).unwrap();
    let second = layout.plane(PlaneIdx::Second).unwrap();
    assert!(layout.plane(PlaneIdx::Third).is_none());

    assert_eq!(first.width(), 900);
    assert_eq!(first.height(), 600);
    let first = first.strided().spec();
    assert_eq!(first.width, 900);
    assert_eq!(first.height, 600);
    assert_eq!(first.element.size(), 1);

    assert_eq!(second.width(), 450);
    assert_eq!(second.height(), 300);
    let second = second.strided().spec();
    assert_eq!(second.width, 450);
    assert_eq!(second.height, 300);
    assert_eq!(second.element.size(), 2);
}
