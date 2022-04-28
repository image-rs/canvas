//! Conversion between texels, mostly color.
//!
//! Takes quite a lot of inspiration from how GPUs work. We have a primitive sampler unit, a
//! fragment unit, and pipeline multiple texels in parallel.
use canvas::canvas::{CanvasMut, CanvasRef};
use canvas::{canvas::Coord, AsTexel, Texel, TexelBuffer};
use core::ops::Range;

use crate::frame::Frame;
use crate::layout::{ByteLayout, FrameLayout, SampleBits, Texel as TexelBits};

/// A buffer for conversion.
pub struct Converter {
    /// The underlying conversion info.
    info: Info,
    /// How many texels to do at once.
    chunk: usize,

    super_blocks: Vec<TexelCoord>,
    /// Buffer where we store input texels after reading them.
    in_texels: TexelBuffer,
    /// Texel coordinates of stored texels.
    in_coords: Vec<TexelCoord>,
    /// Index in the input planes.
    in_index: Vec<usize>,
    /// Buffer where we store input texels before writing.
    out_texels: TexelBuffer,
    /// Texel coordinates of stored texels.
    out_coords: Vec<TexelCoord>,
    /// Index in the input planes.
    out_index: Vec<usize>,

    /// The input texels, split into pixels in the color's natural order.
    pixel_in_buffer: TexelBuffer,
    /// The pixels in a color space in the middle of in and out, mostly CIE XYZ+Alpha.
    neutral_color_buffer: TexelBuffer,
    /// The output texels, split into pixels in the color's natural order.
    pixel_out_buffer: TexelBuffer,

    /// The ops (functions) used for conversion.
    ops: ConvertOps,
}

#[repr(transparent)]
struct TexelCoord(Coord);

struct Info {
    /// Layout of the input frame.
    in_layout: FrameLayout,
    /// Layout of the output frame.
    out_layout: FrameLayout,
    /// The selected way to represent pixels in common parameter space.
    common_pixel: CommonPixel,
    /// The selected common color space, midpoint for conversion.
    common_color: CommonColor,
    /// The texel fetch we perform for input.
    /// Note that this is not necessarily the underlying texel as we throw away parts of
    /// interpretation, as long as it preserves size and alignment in a matter that produces the
    /// correct bits on indexing.
    in_kind: TexelKind,
    /// The texel fetch we perform for output.
    /// Note that this is not necessarily the underlying texel as we throw away parts of
    /// interpretation, as long as it preserves size and alignment in a matter that produces the
    /// correct bits on indexing.
    out_kind: TexelKind,
}

/// Denotes the type we pass to the color decoder.
///
/// This is an internal type due to the type assigned to each color being an implementation detail.
/// Consider that rgb565 can be passed as u16 or a special wrapper type for example. Or that
/// `[f16; 2]` can be a `u32` or a `[u16; 2]` or a wrapper. Until there's indication that this
/// needs stabilization it's kept secret.
///
/// For a valid layout it also fits to the indicated color components. There may be more than one
/// pixel in each texel.
#[derive(Clone, Copy)]
pub enum TexelKind {
    U8,
    U8x2,
    U8x3,
    U8x4,
    U16,
    U16x2,
    U16x3,
    U16x4,
    F32,
    F32x2,
    F32x3,
    F32x4,
}

pub(crate) trait GenericTexelAction<R = ()> {
    fn run<T>(self, texel: Texel<T>) -> R;
}

enum CommonPixel {
    U8x4,
    U16x4,
    U32x4,
    F32x4,
}

enum CommonColor {
    CieXyz,
}

type PlaneSource<'data, 'layout> = CanvasRef<'data, &'layout FrameLayout>;
type PlaneTarget<'data, 'layout> = CanvasMut<'data, &'layout mut FrameLayout>;

/// The function pointers doing the conversion.
///
/// Note how there are no types involved here. Instead, `TexelCoord` is a polymorphic buffer that
/// each function can access with any type it sees feed. We expect the constructor to ensure only
/// matching types are being used.
struct ConvertOps {
    /// Convert in texel coordinates to an index in the color plane.
    in_index: fn(&Info, &[TexelCoord], &mut [usize]),
    /// Convert out texel coordinates to an index in the color plane.
    out_index: fn(&Info, &[TexelCoord], &mut [usize]),

    /// Expand all texels into pixels in normalized channel order.
    expand: fn(&Info, &TexelBuffer, &mut TexelBuffer),
    /// Take pixels in normalized channel order and apply color conversion.
    recolor: Option<RecolorOps>,
    /// Join all pixels from normalized channel order to texels, clamping.
    join: fn(&Info, &TexelBuffer, &mut TexelBuffer),
}

struct RecolorOps {
    from: fn(&Info, &TexelBuffer, &mut TexelBuffer),
    into: fn(&Info, &TexelBuffer, &mut TexelBuffer),
}

struct SuperTexel {
    blocks: Range<u32>,
    /// In blocks per super block.
    in_super: u32,
    /// Out blocks per super block.
    out_super: u32,
}

impl Converter {
    pub fn new(frame_in: &Frame, frame_out: &Frame) -> Self {
        Converter {
            info: Info {
                in_layout: frame_in.layout().clone(),
                out_layout: frame_out.layout().clone(),
                common_pixel: CommonPixel::F32x4,
                common_color: CommonColor::CieXyz,
                in_kind: TexelKind::from(frame_in.layout().texel.bits),
                out_kind: TexelKind::from(frame_in.layout().texel.bits),
            },
            chunk: 512,
            super_blocks: vec![],
            in_texels: TexelBuffer::default(),
            in_coords: vec![],
            in_index: vec![],
            out_texels: TexelBuffer::default(),
            out_coords: vec![],
            out_index: vec![],
            pixel_in_buffer: TexelBuffer::default(),
            neutral_color_buffer: TexelBuffer::default(),
            pixel_out_buffer: TexelBuffer::default(),
            ops: ConvertOps {
                in_index: Self::index_from_in_info,
                out_index: Self::index_from_out_info,
                expand: CommonPixel::F32x4.expand_from_info(),
                recolor: None,
                join: CommonPixel::F32x4.join_from_info(),
            },
        }
    }

    pub fn run_on(&mut self, frame_in: &Frame, frame_out: &mut Frame) {
        // Check that the layout is accurate..

        self.with_filled_texels(
            |that| {
                (that.ops.expand)(&that.info, &that.in_texels, &mut that.pixel_in_buffer);

                let pixel_out = if let Some(ref recolor) = that.ops.recolor {
                    (recolor.from)(
                        &that.info,
                        &that.pixel_in_buffer,
                        &mut that.neutral_color_buffer,
                    );
                    (recolor.into)(
                        &that.info,
                        &that.neutral_color_buffer,
                        &mut that.pixel_out_buffer,
                    );
                    &that.pixel_out_buffer
                } else {
                    &that.pixel_in_buffer
                };

                // FIXME: necessary to do a reorder of pixels here? Or let join do this?
                (that.ops.join)(&that.info, pixel_out, &mut that.out_texels);
            },
            frame_in,
            frame_out,
        )
    }

    /// Choose iteration order of texels, fill with texels and then put them back.
    fn with_filled_texels(
        &mut self,
        mut texel_conversion: impl FnMut(&mut Self),
        frame_in: &Frame,
        frame_out: &mut Frame,
    ) {
        // We use a notion of 'supertexels', the common multiple of input and output texel blocks.
        // That is, if the input is a 2-by-2 pixel block and the output is single pixels then we
        // have 4 times as many outputs as inputs, respectively coordinates.
        //
        // Anyways, first we fill the coordinate buffers, then calculate the planar indices.
        let (sb_x, sb_y) = self.super_texel();
        let mut blocks = Self::blocks(sb_x.blocks.clone(), sb_y.blocks.clone());

        loop {
            self.super_blocks.clear();
            self.super_blocks.extend(blocks.by_ref().take(self.chunk));

            if self.super_blocks.is_empty() {
                break;
            }

            self.generate_coords(&sb_x, &sb_y);
            self.read_texels(frame_in.as_ref());
            texel_conversion(self);
            self.write_texels(frame_out.as_mut());
        }
    }

    fn super_texel(&self) -> (SuperTexel, SuperTexel) {
        let b0 = self.info.in_layout.texel.block;
        let b1 = self.info.out_layout.texel.block;

        let super_width = core::cmp::max(b0.width(), b1.width());
        let super_height = core::cmp::max(b0.height(), b1.height());

        let sampled_with = |w, bs| w / bs + if w % bs == 0 { 0 } else { 1 };

        let sb_width = sampled_with(self.info.in_layout.bytes.width, super_width);
        let sb_height = sampled_with(self.info.in_layout.bytes.height, super_height);

        (
            SuperTexel {
                blocks: 0..sb_width,
                in_super: super_width / b0.width(),
                out_super: super_width / b1.width(),
            },
            SuperTexel {
                blocks: 0..sb_height,
                in_super: super_height / b0.height(),
                out_super: super_height / b1.height(),
            },
        )
    }

    fn generate_coords(&mut self, sb_x: &SuperTexel, sb_y: &SuperTexel) {
        self.in_coords.clear();
        self.out_coords.clear();

        let in_blocks = Self::blocks(0..sb_x.in_super, 0..sb_y.in_super);
        let out_blocks = Self::blocks(0..sb_x.out_super, 0..sb_y.out_super);

        for &TexelCoord(Coord(bx, by)) in self.super_blocks.iter() {
            for TexelCoord(Coord(ix, iy)) in in_blocks.clone() {
                self.in_coords.push(TexelCoord(Coord(bx + ix, by + iy)));
            }

            for TexelCoord(Coord(ox, oy)) in out_blocks.clone() {
                self.out_coords.push(TexelCoord(Coord(bx + ox, by + oy)));
            }
        }

        self.in_index.resize_with(self.in_coords.len(), || 0);
        self.out_index.resize_with(self.out_coords.len(), || 0);

        (self.ops.in_index)(&self.info, &self.in_coords, &mut self.in_index);
        (self.ops.out_index)(&self.info, &self.out_coords, &mut self.out_index);
    }

    fn read_texels(&mut self, from: PlaneSource) {
        fn fetchFromTexelArray<T>(
            from: PlaneSource,
            idx: &[usize],
            into: &mut TexelBuffer,
            texel: Texel<T>,
        ) {
            into.resize_for_texel(idx.len(), texel);
            for (&index, into) in idx.iter().zip(into.as_mut_texels(texel)) {
                if let Some(from) = from.as_texels(texel).get(index) {
                    *into = texel.copy_val(from);
                }
            }
        }

        struct ReadUnit<'data, 'layout> {
            from: PlaneSource<'data, 'layout>,
            idx: &'data [usize],
            into: &'data mut TexelBuffer,
        }

        impl GenericTexelAction for ReadUnit<'_, '_> {
            fn run<T>(self, texel: Texel<T>) {
                fetchFromTexelArray(self.from, self.idx, self.into, texel)
            }
        }

        self.info.in_kind.action(ReadUnit {
            from,
            idx: &self.in_index,
            into: &mut self.in_texels,
        })
    }

    fn write_texels(&mut self, into: PlaneTarget) {
        fn writeFromTexelArray<T>(
            mut into: PlaneTarget,
            idx: &[usize],
            from: &TexelBuffer,
            texel: Texel<T>,
        ) {
            for (&index, from) in idx.iter().zip(from.as_texels(texel)) {
                if let Some(into) = into.as_mut_texels(texel).get_mut(index) {
                    *into = texel.copy_val(from);
                }
            }
        }

        struct WriteUnit<'data, 'layout> {
            into: PlaneTarget<'data, 'layout>,
            idx: &'data [usize],
            from: &'data TexelBuffer,
        }

        impl GenericTexelAction for WriteUnit<'_, '_> {
            fn run<T>(self, texel: Texel<T>) {
                writeFromTexelArray(self.into, self.idx, self.from, texel)
            }
        }

        self.info.out_kind.action(WriteUnit {
            into,
            idx: &self.in_index,
            from: &mut self.out_texels,
        });
    }

    fn blocks(x: Range<u32>, y: Range<u32>) -> impl Iterator<Item = TexelCoord> + Clone {
        x.clone()
            .into_iter()
            .map(move |x| core::iter::repeat(x).zip(y.clone()))
            .flatten()
            .map(|(x, y)| TexelCoord(Coord(x, y)))
    }

    fn index_from_in_info(info: &Info, texel: &[TexelCoord], idx: &mut [usize]) {
        Self::index_from_layer(&info.in_layout, texel, idx)
    }

    fn index_from_out_info(info: &Info, texel: &[TexelCoord], idx: &mut [usize]) {
        Self::index_from_layer(&info.out_layout, texel, idx)
    }

    fn index_from_layer(info: &FrameLayout, texel: &[TexelCoord], idx: &mut [usize]) {
        for (&TexelCoord(Coord(x, y)), idx) in texel.iter().zip(idx) {
            *idx = info.texel_index(x, y) as usize;
        }
    }
}

impl CommonPixel {
    fn expand_from_info(self) -> fn(&Info, &TexelBuffer, &mut TexelBuffer) {
        todo!()
    }

    fn join_from_info(self) -> fn(&Info, &TexelBuffer, &mut TexelBuffer) {
        todo!()
    }
}

impl TexelKind {
    fn byte_len(&self) -> usize {
        use TexelKind::*;
        match self {
            U8 => 1,
            U8x2 => 2,
            U8x3 => 3,
            U8x4 => 4,
            U16 => 2,
            U16x2 => 4,
            U16x3 => 6,
            U16x4 => 8,
            F32 => 4,
            F32x2 => 8,
            F32x3 => 12,
            F32x4 => 16,
        }
    }

    pub(crate) fn action<R>(self, action: impl GenericTexelAction<R>) -> R {
        match self {
            TexelKind::U8 => action.run(u8::texel()),
            TexelKind::U8x2 => action.run(<[u8; 2]>::texel()),
            TexelKind::U8x3 => action.run(<[u8; 3]>::texel()),
            TexelKind::U8x4 => action.run(<[u8; 4]>::texel()),
            TexelKind::U16 => action.run(<[u16; 1]>::texel()),
            TexelKind::U16x2 => action.run(<[u16; 2]>::texel()),
            TexelKind::U16x3 => action.run(<[u16; 3]>::texel()),
            TexelKind::U16x4 => action.run(<[u16; 4]>::texel()),
            TexelKind::F32 => action.run(<[f32; 1]>::texel()),
            TexelKind::F32x2 => action.run(<[f32; 2]>::texel()),
            TexelKind::F32x3 => action.run(<[f32; 3]>::texel()),
            TexelKind::F32x4 => action.run(<[f32; 4]>::texel()),
        }
    }
}

impl From<TexelBits> for TexelKind {
    fn from(texel: TexelBits) -> Self {
        Self::from(texel.bits)
    }
}

impl From<SampleBits> for TexelKind {
    fn from(bits: SampleBits) -> Self {
        match bits {
            SampleBits::Int8 | SampleBits::Int332 | SampleBits::Int233 => TexelKind::U8,
            SampleBits::Int16
            | SampleBits::Int4x4
            | SampleBits::Int_444
            | SampleBits::Int444_
            | SampleBits::Int565 => TexelKind::U16,
            SampleBits::Int8x2 => TexelKind::U8x2,
            SampleBits::Int8x3 => TexelKind::U8x3,
            SampleBits::Int8x4 => TexelKind::U8x4,
            SampleBits::Int16x2 => TexelKind::U16x2,
            SampleBits::Int16x3 => TexelKind::U16x3,
            SampleBits::Int16x4 => TexelKind::U16x4,
            SampleBits::Int1010102
            | SampleBits::Int2101010
            | SampleBits::Int101010_
            | SampleBits::Int_101010 => TexelKind::U16x2,
            SampleBits::Float16x4 => TexelKind::U16x4,
            SampleBits::Float32 => TexelKind::F32,
            SampleBits::Float32x2 => TexelKind::F32x2,
            SampleBits::Float32x3 => TexelKind::F32x3,
            SampleBits::Float32x4 => TexelKind::F32x4,
        }
    }
}
