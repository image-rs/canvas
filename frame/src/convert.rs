//! Conversion between texels, mostly color.
//!
//! Takes quite a lot of inspiration from how GPUs work. We have a primitive sampler unit, a
//! fragment unit, and pipeline multiple texels in parallel.
use canvas::{canvas::Coord, TexelBuffer};
use core::ops::Range;

use crate::buffer::Frame;
use crate::layout::{ByteLayout, Layout};

#[repr(transparent)]
struct TexelCoord(Coord);

struct Info {
    /// Layout of the input frame.
    in_layout: Layout,
    /// Layout of the output frame.
    out_layout: Layout,
    /// The selected way to represent pixels in common parameter space.
    common_pixel: CommonPixel,
    /// The selected common color space, midpoint for conversion.
    common_color: CommonColor,
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
                expand: todo!(),
                recolor: Some(RecolorOps {
                    from: todo!(),
                    into: todo!(),
                }),
                join: todo!(),
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
            self.fetch_texels(frame_in);
            texel_conversion(self);
            self.unfetch_texels(frame_out);
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

    fn fetch_texels(&mut self, from: &Frame) {
        todo!()
    }

    fn unfetch_texels(&mut self, into: &mut Frame) {
        todo!()
    }

    fn blocks(x: Range<u32>, y: Range<u32>) -> impl Iterator<Item = TexelCoord> + Clone {
        x.clone()
            .into_iter()
            .map(move |x| core::iter::repeat(x).zip(y.clone()))
            .flatten()
            .map(|(x, y)| TexelCoord(Coord(x, y)))
    }

    fn index_from_in_info(info: &Info, texel: &[TexelCoord], idx: &mut [usize]) {
        Self::index_from_layer(&info.in_layout.bytes, texel, idx)
    }

    fn index_from_out_info(info: &Info, texel: &[TexelCoord], idx: &mut [usize]) {
        Self::index_from_layer(&info.out_layout.bytes, texel, idx)
    }

    fn index_from_layer(info: &ByteLayout, texel: &[TexelCoord], idx: &mut [usize]) {
        for (&TexelCoord(Coord(x, y)), idx) in texel.iter().zip(idx) {
            *idx = info.texel_index(x, y) as usize;
        }
    }
}
