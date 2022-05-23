//! Conversion between texels, mostly color.
//!
//! Takes quite a lot of inspiration from how GPUs work. We have a primitive sampler unit, a
//! fragment unit, and pipeline multiple texels in parallel.
use core::ops::Range;
use image_texel::image::{ImageMut, ImageRef};
use image_texel::{image::Coord, AsTexel, Texel, TexelBuffer};

use crate::arch::ShuffleOps;
use crate::layout_::{
    BitEncoding, Block, CanvasLayout, SampleBits, SampleParts, SamplePitch, Texel as TexelBits,
};
use crate::Canvas;

/// A buffer for conversion.
pub struct Converter {
    /// How many texels to do at once.
    ///
    /// Each entry in `in_slices` and `out_slices` except for the last has the size of `chunk`.
    chunk: usize,
    /// The number of chunks to do at once.
    chunk_count: usize,

    super_blocks: Vec<TexelCoord>,
    /// Buffer where we store input texels after reading them.
    in_texels: TexelBuffer,
    /// Texel coordinates of stored texels.
    in_coords: Vec<Coord>,
    /// Index in the input planes.
    in_index_list: Vec<usize>,
    /// Runs of texels to be read by anything reading input texels.
    /// Each entry refers to a range of indices in `in_index` and a range of corresponding texels
    /// in `in_texels`, or it can refer directly to the input image.
    in_slices: TexelBuffer<[usize; 2]>,
    /// Buffer where we store input texels before writing.
    out_texels: TexelBuffer,
    /// Texel coordinates of stored texels.
    out_coords: Vec<Coord>,
    /// Index in the input planes.
    out_index_list: Vec<usize>,
    /// Runs of texels to be read by anything writing output texels.
    /// Each entry refers to a range of indices in `out_index` and a range of corresponding texels
    /// in `out_texels`, or it can refer directly to the output image.
    out_slices: TexelBuffer<[usize; 2]>,

    /// The input texels, split into pixels in the color's natural order.
    pixel_in_buffer: TexelBuffer,
    /// The pixels in a color space in the middle of in and out, mostly CIE XYZ+Alpha.
    neutral_color_buffer: TexelBuffer,
    /// The output texels, split into pixels in the color's natural order.
    pixel_out_buffer: TexelBuffer,
}

#[repr(transparent)]
struct TexelCoord(Coord);

struct Info {
    /// Layout of the input frame.
    in_layout: CanvasLayout,
    /// Layout of the output frame.
    out_layout: CanvasLayout,
    /// The selected way to represent pixels in common parameter space.
    common_pixel: CommonPixel,
    /// The selected common color space, midpoint for conversion.
    #[allow(unused)]
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
#[derive(Clone, Copy, Debug)]
pub enum TexelKind {
    U8,
    U8x2,
    U8x3,
    U8x4,
    U8x6,
    U16,
    U16x2,
    U16x3,
    U16x4,
    U16x6,
    F32,
    F32x2,
    F32x3,
    F32x4,
    F32x6,
}

pub(crate) trait GenericTexelAction<R = ()> {
    fn run<T>(self, texel: Texel<T>) -> R;
}

/// Specifies which bits a channel comes from, within a `TexelKind` aggregate.
#[derive(Clone, Copy, Debug)]
struct FromBits {
    begin: usize,
    len: usize,
}

/// FIXME(color): What about colors with more than three stimuli (e.g. scientific instruments such
/// as Mars202 have 13 bands). In general, any observer function that's not a linear combination of
/// the CIE XYZ can not be converted from/to color spaces defined through it without loss because
/// 'observation' is assumed to measure spectrum under a wavelength response curve.
///
/// Examples that fall outside of this, 'yellow' cones in human vision and other awesome mutations
/// such as partial color blindness, that's a shifted response curve of the common cone variant. Do
/// we want to be able to represent those?
#[derive(Clone, Copy, Debug)]
enum CommonPixel {
    F32x4,
}

#[derive(Clone, Copy, Debug)]
enum CommonColor {
    CieXyz,
}

type PlaneSource<'data, 'layout> = ImageRef<'data, &'layout CanvasLayout>;
type PlaneTarget<'data, 'layout> = ImageMut<'data, &'layout mut CanvasLayout>;

/// The function pointers doing the conversion.
///
/// Note how there are no types involved here. Instead, `TexelCoord` is a polymorphic buffer that
/// each function can access with any type it sees feed. We expect the constructor to ensure only
/// matching types are being used.
struct ConvertOps {
    /// Convert in texel coordinates to an index in the color plane.
    in_index: fn(&Info, &[Coord], &mut [usize], ChunkSpec),
    /// Convert out texel coordinates to an index in the color plane.
    out_index: fn(&Info, &[Coord], &mut [usize], ChunkSpec),

    /// Expand all texels into pixels in normalized channel order.
    expand: fn(&Info, &ConvertOps, &TexelBuffer, &mut TexelBuffer, &mut [PlaneSource]),
    /// Take pixels in normalized channel order and apply color conversion.
    recolor: Option<RecolorOps>,
    /// Join all pixels from normalized channel order to texels, clamping.
    join: fn(&Info, &ConvertOps, &TexelBuffer, &mut TexelBuffer, &mut [PlaneTarget]),

    /// Well-define bit/byte/channel shuffle operations on common texel combinations.
    shuffle: ShuffleOps,

    // Ops that are available, dynamically.
    /// Simple int-shuffles, avoiding color decoding.
    int_shuffle: Option<IntShuffleOps>,
}

struct TexelConvertWith<'lt> {
    ops: &'lt mut dyn FnMut(&mut Converter, &mut [PlaneSource], &mut [PlaneTarget]),
    should_defer_texel_read: bool,
    should_defer_texel_write: bool,
}

struct RecolorOps {
    from: fn(&Info, &TexelBuffer, &mut TexelBuffer),
    into: fn(&Info, &TexelBuffer, &mut TexelBuffer),
}

struct IntShuffleOps {
    call: fn(&mut Converter, &ConvertOps, [u8; 4], &[PlaneSource], &mut [PlaneTarget]),
    shuffle: [u8; 4],
    should_defer_texel_read: bool,
    should_defer_texel_write: bool,
}

struct SuperTexel {
    blocks: Range<u32>,
    /// In blocks per super block.
    in_super: u32,
    /// Out blocks per super block.
    out_super: u32,
}

pub(crate) struct ChunkSpec<'ch> {
    pub chunks: &'ch mut [[usize; 2]],
    pub chunk_size: usize,
    pub should_defer_texel_ops: bool,
}

impl Converter {
    pub fn new() -> Self {
        Converter {
            chunk: 1024,
            chunk_count: 1,
            super_blocks: vec![],
            in_texels: TexelBuffer::default(),
            in_coords: vec![],
            in_index_list: vec![],
            in_slices: TexelBuffer::default(),
            out_texels: TexelBuffer::default(),
            out_coords: vec![],
            out_index_list: vec![],
            out_slices: TexelBuffer::default(),
            pixel_in_buffer: TexelBuffer::default(),
            neutral_color_buffer: TexelBuffer::default(),
            pixel_out_buffer: TexelBuffer::default(),
        }
    }

    fn recolor_ops(lhs: &CanvasLayout, rhs: &CanvasLayout) -> Option<RecolorOps> {
        match (lhs.color.as_ref()?, rhs.color.as_ref()?) {
            (c0, c1) if c0 == c1 => None,
            // Some more special methods?
            (_, _) => Some(RecolorOps {
                from: CommonColor::cie_xyz_from_info,
                into: CommonColor::cie_xyz_into_info,
            }),
        }
    }

    pub fn run_on(&mut self, frame_in: &Canvas, frame_out: &mut Canvas) {
        let info = Info {
            in_layout: frame_in.layout().clone(),
            out_layout: frame_out.layout().clone(),
            // FIXME(perf): not optimal in all cases, but necessary for accurate conversion.
            // allow configuration / detect trivial conversion.
            common_pixel: CommonPixel::F32x4,
            // FIXME(color): currently the only case, we also go through this if any conversion is
            // required, but of course in general a potential loss of accuracy. General enough?
            common_color: CommonColor::CieXyz,
            in_kind: TexelKind::from(frame_in.layout().texel.bits),
            out_kind: TexelKind::from(frame_out.layout().texel.bits),
        };

        let recolor = Self::recolor_ops(frame_in.layout(), frame_out.layout());
        let int_shuffle = self
            .convert_intbuf_with_nocolor_ops(&info)
            .filter(|_| recolor.is_none());

        let ops = ConvertOps {
            in_index: Self::index_from_in_info,
            out_index: Self::index_from_out_info,
            expand: CommonPixel::expand_from_info,
            recolor,
            join: CommonPixel::join_from_info,
            shuffle: ShuffleOps::default().with_arch(),

            int_shuffle,
        };

        // Choose how we actually perform conversion.
        let mut convert_texelbuf_with_ops;
        let mut convert_with_intshuffle;
        let convert_with: TexelConvertWith = {
            if let Some(int_ops) = &ops.int_shuffle {
                convert_with_intshuffle =
                    |that: &mut Self, fi: &mut [PlaneSource], fo: &mut [PlaneTarget]| {
                        (int_ops.call)(that, &ops, int_ops.shuffle, fi, fo)
                    };
                TexelConvertWith {
                    ops: &mut convert_with_intshuffle,
                    should_defer_texel_read: int_ops.should_defer_texel_read,
                    should_defer_texel_write: int_ops.should_defer_texel_write,
                }
            } else {
                convert_texelbuf_with_ops =
                    |that: &mut Self, fi: &mut [PlaneSource], fo: &mut [PlaneTarget]| {
                        that.convert_texelbuf_with_ops(&info, &ops, fi, fo)
                    };

                TexelConvertWith {
                    ops: &mut convert_texelbuf_with_ops,
                    should_defer_texel_read: false,
                    should_defer_texel_write: false,
                }
            }
        };

        self.with_filled_texels(convert_with, &info, &ops, frame_in, frame_out)
    }

    /// Convert all loaded texels, using the provided `ConvertOps` as dynamic function selection.
    ///
    /// Assumes that the caller resized all buffers appropriately (TODO: should be a better
    /// contract for this, with explicit data flow of this invariant and what 'proper' size means,
    /// because it depends on the chosen ops).
    fn convert_texelbuf_with_ops(
        &mut self,
        info: &Info,
        ops: &ConvertOps,
        frame_in: &mut [PlaneSource],
        frame_out: &mut [PlaneTarget],
    ) {
        (ops.expand)(
            &info,
            ops,
            &self.in_texels,
            &mut self.pixel_in_buffer,
            frame_in,
        );

        let pixel_out = if let Some(ref recolor) = ops.recolor {
            (recolor.from)(&info, &self.pixel_in_buffer, &mut self.neutral_color_buffer);
            (recolor.into)(
                &info,
                &self.neutral_color_buffer,
                &mut self.pixel_out_buffer,
            );
            &self.pixel_out_buffer
        } else {
            &self.pixel_in_buffer
        };

        // FIXME: necessary to do a reorder of pixels here? Or let join do this?
        (ops.join)(&info, ops, pixel_out, &mut self.out_texels, frame_out);
    }

    /// Special case on `convert_texelbuf_with_ops`, when both buffers:
    /// * utilize an expasion-roundtrip-safe color/bit combination
    /// * have the same bit depths on all channels
    /// * do not require any color conversion between them
    ///
    /// This avoids expanding them into `pixel_in_buffer` where they'd be represented as `f32x4`
    /// and thus undergo an expensive `u8->f32->u8` cast chain.
    fn convert_intbuf_with_nocolor_ops(&mut self, info: &Info) -> Option<IntShuffleOps> {
        fn determine_shuffle(inp: SampleParts, outp: SampleParts) -> Option<[u8; 4]> {
            let mut ch_from_common = [0x80u8; 4];
            let mut ch_from_input = [0x80u8; 4];

            // Not yet handled, we need independent channels and the same amount.
            // FIXME(perf): we could use very similar code to expand pixels from blocks but that
            // requires specialized shuffle methods.
            // FIXME(perf): for simple linear combinations in non-linear space (e.g. both Rec.601
            // and Rec.709 specify their YUV in the electric domain even though that's not
            // accurate) we could do them here, too.
            // FIXME(perf): Utilize a library for this, e.g. `dcv-color-primitives`, but those are
            // heavy and may have different implementation goals.
            // - `dvc-color-primitives` uses an unsafe globals to fetch the `fn` to use...
            // - `dvc-color-primitives` also depends on `paste`, a proc-macro crate.
            if inp.pitch != SamplePitch::PixelBits || outp.pitch != SamplePitch::PixelBits {
                return None;
            }

            for ((ch, common_pos), idx) in inp.channels().zip(0..4) {
                if ch.is_some() {
                    ch_from_input[common_pos as usize] = idx;
                }
            }

            for ((ch, common_pos), idx) in outp.channels().zip(0..4) {
                if ch.is_some() {
                    ch_from_common[idx] = ch_from_input[common_pos as usize];
                }
            }

            Some(ch_from_common)
        }

        let in_texel = &info.in_layout.texel;
        let out_texel = &info.out_layout.texel;

        // We can't handle color conversion inside the shuffles.
        if info.in_layout.color != info.out_layout.color {
            return None;
        }

        let shuffle = determine_shuffle(in_texel.parts, out_texel.parts)?;

        trait Shuffle<T, const N: usize, const M: usize> {
            fn run(_: &ConvertOps, _: &[[T; N]], _: &mut [[T; M]], _: [u8; 4]);
        }

        fn shuffle_with_texel<T, S: Shuffle<T, N, M>, const N: usize, const M: usize>(
            that: &mut Converter,
            ops: &ConvertOps,
            shuffle: [u8; 4],
            source: &[PlaneSource],
            target: &mut [PlaneTarget],
        ) where
            T: AsTexel,
        {
            let in_texel = T::texel().array::<N>();
            let out_texel = T::texel().array::<M>();

            let source_texels = source[0].as_texels(in_texel);
            let target_texels = target[0].as_mut_texels(out_texel);

            let in_texels = that.in_texels.as_texels(in_texel);
            let out_texels = that.out_texels.as_mut_texels(out_texel);

            let in_slices = that.in_slices.iter_mut();
            let out_slices = that.out_slices.iter_mut();
            let chunks = (0..in_texels.len()).step_by(that.chunk);

            for ((islice, oslice), chunk_start) in in_slices.zip(out_slices).zip(chunks) {
                let length = in_texels[chunk_start..].len().min(that.chunk);

                let input_slice = if islice[1] > 0 {
                    debug_assert!(length == islice[1]);
                    let length = core::mem::replace(&mut islice[1], 0);
                    &source_texels[islice[0]..][..length]
                } else {
                    &in_texels[chunk_start..][..length]
                };

                let output_slice = if oslice[1] > 0 {
                    debug_assert!(length == oslice[1]);
                    let length = core::mem::replace(&mut oslice[1], 0);
                    &mut target_texels[oslice[0]..][..length]
                } else {
                    &mut out_texels[chunk_start..][..length]
                };

                S::run(ops, input_slice, output_slice, shuffle)
            }
        }

        struct ShuffleInt8;
        struct ShuffleInt16;

        impl Shuffle<u8, 4, 4> for ShuffleInt8 {
            fn run(ops: &ConvertOps, inp: &[[u8; 4]], outp: &mut [[u8; 4]], shuffle: [u8; 4]) {
                outp.copy_from_slice(inp);
                (ops.shuffle.shuffle_u8x4)(outp, shuffle);
            }
        }

        impl Shuffle<u8, 3, 4> for ShuffleInt8 {
            fn run(ops: &ConvertOps, inp: &[[u8; 3]], outp: &mut [[u8; 4]], shuffle: [u8; 4]) {
                (ops.shuffle.shuffle_u8x3_to_u8x4)(inp, outp, shuffle);
            }
        }

        impl Shuffle<u8, 4, 3> for ShuffleInt8 {
            fn run(ops: &ConvertOps, inp: &[[u8; 4]], outp: &mut [[u8; 3]], shuffle: [u8; 4]) {
                let shuffle = [shuffle[0], shuffle[1], shuffle[2]];
                (ops.shuffle.shuffle_u8x4_to_u8x3)(inp, outp, shuffle);
            }
        }

        impl Shuffle<u16, 4, 4> for ShuffleInt16 {
            fn run(ops: &ConvertOps, inp: &[[u16; 4]], outp: &mut [[u16; 4]], shuffle: [u8; 4]) {
                outp.copy_from_slice(inp);
                (ops.shuffle.shuffle_u16x4)(outp, shuffle);
            }
        }

        impl Shuffle<u16, 3, 4> for ShuffleInt16 {
            fn run(ops: &ConvertOps, inp: &[[u16; 3]], outp: &mut [[u16; 4]], shuffle: [u8; 4]) {
                (ops.shuffle.shuffle_u16x3_to_u16x4)(inp, outp, shuffle);
            }
        }

        impl Shuffle<u16, 4, 3> for ShuffleInt16 {
            fn run(ops: &ConvertOps, inp: &[[u16; 4]], outp: &mut [[u16; 3]], shuffle: [u8; 4]) {
                let shuffle = [shuffle[0], shuffle[1], shuffle[2]];
                (ops.shuffle.shuffle_u16x4_to_u16x3)(inp, outp, shuffle);
            }
        }

        Some(match (in_texel.bits, out_texel.bits) {
            (SampleBits::UInt8x4, SampleBits::UInt8x4)
            | (SampleBits::Int8x4, SampleBits::Int8x4) => IntShuffleOps {
                call: shuffle_with_texel::<u8, ShuffleInt8, 4, 4>,
                shuffle,
                should_defer_texel_read: true,
                should_defer_texel_write: true,
            },
            (SampleBits::UInt8x3, SampleBits::UInt8x4)
            | (SampleBits::Int8x3, SampleBits::Int8x4) => IntShuffleOps {
                call: shuffle_with_texel::<u8, ShuffleInt8, 3, 4>,
                shuffle,
                should_defer_texel_read: true,
                should_defer_texel_write: true,
            },
            (SampleBits::UInt8x4, SampleBits::UInt8x3)
            | (SampleBits::Int8x4, SampleBits::Int8x3) => IntShuffleOps {
                call: shuffle_with_texel::<u8, ShuffleInt8, 4, 3>,
                shuffle,
                should_defer_texel_read: true,
                should_defer_texel_write: true,
            },

            // Simple U16 cases.
            (SampleBits::UInt16x4, SampleBits::UInt16x4)
            | (SampleBits::Int16x4, SampleBits::Int16x4) => IntShuffleOps {
                call: shuffle_with_texel::<u16, ShuffleInt16, 4, 4>,
                shuffle,
                should_defer_texel_read: true,
                should_defer_texel_write: true,
            },
            (SampleBits::UInt16x3, SampleBits::UInt16x4)
            | (SampleBits::Int16x3, SampleBits::Int16x4) => IntShuffleOps {
                call: shuffle_with_texel::<u16, ShuffleInt16, 3, 4>,
                shuffle,
                should_defer_texel_read: true,
                should_defer_texel_write: true,
            },
            (SampleBits::UInt16x4, SampleBits::UInt16x3)
            | (SampleBits::Int16x4, SampleBits::Int16x3) => IntShuffleOps {
                call: shuffle_with_texel::<u16, ShuffleInt16, 4, 3>,
                shuffle,
                should_defer_texel_read: true,
                should_defer_texel_write: true,
            },
            _ => return None,
        })
    }

    /// Choose iteration order of texels, fill with texels and then put them back.
    fn with_filled_texels(
        &mut self,
        texel_conversion: TexelConvertWith,
        info: &Info,
        ops: &ConvertOps,
        frame_in: &Canvas,
        frame_out: &mut Canvas,
    ) {
        use core::slice::from_mut;
        // We use a notion of 'supertexels', the common multiple of input and output texel blocks.
        // That is, if the input is a 2-by-2 pixel block and the output is single pixels then we
        // have 4 times as many outputs as inputs, respectively coordinates.
        //
        // Anyways, first we fill the coordinate buffers, then calculate the planar indices.
        let (sb_x, sb_y) = self.super_texel(info);
        let mut blocks = Self::blocks(sb_x.blocks.clone(), sb_y.blocks.clone());

        loop {
            self.super_blocks.clear();
            let at_once = self.chunk * self.chunk_count;
            let blocks = blocks.by_ref().take(at_once);
            self.super_blocks.extend(blocks.map(TexelCoord));

            if self.super_blocks.is_empty() {
                break;
            }

            self.generate_coords(info, ops, &texel_conversion, &sb_x, &sb_y);
            self.reserve_buffers(info, ops);
            // FIXME(planar): should be repeated for all planes?
            self.read_texels(info, ops, &texel_conversion, frame_in.as_ref());

            let mut frame_in = frame_in.as_ref();
            let mut frame_out = frame_out.as_mut();
            (texel_conversion.ops)(self, from_mut(&mut frame_in), from_mut(&mut frame_out));

            // FIXME(planar): should be repeated for all planes?
            self.write_texels(info, ops, &texel_conversion, frame_out);
        }
    }

    fn super_texel(&self, info: &Info) -> (SuperTexel, SuperTexel) {
        let b0 = info.in_layout.texel.block;
        let b1 = info.out_layout.texel.block;

        let super_width = core::cmp::max(b0.width(), b1.width());
        let super_height = core::cmp::max(b0.height(), b1.height());

        let sampled_with = |w, bs| w / bs + if w % bs == 0 { 0 } else { 1 };

        let sb_width = sampled_with(info.in_layout.bytes.width, super_width);
        let sb_height = sampled_with(info.in_layout.bytes.height, super_height);

        (
            SuperTexel {
                blocks: 0..sb_height,
                in_super: super_height / b0.height(),
                out_super: super_height / b1.height(),
            },
            SuperTexel {
                blocks: 0..sb_width,
                in_super: super_width / b0.width(),
                out_super: super_width / b1.width(),
            },
        )
    }

    fn generate_coords(
        &mut self,
        info: &Info,
        ops: &ConvertOps,
        converter: &TexelConvertWith,
        sb_x: &SuperTexel,
        sb_y: &SuperTexel,
    ) {
        fn is_trivial_super(sup: &SuperTexel) -> bool {
            sup.in_super == 1 && sup.out_super == 1
        }

        self.in_coords.clear();
        self.out_coords.clear();

        if is_trivial_super(sb_x) && is_trivial_super(sb_y) {
            // Faster than rustc having to look through and special case the iteration/clones
            // below. For some reason, it doesn't do well on `Range::zip()::flatten`.
            let blocks = self
                .super_blocks
                .iter()
                .map(|&TexelCoord(Coord(bx, by))| Coord(bx, by));
            // FIXME(perf): actually, we'd like to just reuse the `super_blocks` vector where ever
            // possible. This is a pure copy at the byte-level.
            self.in_coords.extend(blocks.clone());
            self.out_coords.extend(blocks.clone());
        } else {
            let in_blocks = Self::blocks(0..sb_x.in_super, 0..sb_y.in_super);
            let out_blocks = Self::blocks(0..sb_x.out_super, 0..sb_y.out_super);

            // FIXME(perf): the other iteration order would serve us better. Then there is a larger
            // bulk of coordinates looped through at the same time, with less branching as a call
            // to std::vec::Vec::extend could rely on the exact length of the iterator.
            for &TexelCoord(Coord(bx, by)) in self.super_blocks.iter() {
                for Coord(ix, iy) in in_blocks.clone() {
                    self.in_coords.push(Coord(bx + ix, by + iy));
                }

                for Coord(ox, oy) in out_blocks.clone() {
                    self.out_coords.push(Coord(bx + ox, by + oy));
                }
            }
        }

        self.in_index_list.resize_with(self.in_coords.len(), || 0);
        self.out_index_list.resize_with(self.out_coords.len(), || 0);

        self.in_slices.resize(self.chunk_count);
        self.out_slices.resize(self.chunk_count);

        let in_chunk = ChunkSpec {
            chunks: self.in_slices.as_mut_slice(),
            chunk_size: self.chunk,
            should_defer_texel_ops: converter.should_defer_texel_read,
        };

        let out_chunk = ChunkSpec {
            chunks: self.out_slices.as_mut_slice(),
            chunk_size: self.chunk,
            should_defer_texel_ops: converter.should_defer_texel_write,
        };

        (ops.in_index)(&info, &self.in_coords, &mut self.in_index_list, in_chunk);
        (ops.out_index)(&info, &self.out_coords, &mut self.out_index_list, out_chunk);
    }

    fn reserve_buffers(&mut self, info: &Info, ops: &ConvertOps) {
        struct ResizeAction<'data>(&'data mut TexelBuffer, usize);

        impl GenericTexelAction for ResizeAction<'_> {
            fn run<T>(self, texel: Texel<T>) {
                self.0.resize_for_texel(self.1, texel)
            }
        }

        let num_in_texels = self.in_coords.len();
        let in_block = info.in_layout.texel.block;
        let in_pixels = (in_block.width() * in_block.height()) as usize * num_in_texels;
        info.in_kind
            .action(ResizeAction(&mut self.in_texels, num_in_texels));

        let num_out_texels = self.out_coords.len();
        let out_block = info.out_layout.texel.block;
        let out_pixels = (out_block.width() * out_block.height()) as usize * num_out_texels;
        info.out_kind
            .action(ResizeAction(&mut self.out_texels, num_out_texels));

        debug_assert!(
            in_pixels == out_pixels,
            "Mismatching in super block layout: {} {}",
            in_pixels,
            out_pixels
        );

        let pixels = in_pixels.max(out_pixels);
        info.common_pixel
            .action(ResizeAction(&mut self.pixel_in_buffer, pixels));

        if let Some(_) = ops.recolor {
            info.common_pixel
                .action(ResizeAction(&mut self.neutral_color_buffer, pixels));
            info.common_pixel
                .action(ResizeAction(&mut self.pixel_out_buffer, pixels));
        }
    }

    fn read_texels(
        &mut self,
        info: &Info,
        _: &ConvertOps,
        converter: &TexelConvertWith,
        from: PlaneSource,
    ) {
        fn fetch_from_texel_array<T>(
            from: &PlaneSource,
            idx: &[usize],
            into: &mut TexelBuffer,
            range: Range<usize>,
            texel: Texel<T>,
        ) {
            into.resize_for_texel(idx.len(), texel);
            let idx = idx[range.clone()].iter();
            let texels = &mut into.as_mut_texels(texel)[range];

            // FIXME(planar):
            // FIXME(color): multi-planar texel fetch.
            for (&index, into) in idx.zip(texels) {
                if let Some(from) = from.as_texels(texel).get(index) {
                    *into = texel.copy_val(from);
                }
            }
        }

        struct ReadUnit<'plane, 'data, 'layout> {
            from: &'plane PlaneSource<'data, 'layout>,
            idx: &'plane [usize],
            into: &'plane mut TexelBuffer,
            range: Range<usize>,
        }

        impl GenericTexelAction for ReadUnit<'_, '_, '_> {
            fn run<T>(self, texel: Texel<T>) {
                fetch_from_texel_array(self.from, self.idx, self.into, self.range, texel)
            }
        }

        if converter.should_defer_texel_read {
            /* For deferred reading, we expect some functions to do the transfer for us allowing us
             * to leave the source texel blank, uninitialized, or in an otherwise unreadable state.
             * We should skip them. The protocol here is that each chunk has two indices; the index
             * in the plane texture and the index up-to-which the texels are to be ignored in the
             * `in_texels`.
             */
            let chunks = self.in_slices.as_mut_slice();
            let indexes = self.in_index_list.chunks(self.chunk);
            let range = (0..self.in_index_list.len()).step_by(self.chunk);

            for (chunk, (indexes, start)) in chunks.iter_mut().zip(indexes.zip(range)) {
                let [_, available] = chunk;

                if *available == indexes.len() {
                    continue;
                }

                // Only use the input frame if all indexes are available in the layout.
                // For this reason read all texels individually into the texel buffer otherwise and
                // the indicate that no texels are available from the layout.
                *available = 0;
                info.in_kind.action(ReadUnit {
                    from: &from,
                    idx: &self.in_index_list,
                    into: &mut self.in_texels,
                    range: start..start + indexes.len(),
                });
            }
        } else {
            info.in_kind.action(ReadUnit {
                from: &from,
                idx: &self.in_index_list,
                into: &mut self.in_texels,
                range: 0..self.in_index_list.len(),
            })
        }
    }

    /// The job of this function is transferring texel information onto the target plane.
    ///
    fn write_texels(
        &mut self,
        info: &Info,
        _: &ConvertOps,
        converter: &TexelConvertWith,
        mut into: PlaneTarget,
    ) {
        fn write_from_texel_array<T>(
            into: &mut PlaneTarget,
            idx: &[usize],
            from: &TexelBuffer,
            range: Range<usize>,
            texel: Texel<T>,
        ) {
            // FIXME(planar):
            // FIXME(color): multi-planar texel write.
            let idx = idx[range.clone()].iter();
            let texels = &from.as_texels(texel)[range];

            for (&index, from) in idx.zip(texels) {
                if let Some(into) = into.as_mut_texels(texel).get_mut(index) {
                    *into = texel.copy_val(from);
                }
            }
        }

        struct WriteUnit<'plane, 'data, 'layout> {
            into: &'plane mut PlaneTarget<'data, 'layout>,
            idx: &'plane [usize],
            from: &'plane TexelBuffer,
            range: Range<usize>,
        }

        impl GenericTexelAction for WriteUnit<'_, '_, '_> {
            fn run<T>(self, texel: Texel<T>) {
                write_from_texel_array(self.into, self.idx, self.from, self.range, texel)
            }
        }

        if converter.should_defer_texel_write {
            /* For deferred writing, we expect some functions to have already done the transfer for
             * us and left the source texel blank, uninitialized, or in an otherwise unreadable
             * state. We must skip them. The protocol here is that each chunk has two indices; the
             * index in the plane texture and the index up-to-which the texels are to be ignored in
             * the `out_texels`.
             */
            let chunks = self.out_slices.as_slice();
            let indexes = self.out_index_list.chunks(self.chunk);
            let range = (0..self.out_index_list.len()).step_by(self.chunk);

            for (&chunk, (indexes, start)) in chunks.iter().zip(indexes.zip(range)) {
                let [_, unwritten] = chunk;
                debug_assert!(unwritten <= indexes.len());

                if unwritten > indexes.len() {
                    continue;
                }

                if unwritten == 0 {
                    continue;
                }

                let offset = indexes.len() - unwritten;
                info.out_kind.action(WriteUnit {
                    into: &mut into,
                    idx: &self.out_index_list,
                    from: &self.out_texels,
                    range: start + offset..start + indexes.len(),
                });
            }
        } else {
            info.out_kind.action(WriteUnit {
                into: &mut into,
                idx: &self.out_index_list,
                from: &self.out_texels,
                range: 0..self.out_index_list.len(),
            });
        }
    }

    fn blocks(x: Range<u32>, y: Range<u32>) -> impl Iterator<Item = Coord> + Clone {
        x.clone()
            .into_iter()
            .map(move |x| core::iter::repeat(x).zip(y.clone()))
            .flatten()
            .map(|(x, y)| Coord(x, y))
    }

    fn index_from_in_info(info: &Info, texel: &[Coord], idx: &mut [usize], chunks: ChunkSpec) {
        Self::index_from_layer(&info.in_layout, texel, idx, chunks)
    }

    fn index_from_out_info(info: &Info, texel: &[Coord], idx: &mut [usize], chunks: ChunkSpec) {
        Self::index_from_layer(&info.out_layout, texel, idx, chunks)
    }

    fn index_from_layer(
        info: &CanvasLayout,
        texel: &[Coord],
        idx: &mut [usize],
        chunks: ChunkSpec,
    ) {
        // FIXME(perf): review performance. Could probably be vectorized by hand.
        info.fill_texel_indices_impl(idx, texel, chunks)
    }
}

trait ExpandYuvLike<const IN: usize, const OUT: usize> {
    fn expand<T: Copy>(_: [T; IN], fill: T) -> [[T; 4]; OUT];
}

impl CommonPixel {
    /// Create pixels from our aggregated block information.
    ///
    /// For each pixel in each texel block, our task is to extract all channels (at most 4) and
    /// convert their bit representation to the `CommonPixel` representation, then put them into
    /// the expected channel give by the color channel's normal form.
    fn expand_from_info(
        info: &Info,
        // FIXME(perf): similar to join_from_info we could use shuffle sometimes..
        _: &ConvertOps,
        in_texel: &TexelBuffer,
        pixel_buf: &mut TexelBuffer,
        _: &mut [PlaneSource],
    ) {
        // FIXME(perf): some bit/part combinations require no reordering of bits and could skip
        // large parts of this phase, or be done vectorized, effectively amounting to a memcpy when
        // the expanded value has the same representation as the texel.
        let TexelBits { bits, parts, .. } = info.in_layout.texel;

        match parts.pitch {
            SamplePitch::PixelBits => {
                Self::expand_bits(info, FromBits::for_pixel(bits, parts), in_texel, pixel_buf)
            }
            SamplePitch::Yuv422 => {
                debug_assert!(matches!(info.in_layout.texel.block, Block::Sub1x2));
                debug_assert!(matches!(info.in_layout.texel.parts.num_components(), 3));
                Self::expand_yuv422(info, in_texel, pixel_buf);
            }
            SamplePitch::Yuy2 => {
                debug_assert!(matches!(info.in_layout.texel.block, Block::Sub1x2));
                debug_assert!(matches!(info.in_layout.texel.parts.num_components(), 3));
                Self::expand_yuy2(info, in_texel, pixel_buf);
            }
            SamplePitch::Yuv411 => {
                debug_assert!(matches!(info.in_layout.texel.block, Block::Sub1x4));
                debug_assert!(matches!(info.in_layout.texel.parts.num_components(), 3));
                Self::expand_yuv411(info, in_texel, pixel_buf);
            }
            // FIXME(color): BC1-6
            other => {
                debug_assert!(false, "{:?}", other);
            }
        }
    }

    fn expand_bits(
        info: &Info,
        bits: [FromBits; 4],
        in_texel: &TexelBuffer,
        pixel_buf: &mut TexelBuffer,
    ) {
        let (encoding, len) = info.in_layout.texel.bits.bit_encoding();

        if encoding[..len as usize] == [BitEncoding::UInt; 6][..len as usize] {
            return Self::expand_ints(info, bits, in_texel, pixel_buf);
        } else if encoding[..len as usize] == [BitEncoding::Float; 6][..len as usize] {
            return Self::expand_floats(info, bits, in_texel, pixel_buf);
        } else {
            // FIXME(color): error treatment..
            debug_assert!(false, "{:?}", &encoding[..len as usize]);
        }
    }

    fn expand_ints(
        info: &Info,
        bits: [FromBits; 4],
        in_texel: &TexelBuffer,
        pixel_buf: &mut TexelBuffer,
    ) {
        struct ExpandAction<'data, T> {
            expand: Texel<T>,
            expand_fn: fn([u32; 4], &[FromBits; 4]) -> T,
            bits: [FromBits; 4],
            in_texel: &'data TexelBuffer,
            pixel_buf: &'data mut TexelBuffer,
        }

        impl<Expanded> GenericTexelAction<()> for ExpandAction<'_, Expanded> {
            fn run<T>(self, texel: Texel<T>) -> () {
                let texel_slice = self.in_texel.as_texels(texel);
                let pixel_slice = self.pixel_buf.as_mut_texels(self.expand);

                // FIXME(color): block expansion to multiple pixels.
                // FIXME(color): adjust the FromBits for multiple planes.
                for (texbits, expand) in texel_slice.iter().zip(pixel_slice) {
                    *expand = (self.expand_fn)(
                        self.bits.map(|b| b.extract_as_lsb(texel, texbits)),
                        &self.bits,
                    );
                }
            }
        }

        match info.common_pixel {
            // FIXME(color): rescaling of channels, and their bit interpretation.
            // Should we scale so that they occupy the full dynamic range, and scale floats from [0;
            // 1.0) or the respective HDR upper bound, i.e. likely 100.0 to represent 10_000 cd/m².
            CommonPixel::F32x4 => info.in_kind.action(ExpandAction {
                expand: <[f32; 4]>::texel(),
                expand_fn: |num, bits| {
                    [0, 1, 2, 3].map(|idx| {
                        let max_val = bits[idx].mask() as u64;
                        num[idx] as f32 / max_val as f32
                    })
                },
                bits,
                in_texel,
                pixel_buf,
            }),
        }
    }

    fn expand_floats(
        info: &Info,
        bits: [FromBits; 4],
        in_texel: &TexelBuffer,
        pixel_buf: &mut TexelBuffer,
    ) {
        debug_assert!(
            matches!(info.common_pixel, CommonPixel::F32x4),
            "Improper common choices {:?}",
            info.common_pixel
        );
        let destination = pixel_buf.as_mut_texels(<[f32; 4]>::texel());

        // FIXME(color): Assumes that we only read f32 channels..
        let pitch = info.in_kind.size() / 4;

        for (&ch, ch_idx) in bits.iter().zip(0..4) {
            match ch.len {
                0 => continue,
                32 => {}
                // FIXME(color): half-floats?
                _ => continue,
            }

            let position = ch.begin / 32;
            let texels = in_texel.as_texels(<f32>::texel());
            let pitched = texels[position..].chunks(pitch);

            for (pix, texel) in destination.iter_mut().zip(pitched) {
                pix[ch_idx] = texel[0];
            }
        }
    }

    fn expand_yuv422(info: &Info, in_texel: &TexelBuffer, pixel_buf: &mut TexelBuffer) {
        struct ExpandYuv422;

        impl ExpandYuvLike<4, 2> for ExpandYuv422 {
            fn expand<T: Copy>(yuyv: [T; 4], fill: T) -> [[T; 4]; 2] {
                let [u, y1, v, y2] = yuyv;

                [[y1, u, v, fill], [y2, u, v, fill]]
            }
        }

        Self::expand_yuv_like::<ExpandYuv422, 4, 2>(
            info,
            in_texel,
            pixel_buf,
            <[u8; 4]>::texel(),
            <[u16; 4]>::texel(),
            <[f32; 4]>::texel(),
        )
    }

    fn expand_yuy2(info: &Info, in_texel: &TexelBuffer, pixel_buf: &mut TexelBuffer) {
        struct ExpandYuy2;

        impl ExpandYuvLike<4, 2> for ExpandYuy2 {
            fn expand<T: Copy>(yuyv: [T; 4], fill: T) -> [[T; 4]; 2] {
                let [y1, u, y2, v] = yuyv;

                [[y1, u, v, fill], [y2, u, v, fill]]
            }
        }

        Self::expand_yuv_like::<ExpandYuy2, 4, 2>(
            info,
            in_texel,
            pixel_buf,
            <[u8; 4]>::texel(),
            <[u16; 4]>::texel(),
            <[f32; 4]>::texel(),
        )
    }

    fn expand_yuv411(info: &Info, in_texel: &TexelBuffer, pixel_buf: &mut TexelBuffer) {
        struct ExpandYuv411;

        impl ExpandYuvLike<6, 4> for ExpandYuv411 {
            fn expand<T: Copy>(yuyv: [T; 6], fill: T) -> [[T; 4]; 4] {
                let [u, y1, y2, v, y3, y4] = yuyv;

                [
                    [y1, u, v, fill],
                    [y2, u, v, fill],
                    [y3, u, v, fill],
                    [y4, u, v, fill],
                ]
            }
        }

        Self::expand_yuv_like::<ExpandYuv411, 6, 4>(
            info,
            in_texel,
            pixel_buf,
            <[u8; 6]>::texel(),
            <[u16; 6]>::texel(),
            <[f32; 6]>::texel(),
        )
    }

    fn expand_yuv_like<F, const N: usize, const M: usize>(
        info: &Info,
        in_texel: &TexelBuffer,
        pixel_buf: &mut TexelBuffer,
        tex_u8: Texel<[u8; N]>,
        tex_u16: Texel<[u16; N]>,
        tex_f32: Texel<[f32; N]>,
    ) where
        F: ExpandYuvLike<N, M>,
    {
        // FIXME(perf): it makes sense to loop-remove this match into `ops` construction?
        // In particular, instruction cache if each case is treated separately should be decent..
        match info.in_layout.texel.bits {
            SampleBits::UInt8x4 => {
                let texels = in_texel.as_texels(tex_u8).iter();
                match info.common_pixel {
                    CommonPixel::F32x4 => {
                        let pixels = pixel_buf
                            .as_mut_texels(<[f32; 4]>::texel())
                            .chunks_exact_mut(M);
                        debug_assert!(pixels.len() == texels.len());

                        for (texel, pixel_chunk) in texels.zip(pixels) {
                            let pixels: &mut [_; M] = pixel_chunk.try_into().unwrap();
                            let expand = F::expand(*texel, u8::MAX);
                            let remap = |v: u8| (v as f32) / 255.0f32;
                            *pixels = expand.map(|v| v.map(remap));
                        }
                    }
                }
            }
            SampleBits::UInt16x4 => {
                let texels = in_texel.as_texels(tex_u16).iter();
                match info.common_pixel {
                    CommonPixel::F32x4 => {
                        let pixels = pixel_buf
                            .as_mut_texels(<[f32; 4]>::texel())
                            .chunks_exact_mut(M);
                        debug_assert!(pixels.len() == texels.len());

                        for (texel, pixel_chunk) in texels.zip(pixels) {
                            let pixels: &mut [_; M] = pixel_chunk.try_into().unwrap();
                            let expand = F::expand(*texel, u16::MAX);
                            let remap = |v: u16| (v as f32) / 65535.0f32;
                            *pixels = expand.map(|v| v.map(remap));
                        }
                    }
                }
            }
            SampleBits::Float32x4 => {
                let texels = in_texel.as_texels(tex_f32).iter();
                match info.common_pixel {
                    CommonPixel::F32x4 => {
                        let pixels = pixel_buf
                            .as_mut_texels(<[f32; 4]>::texel())
                            .chunks_exact_mut(2);
                        debug_assert!(pixels.len() == texels.len());

                        for (texel, pixel_chunk) in texels.zip(pixels) {
                            let pixels: &mut [_; M] = pixel_chunk.try_into().unwrap();
                            *pixels = F::expand(*texel, 1.0);
                        }
                    }
                }
            }
            other => {
                debug_assert!(false, "Bad YUV spec {:?}", other);
            }
        }
    }

    fn join_from_info(
        info: &Info,
        ops: &ConvertOps,
        pixel_buf: &TexelBuffer,
        out_texels: &mut TexelBuffer,
        // FIXME(perf): see `join_bits` which could use it but requires chunk information.
        _: &mut [PlaneTarget],
    ) {
        // FIXME(perf): some bit/part combinations require no reordering of bits and could skip
        // large parts of this phase, or be done vectorized, effectively amounting to a memcpy when
        // the expanded value had the same representation as the texel.
        let TexelBits { bits, parts, .. } = info.out_layout.texel;

        match parts.pitch {
            SamplePitch::PixelBits => Self::join_bits(
                info,
                ops,
                FromBits::for_pixel(bits, parts),
                pixel_buf,
                out_texels,
            ),
            SamplePitch::Yuv422 => {
                // Debug assert: common_pixel
                debug_assert!(matches!(info.out_layout.texel.block, Block::Sub1x2));
                debug_assert!(matches!(info.out_layout.texel.parts.num_components(), 3));
                Self::join_yuv422(info, pixel_buf, out_texels)
            }
            other => {
                debug_assert!(false, "{:?}", other);
            }
        }
    }

    // FIXME(perf): for single-plane, in particular integer cases, we could write directly into the
    // target buffer by chunks if this is available.
    fn join_bits(
        info: &Info,
        ops: &ConvertOps,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        out_texels: &mut TexelBuffer,
    ) {
        let (encoding, len) = info.out_layout.texel.bits.bit_encoding();

        if let SampleBits::UInt8x4 = info.out_layout.texel.bits {
            // TODO: pre-select SIMD version from info.ops.
            return Self::join_uint8x4(ops, bits, pixel_buf, out_texels);
        } else if let SampleBits::UInt16x4 = info.out_layout.texel.bits {
            // TODO: pre-select SIMD version from info.ops.
            return Self::join_uint16x4(ops, bits, pixel_buf, out_texels);
        } else if let SampleBits::UInt8x3 = info.out_layout.texel.bits {
            // TODO: pre-select version from info.ops.
            return Self::join_uint8x3(ops, bits, pixel_buf, out_texels);
        } else if let SampleBits::UInt16x3 = info.out_layout.texel.bits {
            // TODO: pre-select SIMD version from info.ops.
            return Self::join_uint16x3(ops, bits, pixel_buf, out_texels);
        } else if encoding[..len as usize] == [BitEncoding::UInt; 6][..len as usize] {
            return Self::join_ints(info, bits, pixel_buf, out_texels);
        } else if encoding[..len as usize] == [BitEncoding::Float; 6][..len as usize] {
            return Self::join_floats(info, bits, pixel_buf, out_texels);
        } else {
            // FIXME(color): error treatment..
            debug_assert!(false, "{:?}", &encoding[..len as usize]);
        }
    }

    // FIXME(color): int component bias
    fn join_ints(
        info: &Info,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        out_texels: &mut TexelBuffer,
    ) {
        struct JoinAction<'data, T, F: FnMut(&T, &FromBits, u8) -> u32> {
            join: Texel<T>,
            join_fn: F,
            bits: [FromBits; 4],
            out_texels: &'data mut TexelBuffer,
            pixel_buf: &'data TexelBuffer,
        }

        impl<Expanded, F> GenericTexelAction<()> for JoinAction<'_, Expanded, F>
        where
            F: FnMut(&Expanded, &FromBits, u8) -> u32,
        {
            fn run<T>(mut self, texel: Texel<T>) -> () {
                let texel_slice = self.out_texels.as_mut_texels(texel);
                let pixel_slice = self.pixel_buf.as_texels(self.join);

                for idx in [0u8, 1, 2, 3] {
                    let bits = &self.bits[idx as usize];
                    // FIXME(color): block from multiple pixels—how to?
                    // FIXME(color): adjust the FromBits for multiple planes.
                    for (texbits, joined) in texel_slice.iter_mut().zip(pixel_slice) {
                        let value = (self.join_fn)(joined, bits, idx);
                        bits.insert_as_lsb(texel, texbits, value);
                    }
                }
            }
        }

        match info.common_pixel {
            // FIXME(color): rescaling of channels, and their bit interpretation.
            // Should we scale so that they occupy the full dynamic range, and scale floats from [0;
            // 1.0) or the respective HDR upper bound, i.e. likely 100.0 to represent 10_000 cd/m².
            CommonPixel::F32x4 => info.out_kind.action(JoinAction {
                join: <[f32; 4]>::texel(),
                // FIXME: do the transform u32::from_ne_bytes(x.as_ne_bytes()) when appropriate.
                join_fn: |num, bits, idx| {
                    let max_val = bits.mask();
                    let raw = (num[(idx & 0x3) as usize] * max_val as f32).round() as u32;
                    raw.min(max_val)
                },
                bits,
                out_texels,
                pixel_buf,
            }),
        }
    }

    /// Specialized join when channels are a uniform reordering of color channels, as u8.
    fn join_uint8x4(
        ops: &ConvertOps,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        out_texels: &mut TexelBuffer,
    ) {
        let src = pixel_buf.as_texels(f32::texel());
        let dst = out_texels.as_mut_texels(u8::texel());

        // Do one quick SIMD cast to u8. Much faster than the general round and clamp.
        // Note: fma is for some reason a call to a libc function…
        for (tex, &pix) in dst.iter_mut().zip(src) {
            *tex = ((pix * (u8::MAX as f32)) + 0.5) as u8;
        }

        // prepare re-ordering step. Note how we select 0x80 as invalid, which works perfectly with
        // an SSE shuffle instruction which encodes this as a negative offset. Trust llvm to do the
        // transform.
        let mut shuffle = [0x80u8; 4];
        for (idx, bits) in (0u8..4).zip(&bits) {
            if bits.len > 0 {
                shuffle[(bits.begin / 8) as usize] = idx;
            }
        }

        (ops.shuffle.shuffle_u8x4)(out_texels.as_mut_texels(<[u8; 4]>::texel()), shuffle);
    }

    fn join_uint16x4(
        ops: &ConvertOps,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        out_texels: &mut TexelBuffer,
    ) {
        let src = pixel_buf.as_texels(f32::texel());
        let dst = out_texels.as_mut_texels(u16::texel());

        // Do one quick SIMD cast to u8. Faster than the general round and clamp.
        // Note: fma is for some reason a call to a libc function…
        for (tex, &pix) in dst.iter_mut().zip(src) {
            *tex = ((pix * (u16::MAX as f32)) + 0.5) as u16;
        }

        // prepare re-ordering step. Note how we select 0x80 as invalid, which works perfectly with
        // an SSE shuffle instruction which encodes this as a negative offset. Trust llvm to do the
        // transform.
        let mut shuffle = [0x80u8; 4];
        for (idx, bits) in (0u8..4).zip(&bits) {
            if bits.len > 0 {
                shuffle[(bits.begin / 16) as usize] = idx;
            }
        }

        (ops.shuffle.shuffle_u16x4)(out_texels.as_mut_texels(<[u16; 4]>::texel()), shuffle);
    }

    fn join_uint8x3(
        _: &ConvertOps,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        out_texels: &mut TexelBuffer,
    ) {
        let src = pixel_buf.as_texels(<[f32; 4]>::texel());
        let dst = out_texels.as_mut_texels(<[u8; 3]>::texel());

        // prepare re-ordering step. Note how we select 0x80 as invalid, which works perfectly with
        // an SSE shuffle instruction which encodes this as a negative offset. Trust llvm to do the
        // transform.
        let mut shuffle = [0x80u8; 3];
        for (idx, bits) in (0u8..4).zip(&bits) {
            if bits.len > 0 {
                shuffle[(bits.begin / 8) as usize] = idx;
            }
        }

        for (tex, pix) in dst.iter_mut().zip(src) {
            *tex = shuffle.map(|i| {
                let val = *pix.get(i as usize).unwrap_or(&0.0);
                (val * u8::MAX as f32 + 0.5) as u8
            });
        }
    }

    fn join_uint16x3(
        _: &ConvertOps,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        out_texels: &mut TexelBuffer,
    ) {
        let src = pixel_buf.as_texels(<[f32; 4]>::texel());
        let dst = out_texels.as_mut_texels(<[u16; 3]>::texel());

        // prepare re-ordering step. Note how we select 0x80 as invalid, which works perfectly with
        // an SSE shuffle instruction which encodes this as a negative offset. Trust llvm to do the
        // transform.
        let mut shuffle = [0x80u8; 3];
        for (idx, bits) in (0u8..4).zip(&bits) {
            if bits.len > 0 {
                shuffle[(bits.begin / 16) as usize] = idx;
            }
        }

        for (tex, pix) in dst.iter_mut().zip(src) {
            *tex = shuffle.map(|i| {
                let val = *pix.get(i as usize).unwrap_or(&0.0);
                (val * u16::MAX as f32 + 0.5) as u16
            });
        }
    }

    fn join_floats(
        info: &Info,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        out_texels: &mut TexelBuffer,
    ) {
        debug_assert!(
            matches!(info.common_pixel, CommonPixel::F32x4),
            "Improper common choices {:?}",
            info.common_pixel
        );
        let source = pixel_buf.as_texels(<[f32; 4]>::texel());
        // Assume that we only write floating channels..
        let pitch = info.out_kind.size() / 4;

        for (&ch, ch_idx) in bits.iter().zip(0..4) {
            match ch.len {
                0 => continue,
                32 => {}
                // FIXME(color): half-floats?
                _ => continue,
            }

            let position = ch.begin / 32;
            let texels = out_texels.as_mut_texels(<f32>::texel());
            let pitched = texels[position..].chunks_mut(pitch);

            for (pix, texel) in source.iter().zip(pitched) {
                texel[0] = pix[ch_idx];
            }
        }
    }

    fn join_yuv422(_: &Info, _: &TexelBuffer, _: &mut TexelBuffer) {
        // FIXME(color): actually implement this..
        debug_assert!(false);
    }

    fn action<R>(self, action: impl GenericTexelAction<R>) -> R {
        match self {
            CommonPixel::F32x4 => action.run(<[f32; 4]>::texel()),
        }
    }
}

impl CommonColor {
    fn cie_xyz_from_info(info: &Info, pixel: &TexelBuffer, xyz: &mut TexelBuffer) {
        // If we do color conversion, we always choose [f32; 4] representation.
        // Or, at least we should. Otherwise, do nothing..
        if !matches!(info.common_pixel, CommonPixel::F32x4) {
            // FIXME(color): report this error somehow?
            return;
        }

        let texel = <[f32; 4]>::texel();
        let pixel = pixel.as_texels(texel);
        let xyz = xyz.as_mut_texels(texel);
        assert_eq!(
            pixel.len(),
            xyz.len(),
            "Setup create invalid conversion buffer"
        );

        match info.in_layout.color.as_ref() {
            None => xyz.copy_from_slice(pixel),
            Some(color) => color.to_xyz_slice(pixel, xyz),
        }
    }

    fn cie_xyz_into_info(info: &Info, xyz: &TexelBuffer, pixel: &mut TexelBuffer) {
        // If we do color conversion, we always choose [f32; 4] representation.
        // Or, at least we should. Otherwise, do nothing..
        if !matches!(info.common_pixel, CommonPixel::F32x4) {
            // FIXME(color): report this error somehow?
            return;
        }

        let texel = <[f32; 4]>::texel();
        let xyz = xyz.as_texels(texel);
        let pixel = pixel.as_mut_texels(texel);
        assert_eq!(
            pixel.len(),
            xyz.len(),
            "Setup create invalid conversion buffer"
        );

        match info.out_layout.color.as_ref() {
            None => pixel.copy_from_slice(xyz),
            Some(color) => color.from_xyz_slice(xyz, pixel),
        }
    }
}

macro_rules! from_bits {
    ($bits:ident = { $($variant:pat => $($value:expr)+);* }) => {
        match $bits {
            $($variant => from_bits!(@ $($value);*)),*,
        }
    };
    (@ $v0:expr) => {
        [Some(FromBits::from_range($v0)), None, None, None, None, None]
    };
    (@ $v0:expr; $v1:expr) => {
        [Some(FromBits::from_range($v0)), Some(FromBits::from_range($v1)), None, None, None, None]
    };
    (@ $v0:expr; $v1:expr; $v2:expr) => {
        [
            Some(FromBits::from_range($v0)),
            Some(FromBits::from_range($v1)),
            Some(FromBits::from_range($v2)),
            None,
            None,
            None
        ]
    };
    (@ $v0:expr; $v1:expr; $v2:expr; $v3:expr) => {
        [
            Some(FromBits::from_range($v0)),
            Some(FromBits::from_range($v1)),
            Some(FromBits::from_range($v2)),
            Some(FromBits::from_range($v3)),
            None,
            None
        ]
    };
    (@ $v0:expr; $v1:expr; $v2:expr; $v3:expr; $v4:expr; $v5:expr) => {
        [
            Some(FromBits::from_range($v0)),
            Some(FromBits::from_range($v1)),
            Some(FromBits::from_range($v2)),
            Some(FromBits::from_range($v3)),
            Some(FromBits::from_range($v4)),
            Some(FromBits::from_range($v5)),
        ]
    };
}

impl FromBits {
    const NO_BITS: Self = FromBits { begin: 0, len: 0 };

    const fn from_range(range: core::ops::Range<usize>) -> Self {
        FromBits {
            begin: range.start,
            len: range.end - range.start,
        }
    }

    pub(crate) fn for_pixel(bits: SampleBits, parts: SampleParts) -> [Self; 4] {
        assert_eq!(parts.pitch, SamplePitch::PixelBits);
        let mut vals = [Self::NO_BITS; 4];

        let bits = Self::bits(bits);
        let channels = parts.channels();

        for (bits, (channel, pos)) in bits.zip(channels) {
            if let Some(_) = channel {
                vals[pos as usize] = bits;
            }
        }

        vals
    }

    pub(crate) const fn mask(self) -> u32 {
        ((-1i64 as u64) ^ u32::MAX as u64).rotate_left(self.len as u32) as u32
    }

    fn bits(bits: SampleBits) -> impl Iterator<Item = Self> {
        use SampleBits::*;
        let filled: [Option<Self>; 6] = from_bits!(bits = {
            Int8 | UInt8 => 0..8;
            UInt332 => 0..3 3..6 6..8;
            UInt233 => 0..2 2..5 5..8;
            Int16 | UInt16 => 0..16;
            UInt4x4 => 0..4 4..8 8..12 12..16;
            UInt_444 => 4..8 8..12 12..16;
            UInt444_ => 0..4 4..8 8..12;
            UInt565 => 0..5 5..11 11..16;
            Int8x2 | UInt8x2 => 0..8 8..16;
            Int8x3 | UInt8x3 => 0..8 8..16 16..24;
            Int8x4 | UInt8x4 => 0..8 8..16 16..24 24..32;
            UInt8x6 => 0..8 8..16 16..24 24..32 32..40 40..48;
            Int16x2 | UInt16x2 => 0..16 16..32;
            Int16x3 | UInt16x3 => 0..16 16..32 32..48;
            Int16x4 | UInt16x4 => 0..16 16..32 32..48 48..64;
            UInt16x6 => 0..16 16..32 32..48 48..64 64..80 80..96;
            UInt1010102 => 0..10 10..20 20..30 30..32;
            UInt2101010 => 0..2 2..12 12..22 22..32;
            UInt101010_ => 0..10 10..20 20..30;
            UInt_101010 => 2..12 12..22 22..32;
            Float16x4 => 0..16 16..32 32..48 48..64;
            Float32 => 0..32;
            Float32x2 => 0..32 32..64;
            Float32x3 => 0..32 32..64 64..96;
            Float32x4 => 0..32 32..64 64..96 96..128;
            Float32x6 => 0..32 32..64 64..96 96..128 128..160 160..192
        });

        filled.into_iter().filter_map(|x| x)
    }

    /// Extract bit as a big-endian interpretation.
    ///
    /// The highest bit of each byte being the first. Returns a value as `u32` with the same
    /// interpretation where the lowest bits are filled.
    ///
    /// FIXME: there's **a lot** of constant pre-processing. For example, if always access through
    /// either 32-bit boundary or 64-bit boundary then the startu64 is also one of two constants.
    #[inline]
    fn extract_as_lsb<T>(&self, texel: Texel<T>, val: &T) -> u32 {
        // FIXME(perf): vectorized form for all texels where possible.
        // Grab up to 8 bytes surrounding the bits, convert using u64 intermediate, then shift
        // upwards (by at most 7 bit) and mask off any remaining bits.
        let ne_bytes = texel.to_bytes(core::slice::from_ref(val));
        let startu64 = self.begin / 8;
        let from_bytes = &ne_bytes[startu64.min(ne_bytes.len())..];

        let shift = self.begin - startu64 * 8;
        let bitlen = self.len + shift;
        let copylen = if bitlen % 8 == 0 {
            bitlen / 8
        } else {
            bitlen / 8 + 1
        };

        let mut be_bytes = [0; 8];
        let initlen = copylen.min(8).min(from_bytes.len());
        be_bytes[..initlen].copy_from_slice(&from_bytes[..initlen]);

        let val = u64::from_le_bytes(be_bytes) >> shift.min(63);
        // Start with a value where the 32-low bits are clear, high bits are set.
        val as u32 & self.mask()
    }

    fn insert_as_lsb<T>(&self, texel: Texel<T>, val: &mut T, bits: u32) {
        // FIXME(perf): vectorized form for all texels where possible.
        let ne_bytes = texel.to_mut_bytes(core::slice::from_mut(val));
        let startu64 = self.begin / 8;
        let bytestart = startu64.min(ne_bytes.len());
        let texel_bytes = &mut ne_bytes[bytestart..];

        let shift = self.begin - startu64 * 8;
        let bitlen = self.len + shift;
        let copylen = if bitlen % 8 == 0 {
            bitlen / 8
        } else {
            bitlen / 8 + 1
        };

        let mut be_bytes = [0; 8];
        let initlen = copylen.min(8).min(texel_bytes.len());
        be_bytes[..initlen].copy_from_slice(&texel_bytes[..initlen]);

        let mask = ((-1i64 as u64) ^ u32::MAX as u64).rotate_left((self.len as u32).min(32))
            & (u32::MAX as u64);

        let newval =
            (u64::from_le_bytes(be_bytes) & !(mask << shift)) | (u64::from(bits) & mask) << shift;

        be_bytes = newval.to_le_bytes();
        texel_bytes[..initlen].copy_from_slice(&be_bytes[..initlen]);
    }
}

impl TexelKind {
    pub(crate) fn action<R>(self, action: impl GenericTexelAction<R>) -> R {
        match self {
            TexelKind::U8 => action.run(u8::texel()),
            TexelKind::U8x2 => action.run(<[u8; 2]>::texel()),
            TexelKind::U8x3 => action.run(<[u8; 3]>::texel()),
            TexelKind::U8x4 => action.run(<[u8; 4]>::texel()),
            TexelKind::U8x6 => action.run(<[u8; 6]>::texel()),
            TexelKind::U16 => action.run(<[u16; 1]>::texel()),
            TexelKind::U16x2 => action.run(<[u16; 2]>::texel()),
            TexelKind::U16x3 => action.run(<[u16; 3]>::texel()),
            TexelKind::U16x4 => action.run(<[u16; 4]>::texel()),
            TexelKind::U16x6 => action.run(<[u16; 6]>::texel()),
            TexelKind::F32 => action.run(<[f32; 1]>::texel()),
            TexelKind::F32x2 => action.run(<[f32; 2]>::texel()),
            TexelKind::F32x3 => action.run(<[f32; 3]>::texel()),
            TexelKind::F32x4 => action.run(<[f32; 4]>::texel()),
            TexelKind::F32x6 => action.run(<[f32; 6]>::texel()),
        }
    }

    fn size(self) -> usize {
        struct ToSize;

        impl GenericTexelAction<usize> for ToSize {
            fn run<T>(self, texel: image_texel::Texel<T>) -> usize {
                image_texel::layout::TexelLayout::from(texel).size()
            }
        }

        TexelKind::from(self).action(ToSize)
    }
}

impl From<TexelBits> for TexelKind {
    fn from(texel: TexelBits) -> Self {
        Self::from(texel.bits)
    }
}

impl From<SampleBits> for TexelKind {
    fn from(bits: SampleBits) -> Self {
        use SampleBits::*;
        // We only need to match size and align here.
        match bits {
            Int8 | UInt8 | UInt332 | UInt233 => TexelKind::U8,
            Int16 | UInt16 | UInt4x4 | UInt_444 | UInt444_ | UInt565 => TexelKind::U16,
            Int8x2 | UInt8x2 => TexelKind::U8x2,
            Int8x3 | UInt8x3 => TexelKind::U8x3,
            Int8x4 | UInt8x4 => TexelKind::U8x4,
            UInt8x6 => TexelKind::U8x6,
            Int16x2 | UInt16x2 => TexelKind::U16x2,
            Int16x3 | UInt16x3 => TexelKind::U16x3,
            Int16x4 | UInt16x4 => TexelKind::U16x4,
            UInt16x6 => TexelKind::U16x6,
            UInt1010102 | UInt2101010 | UInt101010_ | UInt_101010 => TexelKind::U16x2,
            Float16x4 => TexelKind::U16x4,
            Float32 => TexelKind::F32,
            Float32x2 => TexelKind::F32x2,
            Float32x3 => TexelKind::F32x3,
            Float32x4 => TexelKind::F32x4,
            Float32x6 => TexelKind::F32x6,
        }
    }
}

#[test]
fn from_bits() {
    let bits = FromBits::for_pixel(SampleBits::UInt332, SampleParts::Rgb);
    let (texel, value) = (u8::texel(), &0b01010110);
    assert_eq!(bits[0].extract_as_lsb(texel, value), 0b110);
    assert_eq!(bits[1].extract_as_lsb(texel, value), 0b010);
    assert_eq!(bits[2].extract_as_lsb(texel, value), 0b01);
    assert_eq!(bits[3].extract_as_lsb(texel, value), 0b0);
}

#[test]
fn to_bits() {
    let bits = FromBits::for_pixel(SampleBits::UInt332, SampleParts::Rgb);
    let (texel, ref mut value) = (u8::texel(), 0);
    bits[0].insert_as_lsb(texel, value, 0b110);
    bits[1].insert_as_lsb(texel, value, 0b010);
    bits[2].insert_as_lsb(texel, value, 0b01);
    bits[3].insert_as_lsb(texel, value, 0b0);
    assert_eq!(*value, 0b01010110);
}
