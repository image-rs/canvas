//! Conversion between texels, mostly color.
//!
//! Takes quite a lot of inspiration from how GPUs work. We have a primitive sampler unit, a
//! fragment unit, and pipeline multiple texels in parallel.
use core::ops::Range;
use image_texel::image::{ImageMut, ImageRef};
use image_texel::{image::Coord, AsTexel, Texel, TexelBuffer};

use crate::layout_::{
    BitEncoding, Block, CanvasLayout, SampleBits, SampleParts, SamplePitch, Texel as TexelBits,
};
use crate::Canvas;

/// A buffer for conversion.
pub struct Converter {
    /// How many texels to do at once.
    chunk: usize,

    super_blocks: Vec<TexelCoord>,
    /// Buffer where we store input texels after reading them.
    in_texels: TexelBuffer,
    /// Texel coordinates of stored texels.
    in_coords: Vec<Coord>,
    /// Index in the input planes.
    in_index: Vec<usize>,
    /// Buffer where we store input texels before writing.
    out_texels: TexelBuffer,
    /// Texel coordinates of stored texels.
    out_coords: Vec<Coord>,
    /// Index in the input planes.
    out_index: Vec<usize>,

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
    in_index: fn(&Info, &[Coord], &mut [usize]),
    /// Convert out texel coordinates to an index in the color plane.
    out_index: fn(&Info, &[Coord], &mut [usize]),

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
    pub fn new() -> Self {
        Converter {
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

        let ops = ConvertOps {
            in_index: Self::index_from_in_info,
            out_index: Self::index_from_out_info,
            expand: CommonPixel::expand_from_info,
            // FIXME(color):
            recolor: Self::recolor_ops(frame_in.layout(), frame_out.layout()),
            join: CommonPixel::join_from_info,
        };

        // Check that the layout is accurate..
        self.with_filled_texels(
            |that| {
                (ops.expand)(&info, &that.in_texels, &mut that.pixel_in_buffer);

                let pixel_out = if let Some(ref recolor) = ops.recolor {
                    (recolor.from)(&info, &that.pixel_in_buffer, &mut that.neutral_color_buffer);
                    (recolor.into)(
                        &info,
                        &that.neutral_color_buffer,
                        &mut that.pixel_out_buffer,
                    );
                    &that.pixel_out_buffer
                } else {
                    &that.pixel_in_buffer
                };

                // FIXME: necessary to do a reorder of pixels here? Or let join do this?
                (ops.join)(&info, pixel_out, &mut that.out_texels);
            },
            &info,
            &ops,
            frame_in,
            frame_out,
        )
    }

    /// Choose iteration order of texels, fill with texels and then put them back.
    fn with_filled_texels(
        &mut self,
        mut texel_conversion: impl FnMut(&mut Self),
        info: &Info,
        ops: &ConvertOps,
        frame_in: &Canvas,
        frame_out: &mut Canvas,
    ) {
        // We use a notion of 'supertexels', the common multiple of input and output texel blocks.
        // That is, if the input is a 2-by-2 pixel block and the output is single pixels then we
        // have 4 times as many outputs as inputs, respectively coordinates.
        //
        // Anyways, first we fill the coordinate buffers, then calculate the planar indices.
        let (sb_x, sb_y) = self.super_texel(info);
        let mut blocks = Self::blocks(sb_x.blocks.clone(), sb_y.blocks.clone());

        loop {
            self.super_blocks.clear();
            self.super_blocks
                .extend(blocks.by_ref().take(self.chunk).map(TexelCoord));

            if self.super_blocks.is_empty() {
                break;
            }

            self.generate_coords(info, ops, &sb_x, &sb_y);
            self.reserve_buffers(info, ops);
            // FIXME(planar): should be repeated for all planes?
            self.read_texels(info, frame_in.as_ref());
            texel_conversion(self);
            // FIXME(planar): should be repeated for all planes?
            self.write_texels(info, frame_out.as_mut());
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

    fn generate_coords(
        &mut self,
        info: &Info,
        ops: &ConvertOps,
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
            for &TexelCoord(Coord(bx, by)) in self.super_blocks.iter() {
                self.in_coords.push(Coord(bx, by));
                self.out_coords.push(Coord(bx, by));
            }
        } else {
            let in_blocks = Self::blocks(0..sb_x.in_super, 0..sb_y.in_super);
            let out_blocks = Self::blocks(0..sb_x.out_super, 0..sb_y.out_super);

            for &TexelCoord(Coord(bx, by)) in self.super_blocks.iter() {
                for Coord(ix, iy) in in_blocks.clone() {
                    self.in_coords.push(Coord(bx + ix, by + iy));
                }

                for Coord(ox, oy) in out_blocks.clone() {
                    self.out_coords.push(Coord(bx + ox, by + oy));
                }
            }
        }

        self.in_index.resize_with(self.in_coords.len(), || 0);
        self.out_index.resize_with(self.out_coords.len(), || 0);

        (ops.in_index)(&info, &self.in_coords, &mut self.in_index);
        (ops.out_index)(&info, &self.out_coords, &mut self.out_index);
    }

    fn reserve_buffers(&mut self, info: &Info, ops: &ConvertOps) {
        struct ResizeAction<'data>(&'data mut TexelBuffer, usize);

        impl GenericTexelAction for ResizeAction<'_> {
            fn run<T>(self, texel: Texel<T>) {
                self.0.resize_for_texel(self.1, texel)
            }
        }

        let in_texels = self.in_coords.len();
        let in_block = info.in_layout.texel.block;
        let in_pixels = (in_block.width() * in_block.height()) as usize * in_texels;
        info.in_kind
            .action(ResizeAction(&mut self.in_texels, in_texels));

        let out_texels = self.out_coords.len();
        let out_block = info.out_layout.texel.block;
        let out_pixels = (out_block.width() * out_block.height()) as usize * out_texels;
        info.out_kind
            .action(ResizeAction(&mut self.out_texels, out_texels));

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

    fn read_texels(&mut self, info: &Info, from: PlaneSource) {
        fn fetch_from_texel_array<T>(
            from: PlaneSource,
            idx: &[usize],
            into: &mut TexelBuffer,
            texel: Texel<T>,
        ) {
            into.resize_for_texel(idx.len(), texel);
            // FIXME(planar):
            // FIXME(color): multi-planar texel fetch.
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
                fetch_from_texel_array(self.from, self.idx, self.into, texel)
            }
        }

        info.in_kind.action(ReadUnit {
            from,
            idx: &self.in_index,
            into: &mut self.in_texels,
        })
    }

    fn write_texels(&mut self, info: &Info, into: PlaneTarget) {
        fn write_from_texel_array<T>(
            mut into: PlaneTarget,
            idx: &[usize],
            from: &TexelBuffer,
            texel: Texel<T>,
        ) {
            // FIXME(planar):
            // FIXME(color): multi-planar texel write.
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
                write_from_texel_array(self.into, self.idx, self.from, texel)
            }
        }

        info.out_kind.action(WriteUnit {
            into,
            idx: &self.out_index,
            from: &self.out_texels,
        });
    }

    fn blocks(x: Range<u32>, y: Range<u32>) -> impl Iterator<Item = Coord> + Clone {
        x.clone()
            .into_iter()
            .map(move |x| core::iter::repeat(x).zip(y.clone()))
            .flatten()
            .map(|(x, y)| Coord(x, y))
    }

    fn index_from_in_info(info: &Info, texel: &[Coord], idx: &mut [usize]) {
        Self::index_from_layer(&info.in_layout, texel, idx)
    }

    fn index_from_out_info(info: &Info, texel: &[Coord], idx: &mut [usize]) {
        Self::index_from_layer(&info.out_layout, texel, idx)
    }

    fn index_from_layer(info: &CanvasLayout, texel: &[Coord], idx: &mut [usize]) {
        // FIXME(perf): review performance. Could probably be vectorized by hand.
        info.fill_texel_indices_impl(idx, texel)
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
    fn expand_from_info(info: &Info, texel_buf: &TexelBuffer, pixel_buf: &mut TexelBuffer) {
        // FIXME(perf): some bit/part combinations require no reordering of bits and could skip
        // large parts of this phase, or be done vectorized, effectively amounting to a memcpy when
        // the expanded value has the same representation as the texel.
        let TexelBits { bits, parts, .. } = info.in_layout.texel;

        match parts.pitch {
            SamplePitch::PixelBits => {
                Self::expand_bits(info, FromBits::for_pixel(bits, parts), texel_buf, pixel_buf)
            }
            SamplePitch::Yuv422 => {
                debug_assert!(matches!(info.in_layout.texel.block, Block::Sub1x2));
                debug_assert!(matches!(info.in_layout.texel.parts.num_components(), 3));
                Self::expand_yuv422(info, texel_buf, pixel_buf);
            }
            SamplePitch::Yuy2 => {
                debug_assert!(matches!(info.in_layout.texel.block, Block::Sub1x2));
                debug_assert!(matches!(info.in_layout.texel.parts.num_components(), 3));
                Self::expand_yuy2(info, texel_buf, pixel_buf);
            }
            SamplePitch::Yuv411 => {
                debug_assert!(matches!(info.in_layout.texel.block, Block::Sub1x4));
                debug_assert!(matches!(info.in_layout.texel.parts.num_components(), 3));
                Self::expand_yuv411(info, texel_buf, pixel_buf);
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
        texel_buf: &TexelBuffer,
        pixel_buf: &mut TexelBuffer,
    ) {
        let (encoding, len) = info.in_layout.texel.bits.bit_encoding();

        if encoding[..len as usize] == [BitEncoding::UInt; 6][..len as usize] {
            return Self::expand_ints(info, bits, texel_buf, pixel_buf);
        } else if encoding[..len as usize] == [BitEncoding::Float; 6][..len as usize] {
            return Self::expand_floats(info, bits, texel_buf, pixel_buf);
        } else {
            // FIXME(color): error treatment..
            debug_assert!(false, "{:?}", &encoding[..len as usize]);
        }
    }

    fn expand_ints(
        info: &Info,
        bits: [FromBits; 4],
        texel_buf: &TexelBuffer,
        pixel_buf: &mut TexelBuffer,
    ) {
        struct ExpandAction<'data, T> {
            expand: Texel<T>,
            expand_fn: fn([u32; 4], &[FromBits; 4]) -> T,
            bits: [FromBits; 4],
            texel_buf: &'data TexelBuffer,
            pixel_buf: &'data mut TexelBuffer,
        }

        impl<Expanded> GenericTexelAction<()> for ExpandAction<'_, Expanded> {
            fn run<T>(self, texel: Texel<T>) -> () {
                let texel_slice = self.texel_buf.as_texels(texel);
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
                texel_buf,
                pixel_buf,
            }),
        }
    }

    fn expand_floats(
        info: &Info,
        bits: [FromBits; 4],
        texel_buf: &TexelBuffer,
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
            let texels = texel_buf.as_texels(<f32>::texel());
            let pitched = texels[position..].chunks(pitch);

            for (pix, texel) in destination.iter_mut().zip(pitched) {
                pix[ch_idx] = texel[0];
            }
        }
    }

    fn expand_yuv422(info: &Info, texel_buf: &TexelBuffer, pixel_buf: &mut TexelBuffer) {
        struct ExpandYuv422;

        impl ExpandYuvLike<4, 2> for ExpandYuv422 {
            fn expand<T: Copy>(yuyv: [T; 4], fill: T) -> [[T; 4]; 2] {
                let [u, y1, v, y2] = yuyv;

                [[y1, u, v, fill], [y2, u, v, fill]]
            }
        }

        Self::expand_yuv_like::<ExpandYuv422, 4, 2>(
            info,
            texel_buf,
            pixel_buf,
            <[u8; 4]>::texel(),
            <[u16; 4]>::texel(),
            <[f32; 4]>::texel(),
        )
    }

    fn expand_yuy2(info: &Info, texel_buf: &TexelBuffer, pixel_buf: &mut TexelBuffer) {
        struct ExpandYuy2;

        impl ExpandYuvLike<4, 2> for ExpandYuy2 {
            fn expand<T: Copy>(yuyv: [T; 4], fill: T) -> [[T; 4]; 2] {
                let [y1, u, y2, v] = yuyv;

                [[y1, u, v, fill], [y2, u, v, fill]]
            }
        }

        Self::expand_yuv_like::<ExpandYuy2, 4, 2>(
            info,
            texel_buf,
            pixel_buf,
            <[u8; 4]>::texel(),
            <[u16; 4]>::texel(),
            <[f32; 4]>::texel(),
        )
    }

    fn expand_yuv411(info: &Info, texel_buf: &TexelBuffer, pixel_buf: &mut TexelBuffer) {
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
            texel_buf,
            pixel_buf,
            <[u8; 6]>::texel(),
            <[u16; 6]>::texel(),
            <[f32; 6]>::texel(),
        )
    }

    fn expand_yuv_like<F, const N: usize, const M: usize>(
        info: &Info,
        texel_buf: &TexelBuffer,
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
                let texels = texel_buf.as_texels(tex_u8).iter();
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
                let texels = texel_buf.as_texels(tex_u16).iter();
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
                let texels = texel_buf.as_texels(tex_f32).iter();
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

    fn join_from_info(info: &Info, pixel_buf: &TexelBuffer, texel_buf: &mut TexelBuffer) {
        // FIXME(perf): some bit/part combinations require no reordering of bits and could skip
        // large parts of this phase, or be done vectorized, effectively amounting to a memcpy when
        // the expanded value had the same representation as the texel.
        let TexelBits { bits, parts, .. } = info.out_layout.texel;

        match parts.pitch {
            SamplePitch::PixelBits => {
                Self::join_bits(info, FromBits::for_pixel(bits, parts), pixel_buf, texel_buf)
            }
            SamplePitch::Yuv422 => {
                // Debug assert: common_pixel
                debug_assert!(matches!(info.out_layout.texel.block, Block::Sub1x2));
                debug_assert!(matches!(info.out_layout.texel.parts.num_components(), 3));
                Self::join_yuv422(info, pixel_buf, texel_buf)
            }
            other => {
                debug_assert!(false, "{:?}", other);
            }
        }
    }

    fn join_bits(
        info: &Info,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        texel_buf: &mut TexelBuffer,
    ) {
        let (encoding, len) = info.out_layout.texel.bits.bit_encoding();

        if let SampleBits::UInt8x4 = info.out_layout.texel.bits {
            // TODO: pre-select SIMD version from info.ops.
            return Self::join_uint8x4(bits, pixel_buf, texel_buf);
        } else if let SampleBits::UInt16x4 = info.out_layout.texel.bits {
            // TODO: pre-select SIMD version from info.ops.
            return Self::join_uint16x4(bits, pixel_buf, texel_buf);
        } else if let SampleBits::UInt8x3 = info.out_layout.texel.bits {
            // TODO: pre-select version from info.ops.
            return Self::join_uint8x3(bits, pixel_buf, texel_buf);
        } else if let SampleBits::UInt16x3 = info.out_layout.texel.bits {
            // TODO: pre-select SIMD version from info.ops.
            return Self::join_uint16x3(bits, pixel_buf, texel_buf);
        } else if encoding[..len as usize] == [BitEncoding::UInt; 6][..len as usize] {
            return Self::join_ints(info, bits, pixel_buf, texel_buf);
        } else if encoding[..len as usize] == [BitEncoding::Float; 6][..len as usize] {
            return Self::join_floats(info, bits, pixel_buf, texel_buf);
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
        texel_buf: &mut TexelBuffer,
    ) {
        struct JoinAction<'data, T, F: FnMut(&T, &FromBits, u8) -> u32> {
            join: Texel<T>,
            join_fn: F,
            bits: [FromBits; 4],
            texel_buf: &'data mut TexelBuffer,
            pixel_buf: &'data TexelBuffer,
        }

        impl<Expanded, F> GenericTexelAction<()> for JoinAction<'_, Expanded, F>
        where
            F: FnMut(&Expanded, &FromBits, u8) -> u32,
        {
            fn run<T>(mut self, texel: Texel<T>) -> () {
                let texel_slice = self.texel_buf.as_mut_texels(texel);
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
                texel_buf,
                pixel_buf,
            }),
        }
    }

    /// Specialized join when channels are a uniform reordering of color channels, as u8.
    fn join_uint8x4(bits: [FromBits; 4], pixel_buf: &TexelBuffer, texel_buf: &mut TexelBuffer) {
        let src = pixel_buf.as_texels(f32::texel());
        let dst = texel_buf.as_mut_texels(u8::texel());

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

        // FIXME(perf): Really, use SIMD here.
        Self::shuffle_u8x4(texel_buf.as_mut_texels(<[u8; 4]>::texel()), shuffle);
    }

    fn join_uint16x4(bits: [FromBits; 4], pixel_buf: &TexelBuffer, texel_buf: &mut TexelBuffer) {
        let src = pixel_buf.as_texels(f32::texel());
        let dst = texel_buf.as_mut_texels(u16::texel());

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

        // FIXME(perf): Really, use SIMD here.
        Self::shuffle_u16x4(texel_buf.as_mut_texels(<[u16; 4]>::texel()), shuffle);
    }

    fn join_uint8x3(bits: [FromBits; 4], pixel_buf: &TexelBuffer, texel_buf: &mut TexelBuffer) {
        let src = pixel_buf.as_texels(<[f32; 4]>::texel());
        let dst = texel_buf.as_mut_texels(<[u8; 3]>::texel());

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

    fn join_uint16x3(bits: [FromBits; 4], pixel_buf: &TexelBuffer, texel_buf: &mut TexelBuffer) {
        let src = pixel_buf.as_texels(<[f32; 4]>::texel());
        let dst = texel_buf.as_mut_texels(<[u16; 3]>::texel());

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

    /// For each pixel, in-place select from the existing channels at the index given by `idx`, or
    /// select a `0` if this index is out-of-range.
    fn shuffle_u8x4(u8s: &mut [[u8; 4]], idx: [u8; 4]) {
        // Naive version. For some reason, LLVM does not figure this out as shuffle instructions.
        // Disappointing.
        for ch in u8s {
            *ch = idx.map(|i| ch[(i & 3) as usize] & (i < 4) as u8);
        }
    }

    /// For each pixel, in-place select from the existing channels at the index given by `idx`, or
    /// select a `0` if this index is out-of-range.
    fn shuffle_u16x4(u8s: &mut [[u16; 4]], idx: [u8; 4]) {
        // Naive version. For some reason, LLVM does not figure this out as shuffle instructions.
        // Disappointing.
        for ch in u8s {
            *ch = idx.map(|i| ch[(i & 3) as usize] & (i < 4) as u16);
        }
    }

    fn join_floats(
        info: &Info,
        bits: [FromBits; 4],
        pixel_buf: &TexelBuffer,
        texel_buf: &mut TexelBuffer,
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
            let texels = texel_buf.as_mut_texels(<f32>::texel());
            let pitched = texels[position..].chunks_mut(pitch);

            for (pix, texel) in source.iter().zip(pitched) {
                texel[0] = pix[ch_idx];
            }
        }
    }

    fn join_yuv422(info: &Info, pixel_buf: &TexelBuffer, texel_buf: &mut TexelBuffer) {
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
