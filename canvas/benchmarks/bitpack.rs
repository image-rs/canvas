//! Benchmarks sRGB to sRGB conversions.
use brunch::Bench;

use image_canvas::color::Color;
use image_canvas::layout::{Block, CanvasLayout, LayoutError, SampleBits, SampleParts, Texel};
use image_canvas::Canvas;

struct Convert {
    texel_in: Texel,
    color_in: Color,
    texel_out: Texel,
    color_out: Color,
    sz: u32,
}

impl Convert {
    fn name(&self) -> String {
        format!(
            "bitpack({:?}/{:?}, {:?}/{:?}, {})",
            self.texel_in, self.color_in, self.texel_out, self.color_out, self.sz
        )
    }

    fn prepare(self) -> Result<impl FnMut(), LayoutError> {
        let layout = CanvasLayout::with_texel(&self.texel_in, self.sz, self.sz)?;
        let mut from = Canvas::new(layout.clone());
        from.set_color(self.color_in.clone())?;

        let layout = CanvasLayout::with_texel(&self.texel_out, self.sz, self.sz)?;
        let mut into = Canvas::new(layout);
        into.set_color(self.color_out.clone())?;

        Ok(move || from.convert(&mut into).unwrap())
    }
}

fn main() {
    let tests = [
        /* conversion between same color luma */
        Convert {
            texel_in: Texel {
                block: Block::Pack1x8,
                parts: SampleParts::Luma,
                bits: SampleBits::UInt1x8,
            },
            color_in: Color::BT709,
            texel_out: Texel {
                block: Block::Pixel,
                parts: SampleParts::Luma,
                bits: SampleBits::UInt8,
            },
            color_out: Color::BT709,
            sz: 128,
        },
        Convert {
            texel_in: Texel {
                block: Block::Pixel,
                parts: SampleParts::Luma,
                bits: SampleBits::UInt8,
            },
            color_in: Color::BT709,
            texel_out: Texel {
                block: Block::Pack1x8,
                parts: SampleParts::Luma,
                bits: SampleBits::UInt1x8,
            },
            color_out: Color::BT709,
            sz: 128,
        },
    ];

    let mut benches = brunch::Benches::default();
    benches.extend(tests.map(|convert| {
        Bench::new(format!("framebuf::conversion::main::{}", convert.name()))
            .run(convert.prepare().expect("Failed to setup benchmark"))
    }));
    benches.finish();
}
