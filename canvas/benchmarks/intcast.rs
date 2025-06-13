//! Benchmarks sRGB to sRGB conversions.
use brunch::Bench;

use image_canvas::color::Color;
use image_canvas::layout::{CanvasLayout, LayoutError, SampleParts, Texel};
use image_canvas::Canvas;

struct Convert {
    texel_in: Texel,
    texel_out: Texel,
    sz: u32,
}

impl Convert {
    fn name(&self) -> String {
        format!(
            "intcast({:?}, {:?}, {})",
            self.texel_in, self.texel_out, self.sz
        )
    }

    fn prepare(self) -> Result<impl FnMut(), LayoutError> {
        let layout = CanvasLayout::with_texel(&self.texel_in, self.sz, self.sz)?;
        let mut from = Canvas::new(layout.clone());
        from.set_color(Color::SRGB)?;

        let layout = CanvasLayout::with_texel(&self.texel_out, self.sz, self.sz)?;
        let mut into = Canvas::new(layout);
        into.set_color(Color::SRGB)?;

        Ok(move || from.convert(&mut into).unwrap())
    }
}

fn main() {
    let tests = [
        /* conversion between same color rgba's */
        // Two no-ops.
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            texel_out: Texel::new_f32(SampleParts::Rgb),
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            texel_out: Texel::new_u8(SampleParts::Rgb),
            sz: 128,
        },
        // Bgr, Rgb
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            texel_out: Texel::new_u8(SampleParts::Bgr),
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u8(SampleParts::RgbA),
            texel_out: Texel::new_u8(SampleParts::BgrA),
            sz: 128,
        },
        // While also changing the bit depth
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            texel_out: Texel::new_u16(SampleParts::Bgr),
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u8(SampleParts::RgbA),
            texel_out: Texel::new_u16(SampleParts::BgrA),
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::Rgb),
            texel_out: Texel::new_u8(SampleParts::Bgr),
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::RgbA),
            texel_out: Texel::new_u8(SampleParts::BgrA),
            sz: 128,
        },
        // Conversions that add or drop channels
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            texel_out: Texel::new_u8(SampleParts::RgbA),
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u8(SampleParts::RgbA),
            texel_out: Texel::new_u8(SampleParts::Rgb),
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::RgbA),
            texel_out: Texel::new_u8(SampleParts::Bgr),
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::Bgr),
            texel_out: Texel::new_u8(SampleParts::RgbA),
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
