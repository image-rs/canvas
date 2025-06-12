use brunch::Bench;

use image_canvas::color::{Color, Whitepoint};
use image_canvas::layout::{CanvasLayout, LayoutError, SampleParts, Texel};
use image_canvas::{
    canvas::{ArcCanvas, RcCanvas},
    Canvas, Converter,
};

#[derive(Debug)]
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
            "convert({:?}/{:?}, {:?}/{:?}, {})",
            self.texel_in, self.color_in, self.texel_out, self.color_out, self.sz
        )
    }

    fn prepare(&self) -> Result<impl FnMut(), LayoutError> {
        let layout = CanvasLayout::with_texel(&self.texel_in, self.sz, self.sz)?;
        let mut from = Canvas::new(layout.clone());
        from.set_color(self.color_in.clone())?;

        let layout = CanvasLayout::with_texel(&self.texel_out, self.sz, self.sz)?;
        let mut into = Canvas::new(layout);
        into.set_color(self.color_out.clone())?;

        Ok(move || from.convert(&mut into))
    }

    fn prepare_atomic(&self) -> Result<impl FnMut(), LayoutError> {
        let layout_from = CanvasLayout::with_texel(&self.texel_in, self.sz, self.sz)?;
        let mut from = Canvas::new(layout_from.clone());
        from.set_color(self.color_in.clone())?;

        let layout_into = CanvasLayout::with_texel(&self.texel_out, self.sz, self.sz)?;
        let mut into = Canvas::new(layout_into.clone());
        into.set_color(self.color_out.clone())?;
        let into = ArcCanvas::from(into);

        Ok(move || {
            let mut converter = Converter::new();
            let mut plan = converter.plan(layout_from.clone(), layout_into.clone());
            plan.add_plane_in(from.plane(0).unwrap());
        })
    }
}

fn main() {
    let tests = [
        /* conversion to oklab with different depths */
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::Lab),
            color_out: Color::Oklab,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u16(SampleParts::Lab),
            color_out: Color::Oklab,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_f32(SampleParts::Lab),
            color_out: Color::Oklab,
            sz: 128,
        },
        /* rgb to yuv conversion with different depths */
        /*
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::Yuv),
            color_out: Color::BT709,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u16(SampleParts::Yuv),
            color_out: Color::BT709,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_f32(SampleParts::Yuv),
            color_out: Color::BT709,
            sz: 128,
        },
        /* rgb conversion with alpha */
        Convert {
            texel_in: Texel::new_u8(SampleParts::RgbA),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::YuvA),
            color_out: Color::BT709,
            sz: 128,
        },
        */
        /* Mainly texel conversion */
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u16(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_f32(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u16(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_f32(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
        /* conversion to oklab with different depths */
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::Lab),
            color_out: Color::Oklab,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u16(SampleParts::Lab),
            color_out: Color::Oklab,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_f32(SampleParts::Lab),
            color_out: Color::Oklab,
            sz: 128,
        },
        /* conversion to SRLAB2 */
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::Lab),
            color_out: Color::SrLab2 {
                whitepoint: Whitepoint::D65,
            },
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u16(SampleParts::Lab),
            color_out: Color::SrLab2 {
                whitepoint: Whitepoint::D65,
            },
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_f32(SampleParts::Lab),
            color_out: Color::SrLab2 {
                whitepoint: Whitepoint::D65,
            },
            sz: 128,
        },
        /* conversion from SRLAB2 */
        Convert {
            texel_in: Texel::new_u8(SampleParts::Lab),
            color_in: Color::SrLab2 {
                whitepoint: Whitepoint::D65,
            },
            texel_out: Texel::new_u8(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::Lab),
            color_in: Color::SrLab2 {
                whitepoint: Whitepoint::D65,
            },
            texel_out: Texel::new_u16(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Lab),
            color_in: Color::SrLab2 {
                whitepoint: Whitepoint::D65,
            },
            texel_out: Texel::new_f32(SampleParts::Rgb),
            color_out: Color::SRGB,
            sz: 128,
        },
    ];

    let mut benches = brunch::Benches::default();
    benches.extend(tests.map(|convert| {
        let bench = match convert.prepare() {
            Ok(bench) => bench,
            Err(err) => panic!("Failed to setup benchmark {:?}: {:?}", convert, err),
        };

        Bench::new(format!("framebuf::conversion::main::{}", convert.name())).run(bench)
    }));
    benches.finish();
}
