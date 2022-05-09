use brunch::Bench;
use image_framebuf::{Color, Frame, FrameLayout, LayoutError, SampleParts, Texel};

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

    fn prepare(self) -> Result<impl FnMut(), LayoutError> {
        let layout = FrameLayout::with_texel(&self.texel_in, self.sz, self.sz)?;
        let mut from = Frame::new(layout.clone());
        from.set_color(self.color_in)?;

        let layout = FrameLayout::with_texel(&self.texel_out, self.sz, self.sz)?;
        let mut into = Frame::new(layout);
        into.set_color(self.color_out)?;

        Ok(move || from.convert(&mut into))
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
        /* rgb conversion with different depths */
        Convert {
            texel_in: Texel::new_u8(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::Rgb),
            color_out: Color::BT709,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_u16(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_u16(SampleParts::Rgb),
            color_out: Color::BT709,
            sz: 128,
        },
        Convert {
            texel_in: Texel::new_f32(SampleParts::Rgb),
            color_in: Color::SRGB,
            texel_out: Texel::new_f32(SampleParts::Rgb),
            color_out: Color::BT709,
            sz: 128,
        },
        /* rgb conversion with alpha */
        Convert {
            texel_in: Texel::new_u8(SampleParts::RgbA),
            color_in: Color::SRGB,
            texel_out: Texel::new_u8(SampleParts::RgbA),
            color_out: Color::BT709,
            sz: 128,
        },
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
    ];

    let mut benches = tests.map(|convert| {
        Bench::new("framebuf::conversion::main", &convert.name())
            .with(convert.prepare().expect("Failed to setup benchmark"))
    });

    // Technically, we're not meant to call this directly but this makes me sad.. Why are we forced
    // to use a macro to setup such a simple data structure. Not like the macro makes it possible
    // to define any more complicated thing than a linear list as well..
    brunch::analyze(&mut benches[..])
}
