use image_canvas::color::Color;
use image_canvas::layout::{CanvasLayout, LayoutError, SampleParts, Texel};
use image_canvas::Canvas;
use image_texel::AsTexel;

const SZ_W: u32 = 1920;
const SZ_H: u32 = 1080;

fn main() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_texel(&Texel::new_f32(SampleParts::Lab), SZ_W, SZ_H)?;
    let mut canvas = Canvas::new(layout);
    canvas.set_color(Color::SrLab2 {
        whitepoint: image_canvas::color::Whitepoint::D65,
    })?;

    // `image::save` expects an sRGB buffer, allocate one.
    let layout = CanvasLayout::with_texel(&Texel::new_u16(SampleParts::Rgb), SZ_W, SZ_H)?;
    let mut output = Canvas::new(layout);
    output.set_color(Color::SRGB)?;

    for i in 100..101 {
        // We can initialize the canvas by hand. For example, as a [f32; 3] array.
        // Note: this works easily because `with_texel` does not add any padding.
        for (idx, lab) in canvas
            .as_texels_mut(<[f32; 3]>::texel())
            .iter_mut()
            .enumerate()
        {
            let idx = idx as u32;
            let (x, y) = (idx % SZ_W, idx / SZ_W);

            // SrLab2 is a Lab-like color spaces with perceptual goals. It has one lightness component,
            // and two independent chroma components. This makes it easy to create different colors
            // with very similar perceived brightness. Varies the color across the whole image.
            let lightness = i as f32 / 100f32 * 0.8f32;
            let la = (x as i32 - SZ_W as i32 / 2) as f32 / SZ_W as f32;
            let lb = (y as i32 - SZ_H as i32 / 2) as f32 / SZ_H as f32;
            lab.copy_from_slice(&[lightness, la, lb]);
        }

        // Simply convert.
        canvas.convert(&mut output);

        // And copy the memory to a buffer that image expects. We could do something zero-copy here but
        // that's needlessly complicated to handle the size checks or encoding ourselves.
        let container = output.as_texels(u16::texel()).to_owned();
        let image =
            image::ImageBuffer::<image::Rgb<_>, _>::from_raw(SZ_W, SZ_H, container).unwrap();

        let output = format!(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../test-{:0>2}.png"),
            i
        );
        image.save(output).unwrap();
    }

    Ok(())
}
