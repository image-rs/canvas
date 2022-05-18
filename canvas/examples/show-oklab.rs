use image_canvas::color::Color;
use image_canvas::layout::{CanvasLayout, LayoutError, SampleParts, Texel};
use image_canvas::Canvas;
use image_texel::AsTexel;

const SZ: u32 = 257;

fn main() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_texel(&Texel::new_f32(SampleParts::Lab), SZ, SZ)?;
    let mut canvas = Canvas::new(layout);
    canvas.set_color(Color::Oklab)?;

    // We can initialize the canvas by hand. For example, as a [f32; 3] array.
    // Note: this works easily because `with_texel` does not add any padding.
    for (idx, lab) in canvas
        .as_texels_mut(<[f32; 3]>::texel())
        .iter_mut()
        .enumerate()
    {
        let idx = idx as u32;
        let (x, y) = (idx % SZ, idx / SZ);

        // Oklab is a Lab-like color spaces with perceptual goals. It has one lightness component,
        // and two independent chroma components. This makes it easy to create different colors
        // with very similar perceived brightness. Varies the color across the whole image.
        let lightness = 0.8f32;
        let la = (x as f32 - 128.) / 256.;
        let lb = (y as f32 - 128.) / 256.;
        lab.copy_from_slice(&[lightness, la, lb]);
    }

    // `image::save` expects an sRGB buffer, allocate one.
    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), SZ, SZ)?;
    let mut output = Canvas::new(layout);
    output.set_color(Color::SRGB)?;

    // Simply convert.
    canvas.convert(&mut output);

    // And copy the memory to a buffer that image expects. We could do something zero-copy here but
    // that's needlessly complicated to handle the size checks or encoding ourselves.
    let container = output.as_bytes().to_owned();
    let image = image::RgbImage::from_raw(SZ, SZ, container).unwrap();

    let output = concat!(env!("CARGO_MANIFEST_DIR"), "/../test.png");
    image.save(output).unwrap();

    Ok(())
}
