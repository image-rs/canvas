use image_canvas::color::Color;
use image_canvas::layout::{CanvasLayout, LayoutError, SampleParts, Texel};
use image_canvas::Canvas;
use image_texel::AsTexel;

const SZ_W: u32 = 1920;
const SZ_H: u32 = 1080;

/// Shows how one would attempt conversion of a BgrA frame into RgbA.
///
/// This example exists, in part, so that we can run `perf`.
fn main() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::BgrA), SZ_W, SZ_H)?;
    let mut canvas = Canvas::new(layout);
    canvas.set_color(Color::SRGB)?;

    // `image::save` expects an sRGB buffer, allocate one.
    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::RgbA), SZ_W, SZ_H)?;
    let mut output = Canvas::new(layout);
    output.set_color(Color::SRGB)?;

    if std::env::var_os("IMAGE_CANVAS_SKIP_IO").is_none() {
        for bgra in canvas.as_texels_mut(<[u8; 4]>::texel()) {
            *bgra = [0xbb, 0x00, 0x00, 0xff];
        }
    }

    canvas.convert(&mut output).unwrap();

    if std::env::var_os("IMAGE_CANVAS_SKIP_IO").is_none() {
        let container = output.as_texels(u8::texel()).to_owned();
        let image =
            image::ImageBuffer::<image::Rgba<_>, _>::from_raw(SZ_W, SZ_H, container).unwrap();

        let output = format!(concat!(env!("CARGO_MANIFEST_DIR"), "/../test.png"),);
        // This will be blue, i.e. color channels were swapped.
        image.save(output).unwrap();
    }

    Ok(())
}
