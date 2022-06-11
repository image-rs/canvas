use crate::color::Color;
use crate::layout::{Block, CanvasLayout, LayoutError, SampleBits, SampleParts, Texel};
use crate::Canvas;

#[test]
fn simple_conversion() -> Result<(), LayoutError> {
    let texel = Texel::new_u8(SampleParts::RgbA);
    let source_layout = CanvasLayout::with_texel(&texel, 32, 32)?;
    let target_layout = CanvasLayout::with_texel(
        &Texel {
            bits: SampleBits::UInt565,
            parts: SampleParts::Bgr,
            ..texel
        },
        32,
        32,
    )?;

    let mut from = Canvas::new(source_layout);
    let mut into = Canvas::new(target_layout);

    from.as_texels_mut(<[u8; 4] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0x7f, 0xff, 0x0, 0xff]);

    // Expecting conversion [0xff, 0xff, 0x0, 0xff] to 0–ff—ff
    from.convert(&mut into);

    into.as_texels_mut(<[u8; 2] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(u16::from_be_bytes(*b), 0x07ef, "at {}", idx));

    Ok(())
}

#[test]
fn frame_as_channels() -> Result<(), LayoutError> {
    let texel = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Luma), 32, 32)?;
    assert!(Canvas::new(texel).channels_u8().is_some());

    let texel = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::LumaA), 32, 32)?;
    assert!(Canvas::new(texel).channels_u8().is_some());

    let texel = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
    assert!(Canvas::new(texel).channels_u8().is_some());

    let texel = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::RgbA), 32, 32)?;
    assert!(Canvas::new(texel).channels_u8().is_some());

    let texel = CanvasLayout::with_texel(&Texel::new_u16(SampleParts::Luma), 32, 32)?;
    assert!(Canvas::new(texel).channels_u16().is_some());

    let texel = CanvasLayout::with_texel(&Texel::new_f32(SampleParts::Luma), 32, 32)?;
    assert!(Canvas::new(texel).channels_f32().is_some());

    Ok(())
}

#[test]
fn color_no_conversion() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
    let mut from = Canvas::new(layout.clone());
    from.set_color(Color::SRGB)?;

    from.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0xff, 0xff, 0x20]);

    let mut into = Canvas::new(layout);
    into.set_color(Color::SRGB)?;

    from.convert(&mut into);

    into.as_texels(<[u8; 3] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(*b, [0xff, 0xff, 0x20], "at {}", idx));

    Ok(())
}

#[test]
fn color_conversion() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
    let mut from = Canvas::new(layout.clone());
    from.set_color(Color::SRGB)?;

    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Lab), 32, 32)?;
    let mut into = Canvas::new(layout);
    into.set_color(Color::Oklab)?;

    let layout = CanvasLayout::with_texel(&Texel::new_f32(SampleParts::Lab), 32, 32)?;
    let mut rt = Canvas::new(layout);
    rt.set_color(Color::Oklab)?;

    let mut check_color_pair = |rgb: [u8; 3], lab: [u8; 3]| {
        from.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter_mut()
            .for_each(|b| *b = rgb);

        from.convert(&mut into);

        into.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter()
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, lab, "at {}", idx));

        from.convert(&mut rt);
        rt.convert(&mut from);

        from.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter()
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, rgb, "at {}", idx));
    };

    // colorio says this should be: array([246.83446829, -18.2111931 ,  50.6162153 ])
    check_color_pair([255, 255, 0], [247, 0, 51]);
    // colorio says this should be: array([100.55198341,  41.28176379,  -6.22791988])
    check_color_pair([128, 0, 80], [101, 41, 0]);
    // easy check, full black is full black
    check_color_pair([0, 0, 0], [0, 0, 0]);
    // full white is only Luma, no chroma
    check_color_pair([255, 255, 255], [255, 0, 0]);

    Ok(())
}

/// Check one aspect of proper indexing into non-rectangular images.
#[test]
fn non_rectantular() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 64)?;
    let mut from = Canvas::new(layout.clone());
    from.set_color(Color::SRGB)?;

    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Lab), 32, 64)?;
    let mut into = Canvas::new(layout);
    into.set_color(Color::Oklab)?;

    // Initializes two 32x32 images stacked vertically on top of each other.
    // Then converts, checks that the correct color was written to each pixel.
    let mut check_color_pair = |(rgb0, lab0): ([u8; 3], [u8; 3]),
                                (rgb1, lab1): ([u8; 3], [u8; 3])| {
        let mut pixels = from
            .as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter_mut();
        pixels.by_ref().take(32 * 32).for_each(|b| *b = rgb0);
        pixels.for_each(|b| *b = rgb1);

        from.convert(&mut into);

        let mut pixels = into
            .as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter();

        pixels
            .by_ref()
            .take(32 * 32)
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, lab0, "at {}", idx));

        pixels
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, lab1, "at {}", idx));
    };

    check_color_pair(([255, 255, 0], [247, 0, 51]), ([128, 0, 80], [101, 41, 0]));

    Ok(())
}

#[test]
fn shuffled_samples() -> Result<(), LayoutError> {
    let texel = Texel::new_u8(SampleParts::RgbA);
    let source_layout = CanvasLayout::with_texel(&texel, 32, 32)?;
    let texel = Texel::new_u8(SampleParts::BgrA);
    let target_layout = CanvasLayout::with_texel(&texel, 32, 32)?;

    let mut from = Canvas::new(source_layout);
    from.set_color(Color::SRGB)?;
    let mut into = Canvas::new(target_layout);
    into.set_color(Color::SRGB)?;

    from.as_texels_mut(<[u8; 4] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0x40, 0x41, 0x42, 0x43]);

    from.convert(&mut into);

    into.as_texels(<[u8; 4] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(*b, [0x42, 0x41, 0x40, 0x43], "at {}", idx));

    Ok(())
}

#[test]
fn drop_shuffled_samples() -> Result<(), LayoutError> {
    let texel = Texel::new_u8(SampleParts::RgbA);
    let source_layout = CanvasLayout::with_texel(&texel, 32, 32)?;
    let texel = Texel::new_u8(SampleParts::Bgr);
    let target_layout = CanvasLayout::with_texel(&texel, 32, 32)?;

    let mut from = Canvas::new(source_layout);
    from.set_color(Color::SRGB)?;
    let mut into = Canvas::new(target_layout);
    into.set_color(Color::SRGB)?;

    from.as_texels_mut(<[u8; 4] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0x40, 0x41, 0x42, 0x43]);

    from.convert(&mut into);

    into.as_texels(<[u8; 3] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(*b, [0x42, 0x41, 0x40], "at {}", idx));

    Ok(())
}

#[test]
fn expand_shuffled_samples() -> Result<(), LayoutError> {
    let texel = Texel::new_u8(SampleParts::Rgb);
    let source_layout = CanvasLayout::with_texel(&texel, 32, 32)?;
    let texel = Texel::new_u8(SampleParts::BgrA);
    let target_layout = CanvasLayout::with_texel(&texel, 32, 32)?;

    let mut from = Canvas::new(source_layout);
    from.set_color(Color::SRGB)?;
    let mut into = Canvas::new(target_layout);
    into.set_color(Color::SRGB)?;

    from.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0x40, 0x41, 0x42]);

    from.convert(&mut into);

    into.as_texels(<[u8; 4] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(*b, [0x42, 0x41, 0x40, 0x00], "at {}", idx));

    Ok(())
}

#[test]
fn split_to_planes() -> Result<(), LayoutError> {
    let texel = Texel::new_u8(SampleParts::Rgb);
    let source_layout = CanvasLayout::with_texel(&texel, 32, 32)?;
    let mut from = Canvas::new(source_layout);

    assert!(from.planes_mut::<0>().is_some());
    let [_] = from
        .planes_mut::<1>()
        .expect("single plane always possible");
    Ok(())
}

#[test]
fn expand_bits() -> Result<(), LayoutError> {
    let source_layout = CanvasLayout::with_texel(
        &Texel {
            block: Block::Sub1x4,
            parts: SampleParts::Rgb,
            bits: SampleBits::UInt8x3,
        },
        32,
        32,
    )?;

    let mut from = Canvas::new(source_layout);
    from.set_color(Color::SRGB)?;

    assert_eq!(
        from.as_texels(<[u8; 3] as image_texel::AsTexel>::texel())
            .len(),
        8 * 32
    );

    let texel = Texel::new_u8(SampleParts::BgrA);
    let target_layout = CanvasLayout::with_texel(&texel, 32, 32)?;

    let mut into = Canvas::new(target_layout);
    into.set_color(Color::SRGB)?;

    from.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0x40, 0x41, 0x42]);

    from.convert(&mut into);

    into.as_texels(<[u8; 4] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(*b, [0x42, 0x41, 0x40, 0xff], "at {}", idx));

    Ok(())
}

#[test]
fn unpack_bits() -> Result<(), LayoutError> {
    let source_layout = CanvasLayout::with_texel(
        &Texel {
            block: Block::Pack1x8,
            parts: SampleParts::Luma,
            bits: SampleBits::UInt1x8,
        },
        8,
        8,
    )?;

    let mut from = Canvas::new(source_layout);
    from.set_color(Color::BT709)?;

    assert_eq!(
        from.as_texels(<u8 as image_texel::AsTexel>::texel()).len(),
        1 * 8
    );

    let texel = Texel::new_u8(SampleParts::LumaA);
    let target_layout = CanvasLayout::with_texel(&texel, 8, 8)?;

    let mut into = Canvas::new(target_layout);
    into.set_color(Color::BT709)?;

    from.as_texels_mut(<u8 as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = 0x44);

    from.convert(&mut into);

    into.as_texels(<[u8; 8] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| {
            assert_eq!(
                *b,
                [0x00, 0xff, 0xff, 0xff, 0x00, 0xff, 0x00, 0xff],
                "at {}",
                idx
            )
        });

    Ok(())
}

#[test]
fn pack_bits() -> Result<(), LayoutError> {
    let texel = Texel::new_u8(SampleParts::LumaA);
    let source_layout = CanvasLayout::with_texel(&texel, 8, 8)?;

    let mut from = Canvas::new(source_layout);
    from.set_color(Color::BT709)?;

    let target_layout = CanvasLayout::with_texel(
        &Texel {
            block: Block::Pack1x8,
            parts: SampleParts::Luma,
            bits: SampleBits::UInt1x8,
        },
        8,
        8,
    )?;

    let mut into = Canvas::new(target_layout);
    into.set_color(Color::BT709)?;

    assert_eq!(
        into.as_texels(<u8 as image_texel::AsTexel>::texel()).len(),
        1 * 8
    );

    from.as_texels_mut(<[u8; 8] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0x00, 0xff, 0xff, 0xff, 0x00, 0xff, 0x00, 0xff]);

    from.convert(&mut into);

    into.as_texels(<u8 as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(*b, 0x44, "at {}", idx));

    Ok(())
}

#[test]
fn yuv_conversion() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
    let mut from = Canvas::new(layout.clone());
    from.set_color(Color::SRGB)?;

    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Yuv), 32, 32)?;
    let mut into = Canvas::new(layout);
    into.set_color(Color::BT709)?;

    let layout = CanvasLayout::with_texel(&Texel::new_f32(SampleParts::Yuv), 32, 32)?;
    let mut rt = Canvas::new(layout);
    rt.set_color(Color::BT709)?;

    let mut check_color_pair = |rgb: [u8; 3], yuv: [u8; 3]| {
        from.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter_mut()
            .for_each(|b| *b = rgb);

        from.convert(&mut into);

        into.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter()
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, yuv, "at {}", idx));

        from.convert(&mut rt);
        rt.convert(&mut from);

        from.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter()
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, rgb, "at {}", idx));
    };

    check_color_pair([255, 255, 0], [237, 0, 12]);
    check_color_pair([128, 0, 80], [29, 19, 55]);
    // easy check, full black is full black
    check_color_pair([0, 0, 0], [0, 0, 0]);
    // full white is only Luma, no chroma
    check_color_pair([255, 255, 255], [255, 0, 0]);

    Ok(())
}
