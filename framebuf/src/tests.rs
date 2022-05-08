use crate::{Color, Frame, FrameLayout, LayoutError, SampleBits, SampleParts, Texel};

#[test]
fn simple_conversion() -> Result<(), LayoutError> {
    let texel = Texel::new_u8(SampleParts::RgbA);
    let source_layout = FrameLayout::with_texel(&texel, 32, 32)?;
    let target_layout = FrameLayout::with_texel(
        &Texel {
            bits: SampleBits::Int565,
            parts: SampleParts::Bgr,
            ..texel
        },
        32,
        32,
    )?;

    let mut from = Frame::new(source_layout);
    let mut into = Frame::new(target_layout);

    from.as_texels_mut(<[u8; 4] as canvas::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0xff, 0xff, 0x0, 0xff]);

    // Expecting conversion [0xff, 0xff, 0x0, 0xff] to 0–ff—ff
    from.convert(&mut into);

    into.as_texels_mut(<u16 as canvas::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(*b, 0xffe0, "at {}", idx));

    Ok(())
}

#[test]
fn frame_as_channels() -> Result<(), LayoutError> {
    let texel = FrameLayout::with_texel(&Texel::new_u8(SampleParts::Luma), 32, 32)?;
    assert!(Frame::new(texel).channels_u8().is_some());

    let texel = FrameLayout::with_texel(&Texel::new_u8(SampleParts::LumaA), 32, 32)?;
    assert!(Frame::new(texel).channels_u8().is_some());

    let texel = FrameLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
    assert!(Frame::new(texel).channels_u8().is_some());

    let texel = FrameLayout::with_texel(&Texel::new_u8(SampleParts::RgbA), 32, 32)?;
    assert!(Frame::new(texel).channels_u8().is_some());

    let texel = FrameLayout::with_texel(&Texel::new_u16(SampleParts::Luma), 32, 32)?;
    assert!(Frame::new(texel).channels_u16().is_some());

    let texel = FrameLayout::with_texel(&Texel::new_f32(SampleParts::Luma), 32, 32)?;
    assert!(Frame::new(texel).channels_f32().is_some());

    Ok(())
}

#[test]
fn color_no_conversion() -> Result<(), LayoutError> {
    let layout = FrameLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
    let mut from = Frame::new(layout.clone());
    from.set_color(Color::SRGB)?;

    from.as_texels_mut(<[u8; 3] as canvas::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0xff, 0xff, 0x20]);

    let mut into = Frame::new(layout);
    into.set_color(Color::SRGB)?;

    from.convert(&mut into);

    into.as_texels(<[u8; 3] as canvas::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(*b, [0xff, 0xff, 0x20], "at {}", idx));

    Ok(())
}

#[test]
fn color_conversion() -> Result<(), LayoutError> {
    let layout = FrameLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
    let mut from = Frame::new(layout.clone());
    from.set_color(Color::SRGB)?;

    let layout = FrameLayout::with_texel(&Texel::new_u8(SampleParts::Lab), 32, 32)?;
    let mut into = Frame::new(layout);
    into.set_color(Color::Oklab)?;

    let mut check_color_pair = |rgb: [u8; 3], lab: [u8; 3]| {
        from.as_texels_mut(<[u8; 3] as canvas::AsTexel>::texel())
            .iter_mut()
            .for_each(|b| *b = rgb);

        from.convert(&mut into);

        into.as_texels_mut(<[u8; 3] as canvas::AsTexel>::texel())
            .iter()
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, lab, "at {}", idx));
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
