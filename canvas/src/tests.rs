use crate::{
    canvas::{ArcCanvas, RcCanvas},
    color::Color,
    layout::{
        Block, CanvasLayout, LayoutError, RowLayoutDescription, SampleBits, SampleParts, Texel,
    },
    Canvas, Converter,
};

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
    from.convert(&mut into).unwrap();

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

    from.convert(&mut into).unwrap();

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

        from.convert(&mut into).unwrap();

        into.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter()
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, lab, "at {}", idx));

        from.convert(&mut rt).unwrap();
        rt.convert(&mut from).unwrap();

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

        from.convert(&mut into).unwrap();

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

    from.convert(&mut into).unwrap();

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

    from.convert(&mut into).unwrap();

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

    from.convert(&mut into).unwrap();

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

    from.convert(&mut into).unwrap();

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

    from.convert(&mut into).unwrap();

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

    from.convert(&mut into).unwrap();

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

        from.convert(&mut into).unwrap();

        into.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter()
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, yuv, "at {}", idx));

        from.convert(&mut rt).unwrap();
        rt.convert(&mut from).unwrap();

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

#[test]
fn luma_conversion() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Rgb), 32, 32)?;
    let mut from = Canvas::new(layout.clone());
    from.set_color(Color::SRGB)?;

    let layout = CanvasLayout::with_texel(&Texel::new_u8(SampleParts::Luma), 32, 32)?;
    let mut into = Canvas::new(layout);
    into.set_color(Color::SRGB_LUMA)?;

    let mut check_color_pair = |rgb: [u8; 3], luma: u8| {
        from.as_texels_mut(<[u8; 3] as image_texel::AsTexel>::texel())
            .iter_mut()
            .for_each(|b| *b = rgb);

        from.convert(&mut into).unwrap();

        into.as_texels_mut(<[u8; 1] as image_texel::AsTexel>::texel())
            .iter()
            .enumerate()
            .for_each(|(idx, b)| assert_eq!(*b, [luma], "at {}", idx));
    };

    check_color_pair([255, 255, 0], 247);
    check_color_pair([128, 0, 80], 64);
    // easy check, full black is full black
    check_color_pair([0, 0, 0], 0);
    // full white is only Luma, no chroma
    check_color_pair([255, 255, 255], 255);
    // sRGB whitepoint D65 is close enough to equal weight on all channels.
    check_color_pair([30, 30, 30], 30);

    // Primary lightnesses for reference.
    check_color_pair([255, 0, 0], 127);
    check_color_pair([0, 255, 0], 220);
    check_color_pair([0, 0, 255], 76);

    Ok(())
}

#[test]
fn strided() -> Result<(), LayoutError> {
    let layout = CanvasLayout::with_row_layout(&RowLayoutDescription {
        width: 32,
        height: 32,
        texel: Texel::new_u8(SampleParts::A),
        // Overaligned.
        row_stride: 256,
    })?;

    assert_eq!(layout.u64_len(), 256 * 32);

    Ok(())
}

#[test]
fn incorrect_strided() {
    let layout = CanvasLayout::with_row_layout(&RowLayoutDescription {
        width: 32,
        height: 32,
        texel: Texel::new_u8(SampleParts::A),
        // Rows would alias, too few bytes.
        row_stride: 16,
    });

    assert!(layout.is_err());
}

/// Verify some bit packing code by manually constructed matching pairs, with partial texel blocks.
#[test]
fn a_bitpack_dilemma_wrapped() {
    let from = Texel::new_u8(SampleParts::Luma);

    let into = Texel {
        block: Block::Pack1x8,
        bits: SampleBits::UInt1x8,
        parts: SampleParts::Luma,
    };

    let mut from = Canvas::new(CanvasLayout::with_texel(&from, 23, 4).unwrap());
    let mut into = Canvas::new(CanvasLayout::with_texel(&into, 23, 4).unwrap());

    #[rustfmt::skip]
    const IMG_BW_LUMA: &[u8; 23 * 4] = &[
        0, 0, 0, 0, 0, 0, 0, 0,  128, 128, 128, 128, 128, 128, 128, 128,  0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,    8,   8,   8,   8,   8,   8,   8,   8,  0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 128,  8,   8,   8,   8,   8,   8,   8,   8,  0, 0, 0, 0, 0, 0, 128,
        0, 0, 0, 0, 0, 0, 0, 0,  128, 128, 128, 128, 128, 128, 128, 128,  0, 0, 0, 0, 0, 0, 0,
    ];

    #[rustfmt::skip]
    const IMG_BW_PACKED: &[u8; 3 * 4] = &[
        0x00, 0xff, 0x00,
        0x00, 0x00, 0x00,
        0x01, 0x00, 0x02,
        0x00, 0xff, 0x00,
    ];

    from.as_bytes_mut().copy_from_slice(IMG_BW_LUMA);
    from.convert(&mut into).unwrap();

    assert_eq!(into.as_bytes(), IMG_BW_PACKED);
}

/// Verify some bit packing code by manually constructed matching pairs, with full texel blocks.
#[test]
fn a_bitpack_dilemma_unwrapped() {
    let texel_from = Texel::new_u8(SampleParts::Luma);

    let texel_into = Texel {
        block: Block::Pack1x8,
        bits: SampleBits::UInt1x8,
        parts: SampleParts::Luma,
    };

    let mut from = Canvas::new(CanvasLayout::with_texel(&texel_from, 24, 9).unwrap());
    let mut into = Canvas::new(CanvasLayout::with_texel(&texel_into, 24, 9).unwrap());

    #[rustfmt::skip]
    const IMG_BW_LUMA: &[u8; 24 * 9] = &[
        0, 0, 0, 0, 0, 0, 0, 0,    8,   8,   8,   8,   8,   8,   8,   8,  0, 0, 0, 0, 0, 0, 0, 128,
        0, 0, 0, 0, 0, 0, 0, 0,  128, 128, 128, 128, 128, 128, 128, 128,  0, 0, 0, 0, 0, 0, 0, 128,
        0, 0, 0, 0, 0, 0, 0, 128,  8,   8,   8,   8,   8,   8,   8,   8,  0, 0, 0, 0, 0, 0, 128, 128,
        0, 0, 0, 0, 0, 0, 0, 0,  128, 128, 128, 128, 128, 128, 128, 128,  0, 0, 0, 0, 0, 0, 0, 128,

        0, 0, 0, 0, 0, 0, 0, 0,    8,   8,   8,   8,   8,   8,   8,   8,  0, 0, 0, 0, 0, 0, 0, 128,
        0, 0, 0, 0, 0, 0, 0, 0,    8,   8,   8,   8,   8,   8,   8,   8,  0, 0, 0, 0, 0, 0, 0, 128,
        0, 0, 0, 0, 0, 0, 0, 0,    8,   8,   8,   8,   8,   8,   8,   8,  0, 0, 0, 0, 0, 0, 0, 128,
        0, 0, 0, 0, 0, 0, 0, 0,    8,   8,   8,   8,   8,   8,   8,   8,  0, 0, 0, 0, 0, 0, 0, 128,

        0, 0, 0, 0, 0, 0, 0, 0,  128, 128, 128, 128, 128, 128, 128, 128,  0, 0, 0, 0, 0, 0, 128, 128,
    ];

    #[rustfmt::skip]
    const IMG_BW_PACKED: &[u8; 3 * 9] = &[
        0x00, 0x00, 0x01,
        0x00, 0xff, 0x01,
        0x01, 0x00, 0x03,
        0x00, 0xff, 0x01,

        0x00, 0x00, 0x01,
        0x00, 0x00, 0x01,
        0x00, 0x00, 0x01,
        0x00, 0x00, 0x01,

        0x00, 0xff, 0x03,
    ];

    from.as_bytes_mut().copy_from_slice(IMG_BW_LUMA);
    from.convert(&mut into).unwrap();

    assert_eq!(into.as_bytes(), IMG_BW_PACKED);

    for i in (0..48).step_by(1) {
        let mut from = Canvas::new(CanvasLayout::with_texel(&texel_from, 24, 9 + i).unwrap());
        let mut into = Canvas::new(CanvasLayout::with_texel(&texel_into, 24, 9 + i).unwrap());
        let i = i as usize;

        from.as_bytes_mut()[24 * i..].copy_from_slice(IMG_BW_LUMA);
        from.convert(&mut into).unwrap();

        assert_eq!(&into.as_bytes()[3 * i..], IMG_BW_PACKED, "{i}");
    }
}

#[test]
fn to_rc_conversion() -> Result<(), LayoutError> {
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

    let mut from = Canvas::new(source_layout.clone());
    let into = RcCanvas::new(target_layout.clone());

    from.as_texels_mut(<[u8; 4] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0x7f, 0xff, 0x0, 0xff]);

    // Expecting conversion [0xff, 0xff, 0x0, 0xff] to 0–ff—ff
    {
        let mut converter = Converter::new();
        let mut plan = converter.plan(source_layout, target_layout).unwrap();
        plan.add_plane_in(from.plane(0).unwrap()).set_as_color();
        plan.add_cell_out(into.plane(0).unwrap()).set_as_color();
        plan.run().unwrap();
    }

    let into = into.to_canvas();
    into.as_texels(<[u8; 2] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(u16::from_be_bytes(*b), 0x07ef, "at {}", idx));

    Ok(())
}

#[test]
fn to_arc_conversion() -> Result<(), LayoutError> {
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

    let mut from = Canvas::new(source_layout.clone());
    let into = ArcCanvas::new(target_layout.clone());

    from.as_texels_mut(<[u8; 4] as image_texel::AsTexel>::texel())
        .iter_mut()
        .for_each(|b| *b = [0x7f, 0xff, 0x0, 0xff]);

    // Expecting conversion [0xff, 0xff, 0x0, 0xff] to 0–ff—ff
    {
        let mut converter = Converter::new();
        let mut plan = converter.plan(source_layout, target_layout).unwrap();
        plan.add_plane_in(from.plane(0).unwrap()).set_as_color();
        plan.add_atomic_out(into.plane(0).unwrap()).set_as_color();
        plan.run().unwrap();
    }

    let into = into.to_canvas();
    into.as_texels(<[u8; 2] as image_texel::AsTexel>::texel())
        .iter()
        .enumerate()
        .for_each(|(idx, b)| assert_eq!(u16::from_be_bytes(*b), 0x07ef, "at {}", idx));

    Ok(())
}
