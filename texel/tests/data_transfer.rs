use std::sync::OnceLock;

use image_texel::image::{AtomicImage, CellImage, DataRef, Image};
use image_texel::layout::{MatrixBytes, PlaneBytes};
use image_texel::texels::U8;

#[test]
fn same_layout_io() {
    let input = TestData::hello_img();

    let initially_empty = MatrixBytes::empty(input.layout.element());
    let mut target = Image::new(initially_empty);

    let data = DataRef::with_layout_at(input.data, input.layout, 0).unwrap();
    target.assign(data.as_source());

    // Layout must match, as must the bytes within the layout.
    assert_eq!(*target.layout(), input.layout);
    assert_eq!(target.as_buf().as_bytes(), input.data);
}

#[test]
fn layout_copy() {
    let input = TestData::hello_img();

    let source = Image::with_bytes(input.layout, input.data);

    let initially_empty = MatrixBytes::empty(input.layout.element());
    let mut target = Image::new(initially_empty);

    target.assign(source.as_source());

    // Layout must match, as must the bytes within the layout.
    assert_eq!(*target.layout(), input.layout);
    assert_eq!(target.as_buf().as_bytes(), input.data);
}

#[test]
fn layout_plane_copy() {
    let input = TestData::hello_img();

    let mut subplanes = {
        let layout = PlaneBytes::new([input.layout, input.layout]);
        let offset = layout.plane_ref(1).unwrap().offset.get();

        let mut data = vec![0u8; offset];
        data.extend_from_slice(input.data);

        Image::with_bytes(layout, &data)
    };

    let (source, buffer) = {
        let [buffer, plane] = subplanes.as_mut().into_planes([0, 1]).unwrap();
        let inner = plane.layout().inner;
        (plane.with_layout(inner).unwrap(), buffer)
    };

    let initially_empty = MatrixBytes::empty(input.layout.element());
    let mut target = Image::new(initially_empty);

    target.assign(source.as_source());

    // Layout must match, as must the bytes within the layout.
    assert_eq!(*target.layout(), input.layout);
    assert_eq!(target.as_buf().as_bytes(), input.data);

    {
        let inner = buffer.layout().inner;
        let mut target = buffer.with_layout(inner).unwrap();
        target.assign(source.as_source()).expect("Enough space");

        // Layout must match, as must the bytes within the layout.
        assert_eq!(*target.layout(), input.layout);
        assert_eq!(target.as_buf().as_bytes(), input.data);
    }
}

#[test]
fn layout_copy_cell() {
    let input = TestData::hello_img();

    let mut subplanes = {
        let layout = PlaneBytes::new([input.layout, input.layout]);
        let offset = layout.plane_ref(1).unwrap().offset.get();

        let mut data = vec![0u8; offset];
        data.extend_from_slice(input.data);

        Image::with_bytes(layout, &data)
    };

    let (source, buffer) = {
        let [buffer, plane] = subplanes.as_mut().into_planes([0, 1]).unwrap();
        let inner = plane.layout().inner;
        (plane.with_layout(inner).unwrap(), buffer)
    };

    // Different to the owned case, we must reserve enough buffer.
    let mut cells = CellImage::new(input.layout);
    cells.assign(source.as_source()).expect("Enough space");

    // Layout must match, as must the bytes within the layout.
    assert_eq!(*cells.layout(), input.layout);
    assert!(cells.as_cell_buf() == input.data);

    {
        let inner = buffer.layout().inner;
        let mut target = buffer.with_layout(inner).unwrap();
        target.assign(cells.as_source()).expect("Enough space");

        // Layout must match, as must the bytes within the layout.
        assert_eq!(*target.layout(), input.layout);
        assert_eq!(target.as_buf().as_bytes(), input.data);
    }
}

#[test]
fn layout_copy_atomic() {
    let input = TestData::hello_img();

    let mut subplanes = {
        let layout = PlaneBytes::new([input.layout, input.layout]);
        let offset = layout.plane_ref(1).unwrap().offset.get();

        let mut data = vec![0u8; offset];
        data.extend_from_slice(input.data);

        Image::with_bytes(layout, &data)
    };

    let (source, buffer) = {
        let [buffer, plane] = subplanes.as_mut().into_planes([0, 1]).unwrap();
        let inner = plane.layout().inner;
        (plane.with_layout(inner).unwrap(), buffer)
    };

    // Different to the owned case, we must reserve enough buffer.
    let mut atomics = AtomicImage::new(input.layout);
    atomics.assign(source.as_source()).expect("Enough space");

    // Layout must match, as must the bytes within the layout.
    assert_eq!(*atomics.layout(), input.layout);
    assert!(atomics.as_texels(U8).to_vec() == input.data);

    {
        let inner = buffer.layout().inner;
        let mut target = buffer.with_layout(inner).unwrap();
        target.assign(atomics.as_source()).expect("Enough space");

        // Layout must match, as must the bytes within the layout.
        assert_eq!(*target.layout(), input.layout);
        assert_eq!(target.as_buf().as_bytes(), input.data);
    }
}

#[test]
fn planar_io() {
    let input = TestData::hello_img();

    let layout: PlaneBytes<3> = PlaneBytes::new({
        let smaller = MatrixBytes::from_width_height(input.layout.element(), 1, 1).unwrap();
        [input.layout, smaller, input.layout]
    });

    let target = Image::new(layout.clone());

    let data = DataRef::with_layout_at(input.data, input.layout, 0).unwrap();
    let reinterpreted = data.as_source().write_to_image(target.decay());

    // Layout must match, as must the bytes within the layout.
    assert_eq!(*reinterpreted.layout(), input.layout);
    assert_eq!(reinterpreted.as_buf().as_bytes(), input.data);

    let mut target = reinterpreted.with_layout(layout);
    // FIXME: see Design Issues in `unaligned`. This is not intuitive.
    let [mut pre, small, post] = target.as_mut().into_planes([0, 1, 2]).unwrap();

    // We've just written that plane.
    assert_eq!(pre.as_buf().as_bytes(), input.data);
    assert_ne!(small.as_buf().as_bytes(), input.data);
    assert_ne!(post.as_buf().as_bytes(), input.data);

    assert!(data.as_source().write_to_mut(small.decay()).is_none());

    let post = data
        .as_source()
        .write_to_mut(post.decay())
        .expect("that plane was large enough");
    // Modify that other independent plane for good measure.
    pre.as_mut_buf().fill(0);

    assert_eq!(*post.layout(), input.layout);
    assert_eq!(post.as_buf().as_bytes(), input.data);
}

struct TestData {
    layout: MatrixBytes,
    data: &'static [u8],
}

impl TestData {
    fn hello_img() -> Self {
        static OFFSET_DATA: OnceLock<Box<[u8]>> = OnceLock::new();
        type Component = u32;

        let texel = <Component as image_texel::AsTexel>::texel();
        let layout = MatrixBytes::from_width_height(texel.into(), 16, 16).expect("Valid, actually");

        let raw_data = OFFSET_DATA.get_or_init(|| {
            let aligned: Vec<_> = (0u32..16 * 16).collect();
            let mut unalign = vec![0x42u8];
            unalign.extend_from_slice(bytemuck::cast_slice(&aligned));
            unalign.into()
        });

        let data = &raw_data[1..];
        TestData { layout, data }
    }
}
