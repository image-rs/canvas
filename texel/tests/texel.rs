use image_texel::{AsTexel, TexelBuffer};

#[test]
fn conversions() {
    let mut buffer: TexelBuffer = TexelBuffer::new(16);

    assert_eq!(buffer.as_texels(u8::texel()).len(), 16);
    assert_eq!(buffer.as_mut_texels(u8::texel()).len(), 16);
    assert_eq!(buffer.as_texels(u16::texel()).len(), 8);
    assert_eq!(buffer.as_mut_texels(u16::texel()).len(), 8);

    let bytes = buffer.as_texels(u8::texel());
    assert_eq!(
        <[u32; 4]>::texel().try_to_slice(bytes),
        Some(&[[0u32; 4]][..])
    );
    let bytes = buffer.as_mut_texels(u8::texel());
    assert_eq!(
        <[u32; 4]>::texel().try_to_slice_mut(bytes),
        Some(&mut [[0u32; 4]][..])
    );

    let texel = <[f32; 4]>::texel();
    assert_eq!(texel.to_bytes(buffer.as_texels(texel)).len(), 16);
    assert_eq!(texel.to_mut_bytes(buffer.as_mut_texels(texel)).len(), 16);
}
