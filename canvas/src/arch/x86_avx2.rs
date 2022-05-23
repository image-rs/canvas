#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_set_m128i, _mm256_shuffle_epi8, _mm_add_epi8, _mm_set1_epi8, _mm_set_epi8,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_set_m128i, _mm256_shuffle_epi8, _mm_add_epi8, _mm_set1_epi8, _mm_set_epi8,
};

#[target_feature(enable = "avx2")]
pub unsafe fn shuffle_u8x4(u8s: &mut [[u8; 4]], idx: [u8; 4]) {
    let (pre, aligned, post) = bytemuck::pod_align_to_mut::<_, __m256i>(u8s);
    super::ShuffleOps::shuffle_u8x4(pre, idx);

    fn in_bounds(idx: u8) -> i8 {
        // The flag bit 0x80 signals that a `0` is written instead of an element.
        // This is also our intention for out-of-bounds indices.
        (if idx < 4 { idx } else { 0x80 }) as i8
    }

    let [a, b, c, d] = idx.map(in_bounds);
    let offset = _mm_set1_epi8(16);
    #[rustfmt::skip]
    let shuffle = _mm_set_epi8(
        d+12, c+12, b+12, a+12,
        d+8, c+8, b+8, a+8,
        d+4, c+4, b+4, a+4,
        d, c, b, a);
    let shuffle = _mm256_set_m128i(_mm_add_epi8(shuffle, offset), shuffle);

    for vec in aligned {
        *vec = _mm256_shuffle_epi8(*vec, shuffle);
    }

    super::ShuffleOps::shuffle_u8x4(post, idx);
}

#[target_feature(enable = "avx2")]
pub unsafe fn shuffle_u16x4(u16s: &mut [[u16; 4]], idx: [u8; 4]) {
    let (pre, aligned, post) = bytemuck::pod_align_to_mut::<_, __m256i>(u16s);
    super::ShuffleOps::shuffle_u16x4(pre, idx);

    fn in_bounds(idx: u8) -> i8 {
        // The flag bit 0x80 signals that a `0` is written instead of an element.
        // This is also our intention for out-of-bounds indices.
        (if idx < 4 { idx } else { 0x80 }) as i8
    }

    let [a, b, c, d] = idx.map(in_bounds);
    let offset = _mm_set1_epi8(16);
    #[rustfmt::skip]
    let shuffle = _mm_set_epi8(
        2*d+9, 2*d+8, 2*c+9, 2*c+8, 2*b+9, 2*b+8, 2*a+9, 2*a+8,
        2*d+1, 2*d, 2*c+1, 2*c, 2*b+1, 2*b, 2*a+1, 2*a);
    let shuffle = _mm256_set_m128i(_mm_add_epi8(shuffle, offset), shuffle);

    for vec in aligned {
        *vec = _mm256_shuffle_epi8(*vec, shuffle);
    }

    super::ShuffleOps::shuffle_u16x4(post, idx);
}
