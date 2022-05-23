#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128i, _mm_set_epi8, _mm_shuffle_epi8};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128i, _mm_set_epi8, _mm_shuffle_epi8};

#[target_feature(enable = "ssse3")]
pub unsafe fn shuffle_u8x4(u8s: &mut [[u8; 4]], idx: [u8; 4]) {
    let (pre, aligned, post) = bytemuck::pod_align_to_mut::<_, __m128i>(u8s);
    if !pre.is_empty() {
        super::ShuffleOps::shuffle_u8x4(pre, idx);
    }

    fn in_bounds(idx: u8) -> i8 {
        // The flag bit 0x80 signals that a `0` is written instead of an element.
        // This is also our intention for out-of-bounds indices.
        (if idx < 4 { idx } else { 0x80 }) as i8
    }

    let [a, b, c, d] = idx.map(in_bounds);
    #[rustfmt::skip]
    let shuffle = _mm_set_epi8(
        d+12, c+12, b+12, a+12,
        d+8, c+8, b+8, a+8,
        d+4, c+4, b+4, a+4,
        d, c, b, a);
    let mut chunks = aligned.chunks_exact_mut(2);
    for vec in &mut chunks {
        // shuffle_epi8 has latency/throughput of 1/1 on everything. Almost, except Ivy Bridge
        // where it has 1/0.5 and therefore we do two at the same time. Doesn't hurt.
        let a = _mm_shuffle_epi8(vec[0], shuffle);
        let b = _mm_shuffle_epi8(vec[1], shuffle);
        vec[0] = a;
        vec[1] = b;
    }
    for vec in chunks.into_remainder() {
        *vec = _mm_shuffle_epi8(*vec, shuffle);
    }

    if !post.is_empty() {
        super::ShuffleOps::shuffle_u8x4(post, idx);
    }
}

#[target_feature(enable = "ssse3")]
pub unsafe fn shuffle_u16x4(u16s: &mut [[u16; 4]], idx: [u8; 4]) {
    let (pre, aligned, post) = bytemuck::pod_align_to_mut::<_, __m128i>(u16s);
    if !pre.is_empty() {
        super::ShuffleOps::shuffle_u16x4(pre, idx);
    }

    fn in_bounds(idx: u8) -> i8 {
        // The flag bit 0x80 signals that a `0` is written instead of an element.
        // This is also our intention for out-of-bounds indices.
        (if idx < 4 { idx } else { 0x80 }) as i8
    }

    let [a, b, c, d] = idx.map(in_bounds);
    #[rustfmt::skip]
    let shuffle = _mm_set_epi8(
        2*d+9, 2*d+8, 2*c+9, 2*c+8, 2*b+9, 2*b+8, 2*a+9, 2*a+8,
        2*d+1, 2*d, 2*c+1, 2*c, 2*b+1, 2*b, 2*a+1, 2*a);
    let mut chunks = aligned.chunks_exact_mut(2);
    for vec in &mut chunks {
        let a = _mm_shuffle_epi8(vec[0], shuffle);
        let b = _mm_shuffle_epi8(vec[1], shuffle);
        vec[0] = a;
        vec[1] = b;
    }
    for vec in chunks.into_remainder() {
        *vec = _mm_shuffle_epi8(*vec, shuffle);
    }

    if !post.is_empty() {
        super::ShuffleOps::shuffle_u16x4(post, idx);
    }
}
