#![allow(unsafe_code)]
// May be unused if no architecture features are detected at compile time or runtime.
#[allow(unused_imports)]
use core::mem::transmute;

// For when we want to make sure we have a texel at compile time based on bytemuck.
#[allow(unused)]
macro_rules! expect_texel {
    (const $ident:ident: image_texel::Texel<$ty:ty> = ...) => {
        const $ident: Texel<$ty> = match Texel::<$ty>::for_type() {
            Some(texel) => texel,
            None => panic!("Compile error, unexpectedly non-texel"),
        };
    };
}

// May be unused if no architecture features are detected at compile time or runtime.
#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_avx2;
// May be unused if no architecture features are detected at compile time or runtime.
#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_ssse3;

#[expect(clippy::type_complexity)]
pub(crate) struct ShuffleOps {
    // 8-bit, note we may use them for unsigned and signed.
    pub(crate) shuffle_u8x4: fn(&mut [[u8; 4]], [u8; 4]),
    pub(crate) shuffle_u8x3_to_u8x4: fn(&[[u8; 3]], &mut [[u8; 4]], [u8; 4]),
    pub(crate) shuffle_u8x4_to_u8x3: fn(&[[u8; 4]], &mut [[u8; 3]], [u8; 3]),
    // 16-bit, note we may use them for unsigned and signed.
    pub(crate) shuffle_u16x4: fn(&mut [[u16; 4]], [u8; 4]),
    pub(crate) shuffle_u16x3_to_u16x4: fn(&[[u16; 3]], &mut [[u16; 4]], [u8; 4]),
    pub(crate) shuffle_u16x4_to_u16x3: fn(&[[u16; 4]], &mut [[u16; 3]], [u8; 3]),
}

impl ShuffleOps {
    /// FIXME(perf): implement and choose arch-specific shuffles.
    // May be unused if no architecture features are detected at compile time or runtime.
    #[allow(unused_mut)]
    pub fn with_arch(mut self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            self = self.with_x86();
        }

        self
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // May be unused if no architecture features are detected at compile time or runtime.
    #[allow(unused_mut)]
    fn with_x86(mut self) -> Self {
        #[cfg(target_feature = "ssse3")]
        // SAFETY: `ssse3` detected at compile time
        unsafe {
            self = self.with_x86_ssse3();
        }

        #[cfg(not(target_feature = "ssse3"))]
        #[cfg(feature = "runtime-features")]
        if std::is_x86_feature_detected!("ssse3") {
            // SAFETY: `ssse3` detected at runtime
            unsafe {
                self = self.with_x86_ssse3();
            }
        }

        #[cfg(target_feature = "avx2")]
        // SAFETY: `avx2` detected at compile time
        unsafe {
            self = self.with_x86_avx2();
        }

        #[cfg(not(target_feature = "avx2"))]
        #[cfg(feature = "runtime-features")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` detected at runtime
            unsafe {
                self = self.with_x86_avx2();
            }
        }

        self
    }

    /// # Safety
    ///
    /// Must only be used when the `ssse3` feature is available.
    // May be unused if no architecture features are detected at compile time or runtime.
    #[allow(dead_code)]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[expect(clippy::missing_transmute_annotations)]
    unsafe fn with_x86_ssse3(mut self) -> Self {
        self.shuffle_u8x4 =
            unsafe { transmute::<unsafe fn(&mut [[u8; 4]], [u8; 4]), _>(x86_ssse3::shuffle_u8x4) };
        self.shuffle_u16x4 = unsafe {
            transmute::<unsafe fn(&mut [[u16; 4]], [u8; 4]), _>(x86_ssse3::shuffle_u16x4)
        };

        self
    }

    /// # Safety
    ///
    /// Must only be used when the `avx2` feature is available.
    // May be unused if no architecture features are detected at compile time or runtime.
    #[allow(dead_code)]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[expect(clippy::missing_transmute_annotations)]
    unsafe fn with_x86_avx2(mut self) -> Self {
        // Note: On Ivy Bridge these have the same *throughput* of 256bit-per-cycle as their SSSE3
        // equivalents until Icelake. With Icelake they are twice as fast at 512bit-per-cycle.
        // Therefore, we don't select them until we find a way to predict/select this.

        self.shuffle_u8x4 =
            unsafe { transmute::<unsafe fn(&mut [[u8; 4]], [u8; 4]), _>(x86_avx2::shuffle_u8x4) };
        self.shuffle_u16x4 =
            unsafe { transmute::<unsafe fn(&mut [[u16; 4]], [u8; 4]), _>(x86_avx2::shuffle_u16x4) };

        self
    }

    /// For each pixel, in-place select from the existing channels at the index given by `idx`, or
    /// select a `0` if this index is out-of-range.
    /// FIXME(perf): this should be chosen arch dependent.
    fn shuffle_u8x4(u8s: &mut [[u8; 4]], idx: [u8; 4]) {
        // Naive version. For some reason, LLVM does not figure this out as shuffle instructions.
        // Disappointing.
        for ch in u8s {
            *ch = idx.map(|i| ch[(i & 3) as usize] & as_u8mask(i < 4));
        }
    }

    /// For each pixel, in-place select from the existing channels at the index given by `idx`, or
    /// select a `0` if this index is out-of-range.
    /// FIXME(perf): this should be chosen arch dependent.
    fn shuffle_u16x4(u8s: &mut [[u16; 4]], idx: [u8; 4]) {
        // Naive version. For some reason, LLVM does not figure this out as shuffle instructions.
        // Disappointing.
        for ch in u8s {
            *ch = idx.map(|i| ch[(i & 3) as usize] & as_u16mask(i < 4));
        }
    }

    fn shuffle_u8x3_to_u8x4(u3: &[[u8; 3]], u4: &mut [[u8; 4]], idx: [u8; 4]) {
        for (dst, src) in u4.iter_mut().zip(u3) {
            *dst = idx.map(|i| src[i.min(2) as usize] & as_u8mask(i < 3));
        }
    }

    fn shuffle_u8x4_to_u8x3(u4: &[[u8; 4]], u3: &mut [[u8; 3]], idx: [u8; 3]) {
        for (dst, src) in u3.iter_mut().zip(u4) {
            *dst = idx.map(|i| src[(i & 3) as usize] & as_u8mask(i < 4));
        }
    }

    fn shuffle_u16x3_to_u16x4(u3: &[[u16; 3]], u4: &mut [[u16; 4]], idx: [u8; 4]) {
        for (dst, src) in u4.iter_mut().zip(u3) {
            *dst = idx.map(|i| src[i.min(2) as usize] & as_u16mask(i < 3));
        }
    }

    fn shuffle_u16x4_to_u16x3(u4: &[[u16; 4]], u3: &mut [[u16; 3]], idx: [u8; 3]) {
        for (dst, src) in u3.iter_mut().zip(u4) {
            *dst = idx.map(|i| src[(i & 3) as usize] & as_u16mask(i < 4));
        }
    }
}

impl Default for ShuffleOps {
    fn default() -> Self {
        ShuffleOps {
            shuffle_u8x4: Self::shuffle_u8x4,
            shuffle_u8x3_to_u8x4: Self::shuffle_u8x3_to_u8x4,
            shuffle_u8x4_to_u8x3: Self::shuffle_u8x4_to_u8x3,
            shuffle_u16x4: Self::shuffle_u16x4,
            shuffle_u16x3_to_u16x4: Self::shuffle_u16x3_to_u16x4,
            shuffle_u16x4_to_u16x3: Self::shuffle_u16x4_to_u16x3,
        }
    }
}

fn as_u8mask(c: bool) -> u8 {
    0u8.wrapping_sub(c as u8)
}

fn as_u16mask(c: bool) -> u16 {
    0u16.wrapping_sub(c as u16)
}
