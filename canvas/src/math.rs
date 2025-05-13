/// Equivalent to `f32::powf` but suitable on `no_std`.
#[inline]
pub(crate) fn powf(base: f32, exp: f32) -> f32 {
    libm::powf(base, exp)
}
