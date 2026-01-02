use crate::layout::{SampleBits, SampleParts};
use image_texel::Texel;

/// Specifies which bits a channel comes from, within a `TexelKind` aggregate.
#[derive(Clone, Copy, Debug)]
pub struct FromBits {
    pub(crate) begin: usize,
    pub(crate) len: usize,
}

macro_rules! from_bits {
    ($bits:ident = { $($variant:pat => $($value:expr)+);* }) => {
        match $bits {
            $($variant => from_bits!(@ $($value);*)),*,
        }
    };
    (@ $v0:expr) => {
        [Some(FromBits::from_range($v0)), None, None, None, None, None, None, None]
    };
    (@ $v0:expr; $v1:expr) => {
        [Some(FromBits::from_range($v0)), Some(FromBits::from_range($v1)), None, None, None, None, None, None]
    };
    (@ $v0:expr; $v1:expr; $v2:expr) => {
        [
            Some(FromBits::from_range($v0)),
            Some(FromBits::from_range($v1)),
            Some(FromBits::from_range($v2)),
            None,
            None,
            None,
            None,
            None,
        ]
    };
    (@ $v0:expr; $v1:expr; $v2:expr; $v3:expr) => {
        [
            Some(FromBits::from_range($v0)),
            Some(FromBits::from_range($v1)),
            Some(FromBits::from_range($v2)),
            Some(FromBits::from_range($v3)),
            None,
            None,
            None,
            None,
        ]
    };
    (@ $v0:expr; $v1:expr; $v2:expr; $v3:expr; $v4:expr; $v5:expr) => {
        [
            Some(FromBits::from_range($v0)),
            Some(FromBits::from_range($v1)),
            Some(FromBits::from_range($v2)),
            Some(FromBits::from_range($v3)),
            Some(FromBits::from_range($v4)),
            Some(FromBits::from_range($v5)),
            None,
            None,
        ]
    };
    (@ $v0:expr; $v1:expr; $v2:expr; $v3:expr; $v4:expr; $v5:expr; $v6:expr; $v7:expr) => {
        [
            Some(FromBits::from_range($v0)),
            Some(FromBits::from_range($v1)),
            Some(FromBits::from_range($v2)),
            Some(FromBits::from_range($v3)),
            Some(FromBits::from_range($v4)),
            Some(FromBits::from_range($v5)),
            Some(FromBits::from_range($v6)),
            Some(FromBits::from_range($v7)),
        ]
    };
}

impl FromBits {
    const NO_BITS: Self = FromBits { begin: 0, len: 0 };

    const fn from_range(range: core::ops::Range<usize>) -> Self {
        FromBits {
            begin: range.start,
            len: range.end - range.start,
        }
    }

    pub(crate) fn for_pixel(bits: SampleBits, parts: SampleParts) -> [Self; 4] {
        let mut vals = [Self::NO_BITS; 4];

        let bits = Self::bits(bits);
        let channels = parts.channels();

        for (bits, (channel, pos)) in bits.zip(channels) {
            if channel.is_some() {
                vals[pos as usize] = bits;
            }
        }

        vals
    }

    pub(crate) fn for_pixels<const N: usize>(
        bits: SampleBits,
        parts: SampleParts,
    ) -> [[Self; 4]; N] {
        let mut vals = [[Self::NO_BITS; 4]; N];

        let mut bits = Self::bits(bits);

        for vals in vals.iter_mut() {
            let channels = parts.channels().filter_map(|(ch, p)| Some((ch?, p)));

            for (_, pos) in channels {
                if let Some(bits) = bits.next() {
                    vals[pos as usize] = bits;
                }
            }
        }

        vals
    }

    pub(crate) const fn mask(self) -> u32 {
        ((-1i64 as u64) ^ u32::MAX as u64).rotate_left(self.len as u32) as u32
    }

    fn bits(bits: SampleBits) -> impl Iterator<Item = Self> {
        use SampleBits::*;

        let filled: [Option<Self>; 8] = from_bits!(bits = {
            Int8 | UInt8 => 0..8;
            UInt332 => 0..3 3..6 6..8;
            UInt233 => 0..2 2..5 5..8;
            Int16 | UInt16 => 0..16;
            UInt4x2 => 0..4 4..8;
            UInt4x4 => 0..4 4..8 8..12 12..16;
            UInt4x6 => 0..4 4..8 8..12 12..16 16..20 20..24;
            UInt_444 => 4..8 8..12 12..16;
            UInt444_ => 0..4 4..8 8..12;
            UInt565 => 0..5 5..11 11..16;
            Int8x2 | UInt8x2 => 0..8 8..16;
            Int8x3 | UInt8x3 => 0..8 8..16 16..24;
            Int8x4 | UInt8x4 => 0..8 8..16 16..24 24..32;
            UInt8x6 => 0..8 8..16 16..24 24..32 32..40 40..48;
            Int16x2 | UInt16x2 => 0..16 16..32;
            Int16x3 | UInt16x3 => 0..16 16..32 32..48;
            Int16x4 | UInt16x4 => 0..16 16..32 32..48 48..64;
            UInt16x6 => 0..16 16..32 32..48 48..64 64..80 80..96;
            UInt1010102 => 0..10 10..20 20..30 30..32;
            UInt2101010 => 0..2 2..12 12..22 22..32;
            UInt101010_ => 0..10 10..20 20..30;
            UInt_101010 => 2..12 12..22 22..32;
            Float16x4 => 0..16 16..32 32..48 48..64;
            Float32 => 0..32;
            Float32x2 => 0..32 32..64;
            Float32x3 => 0..32 32..64 64..96;
            Float32x4 => 0..32 32..64 64..96 96..128;
            Float32x6 => 0..32 32..64 64..96 96..128 128..160 160..192;
            UInt1x8 => 0..1 1..2 2..3 3..4 4..5 5..6 6..7 7..8;
            UInt2x4 => 0..2 2..4 4..6 6..8
        });

        filled.into_iter().flatten()
    }

    /// Extract bit as a big-endian interpretation.
    ///
    /// The highest bit of each byte being the first. Returns a value as `u32` with the same
    /// interpretation where the lowest bits are filled.
    ///
    /// FIXME: there's **a lot** of constant pre-processing. For example, if always access through
    /// either 32-bit boundary or 64-bit boundary then the startu64 is also one of two constants.
    #[inline]
    pub(crate) fn extract_as_lsb<T>(&self, texel: Texel<T>, val: &T) -> u32 {
        // FIXME(perf): vectorized form for all texels where possible.
        // Grab up to 8 bytes surrounding the bits, convert using u64 intermediate, then shift
        // upwards (by at most 7 bit) and mask off any remaining bits.
        let ne_bytes = texel.to_bytes(core::slice::from_ref(val));
        let start_byte = self.begin / 8;
        let from_bytes = &ne_bytes[start_byte.min(ne_bytes.len())..];
        assert!(self.len <= 32);

        let shift = self.begin - start_byte * 8;
        let bitlen = self.len + shift;
        let copylen = bitlen.div_ceil(8);

        let mut be_bytes = [0; 8];
        let initlen = copylen.min(8).min(from_bytes.len());
        be_bytes[..initlen].copy_from_slice(&from_bytes[..initlen]);

        let val = u64::from_be_bytes(be_bytes) >> (64 - bitlen).min(63);
        // Start with a value where the 32-low bits are clear, high bits are set.
        val as u32 & self.mask()
    }

    pub(crate) fn insert_as_lsb<T>(&self, texel: Texel<T>, val: &mut T, bits: u32) {
        // FIXME(perf): vectorized form for all texels where possible.
        let ne_bytes = texel.to_mut_bytes(core::slice::from_mut(val));
        let start_byte = self.begin / 8;
        let bytestart = start_byte.min(ne_bytes.len());
        let texel_bytes = &mut ne_bytes[bytestart..];

        let shift = self.begin - start_byte * 8;
        let bitlen = self.len + shift;
        let copylen = bitlen.div_ceil(8);

        let mut be_bytes = [0; 8];
        let initlen = copylen.min(8).min(texel_bytes.len());
        be_bytes[..initlen].copy_from_slice(&texel_bytes[..initlen]);

        let mask = ((-1i64 as u64) ^ u32::MAX as u64).rotate_left((self.len as u32).min(32))
            & (u32::MAX as u64);

        let bitshift = (64 - bitlen).min(63);
        let newval = (u64::from_be_bytes(be_bytes) & !(mask << bitshift))
            | (u64::from(bits) & mask) << bitshift;

        be_bytes = newval.to_be_bytes();
        texel_bytes[..initlen].copy_from_slice(&be_bytes[..initlen]);
    }
}

#[cfg(test)]
mod tests {
    use super::FromBits;
    use image_texel::AsTexel;

    #[test]
    fn bit_extraction() {
        fn extract_simple(r: core::ops::Range<usize>, val: &u8) -> u32 {
            FromBits {
                begin: r.start,
                len: r.len(),
            }
            .extract_as_lsb(u8::texel(), &val)
        }

        let val = 0b1000_1010u8;
        assert_eq!(extract_simple(0..1, &val), 1);
        assert_eq!(extract_simple(1..2, &val), 0);
        assert_eq!(extract_simple(2..3, &val), 0);
        assert_eq!(extract_simple(6..7, &val), 1);
        assert_eq!(extract_simple(0..7, &val), val as u32 >> 1);
        assert_eq!(extract_simple(1..8, &val), (val & 0x7f) as u32);

        assert_eq!(extract_simple(0..0, &val), 0);
        assert_eq!(extract_simple(1..1, &val), 0);
    }

    #[test]
    fn bit_insertion() {
        // Return the binary diff of insertion.
        fn insert_simple(r: core::ops::Range<usize>, bits: u32, val: &mut u8) -> u8 {
            let before = *val;
            FromBits {
                begin: r.start,
                len: r.len(),
            }
            .insert_as_lsb(u8::texel(), val, bits);
            before ^ *val
        }

        let mut val = 0b1000_1010u8;
        assert_eq!(insert_simple(0..1, 1, &mut val), 0);
        assert_eq!(insert_simple(1..2, 0, &mut val), 0);
        assert_eq!(insert_simple(2..3, 0, &mut val), 0);
        assert_eq!(insert_simple(6..7, 1, &mut val), 0);
        assert_eq!(insert_simple(0..7, val as u32 >> 1, &mut val), 0);
        assert_eq!(insert_simple(1..8, (val & 0x7f) as u32, &mut val), 0);

        assert_eq!(insert_simple(0..0, 0, &mut val), 0);
        assert_eq!(insert_simple(1..1, 0, &mut val), 0);
    }
}
