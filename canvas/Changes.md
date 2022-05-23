## 0.3.0

- Added u8x4 and u16x4 shuffle implementations for SSSE3 and AVX2.
- Speedups for everything, mostly the u8x4 case though because of above.
- Bump `image-texel` to `0.2` to fix an API oversight.

## 0.2.4

- Fixed a bug in conversion to integer formats that would mask some bits.
- Conversion from int-array-to-int-array representations within the same color
  space (e.g. bgra-u8 to rgba-u8 or bgra-u8 to rgb-u8) are now much faster. The
  converter has learned to elide some copies into the texel buffer if the
  canvas already contains the right contiguous texel representation.

## 0.2.3

- Fixed a bug with the texel coordinate generation phase that caused issues in
  non-rectangular images.

## 0.2.2

- Performance optimizations, all around. Some conversions are 100% faster.
- The Readme now contains a performance notice, will be updated irregularly.

## 0.2.1 (Yavin IV)

- Rename `SampleParts::Int*` to `SampleParts::UInt*`.
- Added proper `SampleParts::Int*` as signed integers.
- Added `SampleParts::with_yuv_*`. Those don't work yet and their colors are
  not yet implemented but are to be used for all your YUV-needs.
- This is the second notable release in a row because colors work. See example
  `show-oklab`, you can do proper art now. Float-sRGB buffers included for free
  but less tested.

## 0.2.0 (Tatooine)

- A new beginning, this is now a buffer for a colored frame.
- Previous crate was moved to `image-texel`.
- I'm doing a thing where all notable versions get a name. Choosing Star Wars
  planets, because why not.
