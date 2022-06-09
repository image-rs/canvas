## 0.4.0

This adds support for packed blocks and incomplete YUV color support.

Bug fixes:
- Fixed a buf where bitfield sample parts referred to the wrong bits.

New features:
- Added `Block::Pack1x{2,4,8}` which refer to texel containing multiple pixels
  by bit-packing their sample parts in sequence.
- Added `Block::Yuv422,Yuy2,Yuv411` which refer to texels containing multiple
  pixels by subsampling some of their sample parts. Their implementation with
  regards to bit packing and unpacking is not yet final.
- Added `SampleParts::{UInt1x8,UInt2x4}` to complement bit packing blocks with
  more low-depth channels in a single texel.
- More information in `LayoutError` for debugging.
- Added a hidden `Color::Yuv`, not part of the official SemVer interface yet.

## 0.3.1 (Mustafar)

This release is complete enough to provide all relevant RGB-Lab interaction.

Bug fixes:
- Fixed having default values for missing color channels. Converting `Rgb` to
  `Rgba` now assigns a value corresponding to `1.0` to the alpha channel,
  instead of `NaN`.

New features:
- Added `Canvas::planes_mut` as an interface to split it into multiple planes.
- Added `{BytePlaneRef, BytePlaneMut}::{to_owned, to_canvas}` to get owning
  representations of singular color planes.
- Export containers related to `Canvas` in a new public `canvas` module.
- Added `{Canvas, *}::layout` for container types, returning references to
  their respective layout types.
- Exposed `Channels{Ref,Mut}` as a strongly typed reference to arrays of
  uniform color channels. These offer `{as,mut,into}_slice` methods for direct
  access to individual channel values.
- Added `From<Plane>` for `Canvas`.
- Added `SampleParts::has_alpha` to check for any alpha channel.
- Added `color` and `texel` getter for `CanvasLayout`.
- The `BytePlane` type is now `BytePlaneRef` (hidden alias keeps this from
  being a breaking change).


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
