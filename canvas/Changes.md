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
