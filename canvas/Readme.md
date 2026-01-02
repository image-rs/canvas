# image-canvas

Provides a planar, colored, convertible frame buffer based on `image-texel`.

**Work-In-Progress**: Do not use in production yet. The version is above
`0.*.0` and the API is somewhat stable but will still change. That said, we
will adhere to semantic versioning and most byte-slice-based interfaces won't
be removed as they are the selling point of the library.

## Goals

The buffer should be free of IO, yet provide a rich API for a color image
buffer: that is, a buffer with _correct_ color spaces, color and sample
representations at various bit depths, that can be effectively and efficiently
semantically manipulated on a CPU. Provide a lingua franca for passing portions
of an image to other routines. At some point, we will wrap this into `image`
as a decoding buffer.

The goal for a `1.0` version is further to provide the *same* bit-by-bit
reproducible results on all platforms and ISA combinations for all
representation and conversion functionality. Not sure how feasible this is
while reaching best-in-class performance but we'll try. Not being at the mercy
of hardware texel/shading/interpolation units should have some benefit, right?

This isn't definitive yet, should be considered a rough idea.

## Non-Goals

Do not solve IO, bindings for reading into portions can be written on top of
the exposed methods for mutating and referencing parts of the frame.

Do not provide GPU bindings (but make it easy to translate a layout and prepare
it for copying to a texture buffer).

## Current performance

The benchmarks are measuring conversion of 128x128 pixel images into other
color and texel representations. For comparison's sake, the numbers below are
scaled up to seconds-per-FullHD image (1920×1080 pixels). This isn't exactly
right due to caches but good for future reference.

At 2022-05-19 we have for example:

- sRGB(u8) to sBGR(u8) at 9.84ms/FullHD `= 101 fps`
- sRGBa(u8) to sBGRa(u8) at 8.06ms/FullHD `= 124 fps`
- SrLab2(f32) to sRGBa(f32) at 67.31ms/FullHD `= 14.85 fps`
- sRGB(u8) to Oklab(u8) at 188.58ms/FullHD `= 5.3 fps`
- sRGB(f32) to Oklab(f32) at 79.35ms/FullHD `= 12.60 fps`
