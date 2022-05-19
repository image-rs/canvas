# image-canvas

Provides the definition of a *texel* trait and marker type, several buffer and
layout abstractions, and a color aware, flexible buffer on top.

- `texel` contains `image-texel`, the kernel of image buffer and element types
  that are themselves free of policy. By that we mean, it's always possible to
  access bytes directly and to convert to arbitrary layouts with minimal
  reallocation overhead.
- `canvas` contains `image-canvas`, defining color models and conversion to
  provide an opinionated default within the layout framework of `image-texel`.
- `drm` contains `image-drm`, a work-in-progress to provide native Rust types
  mirroring each of those available through `libdrm` with the goal of
  simplifying frame buffer interaction.

## I just want to see the goods

```bash
cargo run --example --release show-oklab
# Writes an image test.png
cargo run --example --release show-srlab2
# Another image test.png in another color space
```

## Current performance

The benchmarks are measuring conversion of 128x128 pixel images into other
color and texel representations. For comparison's sake, the numbers below are
scaled up to seconds-per-FullHD image (1920×1080 pixels). This isn't exactly
right due to caches but good for future reference.

At 2022-05-19 we have for example:

- SrLab2(f32) to sRGBa(f32) at 70.49ms/FullHD
- sRGBa(u8) to Oklab(u8) at 188.58ms/FullHD
- sRGBa(f32) to Oklab(f32) at 81.86ms/FullHD
