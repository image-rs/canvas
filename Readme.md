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
