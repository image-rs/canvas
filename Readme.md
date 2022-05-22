# image-canvas

Provides the definition of a *texel* trait and marker type, several buffer and
layout abstractions, and a color aware, flexible buffer on top.

- `texel` contains `image-texel`, the kernel of image buffer and element types
  that are themselves free of policy. By that we mean, it's always possible to
  access bytes directly and to convert to arbitrary layouts with minimal
  reallocation overhead. [![texel-crates]](https://crates.io/crates/image-texel)
  [![texel-docs]](https://docs.rs/image-texel)
- `canvas` contains `image-canvas`, defining color models and conversion to
  provide an opinionated default within the layout framework of `image-texel`.
  [![canvas-crates]](https://crates.io/crates/image-canvas)
  [![texel-docs]](https://docs.rs/image-canvas)
- `drm` contains `image-drm`, a work-in-progress to provide native Rust types
  mirroring each of those available through `libdrm` with the goal of
  simplifying frame buffer interaction.

[texel-crates]: https://img.shields.io/crates/v/image-texel
[canvas-crates]: https://img.shields.io/crates/v/image-canvas
[texel-docs]: https://docs.rs/image-texel/badge.svg
[canvas-docs]: httpshttps://docs.rs/image-canvas/badge.svg

## I just want to see the goods

```bash
cargo run --example --release show-oklab
# Writes an image test.png
cargo run --example --release show-srlab2
# Another image test.png in another color space
```
