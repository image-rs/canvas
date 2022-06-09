## v0.3.0

- UB: Fix `Texel::{try_to_slice_mut,cast_mut_bytes}` casting a non-mutable
  pointer to a mutable slice. This is a local error, the input parameter was
  properly mutable. Commit: 67b5bc9f54f2d348415d52aee34ee9b224860bc5
  No LLVM analysis *can* make use of this UB / unsoundness as of LLVM 13.
- `{Image,ImageRef,ImageMut,TexelBuffer}::{as_texels,as_mut_texels}` now return
  a slice truncated to the portion covering the logical size of the respective
  buffers. This is breaking as it required their layouts to be `Layout`.

## v0.2.1

- Added `Image{Ref,Mut}::split_layout` to re-use excess storage with a
  different layout, enabling splitting an image into separated planes. This
  enforces splitting at a maximum alignment boundary.

## v0.2.0

- Fix the signature of `Texel::try_to_slice_mut`.

## v0.1.2

- `Texel::for_type` is now a `const fn`, allowing certifying from a `bytemuck`
  trait instance at compile time.
- The alignment&size guarantee for `x86`/`x86_64` is now `32`.
- Added other architectures: `arm`, `aarch64`, `wasm32`.
- Added impls of `AsTexel` for all stable primitive vector types of architectures.

## v0.1.1

- Implement `AsTexel` for `[T; 5..8]` where `T: AsTexel`.

## v0.1.0

- No large changes from 0.0.6 in terms of API but all texels are now required
  to be non-ZST. A few utility constructors for different texel conversions.

## v0.0.6

- Add mapping operations over pixels of a buffer or canvas
- Fix a bug that caused reallocations to be too short

## v0.0.5

- Update `zerocopy` to a version supporting `f32`
