# v0.5.0

For the primitive buffers:
- `cell_buf` can be indexed by a `TexelRange<P>` for `&Cell<P>`.
- `cell_buf` can now be compared against `[u8]`.
- `atomic_buf` can be indexed by a `TexelRange<P>` for `AtomicSliceRef<P>`.
- `AtomicRef` gained `store` and `load`, mirroring methods on `Cell`.
- `AtomicSliceRef<P>` can be efficiently loaded into an owned `Vec<P>` or an
  aligned `TexelBuffer<P>`.

For images:
- `Decay` is now presumed to result in a layout that is at most the size of the
  previous layout, as documented. The signature of functions no longer return
  an `Option`, panicking instead on a violation. A new set of `checked_decay`
  makes it consistently opt-in to verify this property.
- `as_{capacity_,}{,cell_,atomic_}buf` are now consistently available on images
  and their reference equivalents.
- Shared and reference image types gained `into_owned` to convert them into an
  owned `Image<L>` buffer with the same layout.
- `CellImage` now exposes `as_cell_buf` and `as_capacity_cell_buf`.
- `AtomicImage` now exposes `as_atomic_buf`

The `image::data` module:
- Added this module to interface between aligned image buffers and unaligned
  external byte buffers. You can wrap `&[u8]`, `&mut [u8]` and `&[Cell<u8>]` to
  transfer data between them.
- Images have an `assign` method for transferring data _and_ layout. These are
  fallible on all images other than `Image`, which can always reallocate its
  buffers instead.
- All images have `as_source` and `as_target` to treat them as unaligned slices
  for the sake of interfaces specified as such. Copying relies on dynamic
  dispatch, so these share the exact type of unaligned data buffers.

Documentation:
- received a major overhaul and a preface overview of the library.

# v0.4.0

- Add `AtomicImage`, `CellImage` and several supporting types and interfaces.
  These replicate large parts of the `Image` and `ImageMut` types but allow for
  operations on a shared buffer, wrapping `Arc` and `Rc` respectively.
- Added `PlaneOf` trait, a relationship between a layout, and index, and a
  subcomponent of the layout. All image reference types implement a form of
  `into_planes` that splits them apart by those components.
- Added a `Relocate` trait for layouts that can be move to another location
  in a buffer.

## v0.3.1

- Fix compilation on non-explicit targets, where the maximum alignment will
  default to a standard of `8`. This affects the Tier 1 targets: `mips`,
  `powerpc`, `s390x` and others.

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
