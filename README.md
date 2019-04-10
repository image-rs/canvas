# Canvas [![docs.rs: image-canvas](https://docs.rs/image-canvas/badge.svg)](https://docs.rs/image-canvas) [![Build Status](https://travis-ci.org/image-rs/canvas.svg?branch=master)](https://travis-ci.org/image-rs/canvas) 

An allocated buffer suitable for image data. Compared to a `Vec` of all
samples, it restricts the possible data types but on the other hand focusses on
a more efficient interface to casting the data or rearranging the contents.

## Why

After [some discussion](https://github.com/PistonDevelopers/image/pull/885) it
was concluded that there is likely no safe way to reinterpret the allocation of
a `Vec<T>` for a different `Vec<U>`, except in a very restricted set of cases.
This includes grouping samples in logic pixel structs, even if those are
annotated `#[repr(C)]` or alike. This is hardly the fault of `Vec<_>`, as
images and the necessary transmutations of the representated binary data are
*far* from general but `Vec` must be.

## Not

This library is not yet ready for use, and mostly a reservation in `crates.io`
for future use within the `image-rs` organization managing the `image` crate.

It is also not intended to provide nd-algebra or other matrix style operations.
This firstly keeps the implementation complexity low, allows other
implementation details, and separates concerns. If you nevertheless find it
useful for such purposes, rest asured that we find this incredibly cool even
though we may not accept PRs to introduce additional complexity motivated
solely by such use. Rather, use the code and free license to shape it to those
other needs.

## Todo

* `Rec` and underlying `Buf` need `realloc`
* `Canvas` should implement resizing to smaller image with guaranteed reuse
* `Canvas` mutation to differently sized types blocked by ..
* .. `Rec` logical mutations, keeping `len` but not `byte_len`
* Relaxing internal `Buf` requirements to allow reuse of memory for `f32` (only
  `AsBytes` but not `FromBytes`)
* Use alignment for SIMD iteration/transmutation/map-operation

