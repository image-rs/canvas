# image-texel [![docs.rs: image-texel](https://docs.rs/image-texel/badge.svg)](https://docs.rs/image-texel) [![Build Status](https://github.com/image-rs/canvas/workflows/Rust%20CI/badge.svg)](https://github.com/image-rs/canvas/actions) 

An allocated buffer suitable for image data. Compared to a `Vec` of all
samples, it restricts the possible data types but on the other hand focusses on
a more efficient interface to casting the data or rearranging the contents.

## What it is

The buffer types provided provide several utility operations out of the box, or
they can be used as internal structures on which a more concrete interface is
built.

- As the inner type of opaque image struct that offers conversion.
- As standard forms of particularly shaped image data in interfaces.
- As references to image data with their layout attached.
- As a basis for optimized transform libraries.
- As a mechanism for an FFI interface.

## Why

After [some discussion](https://github.com/image-rs/image/pull/885) it
was concluded that there is likely no safe way to reinterpret the allocation of
a `Vec<T>` for a different `Vec<U>`, except in a very restricted set of cases¹.
This includes grouping samples in logic pixel structs, even if those are
annotated `#[repr(C)]` or alike. This is hardly the fault of `Vec<_>`, as
images and the necessary transmutations of the representative binary data are
*far* from general but `Vec` must be.

¹This was slightly relaxed in Rust 1.61 where few casts such as `Vec<[u8; 3]>`
to `Vec<u8>` are now permitted, and the reverse under a restricted set of
circumstances. This may provide us a good 'out' to perform zero-allocation
conversion into and from a standard vector. It's still far from flexible enough
for most high-performance needs.

## What it is not

It is not optimized to provide nd-algebra or other matrix style operations.
This firstly keeps the implementation complexity low, allows other
implementation details, and separates concerns. If you nevertheless find it
useful for such purposes, rest assured that we find this incredibly cool even
though we may not accept PRs to introduce additional complexity motivated
solely by such use. Rather, use the code and free license to shape it to those
other needs.

## Todo

* Safely wrap SIMD iteration/transmutation/map-operation that are sound under
  the alignment guarantees of the buffer and slice types provided.
