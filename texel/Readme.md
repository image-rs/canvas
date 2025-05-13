# image-texel [![docs.rs: image-texel](https://docs.rs/image-texel/badge.svg)](https://docs.rs/image-texel) [![Build Status](https://github.com/image-rs/canvas/workflows/Rust%20CI/badge.svg)](https://github.com/image-rs/canvas/actions) 

An allocated buffer suitable for image data. Compared to a `Vec` of all
samples, it restricts the possible data types but on the other hand focusses on
a more efficient interface to casting the data or rearranging the contents.

## What it is

The goal is to create a comprehensive library that supplements the standard
`alloc` library specifically for texture buffers, bags of bytes which are
mostly dynamically typed with regards to the data types and layout they
contain. The main problem is the ability to convert from this weak typing to
stronger typing needed for efficient and custom computation in ad-hoc settings.

This library is therefore strictly `no_std` (but with `extern crate alloc`).

This crate defines special buffers around `Texel`s. With that term we mean a
non-empty, plain-old-data type with reasonable alignment, i.e. not needing to
be aligned to a page but aligned to about the largest register size. Since we
can interact with such values on a byte-for-byte level we provide interactions
with byte slices, cells, and atomics, polymorphic buffers that can be
reinterpreted with almost no cost, and layout-imbued image containers.

## Motivation

As `wgpu` discovered early on, it isn't a good idea to try encode too much of
dynamic information such as layouts of textures into the type system. Despite
this realization, most libraries interacting with image data define structs
such as `Rgb` or `Luma` and their buffers in terms of those. And to complicate
matters further, these are often not interchangeable. Thus a host of ad-hoc
casting with `bytemuck` or, worse, `unsafe` code ensues with anything from
dubious to broken effect.

To avoid engaging with the rather Sisyphean task of unifying the ecosystem's
representation of pixel data and enumerating uncountable different texel
formats across GPU vendors etc we let you bring your own type. You describe the
texels and your layout only as much as necessary for the containers to work,
and the library utilizes the extra knowledge of your data being `Copy` to run
on bytes.

## What it is not

It is not optimized to provide nd-algebra or other matrix style operations.
This firstly keeps the implementation complexity low, allows other
implementation details, and separates concerns. If you nevertheless find it
useful for such purposes, rest assured that we find this incredibly cool even
though we may not accept PRs to introduce additional complexity motivated
solely by such use. Rather, use the code and free license to shape it to those
other needs.

It is also not an opinionated interface. This crate offers the general
mechanisms for safely and soundly dealing with texel buffers. The user API for
handling real world data, including image metadata, color interpretation, is
left to a higher level crate.

## Todo

* Safely wrap SIMD iteration/transmutation/map-operation that are sound under
  the alignment guarantees of the buffer and slice types provided. `std::simd`
  is a good attempt at this and we may mostly provide interoperability in the
  form of `Texel` constants.
