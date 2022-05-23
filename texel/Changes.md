## v0.2.0

* Fix the signature of `Texel::try_to_slice_mut`.

## v0.1.2

* `Texel::for_type` is now a `const fn`, allowing certifying from a `bytemuck`
  trait instance at compile time.
* The alignment&size guarantee for `x86`/`x86_64` is now `32`.
* Added other architectures: `arm`, `aarch64`, `wasm32`.
* Added impls of `AsTexel` for all stable primitive vector types of architectures.

## v0.1.1

* Implement `AsTexel` for `[T; 5..8]` where `T: AsTexel`.

## v0.1.0

* No large changes from 0.0.6 in terms of API but all texels are now required
  to be non-ZST. A few utility constructors for different texel conversions.

## v0.0.6

* Add mapping operations over pixels of a buffer or canvas
* Fix a bug that caused reallocations to be too short

## v0.0.5

* Update `zerocopy` to a version supporting `f32`
