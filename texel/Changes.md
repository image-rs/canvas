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
