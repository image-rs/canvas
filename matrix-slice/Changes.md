## Version 1.1.1

Fixes:
- Splitting an existing `VecRef` or `VecMut` at its length would offset a
  pointer by too many elements. Since the resulting value has length zero, this
  could not lead to an out-of-bounds pointer dereference but the offset itself
  was technically undefined behavior.

Additions:
- Added `Vec{Ref,Mut}::split_off`.
- Added `IterVec{,Mut}` to access elements of a vector sequentially.

## Version 1.1.0

Additions:
- Added `VecRef` and `VecMut` types for working with strided slices that must
  not be considered aliasing the intermediate elements.
- Added `Block{Ref,Mut}::{row,col}` that constructs the `Vec{Ref,Mut}`.

## Version 1.0.1

Fixes:
- `BlockMut::new` used `NonNull::from_ref` where it should have used
  `NonNull::from_mut` to preserve mutable access.

Additions:
- More documentation
