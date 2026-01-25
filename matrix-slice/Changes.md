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
