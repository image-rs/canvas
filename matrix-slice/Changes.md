## Version 1.0.1

Fixes:
- `BlockMut::new` used `NonNull::from_ref` where it should have used
  `NonNull::from_mut` to preserve mutable access.

Additions:
- More documentation
