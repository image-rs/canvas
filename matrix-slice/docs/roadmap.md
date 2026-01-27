This document is a wishlist (and note of completion) for features. I guess you
could create a PR against the repository to add items in an RFC style? Items
that are completed or have a design should link to a section further down.

## Utils for matrices

- [ ] `BlockMut`: `rotate` for rotating rows/columns.
- [ ] `BlockMut`: `flip` for flipping rows/columns.
- [x] `BlockMut`: `fill` for efficient initialization.
- [ ] `BlockMut`: `copy_within` which will take some design to make sure the
  interface is not overloaded with parameters.
- [ ] `BlockMut`: `swap`.
- [x] `BlockMut`: `copy_from_block(BlockRef)`, `swap_with_block`.
- [ ] `Block`: `split_*_unchecked` methods that forward the safety requirement to
  the caller.
- [ ] `Block`: `split_{first,last}` to separate into a respective `Vec` and
  block. The same for their `split_off` variants of course.
- [ ] `Block`: additional interactions with arrays for `split_first`.
- [ ] `Block`: `get_disjoint_mut` variants (`1.86` stabilized the error type
  `core::slice::GetDisjointMutError`)

## Utils for vectors

- [ ] `Vec`: `as_array -> Option<&[T; N]>`.
- [ ] `Vec`: `copy_within` which will take some design to make sure the
  interface is not overloaded with parameters.
- [ ] `Vec`: `swap`.
- [ ] `Vec`: `copy_from_vec(VecRef)`, `swap_with_vec(VecMut)`.
- [ ] `Vec`: `copy_from_slice(&[T])`, `swap_with_slice(&mut [T])`.

## New matrix descriptors

- [ ] Dense matrix with two strides that can be flipped.
- [ ] Dense matrix with byte strides.
- [ ] Sparse matrix in CSC or CSR format (only elements mutable), lots of
  design work though be prepared for iteration.
