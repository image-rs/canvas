# matrix-slice

A sound way of handling 2D matrix slices for Rust.

```rust
let mut data = vec![
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9,10,11,
];

let matrix = matrix_slice::BlockMut::new(&mut data, 4);
let (mut lhs, mut rhs) = matrix.split_at_col(2);

assert_eq!(lhs[(0, 0)], 0);
assert_eq!(rhs[(0, 1)], 3);

// Completely independent though data address ranges overlap.
lhs[(0, 0)] = rhs[(0, 1)];
assert_eq!(lhs[(0, 0)], 3);

// Safe to send to another thread.
std::thread::scope(|scope| {
    scope.spawn(move || {
        lhs[(1, 1)] = 42;
    });

    scope.spawn(move || {
        rhs[(1, 1)] = 0xbeef;
    });
});

// After their borrow ends, see the changes:
assert_eq!(data[5], 42);
assert_eq!(data[7], 0xbeef);
```

## The problem

If you've handled multi-dimensional arrays in Rust, you have very likely seen
them often around as flat slices with a descriptor of pitch, width and height.
The underlying reason here is a conflict between Rust's aliasing rules and the
structure of a multi-dimensional slice. 

Consider a `3x3` matrix stored in row-major order with pitch `3`:

```text
+---+---+---+
| 0 | 1 | 2 |
+---+---+---+
| 3 | 4 | 5 |
+---+---+---+
| 6 | 7 | 8 |
+---+---+---+
```

How would a 2x2 sub-block starting at element `1` be represented? Note that it
covers at least five elements in the linear memory order. Even though `3` is
not part of the block we always pass over it when we cross from row 0 to row 1.
It turns out this puts some quite fundamental restrictions on our solution
space.

By the language's rules, _references_ impose requirements on the region of
memory they point to: unique references (`&mut T`) require exclusive access to
the memory region, while shared references (`&T`) (usually¹) require an absence
of mutation for the duration of the value. The region tied to a reference is
defined by a start, the address of the pointer, and a length, the size of the
type (or value for unsized `?Sized` types). This is crucially always a
contiguous region, which heavily conflicts with the structure of a sub-block of
a matrix: a list of packed regions each separated by padding according to
difference between the block's and the pitch of the whole matrix. If our block
references asserted their invariants on the space between rows (assuming a
row-major layout), it would not be possible to split a block into *independent*
sub-blocks.

For instance, `nalgebra`'s `ViewStorageMut` chooses this route.

## The solution

The main technique has already been implied by the careful wording above: we
will avoid forming proper references to the data inside each sub-block except
in circumstances where we can guarantee that all covered elements are part of
the block. Instead of the familiar slice methods having a reference on the
outside (`&'lt mut [T]`), the types manipulate a pointer and metadata
internally.

This comes at a cost, mainly we can not provide any proper pointee type. Note
that `& [T]` and `&mut [T]` are two different type _constructors_ (type
families) instantiated with the common slice type proper (`[T]`). Sharing this
underlying type² is a subtle but highly effective tool of providing language
builtin tools such as automatic re-borrowing, formulating traits such as
`Index`, and generally formalizing Rust's semantics. Most regretfully, we can
not write an impl of the `Index` trait that would return our matrix types since
that invariably requires us to return a reference. Those are road bumps however
nothing fundamentally blocking our type from working.

The question of design will be discussed in more detail in the future
documentation.array

## Footnotes

¹² See the [development log](./docs/development_log.md) in the documentation.
