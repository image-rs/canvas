# matrix-slice

A sound way of handling 2D matrix slices for Rust.

```rust
let mut 
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
documentation.

## Technical footnotes

¹Technically, the sharing predicate / invariant depends on the pointee type but
access is closely guarded to most code. To crates, stable Rust provides *one*
escape hatch in the form of `UnsafeCell<T>`, which allows shared mutation, and
your own types can only make use of this through composition (fields in
`struct`, `enum`, and on nightly within `union`). Additionally, any operation
that such a custom type implements must still conform to the requirements of
the underlying type's operational semantics and thus sharing predicates if
references to it are involved.

²In contrast to C++ where `T*` and `T const *` are both 'pointers' but have
pointee types that _differ_ in their _qualifiers_. So while they are distinct
on a type-level (including type mismatche shenanigans with `decltype(auto)`) on
an operation semantics level we can always `const_cast` the difference away and
the compiler can distinguish and *optimize* only in a very limited number of
cases. Also programmers can accidentally introduce paradoxical `T const&&`
types in their programs, which contributes significantly to a quirky and
lighthearted work environment. Remember: the clearest way of letting people
know you are taking control of a resource is to explicitly say you're copying
it (/s, [summarized from this gem](https://stackoverflow.com/a/60587511)). Also
you know need to explain what it means to return a `const T` even though
returning is a common way of initializing a totally non-const value. I suppose
[the analogy][Sandor Dargo] of buying a house that you're not allowed to modify
is more hilarious in Germany (see UrhG §1 (1) No. 4).

[Sandor Dargo]: https://www.sandordargo.com/blog/2020/11/18/when-use-const-3-return-types)
