## Alternative design: multi-dimensional slices

We limit ourselves to 2-dimensional slices here. This avoids two major
complications. _Potential_ allocations within the 'tag' portion of the
reference would make this metadata incompatible with `Copy` semantics, required
for the unstable custom-unsized type design that we want to be compatible with
in the future, as well many forms of `const` manipulation, which forbid
dropping generally. Additionally, higher-dimensional slices would require us to
be more cautious with manipulations that are trivial for 2-element vectors but
may be non-(log)-linear in higher dimensions. All that would introduce
additional decision points and jump that likely affect codegen. We rather solve
_one_ problem well at a time.

## Alternative design: strided slices

A more general solution to the problem of multi-dimensional slices is to use
two separate explicit strides for both dimensions of the matrix. The position
of an element is then computed by the inner product of the index vector and the
stride vector (or a homogeneous version with an offset). This representation
makes transposition trivial.

However, the flexible stride in both direction implies that we can not rely on
either direction having contiguous elements. (That is, to a degree, the point).
We could now longer offer an conversion to row iterator of element slices,
which also implies quite some performance penalties for all operations that
would be dependent on it. This more general design may be worth exploring in
*addition* to the denser matrix design.

## Alternative design: byte strides

Instead of storing the offset of a block in number of elements we could also
store all strides as a number of bytes. The pointer offset operation would work
practically the same, just moving the scaling operation. We would have an extra
_invariant_ to maintain as all references must be well-aligned which is not
guaranteed by arbitrary byte offsets. As a consequence, the type used for the
pointer tag is not by itself a sufficient description. Its validity would also
depend on the element type `T` for which it is used.

We'd either want to provide a different generic-over-`T` tag type or need to
annotate several operations with the type. This would complicate the design and
split up the responsibility of maintaining the `unsafe` invariants across more
locations. Hence we stay with element-based offsets which mirrors the standard
library choice of representing slice lengths as `usize` in terms of element
count; not in terms of memory size.

Still this design may be worth exploring. Several real-world matrix types pad
their rows in a way that they are *not* guaranteed to be multiples of their
element size, but highly aligned. For instance, a row of `[f32; 3]` elements
may always be padded to a multiple of 4 `f32` elements. This pseudo-3d matrix
appears in several image processing contexts and byte strides are the more
pragmatic way of approach it, rather than making the matrix generic over both
element and underlying type.

## Additional design: sparse matrices

Sparse matrices are very often expressed as a set of three arrays (CSC format):
a slice of raw elements, a slice of row indices, and a set of column offsets.
Accessing an element uses two indirection lookups, first we compute the number
and index of elements in a column from the column offsets at `i` and `i+1`,
then retrieve the sorted row indices and corresponding raw elements by indexing
the two other matrices with the offset range.

Slicing such a matrix into blocks is possible as long as the set of present
elements can not change. Computing the range of allowed elements for a column
is a binary-search within the row indices to determine the lower and upper
bound indices. We must not slice into the element array with any larger indices
to avoid undefined mutable aliasing with other blocks. As an iterator design
this might make sense, very similar to the design of a dense slice. Having no
modification for data indices is a bummer, though insertion and removal being
separate steps comes with the representation so this may be fine.

An additional trickiness here would be the types allowed for the index slices.
While we might natively want to default to `usize`, smaller and
platform-independent types such as `u32` or even `u16` will be desirable in
practice. Unlike the sized length tuple we can not _expect_ a conversion to
always be possible though. Hence the type would need to be generic and that
combination of generic types with critical safety requirements to their value
interpretation would warrant extra care in the implementation. We can't trust
user-chosen types to maintain the sortedness property critical to row
non-aliasing without an `unsafe` or sealed trait—neither of which appear to be
super ergonomic choices in the interface. If you need to have `unsafe` lines
anyways, the motivation to this library goes down quite a bit.

## Footnotes

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
on a type-level (including type mismatch shenanigans with `decltype(auto)`) on
an operation semantics level this is more permissive than the Rust model, and
we can `const_cast` the difference away as well as undo its recursiveness for
fields by setting them `mutable` (only class fields though, not arbitrary
places such as parts of an array). Hence, the compiler can distinguish and
*optimize* only in a very limited number of cases. Also programmers can
accidentally introduce paradoxical `T const&&` types in their programs, which
contributes significantly to a quirky and lighthearted work environment.

The mantra seems to be: the clearest way of letting people know you are taking
control of a resource is to explicitly say you're copying it (/s, [summarized
from this gem](https://stackoverflow.com/a/60587511)). Also you know need to
explain what it means to return a `const T` as a separate type from returning a `T` even though
returning is maybe the most common way of initializing a totally non-const value. I suppose
[the analogy][Sandor Dargo] of buying a house that you're not allowed to modify
is more hilarious in Germany (see UrhG §1 (1) No. 4).

[Sandor Dargo]: https://www.sandordargo.com/blog/2020/11/18/when-use-const-3-return-types
