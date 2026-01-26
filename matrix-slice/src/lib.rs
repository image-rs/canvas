//! Implements references to blocks of a matrix.
//!
//! Consider a reference to a slice (sometimes also just called slice). Typically, it is created by
//! unsizing a reference to an array through type coercion, such as sing `&[0, 1, 2]` in a
//! parameter to function that takes `&[u32]` which turns a reference `&[u32; 3]` to a slice
//! `&[u32]`. The length of the slice, controlling the number of elements and thus the provenance
//! of access that the reference allows, is stored in a tag alongside the pointer to its elements
//! and initialized from the known length of the array. Since this tag is a runtime value we can
//! manipulate it while upholding the invariants required by the type system.
//!
//! This is analogous to that but for blocks of a matrix. A block is a rectangular region of a
//! matrix where the matrix provides an underlying pitch (or stride) between rows and a total
//! number of elements and the block the number of rows and columns that are spanned, i.e. are
//! allowed to be accessed by (mutable) reference.
//!
//! ## Treatment of empty blocks
//!
//! A block may have zero rows or zero columns. In either case the block is empty and provides no
//! access to any elements yet will still return an empty slice for some operations that would
//! otherwise access multiple elements. The memory address of such a block is **not** necessarily
//! at its expected location but it will be in-bounds of the underlying matrix data.
//!
//! Consider the bottom right `2x2` block of a row-major `3x3` matrix.
//!
//! ```text
//! +---+---+---+
//! | x | x | x |
//! +---+---+---+
//! | x | 4 | 5 |
//! +---+---+---+
//! | x | 7 | 8 |
//! +---+---+---+
//! ```
//!
//! This block has a pitch of 3 but only spans 2 columns. If we would naively calculate the address
//! of its past-the-end element we would get an element below `7` which is out-of-bounds. Hence if
//! we split this at row 2, into itself and an empty `0x2` block, the latter block's data pointer
//! would be created with undefined behavior. Instead, we sacrifice the ability to 'locate' such
//! empty blocks and instead have them point at an arbitrary (empty) in-bounds slice within the
//! matrix. (Currently, that is the start of the block from which it was created).
#![no_std]
use core::{cell::Cell, fmt, marker::PhantomData, ops, ptr::NonNull};

/// The Readme of the crate and links to further documentation.
///
/// Note: This module only exists on `cfg(doc)` builds, do not refer to it.
///
#[cfg(doc)]
#[doc = include_str!("../Readme.md")]
pub mod docs {
    /// A discussion of the approach, alternatives, trade-offs and context.
    ///
    #[doc = include_str!("../docs/development_log.md")]
    pub const DEVELOPMENT_NOTES: () = ();

    /// Documentation of each released version.
    ///
    #[doc = include_str!("../Changes.md")]
    pub const CHANGELOG: () = ();
}

/// Create a block reference from a full matrix represented as an array of rows.
///
/// # Examples
///
/// ```
/// let data = &mut [
///    [0, 1, 2],
///    [3, 4, 5],
/// ];
///
/// let mut block = matrix_slice::from_array_rows(data);
///
/// assert_eq!(block.rows(), 2);
/// assert_eq!(block.cols(), 3);
///
/// assert_eq!(block[(1, 1)], 4);
/// ```
pub fn from_array_rows<'a, T, const N: usize>(data: &'a [[T; N]]) -> BlockRef<'a, T> {
    BlockRef {
        block: BlockSlice {
            rows: data.len(),
            cols: N,
            pitch: N,
        },
        data: NonNull::from_ref(data).cast(),
        lifetime: PhantomData,
    }
}

/// A reference to a block of a matrix with shared access to elements.
#[derive(Copy, Clone)]
pub struct BlockRef<'a, T> {
    data: NonNull<T>,
    block: BlockSlice,
    lifetime: PhantomData<&'a [T]>,
}

// SAFETY: See `&[T]`. The reference can be used to, potentially, get a `&T` for each element in
// the block and thus the block itself provides the exact same properties as `T`. The `BlockRef` is
// then `&[T]` itself and thus has properties of a reference to such a type. Refer to the
// reference: <https://doc.rust-lang.org/stable/std/primitive.reference.html>
//
// We have `&T: Sync` iff `T: Sync`
unsafe impl<T> Sync for BlockRef<'_, T> where T: Sync {}
// We have `&T: Send` iff `T: Sync`
unsafe impl<T> Send for BlockRef<'_, T> where T: Sync {}

const _: () = {
    // We can coerce a block to a shorter lifetime.
    fn _coerce_block<'a, 'b: 'a, T>(v: BlockRef<'b, T>) -> BlockRef<'a, T> {
        v
    }

    // We can coerce a reference to a block to a shorter lifetime.
    fn _coerce_covariant<'lt, 'a, 'b: 'a, T>(v: &'lt BlockRef<'b, T>) -> &'lt BlockRef<'a, T> {
        v
    }

    fn _coerce_covariant_fn<'lt, 'a, 'b: 'a, T>(v: fn(BlockRef<'a, T>)) -> fn(BlockRef<'b, T>) {
        v
    }

    fn _coerce_item_covariant<'lt, 'a, 'b: 'a, T>(v: BlockRef<'lt, &'b T>) -> BlockRef<'lt, &'a T> {
        v
    }
};

/// Creates an empty block reference, within a matrix of a dangling slice.
impl<T> Default for BlockRef<'_, T> {
    fn default() -> Self {
        from_array_rows::<T, 0>(&[])
    }
}

impl<'data, T> BlockRef<'data, T> {
    /// Create a new block reference from a raw slice and pitch.
    ///
    /// The resulting block refers to the whole matrix.
    ///
    /// # Panics
    ///
    /// Panics if the length of `data` is not a multiple of `pitch`.
    pub fn new(data: &'data [T], pitch: usize) -> Self {
        assert!(data.len().is_multiple_of(pitch));

        BlockRef {
            block: BlockSlice {
                rows: data.len() / pitch,
                cols: pitch,
                pitch,
            },
            data: NonNull::from_ref(data).cast(),
            lifetime: PhantomData,
        }
    }

    /// Number of rows in this block.
    pub fn rows(&self) -> usize {
        self.block.rows
    }

    /// Number of columns in this block.
    pub fn cols(&self) -> usize {
        self.block.cols
    }

    /// Divide into two blocks at the given column.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &[
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows(data);
    /// let (left, right) = block.split_at_col(2);
    ///
    /// assert_eq!(left[(1, 0)], 3);
    /// assert_eq!(right[(1, 0)], 5);
    /// ```
    pub fn split_at_col(self, mid: usize) -> (BlockRef<'data, T>, BlockRef<'data, T>) {
        self.split_at_col_checked(mid).unwrap()
    }

    /// Divide into two blocks at the given column.
    ///
    /// See [`Self::split_at_col`] but returns `None` if out of bounds.
    pub fn split_at_col_checked(
        self,
        mid: usize,
    ) -> Option<(BlockRef<'data, T>, BlockRef<'data, T>)> {
        if let Some((lhs, rhs, offset)) = self.block.split_at_col(mid) {
            Some((
                BlockRef {
                    data: self.data,
                    block: lhs,
                    lifetime: self.lifetime,
                },
                BlockRef {
                    data: unsafe { self.data.add(offset) },
                    block: rhs,
                    lifetime: self.lifetime,
                },
            ))
        } else {
            None
        }
    }

    /// Divide into two blocks at the given row.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &[
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows(data);
    /// let (top, bot) = block.split_at_row(1);
    ///
    /// assert_eq!(top[(0, 2)], 2);
    /// assert_eq!(bot[(0, 2)], 5);
    /// ```
    pub fn split_at_row(self, mid: usize) -> (BlockRef<'data, T>, BlockRef<'data, T>) {
        self.split_at_row_checked(mid).unwrap()
    }

    /// Divide into two blocks at the given row.
    ///
    /// See [`Self::split_at_row`] but returns `None` if out of bounds.
    pub fn split_at_row_checked(
        self,
        mid: usize,
    ) -> Option<(BlockRef<'data, T>, BlockRef<'data, T>)> {
        if let Some((lhs, rhs, offset)) = self.block.split_at_row(mid) {
            Some((
                BlockRef {
                    data: self.data,
                    block: lhs,
                    lifetime: self.lifetime,
                },
                BlockRef {
                    data: unsafe { self.data.add(offset) },
                    block: rhs,
                    lifetime: self.lifetime,
                },
            ))
        } else {
            None
        }
    }

    /// Choose a single row and refer to its data.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &[
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows(data);
    /// let row = block.row(1);
    /// assert_eq!(row[0], 3);
    /// ```
    pub fn row(self, row: usize) -> VecRef<'data, T> {
        let (_, block, offset) = self.block.split_at_row(row).unwrap();
        assert!(block.rows >= 1);

        VecRef {
            block: VectorSlice {
                count: block.cols,
                pitch: 1,
            },
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        }
    }

    /// Choose a single column and refer to its data.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &[
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows(data);
    /// let row = block.col(1);
    /// assert_eq!(row[0], 1);
    /// ```
    pub fn col(self, col: usize) -> VecRef<'data, T> {
        let (_, block, offset) = self.block.split_at_col(col).unwrap();
        assert!(block.cols >= 1);

        VecRef {
            block: VectorSlice {
                count: block.rows,
                pitch: block.pitch,
            },
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        }
    }

    /// Choose a range of rows and contract the block to that.
    ///
    /// The argument type is flexible, allowing ranges (`1..3`), half open ranges (`2..` and `..2`)
    /// among others. See the [`MatrixIndex`] trait, which is sealed though as its details are not
    /// yet finalized.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &[
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows(data);
    ///
    /// let center = block.select_rows(1..2).unwrap();
    /// assert_eq!(center.rows(), 1);
    /// assert_eq!(center.cols(), 3);
    /// assert_eq!(center[(0, 1)], 4);
    /// ```
    pub fn select_rows<R>(self, range: R) -> Option<BlockRef<'data, T>>
    where
        R: MatrixIndex,
    {
        let (start, len) = range.into_start_and_len(self.block.rows)?;
        let (_, block, offset) = self.block.split_at_row(start)?;
        // Safety: ensures that the resulting block is more constrained, this property should be
        // ensured by our sealed `MatrixIndex` implementations.
        assert!(block.rows >= len);

        Some(BlockRef {
            block: BlockSlice { rows: len, ..block },
            // SAFETY: offset is in-bounds as per `split_at_row` contract.
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        })
    }

    /// Choose a range of columns and contract the block to that.
    ///
    /// The argument type is flexible, allowing ranges (`1..3`), half open ranges (`2..` and `..2`)
    /// among others. See the [`MatrixIndex`] trait, which is sealed though as its details are not
    /// yet finalized.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &[
    ///     [0, 1],
    ///     [3, 4],
    ///     [6, 7],
    /// ];
    ///
    /// let mut block = matrix_slice::from_array_rows(data);
    ///
    /// assert!(block.reborrow().select_cols(1..).is_some_and(|b| b.cols() == 1));
    /// assert!(block.reborrow().select_cols(2..).is_some_and(|b| b.cols() == 0));
    /// assert!(block.reborrow().select_cols(3..).is_none());
    /// ```
    pub fn select_cols<R>(self, range: R) -> Option<BlockRef<'data, T>>
    where
        R: MatrixIndex,
    {
        let (start, len) = range.into_start_and_len(self.block.cols)?;
        let (_, block, offset) = self.block.split_at_col(start)?;
        debug_assert!(block.cols >= len);

        Some(BlockRef {
            block: BlockSlice { cols: len, ..block },
            // SAFETY: `split_at_col` guarantees that `offset` is in-bounds of the
            // provenance tracked by `self.block`, which is in-sync with the pointer field
            // (but we not necessary access these without synchronization). By extension
            // of being in-bounds of the allocation the offset does not overflow `isize`.
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        })
    }

    /// Choose a sub-block by its range of rows and columns.
    pub fn select(
        self,
        row_range: impl MatrixIndex,
        col_range: impl MatrixIndex,
    ) -> Option<BlockRef<'data, T>> {
        let block = self.select_rows(row_range)?;
        block.select_cols(col_range)
    }

    /// Extract a contiguous underlying slice of elements if the block is contiguous.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &[[0u32; 3]; 3];
    /// let block = matrix_slice::from_array_rows(data);
    ///
    /// let (block, _) = block.split_at_row(2);
    /// assert!(block.into_contiguous_slice().is_some());
    ///
    /// let (pre, post) = block.split_at_col(2);
    /// assert!(pre.into_contiguous_slice().is_none());
    /// assert!(post.into_contiguous_slice().is_none());
    ///
    /// let (same, _) = block.split_at_col(3);
    /// assert!(same.into_contiguous_slice().is_some());
    /// ```
    pub fn into_contiguous_slice(self) -> Option<&'data [T]> {
        if let Some(items) = self.block.contiguous_span() {
            Some(unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast(), items) })
        } else {
            None
        }
    }

    /// Turn this into a slice of the first row, assuming it is at most one row.
    fn fake_contiguity(mut self) -> &'data [T] {
        self.block.fake_contiguity();
        self.into_contiguous_slice().unwrap()
    }

    /// Extract access as a slice of arrays if the block is contiguous.
    ///
    /// The caller must choose `N` matching the number of columns.
    pub fn into_array_rows_checked<const N: usize>(self) -> Option<&'data [[T; N]]> {
        if self.block.cols == self.block.pitch && self.block.cols == N {
            Some(unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast(), self.block.rows) })
        } else {
            None
        }
    }

    /// Iterate over the rows of this block.
    pub fn iter_rows(self) -> IterRows<'data, T> {
        IterRows { block: self }
    }

    /// Create a reference to this block with a shorter lifetime.
    pub fn reborrow(&self) -> BlockRef<'_, T> {
        BlockRef {
            data: self.data,
            block: self.block,
            lifetime: PhantomData,
        }
    }
}

impl<T> ops::Index<(usize, usize)> for BlockRef<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let idx = self.block.in_bounds_offset(index.0, index.1);
        // SAFETY: Index is bounded by `total_span` which itself is a lower estimate of the
        // provenance of the pointer.
        unsafe { &*self.data.as_ptr().add(idx) }
    }
}

impl<T: fmt::Debug> fmt::Debug for BlockRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.reborrow().iter_rows()).finish()
    }
}

/// Create a mutable block reference from a full matrix represented as an array of rows.
///
/// # Examples
///
/// ```
/// let data = &mut [
///    [0, 1, 2],
///    [3, 4, 5],
/// ];
///
/// let mut block = matrix_slice::from_array_rows_mut(data);
///
/// assert_eq!(block.rows(), 2);
/// assert_eq!(block.cols(), 3);
///
/// block[(1, 1)] = 42;
///
/// assert_eq!(data[1][1], 42);
/// ```
pub fn from_array_rows_mut<'a, T, const N: usize>(data: &'a mut [[T; N]]) -> BlockMut<'a, T> {
    BlockMut {
        block: BlockSlice {
            rows: data.len(),
            cols: N,
            pitch: N,
        },
        data: NonNull::from_mut(data).cast(),
        lifetime: PhantomData,
    }
}

/// A reference to a block of a matrix with unique access to elements.
pub struct BlockMut<'a, T> {
    data: NonNull<T>,
    block: BlockSlice,
    lifetime: PhantomData<&'a mut [T]>,
}

// SAFETY: See `BlockRef` but with `&mut [T]`.
//
// We have `&mut T: Sync` iff `T: Sync`
unsafe impl<T> Sync for BlockMut<'_, T> where T: Sync {}
// We have `&mut T: Send` iff `T: Send`
unsafe impl<T> Send for BlockMut<'_, T> where T: Sync {}

/// ```compile_fail
/// use matrix_slice::BlockMut;
///
/// // This coercion must *not* be possible. The field `lifetime` ensures the right variance.
/// fn _coerce_item_not_covariant<'lt, 'a, 'b: 'a, T>(
///     v: BlockMut<'lt, &'b T>,
/// ) -> BlockMut<'lt, &'a T> {
///     v
/// //  ^ function was supposed to return data with lifetime `'b` but it is returning data with lifetime `'a`
/// }
///
/// ```compile_fail
/// use matrix_slice::BlockMut;
///
/// fn _copy_block(v: BlockMut<'_, u32>) -> [BlockMut<'_, u32>; 2] {
///    [v, v]
/// }
const _: () = {
    // We can coerce a block to a shorter lifetime.
    fn _coerce_block_mut<'a, 'b: 'a, T>(v: BlockMut<'b, T>) -> BlockMut<'a, T> {
        v
    }

    // We can coerce a reference to a block to a shorter lifetime.
    fn _coerce_covariant<'lt, 'a, 'b: 'a, T>(v: &'lt BlockMut<'b, T>) -> &'lt BlockMut<'a, T> {
        v
    }

    fn _coerce_covariant_fn<'lt, 'a, 'b: 'a, T>(v: fn(BlockMut<'a, T>)) -> fn(BlockMut<'b, T>) {
        v
    }
};

/// Creates an empty block reference, within a matrix of a dangling slice.
impl<T> Default for BlockMut<'_, T> {
    fn default() -> Self {
        from_array_rows_mut::<T, 0>(&mut [])
    }
}

impl<'data, T> BlockMut<'data, T> {
    /// Create a new block reference from a raw slice and pitch.
    ///
    /// The resulting block refers to the whole matrix.
    ///
    /// # Panics
    ///
    /// Panics if the length of `data` is not a multiple of `pitch`.
    pub fn new(data: &'data mut [T], pitch: usize) -> Self {
        assert!(data.len().is_multiple_of(pitch));

        BlockMut {
            block: BlockSlice {
                rows: data.len() / pitch,
                cols: pitch,
                pitch,
            },
            data: NonNull::from_mut(data).cast(),
            lifetime: PhantomData,
        }
    }

    /// Number of rows in this block.
    pub fn rows(&self) -> usize {
        self.block.rows
    }

    /// Number of columns in this block.
    pub fn cols(&self) -> usize {
        self.block.cols
    }

    /// Divide into two blocks at the given column.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &mut [
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows_mut(data);
    /// let (left, right) = block.split_at_col(2);
    ///
    /// assert_eq!(left[(1, 0)], 3);
    /// assert_eq!(right[(1, 0)], 5);
    /// ```
    pub fn split_at_col(self, mid: usize) -> (BlockMut<'data, T>, BlockMut<'data, T>) {
        self.split_at_col_checked(mid).unwrap()
    }

    /// Divide into two blocks at the given column.
    ///
    /// See [`Self::split_at_col`] but returns `None` if out of bounds.
    pub fn split_at_col_checked(
        self,
        mid: usize,
    ) -> Option<(BlockMut<'data, T>, BlockMut<'data, T>)> {
        if let Some((lhs, rhs, offset)) = self.block.split_at_col(mid) {
            Some((
                BlockMut {
                    data: self.data,
                    block: lhs,
                    lifetime: self.lifetime,
                },
                BlockMut {
                    data: unsafe { self.data.add(offset) },
                    block: rhs,
                    lifetime: self.lifetime,
                },
            ))
        } else {
            None
        }
    }

    /// Divide into two blocks at the given row.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &mut [
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows_mut(data);
    /// let (top, bot) = block.split_at_row(1);
    ///
    /// assert_eq!(top[(0, 2)], 2);
    /// assert_eq!(bot[(0, 2)], 5);
    /// ```
    pub fn split_at_row(self, mid: usize) -> (BlockMut<'data, T>, BlockMut<'data, T>) {
        self.split_at_row_checked(mid).unwrap()
    }

    /// Divide into two blocks at the given row.
    ///
    /// See [`Self::split_at_row`] but returns `None` if out of bounds.
    pub fn split_at_row_checked(
        self,
        mid: usize,
    ) -> Option<(BlockMut<'data, T>, BlockMut<'data, T>)> {
        if let Some((lhs, rhs, offset)) = self.block.split_at_row(mid) {
            Some((
                BlockMut {
                    data: self.data,
                    block: lhs,
                    lifetime: self.lifetime,
                },
                BlockMut {
                    data: unsafe { self.data.add(offset) },
                    block: rhs,
                    lifetime: self.lifetime,
                },
            ))
        } else {
            None
        }
    }

    /// Choose a single row and refer to its data.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &mut [
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8],
    /// ];
    ///
    /// let mut block = matrix_slice::from_array_rows_mut(data);
    /// let mut row = block.reborrow().row(1);
    /// row[0] = 0x42;
    /// assert_eq!(block[(1, 0)], 0x42);
    /// ```
    pub fn row(self, row: usize) -> VecMut<'data, T> {
        let (_, block, offset) = self.block.split_at_row(row).unwrap();
        assert!(block.rows >= 1);

        VecMut {
            block: VectorSlice {
                count: block.cols,
                pitch: 1,
            },
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        }
    }

    /// Choose a single column and refer to its data.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &mut [
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8],
    /// ];
    ///
    /// let mut block = matrix_slice::from_array_rows_mut(data);
    /// let mut row = block.reborrow().col(1);
    /// row[0] = 0x42;
    /// assert_eq!(block[(0, 1)], 0x42);
    /// ```
    pub fn col(self, col: usize) -> VecMut<'data, T> {
        let (_, block, offset) = self.block.split_at_col(col).unwrap();
        assert!(block.cols >= 1);

        VecMut {
            block: VectorSlice {
                count: block.rows,
                pitch: block.pitch,
            },
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        }
    }

    /// Choose a range of rows and contract the block to that.
    ///
    /// The argument type is flexible, allowing ranges (`1..3`), half open ranges (`2..` and `..2`)
    /// among others. See the [`MatrixIndex`] trait, which is sealed though as its details are not
    /// yet finalized.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &mut [
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows_mut(data);
    ///
    /// let center = block.select_rows(1..2).unwrap();
    /// assert_eq!(center.rows(), 1);
    /// assert_eq!(center.cols(), 3);
    /// assert_eq!(center[(0, 1)], 4);
    /// ```
    pub fn select_rows<R>(self, range: R) -> Option<BlockMut<'data, T>>
    where
        R: MatrixIndex,
    {
        let (start, len) = range.into_start_and_len(self.block.rows)?;
        let (_, block, offset) = self.block.split_at_row(start)?;
        assert!(block.rows >= len);

        Some(BlockMut {
            block: BlockSlice { rows: len, ..block },
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        })
    }

    /// Choose a range of columns and contract the block to that.
    ///
    /// The argument type is flexible, allowing ranges (`1..3`), half open ranges (`2..` and `..2`)
    /// among others. See the [`MatrixIndex`] trait, which is sealed though as its details are not
    /// yet finalized.
    ///
    /// ```
    /// let data = &mut [
    ///     [0, 1],
    ///     [3, 4],
    ///     [6, 7],
    /// ];
    ///
    /// let mut block = matrix_slice::from_array_rows_mut(data);
    ///
    /// assert!(block.reborrow().select_cols(1..).is_some_and(|b| b.cols() == 1));
    /// assert!(block.reborrow().select_cols(2..).is_some_and(|b| b.cols() == 0));
    /// assert!(block.reborrow().select_cols(3..).is_none());
    /// ```
    pub fn select_cols<R>(self, range: R) -> Option<BlockMut<'data, T>>
    where
        R: MatrixIndex,
    {
        let (start, len) = range.into_start_and_len(self.block.cols)?;
        let (_, block, offset) = self.block.split_at_col(start)?;
        debug_assert!(block.cols >= len);

        Some(BlockMut {
            block: BlockSlice { cols: len, ..block },
            // SAFETY:
            // - `split_at_col` guarantees that `offset` is in-bounds of the provenance tracked by
            // `self.block`, which is in-sync with the pointer field (but we not necessary access
            // these without synchronization). By extension of being in-bounds of the allocation
            // the offset does not overflow `isize`.
            // - the block has access to a subset of elements as `self` which is consumed. This
            // holds because `into_start_and_len` ensures that `start + len <= self.block.cols`.
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        })
    }

    /// Choose a sub-block by its range of rows and columns.
    pub fn select(
        self,
        row_range: impl MatrixIndex,
        col_range: impl MatrixIndex,
    ) -> Option<BlockMut<'data, T>> {
        let block = self.select_rows(row_range)?;
        block.select_cols(col_range)
    }

    /// Extract a contiguous underlying slice of elements if the block is contiguous.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &mut [[0u32; 3]; 3];
    /// let mut block = matrix_slice::from_array_rows_mut(data);
    ///
    /// let (mut part, _) = block.reborrow().split_at_row(2);
    /// assert!(part.into_contiguous_slice().is_some());
    ///
    /// let (pre, post) = block.reborrow().split_at_col(2);
    /// assert!(pre.into_contiguous_slice().is_none());
    /// assert!(post.into_contiguous_slice().is_none());
    ///
    /// let (same, _) = block.reborrow().split_at_col(3);
    /// assert!(same.into_contiguous_slice().is_some());
    /// ```
    pub fn into_contiguous_slice(self) -> Option<&'data mut [T]> {
        if let Some(items) = self.block.contiguous_span() {
            Some(unsafe { core::slice::from_raw_parts_mut(self.data.as_ptr().cast(), items) })
        } else {
            None
        }
    }

    /// Turn this into a slice of the first row, assuming it is at most one row.
    fn fake_contiguity(mut self) -> &'data mut [T] {
        self.block.fake_contiguity();
        self.into_contiguous_slice().unwrap()
    }

    /// Extract access as a slice of arrays if the block is contiguous.
    ///
    /// The caller must choose `N` matching the number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// let data = &mut [[0u32; 3]; 3];
    /// let mut block = matrix_slice::from_array_rows_mut(data);
    ///
    /// // Turns this back into the same type as `data` had.
    /// assert!(block.reborrow().into_array_rows_checked::<3>().is_some());
    ///
    /// // Using an incorrect number of columns fails.
    /// assert!(block.reborrow().into_array_rows_checked::<2>().is_none());
    ///
    /// // Can still be used after splitting at rows.
    /// let (_, mut block) = block.split_at_row(2);
    /// assert!(block.reborrow().into_array_rows_checked::<3>().is_some());
    /// ```
    pub fn into_array_rows_checked<const N: usize>(self) -> Option<&'data mut [[T; N]]> {
        if self.block.cols == self.block.pitch && self.block.cols == N {
            Some(unsafe {
                core::slice::from_raw_parts_mut(self.data.as_ptr().cast(), self.block.rows)
            })
        } else {
            None
        }
    }

    /// Turn this unique reference into a shared reference.
    pub fn cast_const(self) -> BlockRef<'data, T> {
        // SAFETY: shared access can always be re-tagged from unique access.
        BlockRef {
            data: self.data,
            block: self.block,
            lifetime: PhantomData,
        }
    }

    /// Create a unique reference to this block with a shorter lifetime.
    pub fn reborrow(&mut self) -> BlockMut<'_, T> {
        // SAFETY: Unique access is created by deriving it from our current pointer so the
        // provenance is the same, and temporally it can not overlap access through the current
        // value due to the lifetime enforcing a borrow relationship.
        BlockMut {
            data: self.data,
            block: self.block,
            lifetime: PhantomData,
        }
    }

    /// Iterate over the rows of this block.
    pub fn iter_rows(self) -> IterRows<'data, T> {
        self.cast_const().iter_rows()
    }

    /// Iterate over the rows of this block.
    pub fn iter_rows_mut(self) -> IterRowsMut<'data, T> {
        IterRowsMut { block: self }
    }

    /// Modify the item type to a `Cell`, allowing interior mutability.
    ///
    /// This is the equivalent of [`Cell::from_mut`] over elements in this slice.
    pub fn as_cells(self) -> BlockMut<'data, Cell<T>> {
        // SAFETY: `Cell<T>` has the same layout as `T`.
        BlockMut {
            data: self.data.cast(),
            block: self.block,
            lifetime: PhantomData,
        }
    }
}

impl<'data, T> BlockMut<'data, Cell<T>> {
    /// Modify the item type from a `Cell` to its interior type.
    ///
    /// This is the equivalent of [`Cell::get_mut`] over elements in this slice.
    pub fn as_cell_items(self) -> BlockMut<'data, T> {
        // SAFETY: `Cell<T>` has the same layout as `T`.
        BlockMut {
            data: self.data.cast(),
            block: self.block,
            lifetime: PhantomData,
        }
    }
}

impl<T> ops::Index<(usize, usize)> for BlockMut<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let idx = self.block.in_bounds_offset(index.0, index.1);
        // SAFETY: Index is bounded by `total_span` which itself is a lower estimate of the
        // provenance of the pointer.
        unsafe { &*self.data.as_ptr().add(idx) }
    }
}

impl<T> ops::IndexMut<(usize, usize)> for BlockMut<'_, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.block.in_bounds_offset(index.0, index.1);
        // SAFETY: Index is bounded by `total_span` which itself is a lower estimate of the
        // provenance of the pointer.
        unsafe { &mut *self.data.as_ptr().add(idx) }
    }
}

/// Represents the provenance of a pointer to a block of a matrix.
///
/// FIXME: before exposing this consider `PartialEq, … Ord` implications. These were added to
/// satisfy the `Pointee` trait requirements but really what does ordering mean? We have chosen the
/// field `pitch` to be last but that is super arbitrary.
///
/// We assume row major order here for the convention of _naming_ things. That is, when we say row
/// we mean a tightly packed slice of items. This implies that the item pitch is assumed to be `1`.
/// We have two major possible choices in representation a block-subset of a matrix: store the
/// dimensions of the block with a matrix row pitch or store the total size of the matrix and two
/// lengths.
///
/// The former of these allows us to represent both `0×N` and `M×0` blocks naturally, while the
/// latter allows one of them but provides a fast capacity that's pre-calculated. We choose the
/// former. In either case we need to store three `usize` values. Note that the total span of items
/// is *not* `rows * pitch` since the last row might be ragged.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct BlockSlice {
    rows: usize,
    cols: usize,
    pitch: usize,
}

const _: () = {
    // As per Rust 1.92's `Pointee` trait. Suspicious: `Ord`. See comment on `BlockSlice`.
    use core::{fmt, hash};

    fn _can_eventually_be_ptr_metadata<
        // Missing: `Freeze` which is unstable
        Metadata: fmt::Debug + Copy + Send + Sync + Ord + hash::Hash + Unpin,
    >() {
    }

    let _ = _can_eventually_be_ptr_metadata::<BlockSlice>;
};

impl BlockSlice {
    /// The number of elements if this block is contiguous (cols equals pitch).
    fn contiguous_span(&self) -> Option<usize> {
        if self.cols == self.pitch {
            Some(self.rows * self.cols)
        } else {
            None
        }
    }

    /// The number of elements spanned by this block (including those we are not allowed to
    /// access).
    fn total_span(&self) -> usize {
        if let Some(all_but_last) = self.rows.checked_sub(1) {
            all_but_last * self.pitch + self.cols
        } else {
            0
        }
    }

    /// The caller must ensure that this block has at most one row.
    fn fake_contiguity(&mut self) {
        debug_assert!(self.rows <= 1);
        debug_assert!(self.cols <= self.pitch);

        self.rows = self.rows.min(1);
        // SAFETY: Reducing the pitch when we have at most one row does not change the elements we
        // may refer to. The pitch always exceeds the number of columns.
        self.pitch = self.cols;
    }

    /// Split into two block descriptors.
    ///
    /// Returns `Some` with two valid blocks. The first block is in-bounds. Also returns an offset
    /// that is in-bounds of the current block and such that the elements valid for both blocks do
    /// not alias. The second block is in-bounds when interpreted as start at the offset.
    fn split_at_row(self, mid: usize) -> Option<(BlockSlice, BlockSlice, usize)> {
        let n = self.rows.checked_sub(mid)?;

        let lhs = BlockSlice {
            rows: mid,
            cols: self.cols,
            pitch: self.pitch,
        };

        let rhs = BlockSlice {
            rows: n,
            cols: self.cols,
            pitch: self.pitch,
        };

        // Careful: If we split a block after its last row (i.e. lhs and self are identical),
        // the naive offset of rows * pitch may point beyond the total span of elements covered
        // by ourselves. In this case the rhs does not cover any row so we assign it any
        // in-bounds offset.
        let offset = if n > 0 { mid * self.pitch } else { 0 };
        debug_assert!(offset <= self.total_span());

        Some((lhs, rhs, offset))
    }

    fn split_at_col(self, mid: usize) -> Option<(BlockSlice, BlockSlice, usize)> {
        let n = self.cols.checked_sub(mid)?;

        let lhs = BlockSlice {
            rows: self.rows,
            cols: mid,
            pitch: self.pitch,
        };

        let rhs = BlockSlice {
            rows: self.rows,
            cols: n,
            pitch: self.pitch,
        };

        // If we have no rows at all then this block does not cover any elements so we must
        // pick an offset of 0 to guarantee in-bounds access. The good news is that this case
        // also implies that the other side is empty so its offset does not matter.
        let offset = if self.rows > 0 { mid } else { 0 };
        debug_assert!(offset <= self.total_span());

        Some((lhs, rhs, offset))
    }

    /// Return the absolute position of the element, if in bounds. Otherwise, panic.
    fn in_bounds_offset(&self, row: usize, col: usize) -> usize {
        assert!(row < self.rows);
        assert!(col < self.cols);
        let idx = row * self.pitch + col;
        debug_assert!(idx < self.total_span());
        idx
    }
}

/// Represents the provenance of a pointer to a single column/row of a matrix.
///
/// FIXME: before exposing this consider `PartialEq, … Ord` implications. These were added to
/// satisfy the `Pointee` trait requirements but really what does ordering mean? We have chosen the
/// field `pitch` to be last but that is super arbitrary.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct VectorSlice {
    count: usize,
    pitch: usize,
}

const _: () = {
    // As per Rust 1.92's `Pointee` trait. Suspicious: `Ord`. See comment on `VectorSlice`.
    use core::{fmt, hash};

    fn _can_eventually_be_ptr_metadata<
        // Missing: `Freeze` which is unstable
        Metadata: fmt::Debug + Copy + Send + Sync + Ord + hash::Hash + Unpin,
    >() {
    }

    let _ = _can_eventually_be_ptr_metadata::<VectorSlice>;
};

impl VectorSlice {
    /// Split into two vector descriptors.
    ///
    /// Returns `Some` with two valid slice. The first slice is in-bounds. Also returns an offset
    /// that is in-bounds of the current slice and such that the elements valid for both blocks do
    /// not alias. The second block is in-bounds when interpreted as start at the offset.
    fn split_at(self, mid: usize) -> Option<(VectorSlice, VectorSlice, usize)> {
        let right_count = self.count.checked_sub(mid)?;

        let left = VectorSlice {
            count: mid,
            pitch: self.pitch,
        };

        let right = VectorSlice {
            count: right_count,
            pitch: self.pitch,
        };

        // We need to make sure that this is in-bounds of the allocation. Note that the provenance
        // stretches only past our last element, not an additional pitch past it. So we can use the
        // simple formula only if there are elements covered by the right part of the split.
        // Fortunately the opposite of this implies that the right does not cover any elements and
        // we can thus use any inbounds pointer; like the zero offset.
        let offset = if right_count == 0 {
            0
        } else {
            mid * self.pitch
        };

        Some((left, right, offset))
    }

    /// Return the absolute position of the element, if in bounds. Otherwise, panic.
    fn in_bounds_offset(&self, index: usize) -> usize {
        assert!(index < self.count);
        index * self.pitch
    }
}

/// Iterate over the rows of a block in a matrix.
///
/// We assume row-major matrices here, a row is a contiguous slice of items.
pub struct IterRows<'a, T> {
    block: BlockRef<'a, T>,
}

impl<'data, T> Iterator for IterRows<'data, T> {
    type Item = &'data [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.block.rows() == 0 {
            None
        } else {
            // FIXME: add `split_off_rows` instead.
            let (row, rest) = core::mem::take(&mut self.block).split_at_row(1);
            self.block = rest;
            // One row as it was created from `split_at_row(1)`.
            Some(row.fake_contiguity())
        }
    }
}

/// Iterate over mutable rows of a block in a matrix.
///
/// We assume row-major matrices here, a row is a contiguous slice of items.
pub struct IterRowsMut<'a, T> {
    block: BlockMut<'a, T>,
}

impl<'data, T> Iterator for IterRowsMut<'data, T> {
    type Item = &'data mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.block.rows() == 0 {
            None
        } else {
            // FIXME: add `split_off_rows` instead.
            let (row, rest) = core::mem::take(&mut self.block).split_at_row(1);
            self.block = rest;
            // One row as it was created from `split_at_row(1)`.
            Some(row.fake_contiguity())
        }
    }
}

pub trait MatrixIndex: sealed::Sealed {}

impl MatrixIndex for ops::Range<usize> {}
impl MatrixIndex for ops::RangeInclusive<usize> {}
impl MatrixIndex for ops::RangeFrom<usize> {}
impl MatrixIndex for ops::RangeTo<usize> {}
impl MatrixIndex for ops::RangeToInclusive<usize> {}
impl MatrixIndex for ops::RangeFull {}

pub trait OneSidedMatrixIndex: sealed::SealedOneSided {}

impl OneSidedMatrixIndex for ops::RangeFrom<usize> {}
impl OneSidedMatrixIndex for ops::RangeTo<usize> {}
impl OneSidedMatrixIndex for ops::RangeToInclusive<usize> {}

mod sealed {
    use core::ops;

    pub trait Sealed {
        /// SAFETY: It is crucial that `start + len <= dim` holds if `Some` is returned.
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)>;
    }

    pub trait SealedOneSided {
        /// SAFETY: It is crucial that `split <= dim` holds if `Some` is returned.
        fn into_split_point(self, dim: usize) -> Option<(bool, usize)>;
    }

    impl Sealed for ops::Range<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            if self.start <= self.end && self.end <= dim {
                // SAFETY: overflow can not have occurred, so `self.start + len = self.end <= dim`.
                Some((self.start, self.end - self.start))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeInclusive<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            let start = *self.start();
            let end = *self.end();
            if start <= end && end < dim {
                // SAFETY: overflow can not have occurred, so `self.start + len = self.end + 1 <= dim`.
                Some((start, end - start + 1))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeFrom<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            if self.start <= dim {
                // SAFETY: overflow can not have occurred, so `self.start + len = dim <= dim`.
                Some((self.start, dim - self.start))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeTo<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            if self.end <= dim {
                // SAFETY: `self.end <= dim` by test.
                Some((0, self.end))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeToInclusive<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            if self.end < dim {
                // SAFETY: `self.end + 1 <= dim` by test.
                Some((0, self.end + 1))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeFull {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            // SAFETY: `dim <= dim` by tautology.
            Some((0, dim))
        }
    }

    impl SealedOneSided for ops::RangeFrom<usize> {
        fn into_split_point(self, dim: usize) -> Option<(bool, usize)> {
            if self.start <= dim {
                // SAFETY: `self.start <= dim` by test.
                Some((false, self.start))
            } else {
                None
            }
        }
    }

    impl SealedOneSided for ops::RangeTo<usize> {
        fn into_split_point(self, dim: usize) -> Option<(bool, usize)> {
            if self.end <= dim {
                // SAFETY: `self.end <= dim` by test.
                Some((true, self.end))
            } else {
                None
            }
        }
    }

    impl SealedOneSided for ops::RangeToInclusive<usize> {
        fn into_split_point(self, dim: usize) -> Option<(bool, usize)> {
            if self.end < dim {
                // SAFETY: `self.end <= dim` by test.
                Some((true, self.end + 1))
            } else {
                None
            }
        }
    }
}

/// A reference to a single column/row of a matrix.
///
/// This is similar to `&[T]` but with a pitch potentially different from `1` between its elements,
/// i.e. there is no guarantee of contiguity. As a consequence this does not have a simple
/// past-the-end pointer like a slice would have. For an empty slice the only guaranteed-valid
/// pointer is the base pointer itself while for larger slices the last guaranteed-valid pointer is
/// one-past the last element, _not_ one additional pitch.
///
/// Created from its constructors or a block reference via the [`BlockRef::col`] and
/// [`BlockRef::row`] methods.
#[derive(Copy, Clone)]
pub struct VecRef<'a, T> {
    data: NonNull<T>,
    block: VectorSlice,
    lifetime: PhantomData<&'a [T]>,
}

// SAFETY: See `&[T]`. The reference can be used to, potentially, get a `&T` for each element in
// the block and thus the block itself provides the exact same properties as `T`. The `VecRef` is
// then `&[T]` itself and thus has properties of a reference to such a type. Refer to the
// reference: <https://doc.rust-lang.org/stable/std/primitive.reference.html>
//
// We have `&T: Sync` iff `T: Sync`
unsafe impl<T> Sync for VecRef<'_, T> where T: Sync {}
// We have `&T: Send` iff `T: Sync`
unsafe impl<T> Send for VecRef<'_, T> where T: Sync {}

impl<'data, T> VecRef<'data, T> {
    /// Create a new vector reference from a raw slice and pitch.
    ///
    /// The resulting block refers to the first column of the matrix.
    ///
    /// # Panics
    ///
    /// Panics if the pitch is zero.
    pub fn new(data: &'data [T], pitch: usize) -> Self {
        assert_ne!(pitch, 0);

        VecRef {
            // Safety: construction implies `count * pitch <= data.len()`.
            block: VectorSlice {
                count: data.len() / pitch,
                pitch,
            },
            data: NonNull::from(data).cast(),
            lifetime: PhantomData,
        }
    }

    /// Create a new vector reference from a raw slice with pitch `1`.
    pub fn from_slice(data: &'data [T]) -> Self {
        VecRef {
            block: VectorSlice {
                count: data.len(),
                pitch: 1,
            },
            data: NonNull::from(data).cast(),
            lifetime: PhantomData,
        }
    }

    /// Number of elements in this vector.
    pub fn len(&self) -> usize {
        self.block.count
    }

    /// Whether this vector is empty.
    pub fn is_empty(&self) -> bool {
        self.block.count == 0
    }

    /// Divide into two vectors at the given element.
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_slice::VecRef;
    ///
    /// let data = &[0, 1, 2, 3, 4, 5];
    ///
    /// let block = VecRef::new(data, 1);
    /// let (left, right) = block.split_at(2);
    ///
    /// assert_eq!(left[1], 1);
    /// assert_eq!(right[3], 5);
    /// ```
    pub fn split_at(self, mid: usize) -> (VecRef<'data, T>, VecRef<'data, T>) {
        self.split_at_checked(mid).unwrap()
    }

    /// Divide into two vectors at the given element.
    ///
    /// See [`Self::split_at`] but returns `None` if out of bounds.
    pub fn split_at_checked(mut self, mid: usize) -> Option<(VecRef<'data, T>, VecRef<'data, T>)> {
        // Let's assume this will collapse during const-prop after the type here is inserted.
        let tail = self.split_off(mid..)?;
        Some((self, tail))
    }

    /// Take part of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_slice::VecRef;
    ///
    /// let data = &[0, 1, 2, 3, 4, 5];
    /// let mut vec = VecRef::new(data, 1);
    ///
    /// // Does nothing.
    /// assert!(vec.split_off(6..).is_some_and(|v| v.is_empty()));
    /// assert!(vec.split_off(7..).is_none());
    /// assert!(vec.split_off(..7).is_none());
    ///
    /// let right = vec.split_off(2..).unwrap();
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(right[3], 5);
    /// ```
    ///
    /// You can also split off the start:
    ///
    /// ```
    /// use matrix_slice::VecRef;
    ///
    /// let data = &[0, 1, 2, 3, 4, 5];
    /// let mut vec = VecRef::new(data, 1);
    /// let start = vec.split_off(..=2).unwrap();
    ///
    /// assert_eq!(vec[0], 3);
    /// assert_eq!(start[0], 0);
    /// ```
    pub fn split_off<R>(&mut self, range: R) -> Option<Self>
    where
        R: OneSidedMatrixIndex,
    {
        let (is_front, split_point) = range.into_split_point(self.block.count)?;
        let (left, right, offset) = self.block.split_at(split_point)?;

        let ptr = self.data;
        let (ours, theirs) = if is_front {
            ((right, offset), (left, 0))
        } else {
            ((left, 0), (right, offset))
        };

        self.block = ours.0;
        self.data = unsafe { ptr.add(ours.1) };

        Some(VecRef {
            block: theirs.0,
            data: unsafe { ptr.add(theirs.1) },
            lifetime: self.lifetime,
        })
    }

    /// Choose a range of elements and contract the vector to that.
    pub fn select<R>(self, range: R) -> Option<VecRef<'data, T>>
    where
        R: MatrixIndex,
    {
        let (start, len) = range.into_start_and_len(self.block.count)?;
        let (_, block, offset) = self.block.split_at(start)?;
        // Safety: ensures that the resulting block is more constrained, this property should be
        // ensured by our sealed `MatrixIndex` implementations.
        assert!(block.count >= len);

        Some(VecRef {
            block: VectorSlice {
                count: len,
                ..block
            },
            // SAFETY: offset is in-bounds as per `split_at` contract.
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        })
    }

    /// # Examples
    ///
    /// ```
    /// let data = &[
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8],
    /// ];
    ///
    /// let block = matrix_slice::from_array_rows(data);
    /// let column = block.col(1);
    ///
    /// assert!(column.iter().eq(&[1, 4, 7]));
    /// ```
    pub fn iter(self) -> IterVec<'data, T> {
        IterVec { vec: self }
    }
}

impl<T> ops::Index<usize> for VecRef<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let idx = self.block.in_bounds_offset(index);
        // SAFETY: Index is bounded by `total_span` which itself is a lower estimate of the
        // provenance of the pointer.
        unsafe { &*self.data.as_ptr().add(idx) }
    }
}

/// A reference to a single column/row of a matrix.
///
/// This is similar to `&[T]` but with a pitch potentially different from `1` between its elements,
/// i.e. there is no guarantee of contiguity. As a consequence this does not have a simple
/// past-the-end pointer like a slice would have. For an empty slice the only guaranteed-valid
/// pointer is the base pointer itself while for larger slices the last guaranteed-valid pointer is
/// one-past the last element, _not_ one additional pitch.
///
/// Created from its constructors or a block reference via the [`BlockMut::col`] and
/// [`BlockMut::row`] methods.
pub struct VecMut<'a, T> {
    data: NonNull<T>,
    block: VectorSlice,
    lifetime: PhantomData<&'a mut [T]>,
}

// SAFETY: See `VecRef` but with `&mut [T]`.
//
// We have `&mut T: Sync` iff `T: Sync`
unsafe impl<T> Sync for VecMut<'_, T> where T: Sync {}
// We have `&mut T: Send` iff `T: Send`
unsafe impl<T> Send for VecMut<'_, T> where T: Sync {}

impl<'data, T> VecMut<'data, T> {
    /// Create a new vector reference from a raw slice and pitch.
    ///
    /// The resulting block refers to the first column of the matrix.
    pub fn new(data: &'data mut [T], pitch: usize) -> Self {
        VecMut {
            // Safety: construction implies `count * pitch <= data.len()`.
            block: VectorSlice {
                count: data.len() / pitch,
                pitch,
            },
            data: NonNull::from(data).cast(),
            lifetime: PhantomData,
        }
    }

    /// Create a new vector reference from a raw slice with pitch `1`.
    pub fn from_slice(data: &'data mut [T]) -> Self {
        VecMut {
            block: VectorSlice {
                count: data.len(),
                pitch: 1,
            },
            data: NonNull::from(data).cast(),
            lifetime: PhantomData,
        }
    }

    /// Number of elements in this vector.
    pub fn len(&self) -> usize {
        self.block.count
    }

    /// Whether this vector is empty.
    pub fn is_empty(&self) -> bool {
        self.block.count == 0
    }

    /// Divide into two vectors at the given element.
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_slice::VecMut;
    ///
    /// let data = &mut [0, 1, 2, 3, 4, 5];
    ///
    /// let block = VecMut::new(data, 1);
    /// let (left, right) = block.split_at(2);
    ///
    /// assert_eq!(left[1], 1);
    /// assert_eq!(right[3], 5);
    /// ```
    pub fn split_at(self, mid: usize) -> (VecMut<'data, T>, VecMut<'data, T>) {
        self.split_at_checked(mid).unwrap()
    }

    /// Divide into two vectors at the given element.
    ///
    /// See [`Self::split_at`] but returns `None` if out of bounds.
    pub fn split_at_checked(self, mid: usize) -> Option<(VecMut<'data, T>, VecMut<'data, T>)> {
        if let Some((lhs, rhs, offset)) = self.block.split_at(mid) {
            Some((
                VecMut {
                    data: self.data,
                    block: lhs,
                    lifetime: self.lifetime,
                },
                VecMut {
                    data: unsafe { self.data.add(offset) },
                    block: rhs,
                    lifetime: self.lifetime,
                },
            ))
        } else {
            None
        }
    }

    /// Take part of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_slice::VecMut;
    ///
    /// let data = &mut [0, 1, 2, 3, 4, 5];
    /// let mut vec = VecMut::new(data, 1);
    ///
    /// // Does nothing.
    /// assert!(vec.split_off(6..).is_some_and(|v| v.is_empty()));
    /// assert!(vec.split_off(7..).is_none());
    /// assert!(vec.split_off(..7).is_none());
    ///
    /// let mut right = vec.split_off(2..).unwrap();
    /// assert_eq!(vec.len(), 2);
    /// assert_eq!(right[3], 5);
    ///
    /// // The two halves are disjoint:
    /// right[0] = 0x42;
    /// assert_eq!(vec[1], 1);
    /// ```
    ///
    /// You can also split off the start:
    ///
    /// ```
    /// use matrix_slice::VecMut;
    ///
    /// let data = &mut [0, 1, 2, 3, 4, 5];
    /// let mut vec = VecMut::new(data, 1);
    /// let start = vec.split_off(..=2).unwrap();
    ///
    /// assert_eq!(vec[0], 3);
    /// assert_eq!(start[0], 0);
    /// ```
    pub fn split_off<R>(&mut self, range: R) -> Option<Self>
    where
        R: OneSidedMatrixIndex,
    {
        let (is_front, split_point) = range.into_split_point(self.block.count)?;
        let (left, right, offset) = self.block.split_at(split_point)?;

        let ptr = self.data;
        let (ours, theirs) = if is_front {
            ((right, offset), (left, 0))
        } else {
            ((left, 0), (right, offset))
        };

        self.block = ours.0;
        self.data = unsafe { ptr.add(ours.1) };

        Some(VecMut {
            block: theirs.0,
            data: unsafe { ptr.add(theirs.1) },
            lifetime: self.lifetime,
        })
    }

    /// Choose a range of elements and contract the vector to that.
    pub fn select<R>(self, range: R) -> Option<VecMut<'data, T>>
    where
        R: MatrixIndex,
    {
        let (start, len) = range.into_start_and_len(self.block.count)?;
        let (_, block, offset) = self.block.split_at(start)?;
        // Safety: ensures that the resulting block is more constrained, this property should be
        // ensured by our sealed `MatrixIndex` implementations.
        assert!(block.count >= len);

        Some(VecMut {
            block: VectorSlice {
                count: len,
                ..block
            },
            // SAFETY: offset is in-bounds as per `split_at` contract.
            data: unsafe { self.data.add(offset) },
            lifetime: self.lifetime,
        })
    }

    /// Turn this unique reference into a shared reference.
    pub fn cast_const(self) -> VecRef<'data, T> {
        // SAFETY: shared access can always be re-tagged from unique access.
        VecRef {
            data: self.data,
            block: self.block,
            lifetime: PhantomData,
        }
    }

    /// Create a unique reference to this block with a shorter lifetime.
    pub fn reborrow(&mut self) -> VecMut<'_, T> {
        // SAFETY: Unique access is created by deriving it from our current pointer so the
        // provenance is the same, and temporally it can not overlap access through the current
        // value due to the lifetime enforcing a borrow relationship.
        VecMut {
            data: self.data,
            block: self.block,
            lifetime: PhantomData,
        }
    }

    /// Modify the item type to a `Cell`, allowing interior mutability.
    ///
    /// This is the equivalent of [`Cell::from_mut`] over elements in this slice.
    pub fn as_cells(self) -> VecMut<'data, Cell<T>> {
        // SAFETY: `Cell<T>` has the same layout as `T`.
        VecMut {
            data: self.data.cast(),
            block: self.block,
            lifetime: PhantomData,
        }
    }

    /// # Examples
    ///
    /// ```
    /// let data = &mut [
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8],
    /// ];
    ///
    /// let mut block = matrix_slice::from_array_rows_mut(data);
    ///
    /// for item in block.reborrow().col(1) {
    ///     *item *= 2;
    /// }
    ///
    /// assert!(block.col(1).iter().eq(&[2, 8, 14]));
    /// ```
    pub fn iter(self) -> IterVecMut<'data, T> {
        IterVecMut { vec: self }
    }
}

impl<'data, T> VecMut<'data, Cell<T>> {
    /// Modify the item type from a `Cell` to its interior type.
    ///
    /// This is the equivalent of [`Cell::get_mut`] over elements in this slice.
    pub fn as_cell_items(self) -> VecMut<'data, T> {
        // SAFETY: `Cell<T>` has the same layout as `T`.
        VecMut {
            data: self.data.cast(),
            block: self.block,
            lifetime: PhantomData,
        }
    }
}

impl<T> ops::Index<usize> for VecMut<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let idx = self.block.in_bounds_offset(index);
        // SAFETY: Index is bounded by `total_span` which itself is a lower estimate of the
        // provenance of the pointer.
        unsafe { &*self.data.as_ptr().add(idx) }
    }
}

impl<T> ops::IndexMut<usize> for VecMut<'_, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let idx = self.block.in_bounds_offset(index);
        // SAFETY: Index is bounded by `total_span` which itself is a lower estimate of the
        // provenance of the pointer. By construction the `VecMut` has exclusive access to all
        // elements reachable as multiples of its pitch. We access exactly one of them here.
        unsafe { &mut *self.data.as_ptr().add(idx) }
    }
}

/// Iterate over the rows of a block in a matrix.
///
/// We assume row-major matrices here, a row is a contiguous slice of items.
pub struct IterVec<'a, T> {
    // FIXME: see `std::slice::Iter` which stores the end pointer instead of the full
    // representation. That way we do not update two fields each time, i.e. iterating only updates
    // a pointer and not a pointer _and_ a `count` field in `block`.
    vec: VecRef<'a, T>,
}

impl<'data, T> IntoIterator for VecRef<'data, T> {
    type Item = &'data T;
    type IntoIter = IterVec<'data, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// FIXME: if budget allows it we should implement common inner-iteration methods such as
// `for_each`, `collect`, `all` by doing pointer arithmetic on a range iterator which avoids all
// writes to the value's tracking state itself.
impl<'data, T> Iterator for IterVec<'data, T> {
    type Item = &'data T;

    fn next(&mut self) -> Option<Self::Item> {
        let base = self.vec.split_off(..1)?;
        Some(unsafe { &*base.data.as_ptr() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remain = self.vec.len();
        (remain, Some(remain))
    }

    fn count(self) -> usize {
        self.vec.len()
    }
}

impl<'data, T> core::iter::FusedIterator for IterVec<'data, T> {}

/// Iterate over mutable rows of a block in a matrix.
///
/// We assume row-major matrices here, a row is a contiguous slice of items.
pub struct IterVecMut<'a, T> {
    vec: VecMut<'a, T>,
}

impl<'data, T> IntoIterator for VecMut<'data, T> {
    type Item = &'data mut T;
    type IntoIter = IterVecMut<'data, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// FIXME: if budget allows it we should implement common inner-iteration methods such as
// `for_each`, `collect`, `all` by doing pointer arithmetic on a range iterator which avoids all
// writes to the value's tracking state itself.
impl<'data, T> Iterator for IterVecMut<'data, T> {
    type Item = &'data mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let base = self.vec.split_off(..1)?;
        Some(unsafe { &mut *base.data.as_ptr() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remain = self.vec.len();
        (remain, Some(remain))
    }

    fn count(self) -> usize {
        self.vec.len()
    }
}

impl<'data, T> core::iter::FusedIterator for IterVecMut<'data, T> {}

/// Tests should also be ran under MIRI.
#[cfg(test)]
mod tests {
    // Verify that splitting as in the example works.
    #[test]
    fn well_defined_split() {
        let data = &[[0u32; 3]; 3];
        let block = super::from_array_rows(data);
        let (_, block) = block.split_at_row(1);
        let (_, block) = block.split_at_col(1);

        block.split_at_row_checked(2).unwrap();
    }
    #[test]
    fn well_defined_split_mut() {
        let data = &mut [[0u32; 3]; 3];
        let block = super::from_array_rows_mut(data);
        let (_, block) = block.split_at_row(1);
        let (_, block) = block.split_at_col(1);

        block.split_at_row_checked(2).unwrap();
    }

    /// Check our pointer derivation does not cause retagging that would cause any block to lose
    /// provenance over its items. Access individual rows (derived slices) from an overlapping split
    /// concurrently.
    #[test]
    fn soundness_interleaved_block_access() {
        let data = &mut [[0u32; 4]; 4];
        let block = super::from_array_rows_mut(data);

        let (mut lhs, rhs) = block.split_at_col(2);

        for (left, right) in lhs.reborrow().iter_rows_mut().zip(rhs.iter_rows_mut()) {
            left[0] = right[0];
            left[1] = right[1];
            right.fill(1);
        }

        // Check that this pointer is still valid.
        for row in lhs.iter_rows_mut() {
            row.fill(2);
        }

        for row in data.iter() {
            assert_eq!(row, &[2, 2, 1, 1]);
        }
    }
}
