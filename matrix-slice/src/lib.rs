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
        assert!(block.rows >= len);

        Some(BlockRef {
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
    pub fn select_cols<R>(self, range: R) -> Option<BlockRef<'data, T>>
    where
        R: MatrixIndex,
    {
        let (start, len) = range.into_start_and_len(self.block.rows)?;
        let (_, block, offset) = self.block.split_at_col(start)?;
        assert!(block.cols >= len);

        Some(BlockRef {
            block: BlockSlice { cols: len, ..block },
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
        let idx = self.block.in_bounds_index(index.0, index.1);
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
    pub fn select_cols<R>(self, range: R) -> Option<BlockMut<'data, T>>
    where
        R: MatrixIndex,
    {
        let (start, len) = range.into_start_and_len(self.block.rows)?;
        let (_, block, offset) = self.block.split_at_col(start)?;
        assert!(block.cols >= len);

        Some(BlockMut {
            block: BlockSlice { cols: len, ..block },
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
        let idx = self.block.in_bounds_index(index.0, index.1);
        // SAFETY: Index is bounded by `total_span` which itself is a lower estimate of the
        // provenance of the pointer.
        unsafe { &*self.data.as_ptr().add(idx) }
    }
}

impl<T> ops::IndexMut<(usize, usize)> for BlockMut<'_, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.block.in_bounds_index(index.0, index.1);
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
    fn in_bounds_index(&self, row: usize, col: usize) -> usize {
        assert!(row < self.rows);
        assert!(col < self.cols);
        let idx = row * self.pitch + col;
        debug_assert!(idx < self.total_span());
        idx
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

mod sealed {
    use core::ops;

    pub trait Sealed {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)>;
    }

    impl Sealed for ops::Range<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            if self.start <= self.end && self.end <= dim {
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
                Some((start, end - start + 1))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeFrom<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            if self.start <= dim {
                Some((self.start, dim - self.start))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeTo<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            if self.end <= dim {
                Some((0, self.end))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeToInclusive<usize> {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            if self.end < dim {
                Some((0, self.end + 1))
            } else {
                None
            }
        }
    }

    impl Sealed for ops::RangeFull {
        fn into_start_and_len(self, dim: usize) -> Option<(usize, usize)> {
            Some((0, dim))
        }
    }
}

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
}
