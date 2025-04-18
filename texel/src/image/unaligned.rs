//! Buffers that can work with unaligned underlying data.
//!
//! These are mostly for IO purposes since most compute algorithms are not expected to support
//! interactions with these buffers. However, these buffers support more generalized layouts with
//! the goal of admitting a description of arbitrary external resources. In consequence, these
//! buffers interact by references only.
mod sealed {
    use crate::buf::{atomic_buf, buf, cell_buf};
    use crate::layout::Layout;
    use core::ops::Range;

    pub trait LayoutEngineCore {
        type Layout: Layout;

        fn layout(&self) -> &Self::Layout;

        fn ranges(&self) -> impl Iterator<Item = Range<usize>>;
    }

    pub trait Loadable {
        fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize);
        fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize);
        fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize);
    }

    pub trait Storable {
        fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize);
        fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize);
        fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize);
    }
}

use core::{cell::Cell, ops::Range};
use sealed::{LayoutEngineCore, Loadable, Storable};

use crate::buf::{atomic_buf, buf, cell_buf};
use crate::image::{
    AtomicImage, AtomicImageRef, CellImage, CellImageRef, Image, ImageMut, ImageRef,
};
use crate::layout::{Bytes, Layout};
use crate::texels;

pub struct DataRef<'lt, Layout = Bytes> {
    data: &'lt [u8],
    layout: Layout,
    offset: usize,
}

pub struct DataMut<'lt, Layout = Bytes> {
    data: &'lt mut [u8],
    layout: Layout,
    offset: usize,
}

pub struct DataCells<'lt, Layout = Bytes> {
    data: &'lt [Cell<u8>],
    layout: Layout,
    offset: usize,
}

/// Borrows from a data source to read from it data into images.
///
/// The type parameter is the layout engine which defines the byte spans of data to be copied and
/// in doing so controls the overhead of the operation. Note that the type must implement a sealed
/// trait for all the main algorithms. The respective constructors on [`DataRef`], [`DataMut`],
/// [`DataCells`]  choose this parameter.
pub struct AsCopySource<'lt, E> {
    inner: &'lt dyn Storable,
    engine: E,
}

/// Borrows from a mutable data source to fill it with data from mages.
///
/// The type parameter is the layout engine which defines the byte spans of data to be copied and
/// in doing so controls the overhead of the operation. Note that the type must implement a sealed
/// trait for all the main algorithms. The respective constructors on [`DataRef`], [`DataMut`],
/// [`DataCells`]  choose this parameter.
pub struct AsCopyTarget<'lt, E> {
    inner: &'lt mut dyn Loadable,
    engine: E,
}

/// Documents the different layout engines.
///
/// This trait requires a sealed trait, it exists for documentation.
pub trait LayoutEngine: sealed::LayoutEngineCore {}

/// Copies all bytes within the bounds of this layout.
pub struct WholeLayout<'lt, L> {
    inner: &'lt L,
    offset: usize,
}

impl<L: Layout> LayoutEngine for WholeLayout<'_, L> {}

impl<L: Layout> sealed::LayoutEngineCore for WholeLayout<'_, L> {
    type Layout = L;

    fn layout(&self) -> &L {
        self.inner
    }

    fn ranges(&self) -> impl Iterator<Item = Range<usize>> {
        let bytes = self.inner.byte_len();
        [self.offset..self.offset + bytes].into_iter()
    }
}

impl<'lt> DataRef<'lt, Bytes> {
    /// Treat a whole input buffer as image bytes.
    pub fn new(data: &'lt [u8]) -> Self {
        DataRef {
            data,
            layout: Bytes(core::mem::size_of_val(data)),
            offset: 0,
        }
    }
}

impl<'lt, L> DataRef<'lt, L> {
    /// Verifies the data against the layout before construction.
    ///
    /// Note that the type has no hard invariants.
    pub fn with_layout(data: &'lt [u8], layout: L, at: usize) -> Option<Self>
    where
        L: Layout,
    {
        Some(data)
            .filter(|data| {
                if let Some(partial) = data.get(at..) {
                    <dyn Layout>::fits_data(&layout, partial)
                } else {
                    false
                }
            })
            .map(|data| DataRef {
                data,
                layout,
                offset: Default::default(),
            })
    }

    /// Copy the data bytes of the layout to the byte buffer.
    pub fn write_to_buf<'data>(&self, buffer: &mut buf) {
        let len = buffer.len().min(core::mem::size_of_val(self.data));
        buffer[..len].copy_from_slice(self.data);
    }
}

impl Storable for &'_ [u8] {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = &mut buffer.as_bytes_mut()[into..][..len];
        let source = &self[what.start..what.end];
        target.copy_from_slice(source);
    }

    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_texels(texels::U8).as_slice_of_cells()[into..][..len];
        let target = &self[what.start..what.end];
        texels::U8.store_cell_slice(source, target);
    }

    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = buffer.index(texels::U8.to_range(into..into + len).unwrap());
        let source = &self[what.start..what.end];
        texels::U8.store_atomic_slice(target, source);
    }
}

impl<'lt, L> DataMut<'lt, L> {
    /// Verifies the data against the layout before construction.
    ///
    /// Note that the type has no hard invariants.
    pub fn new(data: &'lt mut [u8], layout: L) -> Option<Self>
    where
        L: Layout,
    {
        Some(data)
            .filter(|data| <dyn Layout>::fits_data(&layout, data))
            .map(|data| DataMut {
                data,
                layout,
                offset: Default::default(),
            })
    }

    /// Copy the data bytes of the layout to the byte buffer.
    pub fn write_to_buf(&self, buffer: &mut buf) {
        let len = buffer.len().min(core::mem::size_of_val(self.data));
        buffer[..len].copy_from_slice(self.data);
    }

    /// Copy the data bytes of the layout to the byte buffer.
    pub fn read_from_buf(&mut self, buffer: &buf) {
        let len = buffer.len().min(core::mem::size_of_val(self.data));
        self.data[..len].copy_from_slice(buffer);
    }
}

impl Storable for &'_ mut [u8] {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = &mut buffer.as_bytes_mut()[into..][..len];
        let source = &self[what.start..what.end];
        target.copy_from_slice(source);
    }

    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_texels(texels::U8).as_slice_of_cells()[into..][..len];
        let target = &self[what.start..what.end];
        texels::U8.store_cell_slice(source, target);
    }

    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = buffer.index(texels::U8.to_range(into..into + len).unwrap());
        let source = &self[what.start..what.end];
        texels::U8.store_atomic_slice(target, source);
    }
}

impl Loadable for &'_ mut [u8] {
    #[track_caller]
    fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_bytes()[into..][..len];
        let target = &mut self[what.start..what.end];
        target.copy_from_slice(source);
    }

    fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_texels(texels::U8).as_slice_of_cells()[into..][..len];
        let target = &mut self[what.start..what.end];
        texels::U8.load_cell_slice(source, target);
    }

    fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = buffer.index(texels::U8.to_range(into..into + len).unwrap());
        let target = &mut self[what.start..what.end];
        texels::U8.load_atomic_slice(source, target);
    }
}

impl<'lt, L> DataCells<'lt, L> {
    /// Verifies the data against the layout before construction.
    ///
    /// Note that the type has no hard invariants.
    pub fn new(data: &'lt [Cell<u8>], layout: L) -> Option<Self>
    where
        L: Layout,
    {
        Some(data)
            .filter(|data| <dyn Layout>::fits_data(&layout, data))
            .map(|data| DataCells {
                data,
                layout,
                offset: Default::default(),
            })
    }

    /// Copy the data bytes of the layout to the byte buffer.
    #[track_caller]
    pub fn write_to_buf(&self, buffer: &mut buf) {
        let len = core::mem::size_of_val(self.data);
        self.data.store_to_buf(buffer, 0..len, 0);
    }

    /// Copy the data bytes of the layout to the byte buffer.
    pub fn read_from_buf(&mut self, buffer: &buf) {
        let len = core::mem::size_of_val(self.data);
        self.data.load_from_buf(buffer, 0..len, 0);
    }

    /// Copy all bytes contained in this layout.
    pub fn as_source(&self) -> AsCopySource<'_, WholeLayout<'_, L>>
    where
        L: Layout,
    {
        AsCopySource {
            inner: &self.data,
            engine: WholeLayout {
                inner: &self.layout,
                offset: self.offset,
            },
        }
    }

    /// Copy all bytes contained in this layout.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, WholeLayout<'_, L>>
    where
        L: Layout,
    {
        AsCopyTarget {
            inner: &mut self.data,
            engine: WholeLayout {
                inner: &self.layout,
                offset: self.offset,
            },
        }
    }
}

impl Storable for &'_ [Cell<u8>] {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = &mut buffer.as_bytes_mut()[into..][..len];
        let source = &self[what.start..what.end];
        crate::texels::U8.load_cell_slice(source, target);
    }

    #[track_caller]
    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_texels(texels::U8).as_slice_of_cells()[into..][..len];
        let target = &self[what.start..what.end];
        texels::U8.cell_memory_copy(source, target);
    }

    #[track_caller]
    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        todo!()
    }
}

impl Loadable for &'_ [Cell<u8>] {
    #[track_caller]
    fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_bytes()[into..][..len];
        let target = &self[what.start..what.end];
        texels::U8.store_cell_slice(target, source);
    }

    fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_texels(texels::U8).as_slice_of_cells()[into..][..len];
        let target = &self[what.start..what.end];
        texels::U8.cell_memory_copy(source, target);
    }

    fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = buffer.index(texels::U8.to_range(into..into + len).unwrap());
        todo!()
    }
}

impl<E: LayoutEngine> AsCopySource<'_, E> {
    fn engine_to_buf_at(&self, buffer: &mut buf, offset: usize) {
        // Make sure we compile this once per iterator type. Then for instance there is only one
        // such instance for all LayoutEngine types instead of one per different layout
        #[inline(never)]
        fn ranges_to_buf_at(
            ranges: impl Iterator<Item = Range<usize>>,
            store: &dyn Storable,
            buffer: &mut buf,
            offset: usize,
        ) {
            for range in ranges {
                store.store_to_buf(buffer, range, offset)
            }
        }

        ranges_to_buf_at(self.engine.ranges(), self.inner, buffer, offset);
    }

    fn engine_to_cell_buf_at(&self, buffer: &cell_buf, offset: usize) {
        // Make sure we compile this once per iterator type. Then for instance there is only one
        // such instance for all LayoutEngine types instead of one per different layout
        #[inline(never)]
        fn ranges_to_buf_at(
            ranges: impl Iterator<Item = Range<usize>>,
            store: &dyn Storable,
            buffer: &cell_buf,
            offset: usize,
        ) {
            for range in ranges {
                store.store_to_cell(buffer, range, offset)
            }
        }

        ranges_to_buf_at(self.engine.ranges(), self.inner, buffer, offset);
    }

    fn engine_to_atomic_buf_at(&self, buffer: &atomic_buf, offset: usize) {
        // Make sure we compile this once per iterator type. Then for instance there is only one
        // such instance for all LayoutEngine types instead of one per different layout
        #[inline(never)]
        fn ranges_to_buf_at(
            ranges: impl Iterator<Item = Range<usize>>,
            store: &dyn Storable,
            buffer: &atomic_buf,
            offset: usize,
        ) {
            for range in ranges {
                store.store_to_atomic(buffer, range, offset)
            }
        }

        ranges_to_buf_at(self.engine.ranges(), self.inner, buffer, offset);
    }

    /// Write to an image, changing the layout in the process.
    ///
    /// See [`Self::write_to_image`] but works on a borrowed buffer, only when it has the same
    /// layout type. Clones the layout to the image buffer.
    ///
    /// Reallocates the image buffer when necessary to ensure that the allocated buffer fits the
    /// new data's layout.
    ///
    /// Consider [`Self::write_to_buf`] with the [`Image::as_capacity_buf_mut`] when you want to
    /// ignore the layout value of the target buffer.
    pub fn write_to<L>(&self, buffer: &mut Image<E::Layout>)
    where
        E::Layout: Clone + Layout,
    {
        let layout = self.engine.layout();
        buffer.layout_mut_unguarded().clone_from(layout);
        buffer.ensure_layout();
        self.engine_to_buf_at(buffer.as_capacity_buf_mut(), 0);
    }

    /// Write to an image, changing the layout in the process.
    ///
    /// Reallocates the image buffer when necessary to ensure that the allocated buffer fits the
    /// new data's layout.
    pub fn write_to_image(&self, buffer: Image<impl Layout>) -> Image<E::Layout>
    where
        E::Layout: Clone + Layout,
    {
        let mut buffer = buffer.with_layout(self.engine.layout().clone());
        self.engine_to_buf_at(buffer.as_capacity_buf_mut(), 0);
        buffer
    }

    /// Write to a mutable borrowed buffer with layout.
    ///
    /// First verifies that the data will fit into the target. Then returns `Some` with a new
    /// reference to the target buffer that is using the data's layout. Otherwise, returns `None`.
    pub fn write_to_mut<'data>(
        &self,
        buffer: ImageMut<'data, impl Layout>,
    ) -> Option<ImageMut<'data, E::Layout>>
    where
        E::Layout: Clone + Layout,
    {
        let mut buffer = buffer.with_layout(self.engine.layout().clone())?;
        self.engine_to_buf_at(buffer.as_mut_buf(), 0);
        Some(buffer)
    }

    /// Write to a locally shared buffer with layout.
    ///
    /// First verifies that the data will fit into the target. Then returns `Some` with a new
    /// reference to the target buffer that is using the data's layout. Otherwise, returns `None`.
    pub fn write_to_cell_ref<'data>(
        &self,
        buffer: CellImageRef<'data, impl Layout>,
    ) -> Option<CellImageRef<'data, E::Layout>>
    where
        E::Layout: Clone + Layout,
    {
        let buffer = buffer.checked_with_layout(self.engine.layout().clone())?;
        self.engine_to_cell_buf_at(buffer.as_cell_buf(), 0);
        Some(buffer)
    }

    /// Write to a mutable borrowed buffer with layout.
    ///
    /// First verifies that the data will fit into the target. Then returns `Some` with a new
    /// reference to the target buffer that is using the data's layout. Otherwise, returns `None`.
    pub fn write_to_atomic_ref<'data>(
        &self,
        buffer: AtomicImageRef<'data, impl Layout>,
    ) -> Option<AtomicImageRef<'data, E::Layout>>
    where
        E::Layout: Clone + Layout,
    {
        let buffer = buffer.checked_with_layout(self.engine.layout().clone())?;
        self.engine_to_atomic_buf_at(buffer.as_capacity_atomic_buf(), 0);
        Some(buffer)
    }
}

impl<E: LayoutEngine> AsCopyTarget<'_, E> {
    fn engine_from_buf_at(&mut self, buffer: &buf, offset: usize) {
        // Make sure we compile this once per iterator type. Then for instance there is only one
        // such instance for all LayoutEngine types instead of one per different layout
        #[inline(never)]
        fn ranges_from_buf_at(
            ranges: impl Iterator<Item = Range<usize>>,
            store: &mut dyn Loadable,
            buffer: &buf,
            offset: usize,
        ) {
            for range in ranges {
                store.load_from_buf(buffer, range, offset)
            }
        }

        ranges_from_buf_at(self.engine.ranges(), self.inner, buffer, offset);
    }

    /// Read out data from a borrowed buffer.
    ///
    /// This reads data up to our layout. It does not interpret the data with the layout of
    /// the argument buffer.
    pub fn read_from_ref(&mut self, buffer: ImageRef<'_, impl Layout>) {
        self.engine_from_buf_at(buffer.as_buf(), 0);
    }
}
