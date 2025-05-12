//! Buffers that can work with unaligned underlying data.
//!
//! These are mostly for IO purposes since most compute algorithms are not expected to support
//! interactions with these buffers. However, these buffers support more generalized layouts with
//! the goal of admitting a description of arbitrary external resources. In consequence, these
//! buffers interact by references only.
//!
//! ## Design constraints
//!
//! - Many methods completely disregard some of the layouts. When we write into an `Image` for
//!   instance, we treat it just as a container for bytes under the _input_ layout. We do not
//!   interact with the target's layout at all. In these cases we request a `Bytes` layout as an
//!   explicit opt-in. You can [`decay`][`Image::decay`] all buffer types to conveniently do so at
//!   the call site.
//! - The API may be suggestive, in terms of types, that the layout of an `Image` may be mutated by
//!   interpreting it as a write target of, e.g. `Image<Relocated<Plane>>`. This however is still
//!   wrong. When we write of course the aliasing of other planes is not considered. And we never
//!   read the relocation offset, again just disregarding any existing layout data. This is a big
//!   trap. The design around this is that methods are split into two parts. Those that do *not* take
//!   target layouts into account are inherent methods on [`AsCopySource`] and [`AsCopyTarget`]
//!   respectively. Those that do are located directly on the images such as [`Image::assign`]. This
//!   is not perfect, see deeper considerations on planes in the following.
//!
//! ## Open Design issues
//!
//! - Atomic inputs are probably valuable but we can not command users to use our own atomic type.
//!   Since atomic sizes must not mix, these kinds of buffers require different copy methods for
//!   each single kind of underlying atomic! Not providing these however risks users doing unsound
//!   things, such as improperly casting any of their atomic buffers to `[u8]` or [`Cell`] or
//!   something.
//! - To expand on layouts, of course it is also improper to expect that the input buffer contains
//!   enough data for the whole image, instead it is probably just that single plane. This messes
//!   with the expectations and types involved in the 'copy engine' implementation detail. What is
//!   missing is a discoverable way that hands the plane index to some engine and then resolves the
//!   necessary [`Relocated::offset`] when consuming the target. This should somehow not duplicate too
//!   many interfaces, and really we want to avoid the confusion of having both available, right?
//! - Writing the plane of an allocated image is common, but rather not straightforward. When the
//!   input data has different layout we must modify the containing layout's definition of this
//!   plane, which may fail of course and is not a generic operation at all. It could also entail
//!   relocating some other planes which is far more work than the simple copy engine. So that
//!   should be out-of-scope for the data transfer itself but then ergonomics should have a way to
//!   ensure reserved space for a particular plane in advance.
//! - Do we stack strategies, and how? In particular when copying an array of planes each of which
//!   are matrices we want to copy everything row-by-row but right now this would require a
//!   specialized engine for `impl Planar<impl MatrixLayout>` instead of being able to compose.
//! - Think of blitting. When we write multiple planes, the input data can contain them completely
//!   unaligned but this can not be expressed with properly planar layouts. We not a 'packed
//!   planes' type or so that does not implement `PlaneOf` in terms of `Relocated<T>`.
//!
mod sealed {
    use crate::buf::{atomic_buf, buf, cell_buf};
    use crate::layout::Layout;
    use core::ops::Range;

    pub trait LayoutEngineCore {
        type Layout: Layout;

        fn consume_layout(&mut self) -> Self::Layout;

        /// The byte ranges in the data buffer.
        fn buffer_ranges(&self) -> impl Iterator<Item = Range<usize>>;

        /// The base offset in the image.
        ///
        /// This should be such that the final image can be interpreted with [`Self::layout`].
        fn image_offset(&self) -> usize;
    }

    pub trait Loadable {
        fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, at: usize);
        fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, at: usize);
        fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, at: usize);
    }

    pub trait Storable {
        fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, at: usize);
        fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, at: usize);
        fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, at: usize);
    }

    /// So we can abstract over the invocations of `Loadable::load_from_{buf,cell,atomic}`.
    pub(crate) trait LoadSource {
        fn load(&mut self, into: &mut dyn Loadable, what: Range<usize>, at: usize);
    }

    /// So we can abstract over the invocations of `Storable::store_to_{buf,cell,atomic}`.
    pub(crate) trait StoreTarget {
        fn store(&mut self, into: &dyn Storable, what: Range<usize>, at: usize);
    }

    impl LoadSource for &'_ buf {
        fn load(&mut self, into: &mut dyn Loadable, what: Range<usize>, at: usize) {
            into.load_from_buf(self, what, at)
        }
    }

    impl LoadSource for &'_ cell_buf {
        fn load(&mut self, into: &mut dyn Loadable, what: Range<usize>, at: usize) {
            into.load_from_cell(self, what, at)
        }
    }

    impl LoadSource for &'_ atomic_buf {
        fn load(&mut self, into: &mut dyn Loadable, what: Range<usize>, at: usize) {
            into.load_from_atomic(self, what, at)
        }
    }

    impl StoreTarget for &'_ mut buf {
        fn store(&mut self, into: &dyn Storable, what: Range<usize>, at: usize) {
            into.store_to_buf(self, what, at)
        }
    }

    impl StoreTarget for &'_ cell_buf {
        fn store(&mut self, into: &dyn Storable, what: Range<usize>, at: usize) {
            into.store_to_cell(self, what, at)
        }
    }

    impl StoreTarget for &'_ atomic_buf {
        fn store(&mut self, into: &dyn Storable, what: Range<usize>, at: usize) {
            into.store_to_atomic(self, what, at)
        }
    }
}

use core::{cell::Cell, ops::Range};
use sealed::{Loadable, Storable};

use crate::buf::{atomic_buf, buf, cell_buf, AtomicBuffer, Buffer, CellBuffer};
use crate::image::{
    AtomicImage, AtomicImageRef, CellImage, CellImageRef, Image, ImageMut, ImageRef,
};
use crate::layout::{AlignedOffset, Bytes, Layout, Relocated};
use crate::{texels, BufferReuseError};

/// A buffer with layout, not aligned to any particular boundary.
pub struct DataRef<'lt, Layout = Bytes> {
    data: &'lt [u8],
    layout: Layout,
    offset: usize,
}

/// A mutable buffer with layout, not aligned to any particular boundary.
pub struct DataMut<'lt, Layout = Bytes> {
    data: &'lt mut [u8],
    layout: Layout,
    offset: usize,
}

/// A cell buffer with layout, not aligned to any particular boundary.
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
    /// Construct from an explicit layout.
    ///
    /// This wraps an underlying buffer which has image data of the indicated layout from its
    /// `start` byte onwards.
    pub fn with_layout_at(data: &'lt [u8], layout: L, start: usize) -> Option<Self>
    where
        L: Layout,
    {
        Some(data)
            .filter(|data| {
                if let Some(partial) = data.get(start..) {
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

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: &self.data,
            engine: RangeEngine::new(&self.layout, self.offset),
        }
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
        let target = &buffer.as_texels(texels::U8).as_slice_of_cells()[into..][..len];
        let source = &self[what.start..what.end];
        texels::U8.store_cell_slice(target, source);
    }

    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = buffer.index(texels::U8.to_range(into..into + len).unwrap());
        let source = &self[what.start..what.end];
        texels::U8.store_atomic_slice(target, source);
    }
}

impl<'lt> DataMut<'lt, Bytes> {
    /// Treat a whole input buffer as image bytes.
    pub fn new(data: &'lt mut [u8]) -> Self {
        DataMut {
            layout: Bytes(core::mem::size_of_val(data)),
            data,
            offset: 0,
        }
    }
}

impl<'lt, L> DataMut<'lt, L> {
    /// Construct from an explicit layout.
    ///
    /// This wraps an underlying mutable buffer which has image data of the indicated layout from
    /// its `start` byte onwards.
    pub fn with_layout_at(data: &'lt mut [u8], layout: L, start: usize) -> Option<Self>
    where
        L: Layout,
    {
        Some(data)
            .filter(|data| {
                if let Some(partial) = data.get(start..) {
                    <dyn Layout>::fits_data(&layout, partial)
                } else {
                    false
                }
            })
            .map(|data| DataMut {
                data,
                layout,
                offset: start,
            })
    }

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: &self.data,
            engine: RangeEngine::new(&self.layout, self.offset),
        }
    }

    /// An adapter writing to this buffer in one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopyTarget {
            inner: &mut self.data,
            engine: RangeEngine::new(&self.layout, self.offset),
        }
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

impl<'lt> DataCells<'lt, Bytes> {
    /// Treat a whole input buffer as image bytes.
    pub fn new(data: &'lt [Cell<u8>]) -> Self {
        DataCells {
            layout: Bytes(core::mem::size_of_val(data)),
            data,
            offset: 0,
        }
    }
}

impl<'lt, L> DataCells<'lt, L> {
    /// Verifies the data against the layout before construction.
    ///
    /// This wraps an shared buffer which has image data of the indicated layout from its `start`
    /// byte onwards.
    pub fn with_layout_at(data: &'lt [Cell<u8>], layout: L, start: usize) -> Option<Self>
    where
        L: Layout,
    {
        Some(data)
            .filter(|data| {
                if let Some(partial) = data.get(start..) {
                    <dyn Layout>::fits_data(&layout, partial)
                } else {
                    false
                }
            })
            .map(|data| DataCells {
                data,
                layout,
                offset: Default::default(),
            })
    }

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: &self.data,
            engine: RangeEngine::new(&self.layout, self.offset),
        }
    }

    /// An adapter writing to this buffer in one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopyTarget {
            inner: &mut self.data,
            engine: RangeEngine::new(&self.layout, self.offset),
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
        let len = what.len();
        let target = buffer.index(texels::U8.to_range(into..into + len).unwrap());
        let source = &self[what.start..what.end];
        texels::U8.store_atomic_from_cells(target, source);
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
        let target = &self[what.start..what.end];
        texels::U8.load_atomic_to_cells(source, target);
    }
}

impl<'buf, E: LayoutEngine> AsCopySource<'buf, E> {
    /// Offset the target location of this copy operation, to anther planar location.
    pub fn and_relocated(self, offset: AlignedOffset) -> AsCopySource<'buf, RelocateEngine<E>>
    where
        E::Layout: Clone,
    {
        AsCopySource {
            inner: self.inner,
            engine: RelocateEngine {
                inner: self.engine,
                offset,
            },
        }
    }
}

impl<E: LayoutEngine> AsCopySource<'_, E> {
    fn engine_to_buf_at(&self, buffer: impl sealed::StoreTarget) {
        // Make sure we compile this once per iterator type and buffer type combination. Then for
        // instance there is only one such instance for all LayoutEngine types instead of one per
        // different layout.
        #[inline(never)]
        fn ranges_to_buf_at(
            ranges: impl Iterator<Item = Range<usize>>,
            store: &dyn Storable,
            mut buffer: impl sealed::StoreTarget,
            offset: usize,
        ) {
            for range in ranges {
                buffer.store(store, range, offset)
            }
        }

        ranges_to_buf_at(
            self.engine.buffer_ranges(),
            self.inner,
            buffer,
            self.engine.image_offset(),
        );
    }

    /// Write to an image, changing the layout in the process.
    ///
    /// Reallocates the image buffer when necessary to ensure that the allocated buffer fits the
    /// new data's layout.
    pub fn write_to_image(mut self, buffer: Image<Bytes>) -> Image<E::Layout> {
        let mut buffer = buffer.with_layout(self.engine.consume_layout());
        self.engine_to_buf_at(buffer.as_capacity_buf_mut());
        buffer
    }

    /// Write to a mutable borrowed buffer with layout.
    ///
    /// First verifies that the data will fit into the target. Then returns `Some` with a new
    /// reference to the target buffer that is using the data's layout. Otherwise, returns `None`.
    pub fn write_to_mut<'data>(
        mut self,
        buffer: ImageMut<'data, Bytes>,
    ) -> Option<ImageMut<'data, E::Layout>> {
        let mut buffer = buffer.with_layout(self.engine.consume_layout())?;
        self.engine_to_buf_at(buffer.as_mut_buf());
        Some(buffer)
    }

    /// Write to an image, changing the layout in the process.
    ///
    /// Fails when allocated buffer does not fits the new data's layout.
    pub fn write_to_cell_image(mut self, buffer: CellImage<Bytes>) -> Option<CellImage<E::Layout>>
    where
        E::Layout: Clone + Layout,
    {
        let buffer = buffer.try_with_layout(self.engine.consume_layout()).ok()?;
        self.engine_to_buf_at(buffer.as_capacity_cell_buf());
        Some(buffer)
    }

    /// Write to a locally shared buffer with layout.
    ///
    /// First verifies that the data will fit into the target. Then returns `Some` with a new
    /// reference to the target buffer that is using the data's layout. Otherwise, returns `None`.
    pub fn write_to_cell_ref<'data>(
        mut self,
        buffer: CellImageRef<'data, Bytes>,
    ) -> Option<CellImageRef<'data, E::Layout>> {
        let buffer = buffer.checked_with_layout(self.engine.consume_layout())?;
        self.engine_to_buf_at(buffer.as_cell_buf());
        Some(buffer)
    }

    /// Write to an image, changing the layout in the process.
    ///
    /// Fails when allocated buffer does not fits the new data's layout.
    pub fn write_to_atomic_image(
        mut self,
        buffer: AtomicImage<Bytes>,
    ) -> Option<AtomicImage<E::Layout>>
    where
        E::Layout: Clone + Layout,
    {
        let buffer = buffer.try_with_layout(self.engine.consume_layout()).ok()?;
        self.engine_to_buf_at(buffer.as_capacity_atomic_buf());
        Some(buffer)
    }

    /// Write to a mutable borrowed buffer with layout.
    ///
    /// First verifies that the data will fit into the target. Then returns `Some` with a new
    /// reference to the target buffer that is using the data's layout. Otherwise, returns `None`.
    pub fn write_to_atomic_ref<'data>(
        mut self,
        buffer: AtomicImageRef<'data, Bytes>,
    ) -> Option<AtomicImageRef<'data, E::Layout>>
    where
        E::Layout: Clone + Layout,
    {
        let buffer = buffer.checked_with_layout(self.engine.consume_layout())?;
        self.engine_to_buf_at(buffer.as_capacity_atomic_buf());
        Some(buffer)
    }
}

impl Storable for Buffer {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_buf(&self.as_bytes(), buffer, what, into)
    }

    #[track_caller]
    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_cell(&self.as_bytes(), buffer, what, into)
    }

    #[track_caller]
    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_atomic(&self.as_bytes(), buffer, what, into)
    }
}

impl Loadable for Buffer {
    #[track_caller]
    fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize) {
        <&'_ mut [u8]>::load_from_buf(&mut self.as_bytes_mut(), buffer, what, into)
    }

    fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ mut [u8]>::load_from_cell(&mut self.as_bytes_mut(), buffer, what, into)
    }

    fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ mut [u8]>::load_from_atomic(&mut self.as_bytes_mut(), buffer, what, into)
    }
}

impl<L> Image<L> {
    /// Write to an image, changing the layout in the process.
    ///
    /// Allocates, contrary to `assign` functions on shared and reference types, if the allocated
    /// buffer does not fit the new data's layout. Then copies data and assigns the layout to the
    /// image buffer.
    ///
    /// Consider [`AsCopySource::write_to_mut`] with the whole [`Image::as_mut`] buffer when you
    /// want to instead ignore the keep the current layout and only copy data. See
    /// [`AsCopySource::write_to_image`] for changing the layout type in the process.
    pub fn assign<E>(&mut self, mut data: AsCopySource<'_, E>)
    where
        E: LayoutEngine<Layout = L>,
        L: Layout,
    {
        let layout = data.engine.consume_layout();
        *self.layout_mut_unguarded() = layout;
        self.ensure_layout();
        data.engine_to_buf_at(self.as_capacity_buf_mut());
    }

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: self.inner.get(),
            engine: RangeEngine::new(self.layout(), 0),
        }
    }

    /// An adapter writing to this buffer in one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopyTarget {
            engine: RangeEngine::new(self.layout(), 0),
            inner: self.inner.get_mut(),
        }
    }
}

impl Storable for &'_ buf {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_buf(&self.as_bytes(), buffer, what, into)
    }

    #[track_caller]
    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_cell(&self.as_bytes(), buffer, what, into)
    }

    #[track_caller]
    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_atomic(&self.as_bytes(), buffer, what, into)
    }
}

impl Storable for &'_ mut buf {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_buf(&self.as_bytes(), buffer, what, into)
    }

    #[track_caller]
    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_cell(&self.as_bytes(), buffer, what, into)
    }

    #[track_caller]
    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ [u8]>::store_to_atomic(&self.as_bytes(), buffer, what, into)
    }
}

impl Loadable for &'_ mut buf {
    #[track_caller]
    fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize) {
        <&'_ mut [u8]>::load_from_buf(&mut self.as_bytes_mut(), buffer, what, into)
    }

    fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ mut [u8]>::load_from_cell(&mut self.as_bytes_mut(), buffer, what, into)
    }

    fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ mut [u8]>::load_from_atomic(&mut self.as_bytes_mut(), buffer, what, into)
    }
}

impl<L> ImageMut<'_, L> {
    /// Write to an image, changing the layout in the process.
    ///
    /// Returns an error and keeps the current layout unchanged if the allocated buffer does not
    /// fit the new data's layout. Otherwise copies data and assigns the layout to the image
    /// buffer.
    ///
    /// See [`AsCopySource::write_to_mut`] for changing the layout type in the process.
    pub fn assign<E>(&mut self, mut data: AsCopySource<'_, E>) -> Result<(), BufferReuseError>
    where
        E: LayoutEngine<Layout = L>,
        L: Layout,
    {
        let layout = data.engine.consume_layout();
        self.try_set_layout(layout)?;
        data.engine_to_buf_at(self.as_capacity_buf_mut());
        Ok(())
    }

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: self.inner.get(),
            engine: RangeEngine::new(self.layout(), 0),
        }
    }

    /// An adapter writing to this buffer in one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopyTarget {
            engine: RangeEngine::new(self.layout(), 0),
            inner: self.inner.get_mut(),
        }
    }
}

impl<L> ImageRef<'_, L> {
    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: self.inner.get(),
            engine: RangeEngine::new(self.layout(), 0),
        }
    }
}

impl Storable for CellBuffer {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        <&'_ cell_buf>::store_to_buf(&&**self, buffer, what, into)
    }

    #[track_caller]
    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ cell_buf>::store_to_cell(&&**self, buffer, what, into)
    }

    #[track_caller]
    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ cell_buf>::store_to_atomic(&&**self, buffer, what, into)
    }
}

impl Loadable for CellBuffer {
    #[track_caller]
    fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize) {
        <&'_ cell_buf>::load_from_buf(&mut &**self, buffer, what, into)
    }

    fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ cell_buf>::load_from_cell(&mut &**self, buffer, what, into)
    }

    fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ cell_buf>::load_from_atomic(&mut &**self, buffer, what, into)
    }
}

impl<L> CellImage<L> {
    /// Write to this image, modifying the view of layout in the process.
    ///
    /// Returns an error and keeps the current layout unchanged if the allocated buffer does not
    /// fit the new data's layout. Otherwise copies data and assigns the layout to the image
    /// buffer.
    ///
    /// Consider [`AsCopySource::write_to_cell_ref`] with the whole [`Self::as_ref`] buffer when you
    /// want to instead ignore the keep the current layout and only copy data. See
    /// [`AsCopySource::write_to_cell_image`] for changing the layout type in the process.
    pub fn assign<E>(&mut self, mut data: AsCopySource<'_, E>) -> Result<(), BufferReuseError>
    where
        E: LayoutEngine<Layout = L>,
        L: Layout,
    {
        let layout = data.engine.consume_layout();
        self.try_set_layout(layout)?;
        data.engine_to_buf_at(self.as_capacity_cell_buf());
        Ok(())
    }

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: self.inner.get(),
            engine: RangeEngine::new(self.layout(), 0),
        }
    }

    /// An adapter writing to this buffer in one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopyTarget {
            engine: RangeEngine::new(self.layout(), 0),
            inner: self.inner.get_mut(),
        }
    }
}

impl Storable for &'_ cell_buf {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        let inner = self.as_texels(texels::U8).as_slice_of_cells();
        <&'_ [Cell<u8>]>::store_to_buf(&inner, buffer, what, into)
    }

    #[track_caller]
    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let inner = self.as_texels(texels::U8).as_slice_of_cells();
        <&'_ [Cell<u8>]>::store_to_cell(&inner, buffer, what, into)
    }

    #[track_caller]
    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let inner = self.as_texels(texels::U8).as_slice_of_cells();
        <&'_ [Cell<u8>]>::store_to_atomic(&inner, buffer, what, into)
    }
}

impl Loadable for &'_ cell_buf {
    #[track_caller]
    fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize) {
        let mut inner = self.as_texels(texels::U8).as_slice_of_cells();
        <&'_ [Cell<u8>]>::load_from_buf(&mut inner, buffer, what, into)
    }

    fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let mut inner = self.as_texels(texels::U8).as_slice_of_cells();
        <&'_ [Cell<u8>]>::load_from_cell(&mut inner, buffer, what, into)
    }

    fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let mut inner = self.as_texels(texels::U8).as_slice_of_cells();
        <&'_ [Cell<u8>]>::load_from_atomic(&mut inner, buffer, what, into)
    }
}

impl<L> CellImageRef<'_, L> {
    /// Write to this image, modifying the view of layout in the process.
    ///
    /// Returns an error and keeps the current layout unchanged if the allocated buffer does not
    /// fit the new data's layout. Otherwise copies data and assigns the layout to the image
    /// buffer.
    ///
    /// See [`AsCopySource::write_to_cell_image`] for changing the layout type in the process.
    pub fn assign<E>(&mut self, mut data: AsCopySource<'_, E>) -> Result<(), BufferReuseError>
    where
        E: LayoutEngine<Layout = L>,
        L: Layout,
    {
        let layout = data.engine.consume_layout();
        self.try_set_layout(layout)?;
        data.engine_to_buf_at(self.as_capacity_cell_buf());
        Ok(())
    }

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: self.inner.get(),
            engine: RangeEngine::new(self.layout(), 0),
        }
    }

    /// An adapter writing to this buffer in one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopyTarget {
            engine: RangeEngine::new(self.layout(), 0),
            inner: self.inner.get_mut(),
        }
    }
}

impl Storable for AtomicBuffer {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        <&'_ atomic_buf>::store_to_buf(&&**self, buffer, what, into)
    }

    #[track_caller]
    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ atomic_buf>::store_to_cell(&&**self, buffer, what, into)
    }

    #[track_caller]
    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ atomic_buf>::store_to_atomic(&&**self, buffer, what, into)
    }
}

impl Loadable for AtomicBuffer {
    #[track_caller]
    fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize) {
        <&'_ atomic_buf>::load_from_buf(&mut &**self, buffer, what, into)
    }

    fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        <&'_ atomic_buf>::load_from_cell(&mut &**self, buffer, what, into)
    }

    fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        <&'_ atomic_buf>::load_from_atomic(&mut &**self, buffer, what, into)
    }
}

impl<L> AtomicImage<L> {
    /// Write to this image, modifying the view of layout in the process.
    ///
    /// Returns an error and keeps the current layout unchanged if the allocated buffer does not
    /// fit the new data's layout. Otherwise copies data and assigns the layout to the image
    /// buffer.
    ///
    /// Consider [`AsCopySource::write_to_atomic_ref`] with the whole [`Self::as_ref`] buffer when
    /// you want to instead ignore the keep the current layout and only copy data. See
    /// [`AsCopySource::write_to_atomic_image`] for changing the layout type in the process.
    pub fn assign<E>(&mut self, mut data: AsCopySource<'_, E>) -> Result<(), BufferReuseError>
    where
        E: LayoutEngine<Layout = L>,
        L: Layout,
    {
        let layout = data.engine.consume_layout();
        self.try_set_layout(layout)?;
        data.engine_to_buf_at(self.as_capacity_atomic_buf());
        Ok(())
    }

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: self.inner.get(),
            engine: RangeEngine::new(self.layout(), 0),
        }
    }

    /// An adapter writing to this buffer in one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopyTarget {
            engine: RangeEngine::new(self.layout(), 0),
            inner: self.inner.get_mut(),
        }
    }
}

impl Storable for &'_ atomic_buf {
    #[track_caller]
    fn store_to_buf(&self, buffer: &mut buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = &mut buffer.as_bytes_mut()[into..][..len];
        let source = self.index(texels::U8.to_range(what).unwrap());
        texels::U8.load_atomic_slice(source, target);
    }

    #[track_caller]
    fn store_to_cell(&self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = &buffer.as_texels(texels::U8).as_slice_of_cells()[into..][..len];
        let source = self.index(texels::U8.to_range(what).unwrap());
        texels::U8.load_atomic_to_cells(source, target);
    }

    #[track_caller]
    fn store_to_atomic(&self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let target = buffer.index(texels::U8.to_range(into..into + len).unwrap());
        let source = self.index(texels::U8.to_range(what).unwrap());
        texels::U8.atomic_memory_move(source, target);
    }
}

impl Loadable for &'_ atomic_buf {
    #[track_caller]
    fn load_from_buf(&mut self, buffer: &buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_bytes()[into..][..len];
        let target = self.index(texels::U8.to_range(what).unwrap());
        texels::U8.store_atomic_slice(target, source);
    }

    fn load_from_cell(&mut self, buffer: &cell_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = &buffer.as_texels(texels::U8).as_slice_of_cells()[into..][..len];
        let target = self.index(texels::U8.to_range(what).unwrap());
        texels::U8.store_atomic_from_cells(target, source);
    }

    fn load_from_atomic(&mut self, buffer: &atomic_buf, what: Range<usize>, into: usize) {
        let len = what.len();
        let source = buffer.index(texels::U8.to_range(into..into + len).unwrap());
        let target = self.index(texels::U8.to_range(what).unwrap());
        texels::U8.atomic_memory_move(source, target);
    }
}

impl<L> AtomicImageRef<'_, L> {
    /// Write to this image, modifying the view of layout in the process.
    ///
    /// Returns an error and keeps the current layout unchanged if the allocated buffer does not
    /// fit the new data's layout. Otherwise copies data and assigns the layout to the image
    /// buffer.
    ///
    /// See [`AsCopySource::write_to_atomic_ref`] for changing the layout type in the process.
    pub fn assign<E>(&mut self, mut data: AsCopySource<'_, E>) -> Result<(), BufferReuseError>
    where
        E: LayoutEngine<Layout = L>,
        L: Layout,
    {
        let layout = data.engine.consume_layout();
        self.try_set_layout(layout)?;
        data.engine_to_buf_at(self.as_capacity_atomic_buf());
        Ok(())
    }

    /// An adapter reading from the data as one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_source(&self) -> AsCopySource<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopySource {
            inner: self.inner.get(),
            engine: RangeEngine::new(self.layout(), 0),
        }
    }

    /// An adapter writing to this buffer in one contiguous chunk.
    ///
    /// See [`RangeEngine`] for more explanations.
    pub fn as_target(&mut self) -> AsCopyTarget<'_, RangeEngine<L>>
    where
        L: Clone + Layout,
    {
        AsCopyTarget {
            engine: RangeEngine::new(self.layout(), 0),
            inner: self.inner.get_mut(),
        }
    }
}

impl<E: LayoutEngine> AsCopyTarget<'_, E> {
    fn engine_from_buf_at(&mut self, buffer: impl sealed::LoadSource) {
        // Make sure we compile this once per iterator type and buffer type combination. Then for
        // instance there is only one such instance for all LayoutEngine types instead of one per
        // different layout.
        #[inline(never)]
        fn ranges_from_buf_at(
            ranges: impl Iterator<Item = Range<usize>>,
            store: &mut dyn Loadable,
            mut buffer: impl sealed::LoadSource,
            offset: usize,
        ) {
            for range in ranges {
                buffer.load(store, range, offset)
            }
        }

        ranges_from_buf_at(
            self.engine.buffer_ranges(),
            self.inner,
            buffer,
            self.engine.image_offset(),
        );
    }

    /// Read out data from a borrowed buffer.
    ///
    /// This reads data up to our layout. It does not interpret the data with the layout of
    /// the argument buffer.
    pub fn read_from_ref(&mut self, buffer: ImageRef<'_, Bytes>) {
        self.engine_from_buf_at(buffer.as_buf());
    }

    /// Read out data from a borrowed buffer.
    ///
    /// This reads data up to our layout. It does not interpret the data with the layout of
    /// the argument buffer.
    pub fn read_from_cell_ref(&mut self, buffer: CellImageRef<'_, Bytes>) {
        self.engine_from_buf_at(buffer.as_cell_buf());
    }

    /// Read out data from a borrowed buffer.
    ///
    /// This reads data up to our layout. It does not interpret the data with the layout of
    /// the argument buffer.
    pub fn read_from_atomic_ref(&mut self, buffer: CellImageRef<'_, Bytes>) {
        self.engine_from_buf_at(buffer.as_cell_buf());
    }
}

/// Copies all bytes within the bounds of a layout.
///
/// This strategy will do a single copy, of the appropriate type for the targets buffer type, to
/// transfer all the raw byte data of the image. This is optimal if the layout is supposed to be
/// the first or only plane and does not contain any other internal padding buffers either.
pub struct RangeEngine<L> {
    inner: Option<L>,
    bytes: [Range<usize>; 1],
}

/// Applies an interior copy strategy but the bytes in the image are written to another plane.
pub struct RelocateEngine<Inner: sealed::LayoutEngineCore> {
    inner: Inner,
    offset: AlignedOffset,
}

impl<L: Layout> LayoutEngine for RangeEngine<L> {}

impl<L: Layout> RangeEngine<L> {
    fn new(layout: &L, offset: usize) -> Self
    where
        L: Clone,
    {
        let byte_len = layout.byte_len();
        let bytes = [offset..offset + byte_len];

        RangeEngine {
            inner: Some(layout.clone()),
            bytes,
        }
    }
}

impl<L: Layout> sealed::LayoutEngineCore for RangeEngine<L> {
    type Layout = L;

    fn consume_layout(&mut self) -> L {
        self.inner
            .take()
            .expect("Protocol error, layout polled twice")
    }

    // Return a `Cloned<slice::Iter>`, not an array iterator. Note we aim to reduce the number of
    // distinct iterator types between all layout engines, even for distinct layouts etc. This is
    // compatible as best as possible and misses the loop in copy at most once..
    #[allow(refining_impl_trait)]
    fn buffer_ranges(&self) -> core::iter::Cloned<core::slice::Iter<'_, Range<usize>>> {
        (&self.bytes[..]).iter().cloned()
    }

    fn image_offset(&self) -> usize {
        0
    }
}

impl<I: sealed::LayoutEngineCore> LayoutEngine for RelocateEngine<I> {}

impl<I: sealed::LayoutEngineCore> sealed::LayoutEngineCore for RelocateEngine<I> {
    type Layout = Relocated<I::Layout>;

    fn consume_layout(&mut self) -> Self::Layout {
        Relocated {
            inner: self.inner.consume_layout(),
            offset: self.offset,
        }
    }

    fn buffer_ranges(&self) -> impl Iterator<Item = Range<usize>> {
        self.inner.buffer_ranges()
    }

    fn image_offset(&self) -> usize {
        self.inner.image_offset() + self.offset.get()
    }
}
