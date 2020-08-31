// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
use core::{fmt, ops};

use bytemuck::Pod;

use crate::buf::{buf, Buffer, Cog};
use crate::layout::{Bytes, Coord, Decay, DynLayout, Layout, Mend, SampleSlice, Take, TryMend};
use crate::{Rec, ReuseError};

/// A owned canvas, parameterized over the layout.
///
/// This type permits user defined layouts of any kind and does not unsafely depend on the validity
/// of the layouts. Correctness is achieved in the common case by discouraging methods that would
/// lead to a diverging size of the memory buffer and the layout. Hence, access to the image pixels
/// should not lead to panic unless an incorrectly implemented layout is used.
///
/// Since a `Canvas` can not unsafely rely on the layout behaving correctly, direct accessors may
/// have suboptimal behaviour and perform a few (seemingly) redundant checks. More optimal, but
/// much more specialized, wrappers are provided in other types such as `Matrix`.
///
/// Note also that any borrowing canvas can arbitrarily change its own layout and thus overwrite
/// the content with completely different types and layouts. This is intended to maximize the
/// flexibility for users. In complicated cases it could be hard for the type system to reflect the
/// compatibility of a custom pixel layout and a standard one. It is solely the user's
/// responsibility to use the interface sensibly. The _soundness_ of standard channel types (e.g.
/// `u8` or `u32`) is not impacted by this as any byte content is valid for them.
///
/// It is possible to convert the layout to a less strictly typed one without reallocating the
/// buffer. For example, all standard layouts such as `Matrix` can be weakened to `DynLayout`. The
/// reverse can not be done unchecked but is possible with fallible conversions.
///
/// Note also that `Canvas` provides fallible operations, some of them are meant to modify the
/// type. This can obviously not be performed in-place, in the manner with which it would be common
/// if the type did not change. Instead we approximate at least the result type by transferring the
/// buffer on success while leaving it unchanged in case of failure. An example signature for this is:
///
/// > [`fn mend<M>(&mut self, with: L::Item) -> Option<Canvas<M>>`][`mend`]
///
/// [`mend`]: #method.mend
///
/// ## Examples
///
/// ```
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Canvas<Layout = Bytes> {
    inner: RawCanvas<Buffer, Layout>,
}

/// An owned or borrowed canvas, parameterized over the layout.
///
/// The buffer is either owned or _mutably_ borrowed from another `Canvas`. Some allocating methods
/// may lead to an implicit change from a borrowed to an owned buffer. These methods are documented
/// as performing a fallible allocation. Other method calls on the previously borrowing canvas will
/// afterwards no longer change the bytes of the canvas it was borrowed from.
#[derive(Clone, PartialEq, Eq)]
pub struct CopyOnGrow<'buf, Layout = Bytes> {
    inner: RawCanvas<Cog<'buf>, Layout>,
}

/// A read-only view of a canvas.
#[derive(Clone, PartialEq, Eq)]
pub struct View<'buf, Layout = Bytes> {
    inner: RawCanvas<&'buf buf, Layout>,
}

/// A writeable reference to a canvas.
#[derive(PartialEq, Eq)]
pub struct ViewMut<'buf, Layout = Bytes> {
    inner: RawCanvas<&'buf mut buf, Layout>,
}

/// A raster layout.
pub trait Raster<Pixel>: Sized {
    fn dimensions(&self) -> Coord;
    fn get(from: View<Self>, at: Coord) -> Pixel;
}

/// A raster layout where one can change pixel values independently.
pub trait RasterMut<Pixel>: Raster<Pixel> {
    fn put(into: ViewMut<Self>, at: Coord, val: Pixel);
}

/// Inner buffer implementation.
///
/// Not exposed to avoid leaking the implementation detail of the `Buf` type parameter. This allows
/// a single implementation for borrowed and owned buffers while keeping `buf`, `Cog` etc. private.
#[derive(Default, Clone, PartialEq, Eq)]
pub(crate) struct RawCanvas<Buf, Layout> {
    buffer: Buf,
    layout: Layout,
}

pub(crate) trait BufferLike: ops::Deref<Target = buf> {
    fn into_owned(self) -> Buffer;
    fn take(&mut self) -> Self;
}

pub(crate) trait BufferMut: BufferLike + ops::DerefMut {}

pub(crate) trait Growable: BufferLike {
    fn grow_to(&mut self, _: usize);
}

/// Canvas methods for all layouts.
impl<L: Layout> Canvas<L> {
    /// Create a new canvas for a specific layout.
    pub fn new(layout: L) -> Self {
        RawCanvas::<Buffer, L>::new(layout).into()
    }

    /// Get a reference to those bytes used by the layout.
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Get a mutable reference to those bytes used by the layout.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// Decay into a canvas with less specific layout.
    ///
    /// See the [`Decay`] trait for an explanation of this operation.
    ///
    /// [`Decay`]: ../layout/trait.Decay.html
    pub fn decay<M>(self) -> Canvas<M>
    where
        M: Decay<L>,
        M: Layout,
    {
        self.inner.decay().into()
    }

    /// Move the buffer into a new canvas.
    pub fn take(&mut self) -> Canvas<L>
    where
        L: Take,
    {
        self.inner.take().into()
    }

    /// Strengthen the layout of the canvas.
    ///
    /// See the [`Mend`] trait for an explanation of this operation.
    ///
    /// [`Mend`]: ../layout/trait.Mend.html
    pub fn mended<Item>(self, mend: Item) -> Canvas<Item::Into>
    where
        Item: Mend<L>,
        L: Take,
    {
        let new_layout = mend.mend(self.inner.layout());
        self.inner.reinterpret_unguarded(new_layout).into()
    }

    /// Strengthen the layout of the canvas.
    ///
    /// See the [`Mend`] trait for an explanation of this operation.
    ///
    /// This is a fallible operation. In case of success returns `Ok` and the byte buffer of the
    /// image is moved into the result. When mending fails this method returns `Err` and the buffer
    /// is kept by this canvas.
    ///
    /// [`Mend`]: ../layout/trait.Mend.html
    pub fn try_mend<Item>(&mut self, mend: Item) -> Result<Canvas<Item::Into>, Item::Err>
    where
        Item: TryMend<L>,
        L: Take,
    {
        let new_layout = mend.try_mend(self.inner.layout())?;
        Ok(self.inner.take().reinterpret_unguarded(new_layout).into())
    }
}

/// Canvas methods that do not require a layout.
impl<L> Canvas<L> {
    /// Check if the buffer could accommodate another layout without reallocating.
    pub fn fits(&self, other: &impl Layout) -> bool {
        other.byte_len() <= self.as_capacity_bytes().len()
    }

    /// Get a reference to the unstructured bytes of the canvas.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_bytes`].
    ///
    /// [`as_bytes`]: #method.as_bytes
    pub fn as_capacity_bytes(&self) -> &[u8] {
        self.inner.as_capacity_bytes()
    }

    /// Get a mutable reference to the unstructured bytes of the canvas.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_bytes_mut`].
    ///
    /// [`as_bytes_mut`]: #method.as_bytes_mut
    pub fn as_capacity_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_capacity_bytes_mut()
    }

    /// Get a reference to the layout.
    pub fn layout(&self) -> &L {
        self.inner.layout()
    }

    /// Get a mutable reference to the layout.
    ///
    /// Be mindful not to modify the layout to exceed the allocated size. This does not cause any
    /// unsoundness but might lead to panics when calling other methods.
    pub fn layout_mut_unguarded(&mut self) -> &mut L {
        self.inner.layout_mut_unguarded()
    }
}

/// Canvas methods for layouts based on pod samples.
impl<L: SampleSlice> Canvas<L>
where
    L::Sample: Pod,
{
    /// Interpret an existing buffer as a pixel canvas.
    ///
    /// The data already contained within the buffer is not modified so that prior initialization
    /// can be performed or one array of samples reinterpreted for an image of other sample type.
    /// This method will never reallocate data.
    ///
    /// # Panics
    ///
    /// This function will panic if the buffer is shorter than the layout.
    pub fn from_rec(buffer: Rec<L::Sample>, layout: L) -> Self {
        assert!(buffer.byte_len() >= layout.byte_len());
        RawCanvas::from_rec(buffer, layout).into()
    }

    /// Get a slice of the individual samples in the layout.
    pub fn as_slice(&self) -> &[L::Sample] {
        self.inner.as_slice()
    }

    /// Get a mutable slice of the individual samples in the layout.
    pub fn as_mut_slice(&mut self) -> &mut [L::Sample] {
        self.inner.as_mut_slice()
    }

    /// Convert into an vector-like of sample types.
    pub fn into_rec(self) -> Rec<L::Sample> {
        self.inner.into_rec()
    }
}

/// Layout oblivious methods that can allocate and change to another buffer.
impl<B: Growable, L> RawCanvas<B, L> {
    /// Grow the buffer, preparing for another layout.
    ///
    /// This may allocate a new buffer and thus disassociate the image from the currently borrowed
    /// underlying buffer.
    ///
    /// # Panics
    /// This function will panic if an allocation is necessary but fails.
    pub(crate) fn grow(&mut self, layout: &impl Layout) {
        Growable::grow_to(&mut self.buffer, layout.byte_len());
    }

    /// Convert the inner layout.
    ///
    /// This method expects that the converted layout is compatible with the current layout.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn decay<Other>(mut self) -> RawCanvas<B, Other>
    where
        Other: Decay<L>,
    {
        let layout = Other::decay(self.layout);
        Growable::grow_to(&mut self.buffer, layout.byte_len());
        RawCanvas {
            buffer: self.buffer,
            layout,
        }
    }

    /// Convert the inner layout to a dynamic one.
    ///
    /// This is mostly convenience. Also not that `DynLayout` is of course not _completely_ generic
    /// but tries to emulate a large number of known layouts.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn into_dynamic(self) -> RawCanvas<B, DynLayout>
    where
        DynLayout: Decay<L>,
    {
        self.decay()
    }

    /// Change the layout, reusing and growing the buffer.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn into_reinterpreted<Other: Layout>(
        mut self,
        layout: Other,
    ) -> RawCanvas<B, Other> {
        Growable::grow_to(&mut self.buffer, layout.byte_len());
        RawCanvas {
            buffer: self.buffer,
            layout,
        }
    }

    /// Mutably borrow this canvas with another arbitrary layout.
    ///
    /// The other layout could be completely incompatible and perform arbitrary mutations. This
    /// seems counter intuitive at first, but recall that these mutations are not unsound as they
    /// can not invalidate the bytes themselves and only write unexpected values. This provides
    /// more flexibility for 'transmutes' than easily expressible in the type system.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn as_reinterpreted<Other>(&mut self, other: Other) -> RawCanvas<&'_ mut buf, Other>
    where
        B: BufferMut,
        Other: Layout,
    {
        self.grow(&other);
        RawCanvas {
            buffer: &mut self.buffer,
            layout: other,
        }
    }

    /// Change the layout and then resize the buffer so that it still fits.
    pub(crate) fn mutate_layout<T>(&mut self, f: impl FnOnce(&mut L) -> T) -> T
    where
        L: Layout,
    {
        let t = f(&mut self.layout);
        self.buffer.grow_to(self.layout.byte_len());
        t
    }
}

/// Layout oblivious methods, these also never allocate or panic.
impl<B: BufferLike, L> RawCanvas<B, L> {
    /// Get a reference to the unstructured bytes of the canvas.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes`].
    ///
    /// [`as_layout_bytes`]: #method.as_layout_bytes
    pub(crate) fn as_capacity_bytes(&self) -> &[u8] {
        self.buffer.as_bytes()
    }

    /// Get a mutable reference to the unstructured bytes of the canvas.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes_mut`].
    ///
    /// [`as_layout_bytes_mut`]: #method.as_layout_bytes_mut
    pub(crate) fn as_capacity_bytes_mut(&mut self) -> &mut [u8]
    where
        B: BufferMut,
    {
        self.buffer.as_bytes_mut()
    }

    /// Get a reference to the layout.
    pub(crate) fn layout(&self) -> &L {
        &self.layout
    }

    /// Get a mutable reference to the layout.
    ///
    /// Be mindful not to modify the layout to exceed the allocated size.
    pub(crate) fn layout_mut_unguarded(&mut self) -> &mut L {
        &mut self.layout
    }

    /// Reinterpret the bits in another layout.
    ///
    /// This method fails if the layout requires more bytes than are currently allocated.
    pub(crate) fn try_reinterpret<Other>(self, layout: Other) -> Result<RawCanvas<B, Other>, Self>
    where
        Other: Layout,
    {
        if self.buffer.len() > layout.byte_len() {
            Err(self)
        } else {
            Ok(RawCanvas {
                buffer: self.buffer,
                layout,
            })
        }
    }

    /// Change the layout without checking the buffer.
    pub(crate) fn reinterpret_unguarded<Other: Layout>(self, layout: Other) -> RawCanvas<B, Other> {
        RawCanvas {
            buffer: self.buffer,
            layout,
        }
    }

    /// Borrow the buffer with the same layout.
    pub(crate) fn borrow_mut(&mut self) -> RawCanvas<&'_ mut buf, L>
    where
        B: BufferMut,
        L: Clone,
    {
        RawCanvas {
            buffer: &mut self.buffer,
            layout: self.layout.clone(),
        }
    }

    /// Take ownership of the image's bytes.
    ///
    /// # Panics
    /// This method panics if allocation fails.
    pub(crate) fn into_owned(self) -> RawCanvas<Buffer, L> {
        RawCanvas {
            buffer: BufferLike::into_owned(self.buffer),
            layout: self.layout,
        }
    }
}

/// Methods specifically with a dynamic layout.
impl<B> RawCanvas<B, DynLayout> {
    pub(crate) fn try_from_dynamic<Other>(self, layout: Other) -> Result<RawCanvas<B, Other>, Self>
    where
        Other: Into<DynLayout> + Clone,
    {
        let reference = layout.clone().into();
        if self.layout == reference {
            Ok(RawCanvas {
                buffer: self.buffer,
                layout,
            })
        } else {
            Err(self)
        }
    }
}

/// Methods for all `Layouts` (the trait).
impl<B: BufferLike, L: Layout> RawCanvas<B, L> {
    /// Allocate a buffer for a particular layout.
    pub(crate) fn new(layout: L) -> Self
    where
        B: From<Buffer>,
    {
        let bytes = layout.byte_len();
        RawCanvas {
            buffer: Buffer::new(bytes).into(),
            layout,
        }
    }

    pub(crate) fn with_buffer(layout: L, buffer: B) -> Self {
        assert!(buffer.as_ref().len() <= layout.byte_len());
        RawCanvas { buffer, layout }
    }

    /// Get a reference to those bytes used by the layout.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.as_capacity_bytes()[..self.layout.byte_len()]
    }

    /// Get a mutable reference to those bytes used by the layout.
    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        B: BufferMut,
    {
        let len = self.layout.byte_len();
        &mut self.as_capacity_bytes_mut()[..len]
    }

    /// Create a canvas from a byte slice specifying the contents.
    ///
    /// If the layout requires more bytes then the remaining bytes are zero initialized.
    pub(crate) fn with_contents(buffer: &[u8], layout: L) -> Self
    where
        B: From<Buffer>,
    {
        let mut buffer = Buffer::from(buffer);
        buffer.grow_to(layout.byte_len());
        RawCanvas {
            buffer: buffer.into(),
            layout,
        }
    }

    /// Reuse the buffer for a new image layout of the same type.
    pub(crate) fn try_reuse(&mut self, layout: L) -> Result<(), ReuseError> {
        if self.as_capacity_bytes().len() >= layout.byte_len() {
            self.layout = layout;
            Ok(())
        } else {
            Err(ReuseError {
                capacity: self.as_capacity_bytes().len(),
                requested: Some(layout.byte_len()),
            })
        }
    }

    /// Change the layout but require that the new layout fits the buffer, never reallocate.
    pub(crate) fn mutate_inplace<T>(&mut self, f: impl FnOnce(&mut L) -> T) -> T
    where
        L: Layout,
    {
        let t = f(&mut self.layout);
        assert!(
            self.layout.byte_len() <= self.buffer.len(),
            "Modification required buffer allocation, was not in-place"
        );
        t
    }

    /// Take the buffer and layout from this canvas, moving content into a new instance.
    ///
    /// Asserts that the moved-from container can hold the emptied layout.
    pub(crate) fn take(&mut self) -> Self
    where
        L: Take,
    {
        let buffer = self.buffer.take();
        let layout = self.mutate_inplace(Take::take);
        RawCanvas::with_buffer(layout, buffer)
    }
}

/// Methods for layouts that are slices of individual samples.
impl<B: BufferLike, L: SampleSlice> RawCanvas<B, L>
where
    L::Sample: Pod,
{
    /// Interpret an existing buffer as a pixel canvas.
    ///
    /// The data already contained within the buffer is not modified so that prior initialization
    /// can be performed or one array of samples reinterpreted for an image of other sample type.
    /// However, the `Rec` will be logically resized which will zero-initialize missing elements if
    /// the current buffer is too short.
    ///
    /// # Panics
    ///
    /// This function will panic if resizing causes a reallocation that fails.
    pub(crate) fn from_rec(buffer: Rec<L::Sample>, layout: L) -> Self
    where
        B: From<Buffer>,
    {
        let buffer = buffer.into_inner();
        assert!(buffer.len() >= layout.byte_len());
        Self {
            buffer: buffer.into(),
            layout,
        }
    }

    pub(crate) fn as_slice(&self) -> &[L::Sample] {
        self.buffer.as_pixels(self.layout.sample())
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [L::Sample]
    where
        B: BufferMut,
    {
        self.buffer.as_mut_pixels(self.layout.sample())
    }

    /// Convert back into an vector-like of sample types.
    pub(crate) fn into_rec(self) -> Rec<L::Sample> {
        let sample = self.layout.sample();
        // Avoid calling any method of `Layout` after this. Not relevant for safety but might be in
        // the future, if we want to avoid the extra check in `resize`.
        let count = self.as_slice().len();
        let buffer = self.buffer.into_owned();
        let mut rec = Rec::from_buffer(buffer, sample);
        // This should never reallocate at this point but we don't really know or care.
        rec.resize(count);
        rec
    }
}

impl<L> From<RawCanvas<Buffer, L>> for Canvas<L> {
    fn from(canvas: RawCanvas<Buffer, L>) -> Self {
        Canvas { inner: canvas }
    }
}

impl BufferLike for Cog<'_> {
    fn into_owned(self) -> Buffer {
        Cog::into_owned(self)
    }

    fn take(&mut self) -> Self {
        core::mem::replace(self, Cog::Owned(Default::default()))
    }
}

impl BufferLike for Buffer {
    fn into_owned(self) -> Self {
        self
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl BufferLike for &'_ mut buf {
    fn into_owned(self) -> Buffer {
        Buffer::from(self.as_bytes())
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl Growable for Cog<'_> {
    fn grow_to(&mut self, bytes: usize) {
        Cog::grow_to(self, bytes);
    }
}

impl Growable for Buffer {
    fn grow_to(&mut self, bytes: usize) {
        Buffer::grow_to(self, bytes);
    }
}

impl BufferMut for Cog<'_> {}

impl BufferMut for Buffer {}

impl BufferMut for &'_ mut buf {}

impl<Layout: Clone> Clone for RawCanvas<Cog<'_>, Layout> {
    fn clone(&self) -> Self {
        use alloc::borrow::ToOwned;
        RawCanvas {
            buffer: Cog::Owned(self.buffer.to_owned()),
            layout: self.layout.clone(),
        }
    }
}

impl<Layout: Default> Default for Canvas<Layout> {
    fn default() -> Self {
        Canvas {
            inner: RawCanvas {
                buffer: Buffer::default(),
                layout: Layout::default(),
            },
        }
    }
}

impl<Layout: Default> Default for CopyOnGrow<'_, Layout> {
    fn default() -> Self {
        CopyOnGrow {
            inner: RawCanvas {
                buffer: Cog::Owned(Buffer::default()),
                layout: Layout::default(),
            },
        }
    }
}

impl<L> fmt::Debug for Canvas<L>
where
    L: SampleSlice + fmt::Debug,
    L::Sample: Pod + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Canvas")
            .field("layout", &self.inner.layout)
            .field("content", &self.inner.as_slice())
            .finish()
    }
}
