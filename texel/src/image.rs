//! Defines the `Image` container, with flexibly type-safe layout.
//!
//! Besides the main type, [`Image`], which is an owned buffer of particular layout there are some
//! supporting types that represent other ways in which layouts interact with buffers. Note that
//! the layout is flexible in the sense that it is up to the user to ultimately ensure correct
//! typing. The type definition will _help_ you by not providing the tools for strong types but
//! it's always _allowed_/_valid_ to refer to the same bytes by a different layout. This makes it
//! possible to use your own texel/pixel wrapper types regardless of the underlying byte
//! representation. Indeed, the byte buffer need not even represent a pixel matrix (but it's
//! advised, probably very common, and the only 'supported' use-case).
// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
use core::{fmt, ops};

use crate::buf::{buf, Buffer, Cog};
use crate::layout::{
    Bytes, Decay, DynLayout, Layout, Mend, Raster, RasterMut, SliceLayout, Take, TryMend,
};
use crate::{BufferReuseError, Texel, TexelBuffer};

pub use crate::stride::{ByteCanvasMut, ByteCanvasRef};

/// A container of allocated bytes, parameterized over the layout.
///
/// This type permits user defined layouts of any kind and does not unsafely depend on the validity
/// of the layouts. Correctness is achieved in the common case by discouraging methods that would
/// lead to a diverging size of the memory buffer and the layout. Hence, access to the image pixels
/// should not lead to panic unless an incorrectly implemented layout is used.
///
/// It possible to convert the layout to a less strictly typed one without reallocating the buffer.
/// For example, all standard layouts such as `Matrix` can be weakened to `DynLayout`. The reverse
/// can not be done unchecked but is possible with fallible conversions.
///
/// Indeed, the image can _arbitrarily_ change its own layout—different `ImageRef` and
/// `ImageMut` may even chose _conflicting layouts—and thus overwrite the content with completely
/// different types and layouts. This is intended to maximize the flexibility for users. In
/// complicated cases it could be hard for the type system to reflect the compatibility of a custom
/// pixel layout and a standard one. It is solely the user's responsibility to use the interface
/// sensibly. The _soundness_ of standard channel types (e.g. `u8` or `u32`) is not impacted by
/// this as any byte content is valid for them.
///
/// ## Examples
///
/// Initialize a matrix as computed `[u8; 4]` rga pixels:
///
/// ```
/// # fn test() -> Option<()> {
/// use image_texel::{Image, Matrix};
///
/// let mut image = Image::from(Matrix::<[u8; 4]>::with_width_and_height(400, 400));
///
/// image.shade(|x, y, rgba| {
///     rgba[0] = x as u8;
///     rgba[1] = y as u8;
///     rgba[3] = 0xff;
/// });
///
/// # Some(()) }
/// # let _ = test();
/// ```
///
/// # Design
///
/// Since a `Image` can not unsafely rely on the layout behaving correctly, direct accessors may
/// have suboptimal behaviour and perform a few (seemingly) redundant checks. More optimal, but
/// much more specialized, wrappers can be provided in other types that first reduce to a
/// first-party layout and byte buffer and then preserve this invariant by never calling
/// second/third-party code from traits. Some of these may be offered in this crate in the future.
///
/// Note also that `Image` provides fallible operations, some of them are meant to modify the
/// type. This can obviously not be performed in-place, in the manner with which it would be common
/// if the type did not change. Instead we approximate at least the result type by transferring the
/// buffer on success while leaving it unchanged in case of failure. An example signature for this is:
///
/// > [`fn mend<M>(&mut self, with: L::Item) -> Option<Image<M>>`][`mend`]
///
/// [`mend`]: #method.mend
#[derive(Clone, PartialEq, Eq)]
pub struct Image<Layout = Bytes> {
    inner: RawImage<Buffer, Layout>,
}

/// An owned or borrowed image, parameterized over the layout.
///
/// The buffer is either owned or _mutably_ borrowed from another `Image`. Some allocating methods
/// may lead to an implicit change from a borrowed to an owned buffer. These methods are documented
/// as performing a fallible allocation. Other method calls on the previously borrowing image will
/// afterwards no longer change the bytes of the image it was borrowed from.
///
/// FIXME: figure out if this is 'right' to expose in this crate.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct CopyOnGrow<'buf, Layout = Bytes> {
    inner: RawImage<Cog<'buf>, Layout>,
}

/// A read-only view of an image.
///
/// Note that this requires its underlying buffer to be highly aligned! For that reason it is not
/// possible to take a reference at an arbitrary number of bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct ImageRef<'buf, Layout = &'buf Bytes> {
    inner: RawImage<&'buf buf, Layout>,
}

/// A writeable reference to an image buffer.
#[derive(PartialEq, Eq)]
pub struct ImageMut<'buf, Layout = &'buf mut Bytes> {
    inner: RawImage<&'buf mut buf, Layout>,
}

/// Describes an image coordinate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coord(pub u32, pub u32);

impl Coord {
    pub fn x(self) -> u32 {
        self.0
    }

    pub fn y(self) -> u32 {
        self.1
    }

    pub fn yx(self) -> (u32, u32) {
        (self.1, self.0)
    }

    pub fn xy(self) -> (u32, u32) {
        (self.0, self.1)
    }
}

/// Inner buffer implementation.
///
/// Not exposed to avoid leaking the implementation detail of the `Buf` type parameter. This allows
/// a single implementation for borrowed and owned buffers while keeping `buf`, `Cog` etc. private.
#[derive(Default, Clone, PartialEq, Eq)]
pub(crate) struct RawImage<Buf, Layout> {
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

/// Image methods for all layouts.
impl<L: Layout> Image<L> {
    /// Create a new image for a specific layout.
    pub fn new(layout: L) -> Self {
        RawImage::<Buffer, L>::new(layout).into()
    }

    /// Create a new image with initial byte content.
    pub fn with_bytes(layout: L, bytes: &[u8]) -> Self {
        RawImage::with_contents(bytes, layout).into()
    }

    /// Create a new image with initial texel contents.
    ///
    /// The memory is reused as much as possible. If the layout is too large for the buffer then
    /// the remainder is filled up with zeroed bytes.
    pub fn with_buffer<T>(layout: L, bytes: TexelBuffer<T>) -> Self {
        RawImage::with_buffer(Bytes(0), bytes.into_inner())
            .with_layout(layout)
            .into()
    }

    /// Get a reference to those bytes used by the layout.
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Get a mutable reference to those bytes used by the layout.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// If necessary, reallocate the buffer to fit the layout.
    ///
    /// Call this method after having mutated a layout with [`Image::layout_mut_unguarded`]
    /// whenever you are not sure that the layout did not grow. This will ensure the contract that
    /// the internal buffer is large enough for the layout.
    ///
    /// # Panics
    ///
    /// This method panics when the allocation of the new buffer fails.
    pub fn ensure_layout(&mut self) {
        self.inner.mutate_layout(|_| ());
    }

    /// Change the layer of the image.
    ///
    /// Reallocates the buffer when growing a layout. Call [`Image::fits`] to check this property.
    pub fn with_layout<M>(self, layout: M) -> Image<M>
    where
        M: Layout,
    {
        self.inner.with_layout(layout).into()
    }

    /// Decay into a image with less specific layout.
    ///
    /// See the [`Decay`] trait for an explanation of this operation.
    ///
    /// # Example
    ///
    /// The common layouts define ways to decay into a dynamically typed variant.
    ///
    /// ```
    /// # use image_texel::{Image, Matrix, layout};
    /// let matrix = Matrix::<u8>::with_width_and_height(400, 400);
    /// let image: Image<layout::Matrix<u8>> = Image::from(matrix);
    ///
    /// // to turn hide the `u8` type but keep width, height, texel layout
    /// let image: Image<layout::MatrixBytes> = image.decay();
    /// assert_eq!(image.layout().width(), 400);
    /// assert_eq!(image.layout().height(), 400);
    /// ```
    ///
    /// See also [`Image::mend`] and [`Image::try_mend`] for operations that reverse the effects.
    ///
    /// Can also be used to forget specifics of the layout, turning the image into a more general
    /// container type. For example, to use a uniform type as an allocated buffer waiting on reuse.
    ///
    /// ```
    /// # use image_texel::{Image, Matrix, layout};
    /// let matrix = Matrix::<u8>::with_width_and_height(400, 400);
    ///
    /// // Can always decay to a byte buffer.
    /// let bytes: Image = Image::from(matrix).decay();
    /// let _: &layout::Bytes = bytes.layout();
    /// ```
    ///
    /// [`Decay`]: ../layout/trait.Decay.html
    pub fn decay<M>(self) -> Image<M>
    where
        M: Decay<L>,
        M: Layout,
    {
        self.inner.decay().into()
    }

    /// Move the buffer into a new image.
    pub fn take(&mut self) -> Image<L>
    where
        L: Take,
    {
        self.inner.take().into()
    }

    /// Strengthen the layout of the image.
    ///
    /// See the [`Mend`] trait for an explanation of this operation.
    ///
    /// [`Mend`]: ../layout/trait.Mend.html
    pub fn mend<Item>(self, mend: Item) -> Image<Item::Into>
    where
        Item: Mend<L>,
        L: Take,
    {
        let new_layout = mend.mend(self.inner.layout());
        self.inner.reinterpret_unguarded(|_| new_layout).into()
    }

    /// Strengthen the layout of the image.
    ///
    /// See the [`Mend`] trait for an explanation of this operation.
    ///
    /// This is a fallible operation. In case of success returns `Ok` and the byte buffer of the
    /// image is moved into the result. When mending fails this method returns `Err` and the buffer
    /// is kept by this image.
    ///
    /// [`Mend`]: ../layout/trait.Mend.html
    pub fn try_mend<Item>(&mut self, mend: Item) -> Result<Image<Item::Into>, Item::Err>
    where
        Item: TryMend<L>,
        L: Take,
    {
        let new_layout = mend.try_mend(self.inner.layout())?;
        Ok(self
            .inner
            .take()
            .reinterpret_unguarded(|_| new_layout)
            .into())
    }
}

/// Image methods that do not require a layout.
impl<L> Image<L> {
    /// Check if the buffer could accommodate another layout without reallocating.
    pub fn fits(&self, other: &impl Layout) -> bool {
        self.inner.fits(other)
    }

    /// Get a reference to the unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_bytes`].
    ///
    /// [`as_bytes`]: #method.as_bytes
    pub fn as_capacity_bytes(&self) -> &[u8] {
        self.inner.as_capacity_bytes()
    }

    /// Get a mutable reference to the unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_bytes_mut`].
    ///
    /// [`as_bytes_mut`]: #method.as_bytes_mut
    pub fn as_capacity_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_capacity_bytes_mut()
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_slice`].
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> &[P] {
        self.inner.buffer.as_texels(pixel)
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_mut_slice`].
    pub fn as_mut_texels<P>(&mut self, pixel: Texel<P>) -> &mut [P] {
        self.inner.buffer.as_mut_texels(pixel)
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

    /// Get a view of this image.
    pub fn as_ref(&self) -> ImageRef<'_, &'_ L> {
        self.inner.borrow().into()
    }

    /// Get a view of this image, if the alternate layout fits.
    pub fn try_to_ref<M: Layout>(&self, layout: M) -> Option<ImageRef<'_, M>> {
        self.as_ref().with_layout(layout)
    }

    /// Get a mutable view of this image.
    pub fn as_mut(&mut self) -> ImageMut<'_, &'_ mut L> {
        self.inner.borrow_mut().into()
    }

    /// Get a mutable view under an alternate layout.
    pub fn to_mut<M: Layout>(&mut self, layout: M) -> ImageMut<'_, M> {
        self.inner.as_reinterpreted(layout).into()
    }

    /// Get a mutable view of this image, if the alternate layout fits.
    pub fn try_to_mut<M: Layout>(&mut self, layout: M) -> Option<ImageMut<'_, M>> {
        self.as_mut().with_layout(layout)
    }

    /// Get a single texel from a raster image.
    pub fn get_texel<P>(&self, coord: Coord) -> Option<P>
    where
        L: Raster<P>,
    {
        L::get(self.as_ref(), coord)
    }

    /// Put a single texel to a raster image.
    pub fn put_texel<P>(&mut self, coord: Coord, texel: P)
    where
        L: RasterMut<P>,
    {
        L::put(self.as_mut(), coord, texel)
    }

    /// Call a function on each texel of this raster image.
    ///
    /// The order of evaluation is _not_ defined although certain layouts may offer more specific
    /// guarantees. In general, one can expect that layouts call the function in a cache-efficient
    /// manner if they are aware of a better iteration strategy.
    pub fn shade<P>(&mut self, f: impl FnMut(u32, u32, &mut P))
    where
        L: RasterMut<P>,
    {
        L::shade(self.as_mut(), f)
    }
}

/// Image methods for layouts based on pod samples.
impl<L: SliceLayout> Image<L> {
    /// Interpret an existing buffer as a pixel image.
    ///
    /// The data already contained within the buffer is not modified so that prior initialization
    /// can be performed or one array of samples reinterpreted for an image of other sample type.
    /// This method will never reallocate data.
    ///
    /// # Panics
    ///
    /// This function will panic if the buffer is shorter than the layout.
    pub fn from_buffer(buffer: TexelBuffer<L::Sample>, layout: L) -> Self {
        assert!(buffer.byte_len() >= layout.byte_len());
        RawImage::from_buffer(buffer, layout).into()
    }

    /// Get a slice of the individual samples in the layout.
    ///
    /// An alternative way to get a slice of texels when a layout does _not_ have an inherent texel
    /// _type_ is [`Self::as_texels`].
    pub fn as_slice(&self) -> &[L::Sample] {
        self.inner.as_slice()
    }

    /// Get a mutable slice of the individual samples in the layout.
    ///
    /// An alternative way to get a slice of texels when a layout does _not_ have an inherent texel
    /// _type_ is [`Self::as_mut_texels`].
    pub fn as_mut_slice(&mut self) -> &mut [L::Sample] {
        self.inner.as_mut_slice()
    }

    /// Convert into an vector-like of sample types.
    pub fn into_buffer(self) -> TexelBuffer<L::Sample> {
        self.inner.into_buffer()
    }
}

impl<'data, L> ImageRef<'data, L> {
    /// Get a reference to those bytes used by the layout.
    pub fn as_bytes(&self) -> &[u8]
    where
        L: Layout,
    {
        self.inner.as_bytes()
    }

    pub fn layout(&self) -> &L {
        &self.inner.layout
    }

    /// Get a view of this image.
    pub fn as_ref(&self) -> ImageRef<'_, &'_ L> {
        self.inner.borrow().into()
    }

    /// Check if a call to [`ImageRef::with_layout`] would succeed.
    pub fn fits(&self, other: &impl Layout) -> bool {
        self.inner.fits(other)
    }

    /// Change this view to a different layout.
    ///
    /// This returns `Some` if the layout fits the underlying data, and `None` otherwise. Use
    /// [`ImageRef::fits`] to check this property in a separate call. Note that the new layout
    /// need not be related to the old layout in any other way.
    pub fn with_layout<M>(self, layout: M) -> Option<ImageRef<'data, M>>
    where
        M: Layout,
    {
        let image = self.inner.try_reinterpret(layout).ok()?;
        Some(image.into())
    }

    /// Decay into a image with less specific layout.
    ///
    /// See [`Image::decay`].
    pub fn decay<M>(self) -> Option<ImageRef<'data, M>>
    where
        M: Decay<L>,
        M: Layout,
    {
        let layout = M::decay(self.inner.layout);
        let image = RawImage {
            layout,
            buffer: self.inner.buffer,
        };
        if image.fits(&image.layout) {
            Some(image.into())
        } else {
            None
        }
    }

    /// Copy all bytes to a newly allocated image.
    pub fn to_owned(&self) -> Image<L>
    where
        L: Layout + Clone,
    {
        Image::with_bytes(self.inner.layout.clone(), self.inner.as_bytes())
    }

    /// Get a slice of the individual samples in the layout.
    pub fn as_slice(&self) -> &[L::Sample]
    where
        L: SliceLayout,
    {
        self.inner.as_slice()
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_slice`].
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> &[P] {
        self.inner.buffer.as_texels(pixel)
    }

    /// Turn into a slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_slice(self) -> &'data [L::Sample]
    where
        L: SliceLayout,
    {
        self.inner.buffer.as_texels(self.inner.layout.sample())
    }

    /// Retrieve a single texel from a raster image.
    pub fn get_texel<P>(&self, coord: Coord) -> Option<P>
    where
        L: Raster<P>,
    {
        L::get(self.as_ref(), coord)
    }
}

impl<'data, L> ImageMut<'data, L> {
    /// Get a reference to those bytes used by the layout.
    pub fn as_bytes(&self) -> &[u8]
    where
        L: Layout,
    {
        self.inner.as_bytes()
    }

    /// Get a mutable reference to those bytes used by the layout.
    pub fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        L: Layout,
    {
        self.inner.as_bytes_mut()
    }

    pub fn layout(&self) -> &L {
        &self.inner.layout
    }

    /// Get a view of this image.
    pub fn as_ref(&self) -> ImageRef<'_, &'_ L> {
        self.inner.borrow().into()
    }

    /// Get a mutable view of this image.
    pub fn as_mut(&mut self) -> ImageMut<'_, &'_ mut L> {
        self.inner.borrow_mut().into()
    }

    /// Convert to a view of this image.
    pub fn into_ref(self) -> ImageRef<'data, L> {
        RawImage {
            layout: self.inner.layout,
            buffer: &*self.inner.buffer,
        }
        .into()
    }

    /// Check if a call to [`ImageMut::with_layout`] would succeed, without consuming this reference.
    pub fn fits(&self, other: &impl Layout) -> bool {
        self.inner.fits(other)
    }

    /// Change this view to a different layout.
    ///
    /// This returns `Some` if the layout fits the underlying data, and `None` otherwise. Use
    /// [`ImageMut::fits`] to check this property in a separate call. Note that the new layout
    /// need not be related to the old layout in any other way.
    pub fn with_layout<M>(self, layout: M) -> Option<ImageMut<'data, M>>
    where
        M: Layout,
    {
        let image = self.inner.try_reinterpret(layout).ok()?;
        Some(image.into())
    }

    /// Decay into a image with less specific layout.
    ///
    /// See [`Image::decay`].
    pub fn decay<M>(self) -> Option<ImageMut<'data, M>>
    where
        M: Decay<L>,
        M: Layout,
    {
        let layout = M::decay(self.inner.layout);
        let image = RawImage {
            layout,
            buffer: self.inner.buffer,
        };
        if image.fits(&image.layout) {
            Some(image.into())
        } else {
            None
        }
    }

    /// Copy the bytes and layout to an owned container.
    pub fn to_owned(&self) -> Image<L>
    where
        L: Layout + Clone,
    {
        Image::with_bytes(self.inner.layout.clone(), self.inner.as_bytes())
    }

    /// Get a slice of the individual samples in the layout.
    pub fn as_slice(&self) -> &[L::Sample]
    where
        L: SliceLayout,
    {
        self.inner.as_slice()
    }

    /// Get a mutable slice of the individual samples in the layout.
    pub fn as_mut_slice(&mut self) -> &mut [L::Sample]
    where
        L: SliceLayout,
    {
        self.inner.as_mut_slice()
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_slice`].
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> &[P] {
        self.inner.buffer.as_texels(pixel)
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_mut_slice`].
    pub fn as_mut_texels<P>(&mut self, pixel: Texel<P>) -> &mut [P] {
        self.inner.buffer.as_mut_texels(pixel)
    }

    /// Turn into a slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_slice(self) -> &'data [L::Sample]
    where
        L: SliceLayout,
    {
        self.inner.buffer.as_texels(self.inner.layout.sample())
    }

    /// Turn into a mutable slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_mut_slice(self) -> &'data mut [L::Sample]
    where
        L: SliceLayout,
    {
        self.inner.buffer.as_mut_texels(self.inner.layout.sample())
    }

    /// Retrieve a single texel from a raster image.
    pub fn get_texel<P>(&self, coord: Coord) -> Option<P>
    where
        L: Raster<P>,
    {
        L::get(self.as_ref(), coord)
    }

    /// Put a single texel to a raster image.
    pub fn put_texel<P>(&mut self, coord: Coord, texel: P)
    where
        L: RasterMut<P>,
    {
        L::put(self.as_mut(), coord, texel)
    }

    /// Call a function on each texel of this raster image.
    ///
    /// The order of evaluation is _not_ defined although certain layouts may offer more specific
    /// guarantees. In general, one can expect that layouts call the function in a cache-efficient
    /// manner if they are aware of a better iteration strategy.
    pub fn shade<P>(&mut self, f: impl FnMut(u32, u32, &mut P))
    where
        L: RasterMut<P>,
    {
        L::shade(self.as_mut(), f)
    }
}

// TODO: how to expose?
// This is used internally in `RasterMut::shade` however only for the special case of
// * `&mut &mut L` -> `&mut L`
// * `&&mut L` -> `&L`
// which we know are semantically equivalent. In the general case these would go through checks
// that ensure the new layout is consistent with the data.
impl<'data, 'l, L: Layout> ImageRef<'data, &'l L> {
    pub(crate) fn as_deref(self) -> ImageRef<'data, &'l L::Target>
    where
        L: core::ops::Deref,
        L::Target: Layout,
    {
        self.inner.reinterpret_unguarded(|l| &**l).into()
    }
}

impl<'data, 'l, L: Layout> ImageMut<'data, &'l mut L> {
    pub(crate) fn as_deref_mut(self) -> ImageMut<'data, &'l mut L::Target>
    where
        L: core::ops::DerefMut,
        L::Target: Layout,
    {
        self.inner.reinterpret_unguarded(|l| &mut **l).into()
    }
}

/// Layout oblivious methods that can allocate and change to another buffer.
impl<B: Growable, L> RawImage<B, L> {
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
    pub(crate) fn decay<Other>(mut self) -> RawImage<B, Other>
    where
        Other: Decay<L>,
    {
        let layout = Other::decay(self.layout);
        Growable::grow_to(&mut self.buffer, layout.byte_len());
        RawImage {
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
    pub(crate) fn into_dynamic(self) -> RawImage<B, DynLayout>
    where
        DynLayout: Decay<L>,
    {
        self.decay()
    }

    /// Change the layout, reusing and growing the buffer.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn with_layout<Other: Layout>(mut self, layout: Other) -> RawImage<B, Other> {
        Growable::grow_to(&mut self.buffer, layout.byte_len());
        RawImage {
            buffer: self.buffer,
            layout,
        }
    }

    /// Mutably borrow this image with another arbitrary layout.
    ///
    /// The other layout could be completely incompatible and perform arbitrary mutations. This
    /// seems counter intuitive at first, but recall that these mutations are not unsound as they
    /// can not invalidate the bytes themselves and only write unexpected values. This provides
    /// more flexibility for 'transmutes' than easily expressible in the type system.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn as_reinterpreted<Other>(&mut self, other: Other) -> RawImage<&'_ mut buf, Other>
    where
        B: BufferMut,
        Other: Layout,
    {
        self.grow(&other);
        RawImage {
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
impl<B: BufferLike, L> RawImage<B, L> {
    /// Get a mutable reference to the unstructured bytes of the image.
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

    /// Take ownership of the image's bytes.
    ///
    /// # Panics
    /// This method panics if allocation fails.
    pub(crate) fn into_owned(self) -> RawImage<Buffer, L> {
        RawImage {
            buffer: BufferLike::into_owned(self.buffer),
            layout: self.layout,
        }
    }
}

/// Methods specifically with a dynamic layout.
impl<B> RawImage<B, DynLayout> {
    pub(crate) fn try_from_dynamic<Other>(self, layout: Other) -> Result<RawImage<B, Other>, Self>
    where
        Other: Into<DynLayout> + Clone,
    {
        let reference = layout.clone().into();
        if self.layout == reference {
            Ok(RawImage {
                buffer: self.buffer,
                layout,
            })
        } else {
            Err(self)
        }
    }
}

impl<B, L> RawImage<B, L> {
    /// Allocate a buffer for a particular layout.
    pub(crate) fn new(layout: L) -> Self
    where
        L: Layout,
        B: From<Buffer>,
    {
        let bytes = layout.byte_len();
        RawImage {
            buffer: Buffer::new(bytes).into(),
            layout,
        }
    }

    /// Create a image from a byte slice specifying the contents.
    ///
    /// If the layout requires more bytes then the remaining bytes are zero initialized.
    pub(crate) fn with_contents(buffer: &[u8], layout: L) -> Self
    where
        L: Layout,
        B: From<Buffer>,
    {
        let mut buffer = Buffer::from(buffer);
        buffer.grow_to(layout.byte_len());
        RawImage {
            buffer: buffer.into(),
            layout,
        }
    }

    pub(crate) fn with_buffer(layout: L, buffer: B) -> Self
    where
        B: ops::Deref<Target = buf>,
        L: Layout,
    {
        assert!(buffer.as_ref().len() <= layout.byte_len());
        RawImage { buffer, layout }
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

    /// Get a reference to the unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes`].
    ///
    /// [`as_layout_bytes`]: #method.as_layout_bytes
    pub(crate) fn as_capacity_bytes(&self) -> &[u8]
    where
        B: ops::Deref<Target = buf>,
    {
        self.buffer.as_bytes()
    }

    /// Get a reference to those bytes used by the layout.
    pub(crate) fn as_bytes(&self) -> &[u8]
    where
        B: ops::Deref<Target = buf>,
        L: Layout,
    {
        &self.as_capacity_bytes()[..self.layout.byte_len()]
    }

    pub(crate) fn as_slice(&self) -> &[L::Sample]
    where
        B: ops::Deref<Target = buf>,
        L: SliceLayout,
    {
        self.buffer.as_texels(self.layout.sample())
    }

    /// Borrow the buffer with the same layout.
    pub(crate) fn borrow(&self) -> RawImage<&'_ buf, &'_ L>
    where
        B: ops::Deref<Target = buf>,
    {
        RawImage {
            buffer: &self.buffer,
            layout: &self.layout,
        }
    }

    /// Borrow the buffer mutably with the same layout.
    pub(crate) fn borrow_mut(&mut self) -> RawImage<&'_ mut buf, &'_ mut L>
    where
        B: ops::DerefMut<Target = buf>,
    {
        RawImage {
            buffer: &mut self.buffer,
            layout: &mut self.layout,
        }
    }

    pub(crate) fn fits(&self, other: &impl Layout) -> bool
    where
        B: ops::Deref<Target = buf>,
    {
        other.byte_len() <= self.as_capacity_bytes().len()
    }

    /// Change the layout without checking the buffer.
    pub(crate) fn reinterpret_unguarded<Other: Layout>(
        self,
        layout: impl FnOnce(L) -> Other,
    ) -> RawImage<B, Other> {
        RawImage {
            buffer: self.buffer,
            layout: layout(self.layout),
        }
    }

    /// Reinterpret the bits in another layout.
    ///
    /// This method fails if the layout requires more bytes than are currently allocated.
    pub(crate) fn try_reinterpret<Other>(self, layout: Other) -> Result<RawImage<B, Other>, Self>
    where
        B: ops::Deref<Target = buf>,
        Other: Layout,
    {
        if self.buffer.len() > layout.byte_len() {
            Err(self)
        } else {
            Ok(RawImage {
                buffer: self.buffer,
                layout,
            })
        }
    }
}

/// Methods for all `Layouts` (the trait).
impl<B: BufferLike, L: Layout> RawImage<B, L> {
    /// Get a mutable reference to those bytes used by the layout.
    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        B: BufferMut,
    {
        let len = self.layout.byte_len();
        &mut self.as_capacity_bytes_mut()[..len]
    }

    /// Reuse the buffer for a new image layout of the same type.
    pub(crate) fn try_reuse(&mut self, layout: L) -> Result<(), BufferReuseError> {
        if self.as_capacity_bytes().len() >= layout.byte_len() {
            self.layout = layout;
            Ok(())
        } else {
            Err(BufferReuseError {
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

    /// Take the buffer and layout from this image, moving content into a new instance.
    ///
    /// Asserts that the moved-from container can hold the emptied layout.
    pub(crate) fn take(&mut self) -> Self
    where
        L: Take,
    {
        let buffer = self.buffer.take();
        let layout = self.mutate_inplace(Take::take);
        RawImage::with_buffer(layout, buffer)
    }
}

/// Methods for layouts that are slices of individual samples.
impl<B: BufferLike, L: SliceLayout> RawImage<B, L> {
    /// Interpret an existing buffer as a pixel image.
    ///
    /// The data already contained within the buffer is not modified so that prior initialization
    /// can be performed or one array of samples reinterpreted for an image of other sample type.
    /// However, the `TexelBuffer` will be logically resized which will zero-initialize missing elements if
    /// the current buffer is too short.
    ///
    /// # Panics
    ///
    /// This function will panic if resizing causes a reallocation that fails.
    pub(crate) fn from_buffer(buffer: TexelBuffer<L::Sample>, layout: L) -> Self
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

    pub(crate) fn as_mut_slice(&mut self) -> &mut [L::Sample]
    where
        B: BufferMut,
    {
        self.buffer.as_mut_texels(self.layout.sample())
    }

    /// Convert back into an vector-like of sample types.
    pub(crate) fn into_buffer(self) -> TexelBuffer<L::Sample> {
        let sample = self.layout.sample();
        // Avoid calling any method of `Layout` after this. Not relevant for safety but might be in
        // the future, if we want to avoid the extra check in `resize`.
        let count = self.as_slice().len();
        let buffer = self.buffer.into_owned();
        let mut rec = TexelBuffer::from_buffer(buffer, sample);
        // This should never reallocate at this point but we don't really know or care.
        rec.resize(count);
        rec
    }
}

impl<'lt, L: Layout + Clone> From<Image<&'lt L>> for Image<L> {
    fn from(image: Image<&'lt L>) -> Self {
        let layout: L = (*image.layout()).clone();
        RawImage::with_buffer(layout, image.inner.buffer).into()
    }
}

impl<'lt, L: Layout + Clone> From<Image<&'lt mut L>> for Image<L> {
    fn from(image: Image<&'lt mut L>) -> Self {
        let layout: L = (*image.layout()).clone();
        RawImage::with_buffer(layout, image.inner.buffer).into()
    }
}

impl<'lt, L> From<&'lt Image<L>> for ImageRef<'lt, &'lt L> {
    fn from(image: &'lt Image<L>) -> Self {
        image.as_ref()
    }
}

impl<'lt, L> From<&'lt mut Image<L>> for ImageMut<'lt, &'lt mut L> {
    fn from(image: &'lt mut Image<L>) -> Self {
        image.as_mut()
    }
}

impl<'lt, L: Layout + Clone> From<&'lt Image<L>> for ImageRef<'lt, L> {
    fn from(image: &'lt Image<L>) -> Self {
        image.as_ref().into()
    }
}

impl<'lt, L: Layout + Clone> From<&'lt mut Image<L>> for ImageMut<'lt, L> {
    fn from(image: &'lt mut Image<L>) -> Self {
        image.as_mut().into()
    }
}

impl<'lt, L: Layout + Clone> From<ImageRef<'lt, &'_ L>> for ImageRef<'lt, L> {
    fn from(image: ImageRef<'lt, &'_ L>) -> Self {
        let layout: L = (*image.layout()).clone();
        RawImage::with_buffer(layout, image.inner.buffer).into()
    }
}

impl<'lt, L: Layout + Clone> From<ImageRef<'lt, &'_ mut L>> for ImageRef<'lt, L> {
    fn from(image: ImageRef<'lt, &'_ mut L>) -> Self {
        let layout: L = (*image.layout()).clone();
        RawImage::with_buffer(layout, image.inner.buffer).into()
    }
}

impl<'lt, L: Layout + Clone> From<ImageMut<'lt, &'_ L>> for ImageMut<'lt, L> {
    fn from(image: ImageMut<'lt, &'_ L>) -> Self {
        let layout: L = (*image.layout()).clone();
        RawImage::with_buffer(layout, image.inner.buffer).into()
    }
}

impl<'lt, L: Layout + Clone> From<ImageMut<'lt, &'_ mut L>> for ImageMut<'lt, L> {
    fn from(image: ImageMut<'lt, &'_ mut L>) -> Self {
        let layout: L = (*image.layout()).clone();
        RawImage::with_buffer(layout, image.inner.buffer).into()
    }
}

impl<L> From<RawImage<Buffer, L>> for Image<L> {
    fn from(image: RawImage<Buffer, L>) -> Self {
        Image { inner: image }
    }
}

impl<'lt, L> From<RawImage<&'lt buf, L>> for ImageRef<'lt, L> {
    fn from(image: RawImage<&'lt buf, L>) -> Self {
        ImageRef { inner: image }
    }
}

impl<'lt, L> From<RawImage<&'lt mut buf, L>> for ImageMut<'lt, L> {
    fn from(image: RawImage<&'lt mut buf, L>) -> Self {
        ImageMut { inner: image }
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

impl<Layout: Clone> Clone for RawImage<Cog<'_>, Layout> {
    fn clone(&self) -> Self {
        use alloc::borrow::ToOwned;
        RawImage {
            buffer: Cog::Owned(self.buffer.to_owned()),
            layout: self.layout.clone(),
        }
    }
}

impl<Layout: Default> Default for Image<Layout> {
    fn default() -> Self {
        Image {
            inner: RawImage {
                buffer: Buffer::default(),
                layout: Layout::default(),
            },
        }
    }
}

impl<Layout: Default> Default for CopyOnGrow<'_, Layout> {
    fn default() -> Self {
        CopyOnGrow {
            inner: RawImage {
                buffer: Cog::Owned(Buffer::default()),
                layout: Layout::default(),
            },
        }
    }
}

impl<L> fmt::Debug for Image<L>
where
    L: SliceLayout + fmt::Debug,
    L::Sample: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Image")
            .field("layout", &self.inner.layout)
            .field("content", &self.inner.as_slice())
            .finish()
    }
}
