//! Defines the containers operating on `Sync` shared bytes.
//!
//! Re-exported at its super `image` module.
use crate::buf::{atomic_buf, AtomicBuffer, AtomicSliceRef};
use crate::image::{raw::RawImage, Image, IntoPlanesError};
use crate::layout::{Bytes, Decay, Layout, Mend, PlaneOf, Relocate, SliceLayout, Take, TryMend};
use crate::texel::{constants::U8, MAX_ALIGN};
use crate::{BufferReuseError, Texel, TexelBuffer};

/// A container of allocated bytes, parameterized over the layout.
///
/// This is a synchronized, shared equivalent to [`Image`][`crate::image::Image`]. That is the
/// buffer of bytes of this container is shared between clones of this value and potentially
/// between threads. In particular the same buffer may be owned and viewed with different layouts
/// and modified concurrently. The guarantee, however, is merely that concurrent modification is
/// free of undefined data races. There is no locking, implied synchronization, or ordering
/// guarantees between edits except when you can modify disjoint parts of the buffer.
///
/// ## Differences to owned Image
///
/// Comparing values of this type is possible, but requires calling the method [`Self::compare`] to
/// create a comparator. This is because comparing is inherently racing against modifications made
/// on other threads. While the implementation prevents any unsound *data races* there is no
/// specific meaning to any of its outcomes unless the caller ensure synchronization in some other
/// manner.
///
/// # Examples
///
/// As a type with shared ownership over the underling buffer, this type can be cloned very
/// cheaply. Such duplicates refer to the same buffer, making changes in one visible to the other.
///
/// ```
/// use image_texel::{image::AtomicImage, layout::Matrix};
/// let matrix = Matrix::<u8>::width_and_height(32, 32).unwrap();
/// let image: AtomicImage<_> = AtomicImage::new(matrix);
///
/// let another_reference = image.clone();
/// assert!(AtomicImage::ptr_eq(&image, &another_reference));
///
/// another_reference.as_slice().index_one(0).store(0xff);
/// let value = image.as_slice().index_one(0).load();
/// assert_eq!(value, 0xff);
/// ```
#[derive(Clone)]
pub struct AtomicImage<Layout = Bytes> {
    pub(super) inner: RawImage<AtomicBuffer, Layout>,
}

/// A partial view of an atomic image.
///
/// Note that this requires its underlying buffer to be highly aligned! For that reason it is not
/// possible to take a reference at an arbitrary number of bytes. Values of this type are created
/// by calling [`AtomicImage::as_ref`] or [`AtomicImage::checked_to_ref`].
#[derive(Clone, PartialEq, Eq)]
pub struct AtomicImageRef<'buf, Layout = &'buf Bytes> {
    pub(super) inner: RawImage<&'buf atomic_buf, Layout>,
}

/// Image methods for all layouts.
impl<L: Layout> AtomicImage<L> {
    /// Create a new image for a specific layout.
    pub fn new(layout: L) -> Self {
        RawImage::<AtomicBuffer, L>::new(layout).into()
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
        let (buffer, layout) = RawImage::from_buffer(Bytes(0), bytes.into_inner())
            .with_layout(layout)
            .into_parts();
        RawImage::from_buffer(layout, AtomicBuffer::from(buffer)).into()
    }

    /// Change the layer of the image.
    ///
    /// Call [`AtomicImage::fits`] to check if this will work beforehand. Returns an `Err` with the
    /// original image if the buffer does not fit the new layout. Returns `Ok` with the new image
    /// if the buffer does fit. Never reallocates the buffer, the new image will always alias any
    /// other image sharing the buffer.
    ///
    /// This returns a [`BufferReuseError`] with information about the exceeded limits. If you need
    /// the prior value then you can make a [`AtomicImage::clone`]` of it, it is cheap.
    pub fn try_with_layout<M>(self, layout: M) -> Result<AtomicImage<M>, BufferReuseError>
    where
        M: Layout,
    {
        let requested = layout.byte_len();
        match self.inner.try_reinterpret(layout) {
            Ok(raw) => Ok(raw.into()),
            Err(err) => Err(BufferReuseError {
                capacity: err.as_capacity_atomic_buf().len(),
                requested: Some(requested),
            }),
        }
    }

    /// Attempt to modify the layout to a new value, without modifying its type.
    ///
    /// Returns an `Err` if the layout does not fit the underlying buffer. Otherwise returns `Ok`
    /// and overwrites the layout accordingly.
    ///
    /// TODO: public name and provide a `set_capacity` for `L = Bytes`?
    pub(crate) fn try_set_layout(&mut self, layout: L) -> Result<(), BufferReuseError>
    where
        L: Layout,
    {
        self.inner.try_reuse(layout)
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
    /// # use image_texel::{image::AtomicImage, layout::Matrix, layout};
    /// let matrix = Matrix::<u8>::width_and_height(32, 32).unwrap();
    /// let image: AtomicImage<layout::Matrix<u8>> = AtomicImage::new(matrix);
    ///
    /// // to turn hide the `u8` type but keep width, height, texel layout
    /// let as_bytes: AtomicImage<layout::MatrixBytes> = image.clone().checked_decay().unwrap();
    /// assert_eq!(as_bytes.layout().width(), 32);
    /// assert_eq!(as_bytes.layout().height(), 32);
    /// ```
    ///
    /// See also [`AtomicImage::mend`] and [`AtomicImage::try_mend`] for operations that reverse
    /// the effects.
    ///
    /// Can also be used to forget specifics of the layout, turning the image into a more general
    /// container type. For example, to use a uniform type as an allocated buffer waiting on reuse.
    ///
    /// ```
    /// # use image_texel::{image::AtomicImage, layout::Matrix, layout};
    /// let matrix = Matrix::<u8>::width_and_height(16, 16).unwrap();
    ///
    /// // Can always decay to a byte buffer.
    /// let bytes: AtomicImage = AtomicImage::new(matrix).checked_decay().unwrap();
    /// let _: &layout::Bytes = bytes.layout();
    /// ```
    ///
    /// [`Decay`]: ../layout/trait.Decay.html
    pub fn decay<M>(self) -> AtomicImage<M>
    where
        M: Decay<L>,
        M: Layout,
    {
        self.inner
            .checked_decay()
            .unwrap_or_else(super::decay_failed)
            .into()
    }

    /// Like [`Self::decay`]` but returns `None` rather than panicking. While this is strictly
    /// speaking a violation of the trait contract, you may want to handle this yourself.
    pub fn checked_decay<M>(self) -> Option<AtomicImage<M>>
    where
        M: Decay<L>,
        M: Layout,
    {
        Some(self.inner.checked_decay()?.into())
    }

    /// Copy all bytes to a newly allocated image.
    ///
    /// Note this will allocate a buffer according to the capacity length of this reference, not
    /// merely the layout. When this is not the intention, consider first adjusting the buffer by
    /// reference with [`Self::as_ref`].
    ///
    /// # Examples
    ///
    /// Here we make an independent copy of a pixel matrix image.
    ///
    /// ```
    /// use image_texel::image::{AtomicImage, Image};
    /// use image_texel::layout::{PlaneMatrices, Matrix};
    /// use image_texel::texels::U8;
    ///
    /// let matrix = Matrix::from_width_height(U8, 8, 8).unwrap();
    /// let buffer = AtomicImage::new(matrix);
    ///
    /// // … some code to initialize those planes.
    /// # let data = buffer.as_texels(U8).index(0..8);
    /// # U8.store_atomic_slice(data, b"not zero");
    ///
    /// let clone_of: Image<_> = buffer.clone().into_owned();
    ///
    /// assert!(clone_of.as_bytes() == buffer.as_texels(U8).to_vec());
    /// ```
    pub fn into_owned(self) -> Image<L> {
        self.inner.into_owned().into()
    }

    /// Move the bytes into a new image.
    ///
    /// Afterwards, `self` will refer to an empty but unique new buffer.
    pub fn take(&mut self) -> AtomicImage<L>
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
    pub fn mend<Item>(self, mend: Item) -> AtomicImage<Item::Into>
    where
        Item: Mend<L>,
        L: Take,
    {
        let new_layout = mend.mend(self.inner.layout());
        self.inner.mogrify_layout(|_| new_layout).into()
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
    pub fn try_mend<Item>(&mut self, mend: Item) -> Result<AtomicImage<Item::Into>, Item::Err>
    where
        Item: TryMend<L>,
        L: Take,
    {
        let new_layout = mend.try_mend(self.inner.layout())?;
        Ok(self.inner.take().mogrify_layout(|_| new_layout).into())
    }
}

/// Image methods that do not require a layout.
impl<L> AtomicImage<L> {
    /// Check if the buffer could accommodate another layout without reallocating.
    pub fn fits(&self, layout: &impl Layout) -> bool {
        self.inner.fits(layout)
    }

    /// Check if two images refer to the same buffer.
    ///
    /// Note that two buffers can use different layout types to describe their share of the data or
    /// even to refer to the same data in different ways.
    pub fn ptr_eq<O>(&self, other: &AtomicImage<O>) -> bool {
        AtomicBuffer::ptr_eq(self.inner.get(), other.inner.get())
    }

    /// Create a comparator to another image.
    ///
    /// Note that comparing is inherently racing against modifications made on other threads. While
    /// the implementation prevents any unsound *data races* there is no specific meaning to any of
    /// its outcomes unless the caller ensure synchronization in some other manner.
    ///
    /// You can also compare the allocation with [`Self::ptr_eq`] or ignore the layout and compare
    /// buffer contents with [`Self::as_capacity_atomic_buf`].
    pub fn compare(&self) -> impl core::cmp::Eq + core::cmp::PartialEq + '_
    where
        L: core::cmp::Eq,
    {
        (self.inner.layout(), self.inner.get())
    }

    /// Get a reference to the aligned unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`Self::make_mut`].
    pub fn as_capacity_atomic_buf(&self) -> &atomic_buf {
        self.inner.as_capacity_atomic_buf()
    }

    /// Get a mutable reference to all allocated bytes if this image does not alias any other.
    ///
    /// # Example
    ///
    /// ```
    /// use image_texel::{image::AtomicImage, layout::Matrix};
    ///
    /// let layout = Matrix::<[u8; 4]>::width_and_height(10, 10).unwrap();
    /// let mut image = AtomicImage::new(layout);
    /// assert!(image.get_mut().is_some());
    ///
    /// let mut clone_of = image.clone();
    /// assert!(image.get_mut().is_none());
    /// ```
    pub fn get_mut(&mut self) -> Option<&mut atomic_buf> {
        self.inner.get_mut().get_mut()
    }

    /// Ensure this image does not alias any other.
    ///
    /// Then returns a mutable reference to all the bytes allocated in the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use image_texel::{image::AtomicImage, layout::Matrix, texels::U8};
    /// let texel = U8.array::<4>();
    ///
    /// let layout = Matrix::<[u8; 4]>::width_and_height(10, 10).unwrap();
    /// let image = AtomicImage::new(layout);
    ///
    /// let mut clone_of = image.clone();
    /// let atomic_mut_buf = clone_of.make_mut();
    ///
    /// // Now these are independent buffers.
    /// atomic_mut_buf.as_buf_mut().as_mut_texels(texel)[0] = [0xff; 4];
    /// assert_ne!(texel.load_atomic(image.as_slice().index_one(0)), [0xff; 4]);
    ///
    /// // With mutable reference we initialized the new buffer.
    /// assert_eq!(texel.load_atomic(clone_of.as_slice().index_one(0)), [0xff; 4]);
    /// ```
    pub fn make_mut(&mut self) -> &mut atomic_buf {
        self.inner.get_mut().make_mut()
    }

    /// View this buffer as a slice of texels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_slice`].
    pub fn as_texels<P>(&self, texel: Texel<P>) -> AtomicSliceRef<'_, P>
    where
        L: Layout,
    {
        self.as_ref().into_texels(texel)
    }

    /// View this buffer as a slice of its inherent pixels.
    pub fn as_slice(&self) -> AtomicSliceRef<'_, L::Sample>
    where
        L: SliceLayout,
    {
        self.as_ref().into_slice()
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
    pub fn as_ref(&self) -> AtomicImageRef<'_, &'_ L> {
        self.inner.as_deref().into()
    }

    /// Get a view of this image, if the alternate layout fits.
    pub fn checked_to_ref<M: Layout>(&self, layout: M) -> Option<AtomicImageRef<'_, M>> {
        self.as_ref().checked_with_layout(layout)
    }

    /*
    /// Get a single texel from a raster image.
    #[deprecated = "Do not use yet"]
    pub fn get_texel<P>(&self, _: Coord) -> Option<P>
    where
        L: Raster<P>,
    {
        todo!("Failure of the Raster trait");
    }

    /// Put a single texel to a raster image.
    #[deprecated = "Do not use yet"]
    pub fn put_texel<P>(&mut self, _: Coord, _: P)
    where
        L: RasterMut<P>,
    {
        todo!("Failure of the Raster trait");
    }

    /// Call a function on each texel of this raster image.
    ///
    /// The order of evaluation is _not_ defined although certain layouts may offer more specific
    /// guarantees. In general, one can expect that layouts call the function in a cache-efficient
    /// manner if they are aware of a better iteration strategy.
    pub fn shade<P>(&mut self, _: impl FnMut(u32, u32, &mut P))
    where
        L: RasterMut<P>,
    {
        todo!()
    }
    */
}

impl<'data> AtomicImageRef<'data, Bytes> {
    /// Wrap aligned data as a buffer for an image.
    ///
    /// # Examples
    ///
    /// You can create a buffer, then later transfer data and layout through [`data`].
    ///
    /// ```
    /// use core::cell::Cell;
    /// use image_texel::{image::AtomicImageRef, image::data, layout, texels};
    ///
    /// // Create our stack based image buffer.
    /// let mut data = [texels::MAX.zeroed(); 16];
    /// let buffer = texels::atomic_buf::from_max_mut(&mut data);
    /// let image = AtomicImageRef::new(buffer);
    ///
    /// # fn somewhere() -> data::DataRef<'static, layout::Matrix<u8>> {
    /// #     let l = layout::Matrix::from_width_height(texels::U8, 4, 4).unwrap();
    /// #     data::DataRef::with_layout_at(&[0; 16], l, 0).unwrap()
    /// # }
    /// // Now assume we have data from somewhere:
    /// let data: data::DataRef<'_, layout::Matrix<u8>> = somewhere();
    ///
    /// let Some(image) = data.as_source().write_to_atomic_ref(image) else {
    ///     return; // Okay, our stack buffer was too small..
    /// };
    ///
    /// // We now have a typed image buffer on the stack.
    /// let image: AtomicImageRef<layout::Matrix<u8>> = image;
    /// ```
    pub fn new(data: &'data atomic_buf) -> Self {
        RawImage::from_buffer(Bytes(data.len()), data).into()
    }
}

impl<'data, L> AtomicImageRef<'data, L> {
    /// Wrap an aligned buffer, with a specified layout.
    ///
    /// # Examples
    ///
    /// You can create an image on the stack:
    ///
    /// ```
    /// use image_texel::{image::AtomicImageRef, layout, texels};
    ///
    /// const LEN: usize = 256usize.div_ceil(texels::MAX.size());
    /// let mut data = [texels::MAX.zeroed(); LEN];
    /// let cells = texels::atomic_buf::from_max_mut(&mut data);
    ///
    /// let layout = layout::Matrix::from_width_height(texels::U8, 16, 16).unwrap();
    /// // Convert into an image buffer:
    /// let image = AtomicImageRef::from_layout(cells, layout);
    /// ```
    pub fn from_layout(data: &'data atomic_buf, layout: L) -> Self
    where
        L: Layout,
    {
        RawImage::from_buffer(layout, data).into()
    }

    /// Get a reference to the complete underlying buffer, ignoring the layout.
    pub fn as_capacity_atomic_buf(&self) -> &atomic_buf {
        self.inner.get()
    }

    pub fn layout(&self) -> &L {
        self.inner.layout()
    }

    /// Get a view of this image.
    pub fn as_ref(&self) -> AtomicImageRef<'_, &'_ L> {
        self.inner.as_deref().into()
    }

    /// Check if a call to [`AtomicImageRef::checked_with_layout`] would succeed.
    pub fn fits(&self, other: &impl Layout) -> bool {
        <dyn Layout>::fits_atomic_buf(other, self.inner.as_capacity_atomic_buf())
    }

    /// Change this view to a different layout.
    ///
    /// This returns `Some` if the layout fits the underlying data, and `None` otherwise. Use
    /// [`AtomicImageRef::fits`] to check this property in a separate call. Note that the new layout
    /// need not be related to the old layout in any other way.
    ///
    /// # Usage
    ///
    /// ```rust
    /// # fn not_main() -> Option<()> {
    /// use image_texel::{image::AtomicImage, layout::Matrix, layout::Bytes};
    ///
    /// let layout = Matrix::<[u8; 4]>::width_and_height(10, 10).unwrap();
    /// let image = AtomicImage::new(layout);
    ///
    /// let reference = image.as_ref();
    ///
    /// let as_bytes = reference.checked_with_layout(Bytes(400))?;
    /// assert!(matches!(as_bytes.layout(), Bytes(400)));
    ///
    /// // But not if we request too much.
    /// assert!(as_bytes.checked_with_layout(Bytes(500)).is_none());
    ///
    /// # Some(()) }
    /// # fn main() { not_main(); }
    /// ```
    pub fn checked_with_layout<M>(self, layout: M) -> Option<AtomicImageRef<'data, M>>
    where
        M: Layout,
    {
        Some(self.inner.try_reinterpret(layout).ok()?.into())
    }

    /// Attempt to modify the layout to a new value, without modifying its type.
    ///
    /// Returns an `Err` if the layout does not fit the underlying buffer. Otherwise returns `Ok`
    /// and overwrites the layout accordingly.
    ///
    /// TODO: public name and provide a `set_capacity` for `L = Bytes`?
    pub(crate) fn try_set_layout(&mut self, layout: L) -> Result<(), BufferReuseError>
    where
        L: Layout,
    {
        self.inner.try_reuse(layout)
    }

    /// Decay into a image with less specific layout.
    ///
    /// See [`AtomicImage::decay`].
    pub fn decay<M>(self) -> AtomicImageRef<'data, M>
    where
        M: Decay<L>,
        M: Layout,
    {
        self.inner
            .checked_decay()
            .unwrap_or_else(super::decay_failed)
            .into()
    }

    /// Decay into a image with less specific layout.
    ///
    /// See [`AtomicImage::checked_decay`].
    pub fn checked_decay<M>(self) -> Option<AtomicImageRef<'data, M>>
    where
        M: Decay<L>,
        M: Layout,
    {
        Some(self.inner.checked_decay()?.into())
    }

    /// Copy all bytes to a newly allocated image.
    ///
    /// Note this will allocate a buffer according to the capacity length of this reference, not
    /// merely the layout. When this is not the intention, consider calling [`Self::split_layout`]
    /// or [`Self::truncate_layout`] respectively.
    ///
    /// # Examples
    ///
    /// Here we make an independent copy of a pixel matrix image.
    ///
    /// ```
    /// use image_texel::image::{AtomicImage, Image};
    /// use image_texel::layout::{PlaneMatrices, Matrix};
    /// use image_texel::texels::U8;
    ///
    /// let matrix = Matrix::from_width_height(U8, 8, 8).unwrap();
    /// let buffer = AtomicImage::new(PlaneMatrices::<_, 2>::from_repeated(matrix));
    ///
    /// // … some code to initialize those planes.
    /// # let [plane] = buffer.as_ref().into_planes([1]).unwrap();
    /// # let data = buffer.as_texels(U8).index(0..8);
    /// # U8.store_atomic_slice(data, b"not zero");
    ///
    /// let [plane1] = buffer.as_ref().into_planes([1]).unwrap();
    /// let clone_of: Image<_> = plane1.clone().into_owned();
    ///
    /// assert!(clone_of.as_bytes() == plane1.as_texels(U8).to_vec());
    /// ```
    pub fn into_owned(self) -> Image<L> {
        self.inner.into_owned().into()
    }

    /// Get a slice of the individual samples in the layout.
    pub fn as_slice(&self) -> AtomicSliceRef<'_, L::Sample>
    where
        L: SliceLayout,
    {
        self.as_texels(self.inner.layout().sample())
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_slice`].
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> AtomicSliceRef<'_, P>
    where
        L: Layout,
    {
        let byte_len = self.inner.layout().byte_len();
        let buf = self.inner.as_capacity_atomic_buf();
        buf.as_texels(pixel).truncate_bytes(byte_len)
    }

    /// Turn into a slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_bytes(self) -> alloc::vec::Vec<u8>
    where
        L: Layout,
    {
        let (buffer, layout) = self.inner.into_parts();
        let len = layout.byte_len();

        buffer.as_texels(U8).truncate_bytes(len).to_vec()
    }

    /// Turn into a slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_slice(self) -> AtomicSliceRef<'data, L::Sample>
    where
        L: SliceLayout,
    {
        let sample = self.inner.layout().sample();
        self.into_texels(sample)
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_texels`].
    pub fn into_texels<P>(self, pixel: Texel<P>) -> AtomicSliceRef<'data, P>
    where
        L: Layout,
    {
        let (buffer, layout) = self.inner.into_parts();
        let byte_len = layout.byte_len();
        buffer.as_texels(pixel).truncate_bytes(byte_len)
    }

    /*
    /// Retrieve a single texel from a raster image.
    #[deprecated = "Do not use yet"]
    pub fn get_texel<P>(&self, _: Coord) -> Option<P>
    where
        L: Raster<P>,
    {
        todo!("Failure of the Raster trait");
    }

    /// Retrieve a single texel from a raster image.
    #[deprecated = "Do not use yet"]
    pub fn put_texel<P>(&self, _: Coord) -> Option<P>
    where
        L: Raster<P>,
    {
        todo!("Failure of the Raster trait");
    }
    */

    /// Split off all unused bytes at the tail of the layout.
    pub fn split_layout(&mut self) -> AtomicImageRef<'data, Bytes>
    where
        L: Layout,
    {
        // Need to roundup to correct alignment.
        let size = self.inner.layout().byte_len();
        let round_up = size.next_multiple_of(MAX_ALIGN);
        let buffer = self.inner.get_mut();

        if round_up > buffer.len() {
            return RawImage::from_buffer(Bytes(0), atomic_buf::new(&[])).into();
        }

        let (initial, next) = buffer.split_at(round_up);
        *buffer = initial;

        RawImage::from_buffer(Bytes(next.len()), next).into()
    }

    /// Remove all past-the-layout bytes.
    ///
    /// This is a utility to combine with pipelining. It is equivalent to calling
    /// [`Self::split_layout`] and discarding that result.
    pub fn truncate_layout(mut self) -> Self
    where
        L: Layout,
    {
        let _ = self.split_layout();
        self
    }

    /// Split this reference into independent planes.
    ///
    /// If any plane fails their indexing operation or would not be aligned to the required
    /// alignment or any plane layouts would overlap, an error is returned. The planes are returned
    /// in the order of the descriptors.
    ///
    /// FIXME: the layout type is not what we want. For instance, with `PlaneMatrices` we get a
    /// plane type of `Relocated<Matrix<_>>` but when we relocate that to `0` then we would really
    /// prefer having a simple `Matrix<_>` as the layout type.
    ///
    /// # Examples
    ///
    /// A layout describing a matrix array can be split:
    ///
    /// ```
    /// use image_texel::image::{AtomicImage, AtomicImageRef};
    /// use image_texel::layout::{PlaneMatrices, Matrix};
    /// use image_texel::texels::U8;
    ///
    /// let mat = Matrix::from_width_height(U8, 8, 8).unwrap();
    /// let buffer = AtomicImage::new(PlaneMatrices::<_, 2>::from_repeated(mat));
    /// let image: AtomicImageRef<'_, _> = buffer.as_ref();
    ///
    /// let [p0, p1] = buffer.as_ref().into_planes([0, 1]).unwrap();
    /// ```
    ///
    /// You may select the same plane twice:
    ///
    pub fn into_planes<const N: usize, D>(
        self,
        descriptors: [D; N],
    ) -> Result<[AtomicImageRef<'data, D::Plane>; N], IntoPlanesError>
    where
        D: PlaneOf<L>,
        D::Plane: Relocate,
    {
        let layout = self.layout();
        let mut planes = descriptors.map(|d| {
            let plane = <D as PlaneOf<L>>::get_plane(d, layout);
            let empty_buf = atomic_buf::new(&[]);
            (plane, empty_buf)
        });

        let (mut buffer, _) = self.inner.into_parts();

        for plane in &mut planes {
            let Some(layout) = &mut plane.0 else {
                continue;
            };

            let skip_by = layout.byte_offset();

            // FIXME: do we want failure reasons?
            if skip_by % MAX_ALIGN != 0 {
                plane.0 = None;
                continue;
            }

            if buffer.len() < skip_by {
                plane.0 = None;
                continue;
            }

            layout.relocate(Default::default());
            let len = layout.byte_len().div_ceil(MAX_ALIGN) * MAX_ALIGN;

            // Check this before we consume the buffer. This way the tail can still be used by
            // following layouts, we ignore this.
            if buffer.len() - skip_by < len {
                plane.0 = None;
                continue;
            }

            let (_pre, tail) = buffer.split_at(skip_by);
            let (img_buf, _post) = tail.split_at(len);

            plane.1 = img_buf;
            buffer = tail;
        }

        let planes = IntoPlanesError::from_array(planes)?;
        Ok(planes.map(|(layout, buffer)| RawImage::from_buffer(layout, buffer).into()))
    }
}

impl<L> From<RawImage<AtomicBuffer, L>> for AtomicImage<L> {
    fn from(image: RawImage<AtomicBuffer, L>) -> Self {
        AtomicImage { inner: image }
    }
}

impl<'lt, L> From<RawImage<&'lt atomic_buf, L>> for AtomicImageRef<'lt, L> {
    fn from(image: RawImage<&'lt atomic_buf, L>) -> Self {
        AtomicImageRef { inner: image }
    }
}
