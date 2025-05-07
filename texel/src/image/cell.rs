//! Defines the containers operating on `!Sync` shared bytes.
//!
//! Re-exported at its super `image` module.
use crate::buf::{cell_buf, CellBuffer};
use crate::image::{raw::RawImage, Image, IntoPlanesError};
use crate::layout::{Bytes, Decay, Layout, Mend, PlaneOf, Relocate, SliceLayout, Take, TryMend};
use crate::texel::{constants::U8, MAX_ALIGN};
use crate::{BufferReuseError, Texel, TexelBuffer};
use core::cell::Cell;

/// A container of allocated bytes, parameterized over the layout.
///
/// This is a unsynchronized, shared equivalent to [`Image`][`crate::image::Image`]. That is the
/// buffer of bytes of this container is shared between clones of this value but can not be sent
/// between threads. In particular the same buffer may be owned and viewed with different layouts.
#[derive(Clone, PartialEq, Eq)]
pub struct CellImage<Layout = Bytes> {
    inner: RawImage<CellBuffer, Layout>,
}

/// A partial view of an atomic image.
///
/// Note that this requires its underlying buffer to be highly aligned! For that reason it is not
/// possible to take a reference at an arbitrary number of bytes. Values of this type are created
/// by calling [`CellImage::as_ref`] or [`CellImage::checked_to_ref`].
#[derive(Clone, PartialEq, Eq)]
pub struct CellImageRef<'buf, Layout = &'buf Bytes> {
    inner: RawImage<&'buf cell_buf, Layout>,
}

/// Image methods for all layouts.
impl<L: Layout> CellImage<L> {
    /// Create a new image for a specific layout.
    pub fn new(layout: L) -> Self {
        RawImage::<CellBuffer, L>::new(layout).into()
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
        RawImage::from_buffer(layout, CellBuffer::from(buffer)).into()
    }

    /// Change the layer of the image.
    ///
    /// Call [`CellImage::fits`] to check if this will work beforehand. Returns an `Err` with the
    /// original image if the buffer does not fit the new layout. Returns `Ok` with the new image
    /// if the buffer does fit. Never reallocates the buffer, the new image will always alias any
    /// other image sharing the buffer.
    pub fn try_with_layout<M>(self, layout: M) -> Result<CellImage<M>, Self>
    where
        M: Layout,
    {
        self.inner
            .try_reinterpret(layout)
            .map(Into::into)
            .map_err(Into::into)
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
    /// # use image_texel::{image::CellImage, layout::Matrix, layout};
    /// let matrix = Matrix::<u8>::width_and_height(400, 400).unwrap();
    /// let image: CellImage<layout::Matrix<u8>> = CellImage::new(matrix);
    ///
    /// // to turn hide the `u8` type but keep width, height, texel layout
    /// let as_bytes: CellImage<layout::MatrixBytes> = image.clone().decay();
    /// assert_eq!(as_bytes.layout().width(), 400);
    /// assert_eq!(as_bytes.layout().height(), 400);
    /// ```
    ///
    /// See also [`CellImage::mend`] and [`CellImage::try_mend`] for operations that reverse
    /// the effects.
    ///
    /// Can also be used to forget specifics of the layout, turning the image into a more general
    /// container type. For example, to use a uniform type as an allocated buffer waiting on reuse.
    ///
    /// ```
    /// # use image_texel::{image::CellImage, layout::Matrix, layout};
    /// let matrix = Matrix::<u8>::width_and_height(400, 400).unwrap();
    ///
    /// // Can always decay to a byte buffer.
    /// let bytes: CellImage = CellImage::new(matrix).decay();
    /// let _: &layout::Bytes = bytes.layout();
    /// ```
    ///
    /// [`Decay`]: ../layout/trait.Decay.html
    pub fn decay<M>(self) -> CellImage<M>
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
    pub fn checked_decay<M>(self) -> Option<CellImage<M>>
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
    /// use image_texel::image::{CellImage, Image};
    /// use image_texel::layout::{PlaneMatrices, Matrix};
    /// use image_texel::texels::U8;
    ///
    /// let matrix = Matrix::from_width_height(U8, 8, 8).unwrap();
    /// let buffer = CellImage::new(matrix);
    ///
    /// // … some code to initialize those planes.
    /// # let mut buffer = buffer;
    /// # let data = &buffer.as_cell_buf()[U8.to_range(0..8).unwrap()];
    /// # U8.store_cell_slice(data, b"not zero");
    /// # let buffer = buffer;
    ///
    /// let clone_of: Image<_> = buffer.clone().into_owned();
    ///
    /// assert!(clone_of.as_bytes() == buffer.as_cell_buf());
    /// ```
    pub fn into_owned(self) -> Image<L> {
        self.inner.into_owned().into()
    }

    /// Move the bytes into a new image.
    ///
    /// Afterwards, `self` will refer to an empty but unique new buffer.
    pub fn take(&mut self) -> CellImage<L>
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
    pub fn mend<Item>(self, mend: Item) -> CellImage<Item::Into>
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
    pub fn try_mend<Item>(&mut self, mend: Item) -> Result<CellImage<Item::Into>, Item::Err>
    where
        Item: TryMend<L>,
        L: Take,
    {
        let new_layout = mend.try_mend(self.inner.layout())?;
        Ok(self.inner.take().mogrify_layout(|_| new_layout).into())
    }
}

/// Image methods that do not require a layout.
impl<L> CellImage<L> {
    /// Check if the buffer could accommodate another layout without reallocating.
    pub fn fits(&self, layout: &impl Layout) -> bool {
        self.inner.fits(layout)
    }

    /// Check if two images refer to the same buffer.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        CellBuffer::ptr_eq(self.inner.get(), other.inner.get())
    }

    /// Get a reference to the underlying buffer.
    pub fn as_cell_buf(&self) -> &cell_buf
    where
        L: Layout,
    {
        self.inner.as_cell_buf()
    }

    /// Get a reference to the aligned unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`Self::make_mut`].
    pub fn as_capacity_cell_buf(&self) -> &cell_buf {
        self.inner.as_capacity_cell_buf()
    }

    /// Get a mutable reference to all allocated bytes if this image does not alias any other.
    ///
    /// # Example
    ///
    /// ```
    /// use image_texel::{image::CellImage, layout::Matrix};
    ///
    /// let layout = Matrix::<[u8; 4]>::width_and_height(10, 10).unwrap();
    /// let mut image = CellImage::new(layout);
    /// assert!(image.get_mut().is_some());
    ///
    /// let mut clone_of = image.clone();
    /// assert!(image.get_mut().is_none());
    /// ```
    pub fn get_mut(&mut self) -> Option<&mut cell_buf> {
        self.inner.get_mut().get_mut()
    }

    /// Ensure this image does not alias any other.
    ///
    /// Then returns a mutable reference to all the bytes allocated in the buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use image_texel::{image::CellImage, layout::Matrix, texels::U8};
    /// let texel = U8.array::<4>();
    ///
    /// let layout = Matrix::<[u8; 4]>::width_and_height(10, 10).unwrap();
    /// let image = CellImage::new(layout);
    ///
    /// let mut clone_of = image.clone();
    /// let atomic_mut_buf = clone_of.make_mut();
    ///
    /// // Now these are independent buffers.
    /// atomic_mut_buf.as_texels(texel).as_slice_of_cells()[0].set([0xff; 4]);
    /// assert_ne!(image.as_slice().as_slice_of_cells()[0].get(), [0xff; 4]);
    ///
    /// // With mutable reference we initialized the new buffer.
    /// assert_eq!(clone_of.as_slice().as_slice_of_cells()[0].get(), [0xff; 4]);
    /// ```
    pub fn make_mut(&mut self) -> &mut cell_buf {
        self.inner.get_mut().make_mut()
    }

    /// View this buffer as a slice of texels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_slice`].
    pub fn as_texels<P>(&self, texel: Texel<P>) -> &'_ Cell<[P]>
    where
        L: Layout,
    {
        self.as_ref().into_texels(texel)
    }

    /// View this buffer as a slice of its inherent pixels.
    pub fn as_slice(&self) -> &'_ Cell<[L::Sample]>
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
    pub fn as_ref(&self) -> CellImageRef<'_, &'_ L> {
        self.inner.as_deref().into()
    }

    /// Get a view of this image, if the alternate layout fits.
    pub fn checked_to_ref<M: Layout>(&self, layout: M) -> Option<CellImageRef<'_, M>> {
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

impl<'data, L> CellImageRef<'data, L> {
    /// Get a reference to the underlying buffer.
    pub fn as_cell_buf(&self) -> &cell_buf
    where
        L: Layout,
    {
        self.inner.as_cell_buf()
    }

    /// Get a reference to the complete underlying buffer, ignoring the layout.
    pub fn as_capacity_cell_buf(&self) -> &cell_buf {
        self.inner.get()
    }

    pub fn layout(&self) -> &L {
        self.inner.layout()
    }

    /// Get a view of this image.
    pub fn as_ref(&self) -> CellImageRef<'_, &'_ L> {
        self.inner.as_deref().into()
    }

    /// Check if a call to [`CellImageRef::checked_with_layout`] would succeed.
    pub fn fits(&self, other: &impl Layout) -> bool {
        self.inner.fits(other)
    }

    /// Change this view to a different layout.
    ///
    /// This returns `Some` if the layout fits the underlying data, and `None` otherwise. Use
    /// [`CellImageRef::fits`] to check this property in a separate call. Note that the new layout
    /// need not be related to the old layout in any other way.
    ///
    /// # Usage
    ///
    /// ```rust
    /// # fn not_main() -> Option<()> {
    /// use image_texel::{image::CellImage, layout::Matrix, layout::Bytes};
    ///
    /// let layout = Matrix::<[u8; 4]>::width_and_height(10, 10).unwrap();
    /// let image = CellImage::new(layout);
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
    pub fn checked_with_layout<M>(self, layout: M) -> Option<CellImageRef<'data, M>>
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
    /// See [`CellImage::decay`].
    pub fn decay<M>(self) -> CellImageRef<'data, M>
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
    /// See [`CellImage::checked_decay`].
    pub fn checked_decay<M>(self) -> Option<CellImageRef<'data, M>>
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
    /// use image_texel::image::{CellImage, Image};
    /// use image_texel::layout::{PlaneMatrices, Matrix};
    /// use image_texel::texels::U8;
    ///
    /// let matrix = Matrix::from_width_height(U8, 8, 8).unwrap();
    /// let buffer = CellImage::new(PlaneMatrices::<_, 2>::from_repeated(matrix));
    ///
    /// // … some code to initialize those planes.
    /// # let [plane] = buffer.as_ref().into_planes([1]).unwrap();
    /// # let data = &plane.as_cell_buf()[U8.to_range(0..8).unwrap()];
    /// # U8.store_cell_slice(data, b"not zero");
    ///
    /// let [plane1] = buffer.as_ref().into_planes([1]).unwrap();
    /// let clone_of: Image<_> = plane1.clone().into_owned();
    ///
    /// assert!(clone_of.as_bytes() == plane1.as_cell_buf());
    /// ```
    pub fn into_owned(self) -> Image<L> {
        self.inner.into_owned().into()
    }

    /// Get a slice of the individual samples in the layout.
    pub fn as_slice(&self) -> &'_ Cell<[L::Sample]>
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
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> &'_ Cell<[P]>
    where
        L: Layout,
    {
        self.inner.as_cell_buf().as_texels(pixel)
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
        let mut target = alloc::vec![0; len];
        let source = buffer.truncate(len).as_texels(U8);
        U8.cell_memory_copy(
            source.as_slice_of_cells(),
            Cell::from_mut(&mut target[..]).as_slice_of_cells(),
        );
        target
    }

    /// Turn into a slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_slice(self) -> &'data Cell<[L::Sample]>
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
    pub fn into_texels<P>(self, pixel: Texel<P>) -> &'data Cell<[P]>
    where
        L: Layout,
    {
        let (buffer, layout) = self.inner.into_parts();
        let byte_len = layout.byte_len();
        buffer.truncate(byte_len).as_texels(pixel)
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
    pub fn split_layout(&mut self) -> CellImageRef<'data, Bytes>
    where
        L: Layout,
    {
        // Need to roundup to correct alignment.
        let size = self.inner.layout().byte_len();
        let round_up = size.next_multiple_of(MAX_ALIGN);
        let buffer = self.inner.get_mut();

        if round_up > buffer.len() {
            return RawImage::from_buffer(Bytes(0), cell_buf::new(&[])).into();
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
    /// use image_texel::image::{CellImage, CellImageRef};
    /// use image_texel::layout::{PlaneMatrices, Matrix};
    /// use image_texel::texels::U8;
    ///
    /// let mat = Matrix::from_width_height(U8, 8, 8).unwrap();
    /// let buffer = CellImage::new(PlaneMatrices::<_, 2>::from_repeated(mat));
    /// let image: CellImageRef<'_, _> = buffer.as_ref();
    ///
    /// let [p0, p1] = buffer.as_ref().into_planes([0, 1]).unwrap();
    /// ```
    ///
    /// You may select the same plane twice:
    ///
    pub fn into_planes<const N: usize, D>(
        self,
        descriptors: [D; N],
    ) -> Result<[CellImageRef<'data, D::Plane>; N], IntoPlanesError>
    where
        D: PlaneOf<L>,
        D::Plane: Relocate,
    {
        let layout = self.layout();
        let mut planes = descriptors.map(|d| {
            let plane = <D as PlaneOf<L>>::get_plane(d, layout);
            let empty_buf = cell_buf::new(&[]);
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

impl<L> From<RawImage<CellBuffer, L>> for CellImage<L> {
    fn from(image: RawImage<CellBuffer, L>) -> Self {
        CellImage { inner: image }
    }
}

impl<'lt, L> From<RawImage<&'lt cell_buf, L>> for CellImageRef<'lt, L> {
    fn from(image: RawImage<&'lt cell_buf, L>) -> Self {
        CellImageRef { inner: image }
    }
}
