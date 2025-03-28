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
mod raw;

use core::{fmt, ops};

pub(crate) use self::raw::RawImage;
use crate::buf::{buf, Buffer};
use crate::layout::{
    Bytes, Decay, Layout, Mend, PlaneOf, Raster, RasterMut, Relocate, SliceLayout, Take, TryMend,
};
use crate::texel::MAX_ALIGN;
use crate::{Texel, TexelBuffer};

pub use crate::stride::{StridedBufferMut, StridedBufferRef};

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
        RawImage::from_buffer(Bytes(0), bytes.into_inner())
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

    /// Get a reference to the aligned unstructured bytes of the image.
    pub fn as_buf(&self) -> &buf {
        self.inner.as_buf()
    }

    /// Get a reference to the mutable aligned unstructured bytes of the image.
    pub fn as_mut_buf(&mut self) -> &mut buf {
        self.inner.as_mut_buf()
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
    pub fn try_mend<Item>(&mut self, mend: Item) -> Result<Image<Item::Into>, Item::Err>
    where
        Item: TryMend<L>,
        L: Take,
    {
        let new_layout = mend.try_mend(self.inner.layout())?;
        Ok(self.inner.take().mogrify_layout(|_| new_layout).into())
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

    /// Get a reference to the aligned unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`Self::as_capacity_bytes`].
    pub fn as_capacity_buf(&self) -> &buf {
        self.inner.as_capacity_buf()
    }

    /// Get a mutable reference to the unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`Self::as_capacity_bytes_mut`].
    pub fn as_capacity_buf_mut(&mut self) -> &mut buf {
        self.inner.as_capacity_buf_mut()
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_slice`].
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> &[P]
    where
        L: Layout,
    {
        pixel.cast_buf(self.inner.as_buf())
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_mut_slice`].
    pub fn as_mut_texels<P>(&mut self, pixel: Texel<P>) -> &mut [P]
    where
        L: Layout,
    {
        pixel.cast_mut_buf(self.inner.as_mut_buf())
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
        self.inner.as_borrow().into()
    }

    /// Get a view of this image, if the alternate layout fits.
    pub fn try_to_ref<M: Layout>(&self, layout: M) -> Option<ImageRef<'_, M>> {
        self.as_ref().with_layout(layout)
    }

    /// Get a mutable view of this image.
    pub fn as_mut(&mut self) -> ImageMut<'_, &'_ mut L> {
        self.inner.as_borrow_mut().into()
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
        RawImage::from_texel_buffer(buffer, layout).into()
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
        self.inner.layout()
    }

    /// Get a view of this image.
    pub fn as_ref(&self) -> ImageRef<'_, &'_ L> {
        self.inner.as_borrow().into()
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
    ///
    /// # Usage
    ///
    /// ```rust
    /// # fn not_main() -> Option<()> {
    /// use image_texel::{Image, Matrix, layout::Bytes};
    /// let image = Image::from(Matrix::<[u8; 4]>::with_width_and_height(10, 10));
    ///
    /// let reference = image.as_ref();
    ///
    /// let as_bytes = reference.with_layout(Bytes(400))?;
    /// assert!(matches!(as_bytes.layout(), Bytes(400)));
    ///
    /// // But not if we request too much.
    /// assert!(as_bytes.with_layout(Bytes(500)).is_none());
    ///
    /// # Some(()) }
    /// # fn main() { not_main(); }
    /// ```
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
        Some(self.inner.checked_decay()?.into())
    }

    /// Copy all bytes to a newly allocated image.
    pub fn to_owned(&self) -> Image<L>
    where
        L: Layout + Clone,
    {
        Image::with_bytes(self.inner.layout().clone(), self.inner.as_bytes())
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
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> &[P]
    where
        L: Layout,
    {
        pixel.cast_buf(self.inner.as_buf())
    }

    /// Turn into a slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_bytes(self) -> &'data [u8]
    where
        L: Layout,
    {
        let (visible, layout) = self.inner.into_parts();
        visible.truncate(layout.byte_len())
    }

    /// Turn into a slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_slice(self) -> &'data [L::Sample]
    where
        L: SliceLayout,
    {
        let (visible, layout) = self.inner.into_parts();
        layout.sample().cast_buf(visible)
    }

    /// Retrieve a single texel from a raster image.
    pub fn get_texel<P>(&self, coord: Coord) -> Option<P>
    where
        L: Raster<P>,
    {
        L::get(self.as_ref(), coord)
    }

    /// Split off all unused bytes at the tail of the layout.
    pub fn split_layout(&mut self) -> ImageRef<'data, Bytes>
    where
        L: Layout,
    {
        // Need to roundup to correct alignment.
        let size = self.inner.layout().byte_len();
        let round_up = (size.wrapping_neg() & !(MAX_ALIGN - 1)).wrapping_neg();
        let buffer = self.inner.get_mut();

        if round_up > buffer.len() {
            return RawImage::from_buffer(Bytes(0), buf::new(&[])).into();
        }

        let (initial, next) = buffer.split_at(round_up);
        *buffer = initial;

        RawImage::from_buffer(Bytes(next.len()), next).into()
    }

    /// Split this reference into independent planes.
    ///
    /// Planes that fail their indexing operation or that would not be aligned to the required
    /// alignment are not returned. All other planes are returned in `Some`.
    ///
    /// FIXME: the layout type is not what we want. For instance, with `PlaneMatrices` we get a
    /// plane type of `Relocated<Matrix<_>>` but when we relocate that to `0` then we would really
    /// prefer having a simple `Matrix<_>` as the layout type.
    pub fn into_planes<const N: usize, D>(
        self,
        descriptors: [D; N],
    ) -> [Option<ImageRef<'data, D::Plane>>; N]
    where
        D: PlaneOf<L>,
        D::Plane: Relocate,
    {
        let layout = self.layout();
        let mut planes = descriptors.map(|d| {
            let plane = <D as PlaneOf<L>>::get_plane(d, layout);
            let empty_buf = buf::new(&[]);
            (plane, empty_buf)
        });

        let (mut buffer, _) = self.inner.into_parts();

        for plane in &mut planes {
            let Some(layout) = &mut plane.0 else {
                continue;
            };

            let skip_by = layout.byte_offset();

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

        planes.map(|(layout, buffer)| -> Option<_> {
            Some(RawImage::from_buffer(layout?, buffer).into())
        })
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
        self.inner.layout()
    }

    /// Get a view of this image.
    pub fn as_ref(&self) -> ImageRef<'_, &'_ L> {
        self.inner.as_borrow().into()
    }

    /// Get a mutable view of this image.
    pub fn as_mut(&mut self) -> ImageMut<'_, &'_ mut L> {
        self.inner.as_borrow_mut().into()
    }

    /// Convert to a view of this image.
    pub fn into_ref(self) -> ImageRef<'data, L> {
        let (buffer, layout) = self.inner.into_parts();
        RawImage::with_buffer_unchecked(layout, &*buffer).into()
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
    ///
    /// # Usage
    ///
    /// ```rust
    /// # fn not_main() -> Option<()> {
    /// use image_texel::{Image, Matrix, layout::Bytes};
    /// let mut image = Image::from(Matrix::<[u8; 4]>::with_width_and_height(10, 10));
    ///
    /// let reference = image.as_mut();
    ///
    /// let as_bytes = reference.with_layout(Bytes(400))?;
    /// assert!(matches!(as_bytes.layout(), Bytes(400)));
    ///
    /// // But not if we request too much.
    /// assert!(as_bytes.with_layout(Bytes(500)).is_none());
    ///
    /// # Some(()) }
    /// # fn main() { not_main(); }
    /// ```
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
        Some(self.inner.checked_decay()?.into())
    }

    /// Copy the bytes and layout to an owned container.
    pub fn to_owned(&self) -> Image<L>
    where
        L: Layout + Clone,
    {
        Image::with_bytes(self.inner.layout().clone(), self.inner.as_bytes())
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
    pub fn as_texels<P>(&self, pixel: Texel<P>) -> &[P]
    where
        L: Layout,
    {
        pixel.cast_buf(self.inner.as_buf())
    }

    /// View this buffer as a slice of pixels.
    ///
    /// This reinterprets the bytes of the buffer. It can be used to view the buffer as any kind of
    /// pixel, regardless of its association with the layout. Use it with care.
    ///
    /// An alternative way to get a slice of texels when a layout has an inherent texel type is
    /// [`Self::as_mut_slice`].
    pub fn as_mut_texels<P>(&mut self, pixel: Texel<P>) -> &mut [P]
    where
        L: Layout,
    {
        pixel.cast_mut_buf(self.inner.as_mut_buf())
    }

    /// Turn into a slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_slice(self) -> &'data [L::Sample]
    where
        L: SliceLayout,
    {
        let (visible, layout) = self.inner.into_parts();
        layout.sample().cast_mut_buf(visible)
    }

    /// Turn into a mutable slice of the individual samples in the layout.
    ///
    /// This preserves the lifetime with which the layout is borrowed from the underlying image,
    /// and the `ImageMut` need not stay alive.
    pub fn into_mut_slice(self) -> &'data mut [L::Sample]
    where
        L: SliceLayout,
    {
        let (visible, layout) = self.inner.into_parts();
        layout.sample().cast_mut_buf(visible)
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

    /// Split off unused bytes at the tail of the layout.
    pub fn split_layout(&mut self) -> ImageMut<'data, Bytes>
    where
        L: Layout,
    {
        // Need to roundup to correct alignment.
        let size = self.inner.layout().byte_len();
        let round_up = (size.wrapping_neg() & !(MAX_ALIGN - 1)).wrapping_neg();
        let buffer = self.inner.get_mut();

        let empty = buf::new_mut(&mut []);
        if round_up > buffer.len() {
            return RawImage::from_buffer(Bytes(0), empty).into();
        }

        // replace is needed for the type system as we operate on a mutable reference and must not
        // shorten its lifetime in any way by re-borrowing.
        let (initial, next) = core::mem::replace(buffer, empty).split_at_mut(round_up);
        *buffer = initial;

        RawImage::from_buffer(Bytes(next.len()), next).into()
    }

    /// Split this mutable reference into independent planes.
    ///
    /// This method ignores any of the requested planes which overlap, including if the same plane
    /// is requested multiple times. Only the first requested plane is returned. Planes that would
    /// not be aligned to the required alignment are also ignored.
    pub fn into_planes<const N: usize, D>(
        self,
        descriptors: [D; N],
    ) -> [Option<ImageMut<'data, D::Plane>>; N]
    where
        D: PlaneOf<L>,
        D::Plane: Relocate,
    {
        let layout = self.layout();
        let mut planes = descriptors.map(|d| {
            let plane = <D as PlaneOf<L>>::get_plane(d, layout);
            let empty_buf = buf::new_mut(&mut []);
            (plane, empty_buf)
        });

        // Now re-adjust the planes in order. For this, first collect their associated order.
        let mut remap = planes.each_mut().map(|plane| {
            // Maps all undefined planes to the zero-offset, so that they get skipped.
            let offset = plane.0.as_ref().map_or(0, |p| p.byte_offset());
            (offset, plane)
        });

        // Stable sort, we want to keep the first of each plane that overlaps.
        remap.sort_by_key(|&(offset, _)| offset);

        let mut consumed = 0;
        let (mut buffer, _) = self.inner.into_parts();

        for (offset, plane) in remap {
            let Some(layout) = &mut plane.0 else {
                continue;
            };

            let Some(skip_by) = offset.checked_sub(consumed) else {
                plane.0 = None;
                continue;
            };

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

            buffer = buf::take_at_mut(&mut buffer, skip_by);
            consumed += skip_by;

            let tail = buf::take_at_mut(&mut buffer, len);
            consumed += len;

            plane.1 = buffer;
            buffer = tail;
        }

        planes.map(|(layout, buffer)| -> Option<_> {
            Some(RawImage::from_buffer(layout?, buffer).into())
        })
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
        L: ops::Deref,
        L::Target: Layout,
    {
        self.inner.mogrify_layout(|l| &**l).into()
    }
}

impl<'data, 'l, L: Layout> ImageMut<'data, &'l mut L> {
    pub(crate) fn as_deref_mut(self) -> ImageMut<'data, &'l mut L::Target>
    where
        L: ops::DerefMut,
        L::Target: Layout,
    {
        self.inner.mogrify_layout(|l| &mut **l).into()
    }
}

impl<'lt, L: Layout + Clone> From<Image<&'lt L>> for Image<L> {
    fn from(image: Image<&'lt L>) -> Self {
        let (buffer, layout) = image.inner.into_parts();
        let layout: L = layout.clone();
        RawImage::from_buffer(layout, buffer).into()
    }
}

impl<'lt, L: Layout + Clone> From<Image<&'lt mut L>> for Image<L> {
    fn from(image: Image<&'lt mut L>) -> Self {
        let (buffer, layout) = image.inner.into_parts();
        let layout: L = layout.clone();
        RawImage::from_buffer(layout, buffer).into()
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

/* FIXME: decide if this should be an explicit method. */
impl<'lt, L: Layout + Clone> From<ImageRef<'lt, &'_ L>> for ImageRef<'lt, L> {
    fn from(image: ImageRef<'lt, &'_ L>) -> Self {
        let (buffer, layout) = image.inner.into_parts();
        let layout: L = layout.clone();
        RawImage::from_buffer(layout, buffer).into()
    }
}

impl<'lt, L: Layout + Clone> From<ImageRef<'lt, &'_ mut L>> for ImageRef<'lt, L> {
    fn from(image: ImageRef<'lt, &'_ mut L>) -> Self {
        let (buffer, layout) = image.inner.into_parts();
        let layout: L = layout.clone();
        RawImage::from_buffer(layout, buffer).into()
    }
}

impl<'lt, L: Layout + Clone> From<ImageMut<'lt, &'_ L>> for ImageMut<'lt, L> {
    fn from(image: ImageMut<'lt, &'_ L>) -> Self {
        let (buffer, layout) = image.inner.into_parts();
        let layout: L = layout.clone();
        RawImage::from_buffer(layout, buffer).into()
    }
}

impl<'lt, L: Layout + Clone> From<ImageMut<'lt, &'_ mut L>> for ImageMut<'lt, L> {
    fn from(image: ImageMut<'lt, &'_ mut L>) -> Self {
        let (buffer, layout) = image.inner.into_parts();
        let layout: L = layout.clone();
        RawImage::from_buffer(layout, buffer).into()
    }
}
/* FIXME: until here */

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

impl<L: Layout + Default> Default for Image<L> {
    fn default() -> Self {
        Image {
            inner: RawImage::from_buffer(L::default(), Buffer::default()),
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
            .field("layout", self.inner.layout())
            .field("content", &self.inner.as_slice())
            .finish()
    }
}
