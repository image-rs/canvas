//! A module for different pixel layouts.
//!
//! The `*Layout` traits define generic standard layouts with a normal form. Other traits provide
//! operations to convert between layouts, operations on the underlying image bytes, etc.
use crate::texels::{MaxAligned, TexelRange};
use crate::{AsTexel, Texel};
use ::alloc::boxed::Box;
use core::{alloc, cmp};

mod matrix;
mod planar;
mod relocated;
mod upsampling;

use crate::image::{Coord, ImageMut, ImageRef};
pub use crate::stride::{BadStrideError, StrideSpec, StridedBytes, StridedLayout, Strides};
pub use matrix::{Matrix, MatrixBytes, MatrixLayout};
pub use planar::{PlaneBytes, PlaneMatrices, Planes};
pub use relocated::Relocated;
pub(crate) use upsampling::Yuv420p;

/// A byte layout that only describes the user bytes.
///
/// This is a minimal implementation of the basic `Layout` trait. It does not provide any
/// additional semantics for the buffer bytes described by it. All other layouts may be converted
/// into this layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bytes(pub usize);

/// Describes the byte layout of an texture element, untyped.
///
/// This is not so different from `Texel` and `Layout` but is a combination of both. It has the
/// same invariants on alignment as the former which being untyped like the latter. The alignment
/// of an element must be at most that of [`MaxAligned`] and the size must be a multiple of its
/// alignment.
///
/// This type is a lower semi lattice. That is, given two elements the type formed by taking the
/// minimum of size and alignment individually will always form another valid element. This
/// operation is implemented in the [`Self::infimum`] method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, Hash)]
pub struct TexelLayout {
    size: usize,
    align: usize,
}

/// A descriptor of the layout of image bytes.
///
/// There is no color space and no strict type interpretation here, just some mapping to required
/// bytes for such a fixed buffer and a width and height of the described image. This means that
/// the byte usage for a particular buffer needs to be independent of the content, in particular
/// can not be based on compressibility.
///
/// There is one more thing that differentiates an image from an encoded format. It is expected
/// that the image can be unfolded into some matrix of independent pixels (with potentially
/// multiple channels) without any arithmetic or conversion function. Independent here means that,
/// when supplied with the missing color space and type information, there should exist an
/// `Fn(U) -> T` that can map these pixels independently into some linear color space.
///
/// This property holds for any packed, strided or planar RGB/YCbCr/HSV format as well as chroma
/// subsampled YUV images and even raw Bayer filtered images.
pub trait Layout {
    fn byte_len(&self) -> usize;
}

impl dyn Layout + '_ {
    #[inline]
    pub fn fits_buf(&self, bytes: &crate::buf::buf) -> bool {
        self.byte_len() <= bytes.as_bytes().len()
    }

    #[inline]
    pub fn fits_atomic_buf(&self, bytes: &crate::buf::atomic_buf) -> bool {
        self.byte_len() <= bytes.len()
    }

    #[inline]
    pub fn fits_cell_buf(&self, bytes: &crate::buf::cell_buf) -> bool {
        self.byte_len() <= bytes.len()
    }

    #[inline]
    pub fn fits_data(&self, len: &impl ?Sized) -> bool {
        self.byte_len() <= core::mem::size_of_val(len)
    }
}

/// Convert one layout to a less strict one.
///
/// In contrast to `From`/`Into` which is mostly assumed to model a lossless conversion the
/// conversion here may generalize but need not be lossless. For example, the `Bytes` layout is the
/// least descriptive layout that exists and any layout can decay into it. However, it should be
/// clear that this conversion is never lossless.
///
/// In general, a layout `L` should implement `Decay<T>` if any image with layouts of type `T` is
/// also valid for some layout of type `L`. A common example would be if a crate strictly adds more
/// information to a predefined layout, then it should also decay to that layout. It is invalid to
/// decay to a layout that somehow expands outside the initial layout, you must weaken the buffer
/// required.
///
/// Also note that this trait is not reflexive, in contrast to `From` and `Into` which are. This
/// avoids collisions in impls. In particular, it allows adding blanket impls of the form
///
/// ```ignore
/// struct Local;
///
/// impl Trait for Local { /* … */ }
///
/// impl<T: Trait> Decay<T> for Local { /* … */ }
/// ```
///
/// Otherwise, the instantiation `T = U` would collide with the reflexive impl.
///
/// ## Design
///
/// We consider re-rebalanced coherence rules ([RFC2451]) in this design especially to define the
/// receiver type and the type parameter. Note that adding a blanket impl is a breaking change
/// except under a special rule allowed in that RFC. To quote it here:
///
/// > RFC #1023 is amended to state that adding a new impl to an existing trait is considered a
/// > breaking change unless, given impl<P1..=Pn> Trait<T1..=Tn> for T0:
/// > * At least one of the types T0..=Tn must be a local type, added in this revision. Let Ti be
/// >   the first such type.
/// > * No uncovered type parameters P1..=Pn appear in T0..Ti (excluding Ti)
/// >
/// > [...]
/// >
/// > However, the following impls would not be considered a breaking change: [...]
/// > * `impl<T> OldTrait<T> for NewType`
///
/// Let's say we want to introduce a new desciptor trait for matrix-like layouts. Then we can ship
/// a new type representing the canonical form of this matrix trait and in the same revision define
/// a blanket impl that allows other layouts to decay to it. This wouldn't be possible if the
/// parameters were swapped. We can then let this specific type (it may contain covered type
/// parameters) decay to any other previously defined layout to provide interoperability with older
/// code.
///
/// [RFC2451]: https://rust-lang.github.io/rfcs/2451-re-rebalancing-coherence.html
///
pub trait Decay<T>: Layout {
    fn decay(from: T) -> Self;
}

impl<T: Layout> Decay<T> for Bytes {
    fn decay(from: T) -> Bytes {
        Bytes(from.byte_len())
    }
}

/// Convert a layout to a stricter one.
///
/// ## Design
///
/// A comment on the design space available for this trait.
///
/// (TODO: wrong) We require that the trait is
/// implemented for the type that is _returned_. If we required that the trait be implemented for
/// the receiver then this would restrict third-parties from using it to its full potential. In
/// particular, since `Mend` is a foreign trait the coherence rules make it impossible to specify:
///
/// TODO Terminology: <https://rust-lang.github.io/rfcs/2451-re-rebalancing-coherence.html>
///
/// ```ignore
/// impl<T> Mend<LocalType> for T {}
/// ```
///
/// TODO: rewrite this...
///
/// ```ignore
/// impl<T> Mend<T> for LocalType {}
/// ```
///
/// The forms of evolution that we want to keep open:
/// * Introduce a new form of mending between existing layouts. For example, a new color space
///   transformation should be able to translate between existing types. Note that we will assume
///   that in such a case the type parameters do not appear uncovered in the target or the source
///   so that having either as the trait receiver (T0) allows this.
/// * An *upgrader* type should be able to mend a <T: LocalOrForeignTrait> into a chosen layout.
/// * TODO: When add a new layout type which mender types and targets do we want?
///
/// The exact form thus simply depends on expected use and the allow evolution for this crate.
/// Consider in particular this coherence/SemVer rule:
///
/// > Adding any impl with an uncovered type parameter is considered a major breaking change.
///
/// TODO
///
/// TODO: comment and consider `&self`.
///
pub trait Mend<From> {
    type Into: Layout;
    fn mend(self, from: &From) -> Self::Into;
}

/// Try to convert a layout to a stricter one.
pub trait TryMend<From> {
    type Into: Layout;
    type Err;
    fn try_mend(self, from: &From) -> Result<Self::Into, Self::Err>;
}

/// A layout that can be emptied.
///
/// This trait contains all layout types from which we can steal their memory buffer. This is
/// incredibly useful for fallible operations that change the _type_ of a buffers layout. Instead
/// of being required to take the buffer by value and return the original in case of an error they
/// can use the much natural signature:
///
/// * `fn mutate(&mut self) -> Result<Converted, Err>`
///
/// where semantics are that the buffer is unchanged in case of error but has been moved to the
/// type `Converted` in case of success. This is very similar to the method `Vec::take` and others.
///
/// It is expected that the `byte_len` is `0` after the operation.
///
/// This trait is _not_ simply a clone of `Default`. While we expect that the described image
/// contains no bytes after the operation other data such as channel count, color space
/// information, image plane order, alpha interpretation should be retained.
pub trait Take: Layout {
    fn take(&mut self) -> Self;
}

/// A layout that is a slice of samples.
///
/// These layouts are represented with a slice of a _single_ type of samples. In particular these
/// can be addressed and mutated independently.
pub trait SliceLayout: Layout {
    /// The sample type itself.
    type Sample;

    /// Get the sample description.
    fn sample(&self) -> Texel<Self::Sample>;

    /// The number of samples.
    ///
    /// A slice with the returned length should have the byte length returned in `byte_len`.
    fn len(&self) -> usize {
        self.byte_len() / self.sample().size()
    }

    fn as_index(&self) -> TexelRange<Self::Sample> {
        self.sample()
            .to_range(0..self.len())
            .expect("A layout should fit into memory")
    }
}

// Just assert that `dyn SliceLayout<Sample = T>` is a valid type.
impl<T> dyn SliceLayout<Sample = T> {}

/// A layout of individually addressable raster elements.
///
/// Often referred to as 'pixels', this is a special form of texels that represent a single group
/// of color channels that form one color impression.
///
/// Note that it does not prescribe any particular order of arrangement of these channels. Indeed,
/// they could be in column major format, in row major format, ordered according to some space
/// filling curve, etc. Also, multiple pixels may form one group of subsampled channels.
pub trait Raster<Pixel>: Layout + Sized {
    fn dimensions(&self) -> Coord;
    fn get(from: ImageRef<&Self>, at: Coord) -> Option<Pixel>;
}

/// A raster layout where one can change pixel values independently.
///
/// In other words, requires that texels are actually one-by-one blocks of pixels.
///
/// Note that it does not prescribe any particular order of arrangement of these texels. Indeed,
/// they could be in column major format, in row major format, ordered according to some space
/// filling curve, etc. but subsampled images are not easily possible as pixels can not be written
/// to independently.
pub trait RasterMut<Pixel>: Raster<Pixel> {
    fn put(into: ImageMut<&mut Self>, at: Coord, val: Pixel);

    /// Evaluate a function on each texel of the raster image.
    fn shade(mut image: ImageMut<&mut Self>, mut f: impl FnMut(u32, u32, &mut Pixel)) {
        let Coord(bx, by) = image.layout().dimensions();
        for y in 0..by {
            for x in 0..bx {
                let mut pixel = Self::get(image.as_ref().as_deref(), Coord(x, y)).unwrap();
                f(x, y, &mut pixel);
                Self::put(image.as_mut().as_deref_mut(), Coord(x, y), pixel);
            }
        }
    }
}

/// An index type that identifies a 'plane' of an image layout.
///
/// The byte offset of all planes must adhere to alignment requirements as set by [`MaxAligned`],
/// each individually. This ensures that each plane can be addressed equally.
///
/// Planes may be any part of the image that can be addressed independently. This could be that
/// the component matrices of different channels are stored after each other, or interleaved ways
/// of storing different quantization levels, optimizations for cache oblivious layouts etc. While
/// planes usually constructed to be non-overlapping this requirement is not inherent in the trait.
/// However, consider that mutable access to overlapping planes is not possible.
///
/// There may be multiple different ways of indexing into the same layout. Similar to the standard
/// libraries [Index](`core::ops::Index`) trait, this trait can be implemented to provide an index
/// into a layout defined in a different crate.
///
/// # Examples
///
/// The `PlaneMatrices` wrapper stores an array of matrix layouts. This trait allows accessing them
/// by a `usize` index, conveniently interacting with the `ImageRef` and `ImageMut` types.
///
/// ```
/// use image_texel::image::Image;
/// use image_texel::layout::{PlaneMatrices, Matrix};
/// use image_texel::texels::U8;
///
/// // Imagine a JPEG with progressive DCT coefficient planes.
/// let rough = Matrix::from_width_height(U8, 8, 8).unwrap();
/// let dense = Matrix::from_width_height(U8, 64, 64).unwrap();
///
/// // The assembled layout can be used to access the disjoint planes.
/// let matrices = PlaneMatrices::new(U8, [rough, dense]);
///
/// let buffer = Image::new(&matrices);
/// let [p0, p1] = buffer.as_ref().into_planes([0, 1]).unwrap();
///
/// let rough_coeffs = p0.into_bytes();
/// let dense_coeffs = p1.into_bytes();
///
/// assert_eq!(rough_coeffs.len(), 8 * 8);
/// assert_eq!(dense_coeffs.len(), 64 * 64);
/// ```
pub trait PlaneOf<L: ?Sized> {
    type Plane: Layout;

    /// Get the layout describing the plane.
    fn get_plane(self, layout: &L) -> Option<Self::Plane>;
}

/// A layout that supports being moved in memory.
///
/// Such a layout only occupies a smaller but contiguous range of its full buffer length. One can
/// query that offset and modify it. Note that a relocatable layout should still report its
/// [`Layout::byte_len`] as the past-the-end of its last relevant byte.
pub trait Relocate: Layout {
    /// The offset of the first relevant byte of this layout.
    ///
    /// This should be smaller or equal to the length.
    fn byte_offset(&self) -> usize;

    /// Move the layout to another aligned offset.
    ///
    /// The length of the layout should implicitly be modified by this operation, that is the range
    /// between the start offset and its apparent length should remain the same.
    ///
    /// Moving to an aligned offset must work.
    ///
    /// # Panics
    ///
    /// Implementations are encouraged to panic if the newly chosen offset would make the total
    /// length overflow `isize::MAX`, i.e. possible allocation size.
    fn relocate(&mut self, offset: AlignedOffset);

    /// Attempt to relocate the offset to another start offset, in bytes.
    ///
    /// This method should return `false` if the new offset is not suitable for the layout. The
    /// default implementation requires an aligned offset. Implementations may work for additional
    /// offsets.
    fn relocate_to_byte(&mut self, offset: usize) -> bool {
        if let Some(aligned_offset) = AlignedOffset::new(offset) {
            self.relocate(aligned_offset);
            true
        } else {
            false
        }
    }

    /// Retrieve the contiguous byte range occupied by this layout.
    fn byte_range(&self) -> core::ops::Range<usize> {
        let start = self.byte_offset();
        let end = self.byte_len();

        debug_assert!(
            start <= end,
            "The length calculation of this layout is buggy"
        );

        start..end.min(start)
    }

    /// Get an index addressing all samples covered by the range of this relocated layout.
    fn texel_range(&self) -> TexelRange<Self::Sample>
    where
        Self: SliceLayout + Sized,
    {
        TexelRange::from_byte_range(self.sample(), self.byte_offset()..self.byte_len())
            .expect("A layout should fit into memory")
    }
}

impl dyn Relocate {}

/// An unsigned offset that is maximally aligned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct AlignedOffset(usize);

impl AlignedOffset {
    const ALIGN: usize = core::mem::align_of::<MaxAligned>();

    /// Try to construct an aligned offset.
    pub fn new(offset: usize) -> Option<Self> {
        if offset % Self::ALIGN == 0 && offset <= isize::MAX as usize {
            Some(AlignedOffset(offset))
        } else {
            None
        }
    }

    /// Get the wrapped offset as a `usize`.
    pub fn get(self) -> usize {
        self.0
    }

    /// Get the next valid offset after adding `bytes` to this.
    ///
    /// ```
    /// use image_texel::layout::AlignedOffset;
    ///
    /// let zero = AlignedOffset::default();
    ///
    /// assert_eq!(zero.next_up(0), Some(zero));
    /// assert_eq!(zero.next_up(1), zero.next_up(2));
    ///
    /// assert!(zero.next_up(usize::MAX).is_none());
    /// assert!(zero.next_up(isize::MAX as usize).is_none());
    /// ```
    #[must_use]
    pub fn next_up(self, bytes: usize) -> Option<Self> {
        let range_start = self.0.checked_add(bytes)?;
        let new_offset = range_start.checked_next_multiple_of(Self::ALIGN)?;

        if new_offset > isize::MAX as usize {
            return None;
        }

        AlignedOffset::new(new_offset)
    }
}

/// A dynamic descriptor of an image's layout.
///
/// FIXME: figure out if this is 'right' to expose in this crate.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DynLayout {
    pub(crate) repr: LayoutRepr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LayoutRepr {
    Matrix(MatrixBytes),
    Yuv420p(Yuv420p),
}

/// An error indicating that mending failed due to mismatching pixel attributes.
///
/// This struct is used when a layout with dynamic pixel information should be mended into another
/// layout with static information or a more restrictive combination of layouts. One example is the
/// conversion of a dynamic matrix into a statically typed layout.
#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct MismatchedPixelError {
    _private: (),
}

impl Bytes {
    /// Forget all layout semantics except the number of bytes used.
    pub fn from_layout(layout: impl Layout) -> Self {
        Bytes(layout.byte_len())
    }
}

impl TexelLayout {
    /// Construct an element from a self-evident pixel.
    pub fn from_pixel<P: AsTexel>() -> Self {
        let pix = P::texel();
        TexelLayout {
            size: pix.size(),
            align: pix.align(),
        }
    }

    /// An element with maximum size and no alignment requirements.
    ///
    /// This constructor is mainly useful for the purpose of using it as a modifier. When used with
    /// [`Self::infimum`] it will only shrink the alignment and keep the size unchanged.
    pub const MAX_SIZE: Self = {
        TexelLayout {
            size: isize::MAX as usize,
            align: 1,
        }
    };

    /// Create an element for a fictional type with specific layout.
    ///
    /// It's up to the caller to define or use an actual type with that same layout later. This
    /// skips the check that such a type must not contain any padding and only performs the layout
    /// related checks.
    pub fn with_layout(layout: alloc::Layout) -> Option<Self> {
        if layout.align() > MaxAligned::texel().align() {
            return None;
        }

        if layout.size() % layout.align() != 0 {
            return None;
        }

        Some(TexelLayout {
            size: layout.size(),
            align: layout.align(),
        })
    }

    /// Convert this into a type layout.
    ///
    /// This can never fail as `TexelLayout` refines the standard library layout type.
    pub fn layout(self) -> alloc::Layout {
        alloc::Layout::from_size_align(self.size, self.align).expect("Valid layout")
    }

    /// Reduce the alignment of the element.
    ///
    /// This will perform the same modification as `repr(packed)` on the element's type.
    ///
    /// # Panics
    ///
    /// This method panics if `align` is not a valid alignment.
    #[must_use = "This does not modify `self`."]
    pub fn packed(self, align: usize) -> TexelLayout {
        assert!(align.is_power_of_two());
        let align = self.align.min(align);
        TexelLayout { align, ..self }
    }

    /// Create an element having the smaller of both sizes and alignments.
    #[must_use = "This does not modify `self`."]
    pub fn infimum(self, other: Self) -> TexelLayout {
        // We still have size divisible by align. Whatever the smaller of both, it's divisible by
        // its align and thus also by the min of both alignments.
        TexelLayout {
            size: self.size.min(other.size),
            align: self.align.min(other.align),
        }
    }

    /// Get the size of the element.
    pub const fn size(self) -> usize {
        self.size
    }

    /// Get the minimum required alignment of the element.
    pub const fn align(self) -> usize {
        self.size
    }

    pub const fn superset_of(&self, other: TexelLayout) -> bool {
        self.size >= other.size && self.align >= other.align
    }
}

/// Convert a pixel to an element, discarding the exact type information.
impl<T> From<Texel<T>> for TexelLayout {
    fn from(texel: Texel<T>) -> Self {
        TexelLayout {
            size: texel.size(),
            align: texel.align(),
        }
    }
}

impl DynLayout {
    pub fn byte_len(&self) -> usize {
        match self.repr {
            LayoutRepr::Matrix(matrix) => matrix.byte_len(),
            LayoutRepr::Yuv420p(matrix) => matrix.byte_len(),
        }
    }
}

impl Layout for Bytes {
    fn byte_len(&self) -> usize {
        self.0
    }
}

impl<'lt, T: Layout + ?Sized> Layout for &'lt T {
    fn byte_len(&self) -> usize {
        (**self).byte_len()
    }
}

impl<'lt, T: Layout + ?Sized> Layout for &'lt mut T {
    fn byte_len(&self) -> usize {
        (**self).byte_len()
    }
}

impl Take for Bytes {
    fn take(&mut self) -> Self {
        Bytes(core::mem::take(&mut self.0))
    }
}

impl Layout for DynLayout {
    fn byte_len(&self) -> usize {
        DynLayout::byte_len(self)
    }
}

impl<T> SliceLayout for &'_ T
where
    T: SliceLayout,
{
    type Sample = T::Sample;

    fn sample(&self) -> Texel<Self::Sample> {
        (**self).sample()
    }
}

impl<T> SliceLayout for &'_ mut T
where
    T: SliceLayout,
{
    type Sample = T::Sample;

    fn sample(&self) -> Texel<Self::Sample> {
        (**self).sample()
    }
}

impl<L: Layout + ?Sized> Layout for Box<L> {
    fn byte_len(&self) -> usize {
        (**self).byte_len()
    }
}

impl<L: Layout> Decay<L> for Box<L> {
    fn decay(from: L) -> Box<L> {
        Box::new(from)
    }
}

/// The partial order of elements is defined by comparing size and alignment.
///
/// This turns it into a semi-lattice structure, with infimum implementing the meet operation. For
/// example, the following comparison all hold:
///
/// ```
/// # use image_texel::texels::{U8, U16};
/// # use image_texel::layout::TexelLayout;
/// let u8 = TexelLayout::from(U8);
/// let u8x2 = TexelLayout::from(U8.array::<2>());
/// let u8x3 = TexelLayout::from(U8.array::<3>());
/// let u16 = TexelLayout::from(U16);
///
/// assert!(u8 < u16, "due to size and alignment");
/// assert!(u8x2 < u16, "due to its alignment");
/// assert!(!(u8x3 < u16) && !(u16 < u8x3), "not comparable");
///
/// let meet = u8x3.infimum(u16);
/// assert!(meet <= u8x3);
/// assert!(meet <= u16);
/// assert!(meet == u16.packed(1), "We know it precisely here {:?}", meet);
/// ```
impl cmp::PartialOrd for TexelLayout {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        if self.size == other.size && self.align == other.align {
            Some(cmp::Ordering::Equal)
        } else if self.size <= other.size && self.align <= other.align {
            Some(cmp::Ordering::Less)
        } else if self.size >= other.size && self.align >= other.align {
            Some(cmp::Ordering::Greater)
        } else {
            None
        }
    }
}

macro_rules! bytes_from_layout {
    ($layout:path) => {
        impl From<$layout> for Bytes {
            fn from(layout: $layout) -> Self {
                Bytes::from_layout(layout)
            }
        }
    };
    (<$($bound:ident),*> $layout:ident) => {
        impl<$($bound),*> From<$layout <$($bound),*>> for Bytes {
            fn from(layout: $layout <$($bound),*>) -> Self {
                Bytes::from_layout(layout)
            }
        }
    };
}

bytes_from_layout!(DynLayout);
bytes_from_layout!(MatrixBytes);
bytes_from_layout!(<P> Matrix);

impl From<MatrixBytes> for DynLayout {
    fn from(matrix: MatrixBytes) -> Self {
        DynLayout {
            repr: LayoutRepr::Matrix(matrix),
        }
    }
}

impl From<Yuv420p> for DynLayout {
    fn from(matrix: Yuv420p) -> Self {
        DynLayout {
            repr: LayoutRepr::Yuv420p(matrix),
        }
    }
}

impl<P, L> PlaneOf<&'_ L> for P
where
    P: PlaneOf<L>,
{
    type Plane = <P as PlaneOf<L>>::Plane;

    fn get_plane(self, layout: &&L) -> Option<Self::Plane> {
        <P as PlaneOf<L>>::get_plane(self, *layout)
    }
}

impl<P, L> PlaneOf<&'_ mut L> for P
where
    P: PlaneOf<L>,
{
    type Plane = <P as PlaneOf<L>>::Plane;

    fn get_plane(self, layout: &&mut L) -> Option<Self::Plane> {
        <P as PlaneOf<L>>::get_plane(self, *layout)
    }
}
