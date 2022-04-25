//! A module for different pixel layouts.
//!
//! The `*Layout` traits define generic standard layouts with a normal form. Other traits provide
//! operations to convert between layouts, operations on the underlying image bytes, etc.
use crate::texel::MaxAligned;
use crate::{AsTexel, Texel};
use ::alloc::boxed::Box;
use core::{alloc, cmp};

mod matrix;

use crate::canvas::{CanvasMut, CanvasRef, Coord};
pub use crate::stride::{BadStrideError, StrideSpec, StridedBytes, StridedLayout, StridedTexels};

/// A byte layout that only describes the user bytes.
///
/// This is a minimal implementation of the basic `Layout` trait. It does not provide any
/// additional semantics for the buffer bytes described by it. All other layouts may be converted
/// into this layout.
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

/// Convert one layout to a less strict one.
///
/// In contrast to `From`/`Into` which is mostly assumed to model a lossless conversion the
/// conversion here may generalize but need not be lossless. For example, the `Bytes` layout is the
/// least descriptive layout that exists and any layout can decay into it. However, it should be
/// clear that this conversion is never lossless.
///
/// In general, a layout `L` should implement `Decay<T>` if any image with layouts of type `T` is
/// also valid for some layout of type `L`. A common example would be if a crate strictly adds more
/// information to a predefined layout, then it should also decay to that layout.
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
/// TODO Terminology: https://rust-lang.github.io/rfcs/2451-re-rebalancing-coherence.html
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
}

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
    fn get(from: CanvasRef<&Self>, at: Coord) -> Option<Pixel>;
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
    fn put(into: CanvasMut<&mut Self>, at: Coord, val: Pixel);

    /// Evaluate a function on each texel of the raster image.
    fn shade(mut canvas: CanvasMut<&mut Self>, mut f: impl FnMut(u32, u32, &mut Pixel)) {
        let Coord(bx, by) = canvas.layout().dimensions();
        for y in 0..by {
            for x in 0..bx {
                let mut pixel = Self::get(canvas.as_ref().as_deref(), Coord(x, y)).unwrap();
                f(x, y, &mut pixel);
                Self::put(canvas.as_mut().as_deref_mut(), Coord(x, y), pixel);
            }
        }
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

/// A matrix of packed texels (channel groups).
///
/// This is a simple layout of exactly width·height homogeneous pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatrixBytes {
    pub(crate) element: TexelLayout,
    pub(crate) first_dim: usize,
    pub(crate) second_dim: usize,
}

/// A matrix of packed texels (channel groups).
///
/// The underlying buffer may have more data allocated than this region and cause the overhead to
/// be reused when resizing the image. All ways to construct this already check that all pixels
/// within the resulting image can be addressed via an index.
pub struct Matrix<P> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) pixel: Texel<P>,
}

/// A layout that's a matrix of elements.
pub trait MatrixLayout: Layout {
    /// The valid matrix specification of this layout.
    ///
    /// This call should not fail, or panic. Otherwise, prefer an optional getter for the
    /// `StridedBytes` and have the caller decay their own buffer.
    fn matrix(&self) -> MatrixBytes;
}

/// Planar chroma 2×2 block-wise sub-sampled image.
///
/// FIXME: figure out if this is 'right' to expose in this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Yuv420p {
    channel: TexelLayout,
    width: u32,
    height: u32,
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
}

impl DynLayout {
    pub fn byte_len(&self) -> usize {
        match self.repr {
            LayoutRepr::Matrix(matrix) => matrix.byte_len(),
            LayoutRepr::Yuv420p(matrix) => matrix.byte_len(),
        }
    }
}

impl MatrixBytes {
    pub fn empty(element: TexelLayout) -> Self {
        MatrixBytes {
            element,
            first_dim: 0,
            second_dim: 0,
        }
    }

    pub fn from_width_height(
        element: TexelLayout,
        first_dim: usize,
        second_dim: usize,
    ) -> Option<Self> {
        let max_index = first_dim.checked_mul(second_dim)?;
        let _ = max_index.checked_mul(element.size)?;

        Some(MatrixBytes {
            element,
            first_dim,
            second_dim,
        })
    }

    /// Get the element type of this matrix.
    pub const fn element(&self) -> TexelLayout {
        self.element
    }

    /// Get the width of this matrix.
    pub const fn width(&self) -> usize {
        self.first_dim
    }

    /// Get the height of this matrix.
    pub const fn height(&self) -> usize {
        self.second_dim
    }

    /// Get the required bytes for this layout.
    pub const fn byte_len(self) -> usize {
        // Exactly this does not overflow due to construction.
        self.element.size * self.len()
    }

    /// The number of pixels in this layout
    pub const fn len(self) -> usize {
        self.first_dim * self.second_dim
    }
}

impl Yuv420p {
    pub fn from_width_height(channel: TexelLayout, width: u32, height: u32) -> Option<Self> {
        use core::convert::TryFrom;
        if width % 2 != 0 || height % 2 != 0 {
            return None;
        }

        let mwidth = usize::try_from(width).ok()?;
        let mheight = usize::try_from(height).ok()?;

        let y_count = mwidth.checked_mul(mheight)?;
        let uv_count = y_count / 2;

        let count = y_count.checked_add(uv_count)?;
        let _ = count.checked_mul(channel.size)?;

        Some(Yuv420p {
            channel,
            width,
            height,
        })
    }

    pub const fn byte_len(self) -> usize {
        let ylen = (self.width as usize) * (self.height as usize) * self.channel.size;
        ylen + ylen / 2
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

impl Layout for MatrixBytes {
    fn byte_len(&self) -> usize {
        MatrixBytes::byte_len(*self)
    }
}

impl Take for MatrixBytes {
    fn take(&mut self) -> Self {
        core::mem::replace(self, MatrixBytes::empty(self.element))
    }
}

impl<P> MatrixLayout for Matrix<P> {
    fn matrix(&self) -> MatrixBytes {
        self.into_matrix_bytes()
    }
}

/// Remove the strong typing for dynamic channel type information.
impl<L: MatrixLayout> Decay<L> for MatrixBytes {
    fn decay(from: L) -> MatrixBytes {
        from.matrix()
    }
}

/// Try to use the matrix with a specific pixel type.
impl<P> TryMend<MatrixBytes> for Texel<P> {
    type Into = Matrix<P>;
    type Err = MismatchedPixelError;

    fn try_mend(self, matrix: &MatrixBytes) -> Result<Matrix<P>, Self::Err> {
        Matrix::with_matrix(self, *matrix).ok_or_else(MismatchedPixelError::default)
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

/// Convert a pixel to an element, discarding the exact type information.
impl<P> From<Texel<P>> for TexelLayout {
    fn from(pix: Texel<P>) -> Self {
        TexelLayout {
            size: pix.size(),
            align: pix.align(),
        }
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
/// # use canvas::texels::{U8, U16};
/// # use canvas::layout::TexelLayout;
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

impl<P> From<Matrix<P>> for MatrixBytes {
    fn from(mat: Matrix<P>) -> Self {
        MatrixBytes {
            element: mat.pixel().into(),
            first_dim: mat.width(),
            second_dim: mat.height(),
        }
    }
}

/// Note: on 64-bit targets only the first `u32::MAX` dimensions appear accessible.
impl<P> Raster<P> for Matrix<P> {
    fn dimensions(&self) -> Coord {
        use core::convert::TryFrom;
        let width = u32::try_from(self.width()).unwrap_or(u32::MAX);
        let height = u32::try_from(self.height()).unwrap_or(u32::MAX);
        Coord(width, height)
    }

    fn get(from: CanvasRef<&Self>, Coord(x, y): Coord) -> Option<P> {
        if from.layout().in_bounds(x as usize, y as usize) {
            let index = from.layout().index_of(x as usize, y as usize);
            let texel = from.layout().sample();
            from.as_slice().get(index).map(|v| texel.copy_val(v))
        } else {
            None
        }
    }
}

impl<P> RasterMut<P> for Matrix<P> {
    fn put(into: CanvasMut<&mut Self>, Coord(x, y): Coord, val: P) {
        if into.layout().in_bounds(x as usize, y as usize) {
            let index = into.layout().index_of(x as usize, y as usize);
            if let Some(dst) = into.into_mut_slice().get_mut(index) {
                *dst = val;
            }
        }
    }
}
