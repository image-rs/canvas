use crate::{
    layout::{
        AlignedOffset, Decay, Layout, Matrix, MatrixBytes, MismatchedPixelError, PlaneOf, Relocate,
        SliceLayout, TexelLayout, TryMend,
    },
    texel::Texel,
};

use super::relocated::Relocated;

/// A collection of planes.
///
/// Note that the constructors and [`Layout`] implementations depend on the type parameter.
#[derive(Clone)]
pub struct Planes<Storage: ?Sized> {
    inner: Storage,
}

impl<Storage> Planes<Storage> {
    pub fn into_inner(self) -> Storage {
        self.inner
    }
}

impl<Pl, const N: usize> Planes<[Pl; N]>
where
    Pl: Layout,
{
    pub fn new(inner: [Pl; N]) -> Self {
        Self { inner }
    }

    pub fn as_ref(&self) -> Planes<[&'_ Pl; N]> {
        Planes {
            inner: self.inner.each_ref(),
        }
    }

    pub fn as_mut(&mut self) -> Planes<[&'_ mut Pl; N]> {
        Planes {
            inner: self.inner.each_mut(),
        }
    }
}

impl<Pl, const N: usize> Layout for Planes<[Pl; N]>
where
    Pl: Layout,
{
    fn byte_len(&self) -> usize {
        let lengths: [usize; N] = self.inner.each_ref().map(|p| p.byte_len());
        lengths.iter().copied().max().unwrap_or(0)
    }
}

impl<Pl> Layout for Planes<[Pl]>
where
    Pl: Layout,
{
    fn byte_len(&self) -> usize {
        let lengths = self.inner.iter().map(|p| p.byte_len());
        lengths.max().unwrap_or(0)
    }
}

impl<Pl, const N: usize> PlaneOf<Planes<[Pl; N]>> for usize
where
    Pl: Layout + Clone,
{
    type Plane = Pl;

    fn get_plane(self, layout: &Planes<[Pl; N]>) -> Option<Self::Plane> {
        layout.inner.get(self).cloned()
    }
}

impl<Pl> PlaneOf<Planes<[Pl]>> for usize
where
    Pl: Layout + Clone,
{
    type Plane = Pl;

    fn get_plane(self, layout: &Planes<[Pl]>) -> Option<Self::Plane> {
        layout.inner.get(self).cloned()
    }
}

/// An array of byte matrices.
///
/// This type is optimized for the concrete layout type.
///
/// ```
/// use image_texel::texels::{U8, F32};
/// use image_texel::layout::{Layout, PlaneBytes, MatrixBytes};
///
/// let m0 = MatrixBytes::from_width_height(U8.into(), 4, 4).unwrap();
/// let m1 = MatrixBytes::from_width_height(F32.into(), 16, 16).unwrap();
///
/// let planar = PlaneBytes::new([m0, m1]);
/// let ref_to_m0 = planar.plane_ref(0).unwrap();
/// let ref_to_m1 = planar.plane_ref(1).unwrap();
///
/// assert!(ref_to_m0.byte_len() <= ref_to_m1.offset.get());
/// assert!(ref_to_m1.byte_len() == planar.byte_len());
/// ```
#[derive(Clone)]
pub struct PlaneBytes<const N: usize> {
    planes: Planes<[Relocated<MatrixBytes>; N]>,
}

impl<const N: usize> PlaneBytes<N> {
    /// Construct from separate matrices.
    ///
    /// Relocates each consecutive matrix such that they do not overlap.
    ///
    /// # Panics
    ///
    /// This method panics if the overall layout length would exceed `isize::MAX`.
    pub fn new(inner: [MatrixBytes; N]) -> Self {
        let mut inner = inner.map(Relocated::new);
        let mut offset = AlignedOffset::default();

        for plane in inner.iter_mut() {
            plane.relocate(offset);
            offset = plane.next_aligned_offset().expect("layout too large");
        }

        PlaneBytes {
            planes: Planes::new(inner),
        }
    }

    /// Return a reference to one relocated matrix layout.
    pub fn plane_ref(&self, idx: usize) -> Option<&Relocated<MatrixBytes>> {
        self.planes.inner.get(idx)
    }

    /// Return a layout where only planes with a matching texel layout are preserved.
    ///
    /// All other planes are rewritten to be empty matrices of that layout. All offsets are
    /// preserved.
    ///
    /// ```
    ///
    /// use image_texel::texels::{U8, F32};
    /// use image_texel::layout::{Layout, PlaneBytes, MatrixBytes};
    ///
    /// let m0 = MatrixBytes::from_width_height(U8.into(), 4, 4).unwrap();
    /// let m1 = MatrixBytes::from_width_height(F32.into(), 16, 16).unwrap();
    ///
    /// let planar = PlaneBytes::new([m0, m1]);
    /// let only_u8 = planar.retain_coefficients_like(U8.into());
    ///
    /// assert_eq!(planar.plane_ref(0), only_u8.plane_ref(0));
    /// assert_ne!(planar.plane_ref(1), only_u8.plane_ref(1));
    ///
    /// use image_texel::layout::Relocate;
    /// // That second plane is still offset, but empty
    /// assert!(only_u8.plane_ref(1).unwrap().byte_len() > 0);
    /// assert_eq!(only_u8.plane_ref(1).unwrap().byte_range().len(), 0);
    /// ```
    #[must_use]
    pub fn retain_coefficients_like(&self, texel: TexelLayout) -> Self {
        let matrices = self.planes.inner.clone();
        let inner = matrices.map(|plane| {
            if plane.inner.element() == texel {
                plane
            } else {
                Relocated {
                    offset: plane.offset,
                    inner: MatrixBytes::empty(texel),
                }
            }
        });

        PlaneBytes {
            planes: Planes::new(inner),
        }
    }
}

impl<const N: usize> Layout for PlaneBytes<N> {
    fn byte_len(&self) -> usize {
        if N == 0 {
            0
        } else {
            // We made sure that planes are sorted!
            self.planes.inner[N - 1].byte_len()
        }
    }
}

impl<const N: usize> Relocate for PlaneBytes<N> {
    fn byte_offset(&self) -> usize {
        if N == 0 {
            0
        } else {
            self.planes.inner[0].byte_len()
        }
    }

    fn relocate(&mut self, mut offset: AlignedOffset) {
        for plane in self.planes.inner.iter_mut() {
            plane.relocate(offset);
            offset = plane.next_aligned_offset().expect("layout too large");
        }
    }
}

impl<const N: usize> PlaneOf<PlaneBytes<N>> for usize {
    type Plane = Relocated<MatrixBytes>;

    fn get_plane(self, layout: &PlaneBytes<N>) -> Option<Self::Plane> {
        layout.planes.inner.get(self).copied()
    }
}

/// Upgrade to a collection of planes of the same texel.
impl<T, const N: usize> TryMend<PlaneBytes<N>> for Texel<T> {
    type Into = PlaneMatrices<T, N>;

    type Err = MismatchedPixelError;

    fn try_mend(self, from: &PlaneBytes<N>) -> Result<Self::Into, Self::Err> {
        let planes = from.planes.inner.each_ref();

        // FIXME: use `try_map` once stable.
        let mut results: [Result<_, MismatchedPixelError>; N] = planes.map(|plane| {
            let matrix = self.try_mend(&plane.inner)?;
            Ok(Relocated {
                offset: plane.offset,
                inner: matrix,
            })
        });

        if let Some(err) = results.iter().position(|e| e.is_err()) {
            let mut replacement = Ok(Relocated::new(Matrix::empty(self)));
            core::mem::swap(&mut results[err], &mut replacement);

            return Err(match replacement {
                Err(err) => err,
                Ok(_) => unreachable!(),
            });
        }

        // FIXME: `try_map` until here.
        let inner = results.map(|res| res.unwrap());

        Ok(PlaneMatrices {
            planes: Planes::new(inner),
            texel: self,
        })
    }
}

/// An array of byte matrices.
///
/// This type is optimized for the concrete layout type of matrix planes.
///
/// # Examples
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
/// let rough = matrices.plane_ref(0).unwrap();
/// let dense = matrices.plane_ref(1).unwrap();
///
/// let buffer = Image::new(&matrices);
/// let rough_coeffs = &buffer.as_buf()[rough.texel_range()];
/// assert_eq!(rough_coeffs.len(), 8 * 8);
///
/// let dense_coeffs = &buffer.as_buf()[dense.texel_range()];
/// assert_eq!(dense_coeffs.len(), 64 * 64);
///
/// // The coefficient planes are disjoint and well-ordered.
/// assert!(rough_coeffs.as_ptr_range().end <= dense_coeffs.as_ptr());
/// ```
#[derive(Clone)]
pub struct PlaneMatrices<T, const N: usize> {
    planes: Planes<[Relocated<Matrix<T>>; N]>,
    texel: Texel<T>,
}

impl<T, const N: usize> PlaneMatrices<T, N> {
    /// Construct from separate matrices.
    ///
    /// Relocates each consecutive matrix such that they do not overlap.
    ///
    /// # Panics
    ///
    /// This method panics if the overall layout length would exceed `isize::MAX`.
    pub fn new(texel: Texel<T>, inner: [Matrix<T>; N]) -> Self {
        use crate::layout::{Decay, TryMend};
        let bytes = inner.each_ref().map(|m| MatrixBytes::decay(m));
        let planes = PlaneBytes::new(bytes);
        texel
            .try_mend(&planes)
            .expect("input matrices have this texel")
    }

    /// Return a reference to one relocated matrix layout.
    pub fn plane_ref(&self, idx: usize) -> Option<&Relocated<Matrix<T>>> {
        self.planes.inner.get(idx)
    }
}

impl<T, const N: usize> PlaneOf<PlaneMatrices<T, N>> for usize {
    type Plane = Relocated<Matrix<T>>;

    fn get_plane(self, layout: &PlaneMatrices<T, N>) -> Option<Self::Plane> {
        layout.planes.inner.get(self).copied()
    }
}

impl<T, const N: usize> Layout for PlaneMatrices<T, N> {
    fn byte_len(&self) -> usize {
        if N == 0 {
            0
        } else {
            // We made sure that planes are sorted!
            self.planes.inner[N - 1].byte_len()
        }
    }
}

impl<T, const N: usize> SliceLayout for PlaneMatrices<T, N> {
    type Sample = T;

    fn sample(&self) -> Texel<Self::Sample> {
        self.texel
    }
}

impl<T, const N: usize> Decay<PlaneMatrices<T, N>> for PlaneBytes<N> {
    fn decay(from: PlaneMatrices<T, N>) -> PlaneBytes<N> {
        let bytes = from.planes.inner.each_ref().map(|rel| Relocated {
            offset: rel.offset,
            inner: MatrixBytes::decay(&rel.inner),
        });

        PlaneBytes {
            planes: Planes::new(bytes),
        }
    }
}
