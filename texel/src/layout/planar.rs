use crate::{
    layout::{Decay, Layout, Matrix, MatrixBytes, PlaneOf},
    Texel,
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
#[derive(Clone)]
pub struct PlaneBytes<const N: usize> {
    planes: Planes<[Relocated<MatrixBytes>; N]>,
}

impl<const N: usize> PlaneBytes<N> {
    /// Construct from separate matrices.
    pub fn new(inner: [MatrixBytes; N]) -> Self {
        PlaneBytes {
            planes: Planes::new(inner.map(Relocated::new)),
        }
    }
}

impl<const N: usize> PlaneOf<PlaneBytes<N>> for usize {
    type Plane = Relocated<MatrixBytes>;

    fn get_plane(self, layout: &PlaneBytes<N>) -> Option<Self::Plane> {
        layout.planes.inner.get(self).copied()
    }
}

/// An array of byte matrices.
///
/// This type is optimized for the concrete layout type.
#[derive(Clone)]
pub struct PlaneMatrices<T, const N: usize> {
    planes: Planes<[Relocated<Matrix<T>>; N]>,
    texel: Texel<T>,
}

impl<T, const N: usize> PlaneOf<PlaneMatrices<T, N>> for usize {
    type Plane = Relocated<Matrix<T>>;

    fn get_plane(self, layout: &PlaneMatrices<T, N>) -> Option<Self::Plane> {
        layout.planes.inner.get(self).copied()
    }
}
