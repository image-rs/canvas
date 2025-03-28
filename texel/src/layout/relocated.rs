use crate::layout::{AlignedOffset, Decay, Layout, Matrix, MatrixBytes, PlaneOf, Relocate};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Relocated<T> {
    pub offset: AlignedOffset,
    pub inner: T,
}

impl<T: Layout> Relocated<T> {
    pub fn new(inner: T) -> Self {
        Relocated {
            offset: AlignedOffset::default(),
            inner,
        }
    }

    /// Get the next aligned offset that comes after this relocated layout.
    pub fn next_aligned_offset(&self) -> Option<AlignedOffset> {
        self.offset.next_up(self.inner.byte_len())
    }
}

impl<T: Layout> Layout for Relocated<T> {
    fn byte_len(&self) -> usize {
        self.inner.byte_len() + self.offset.0
    }
}

impl<T: Layout> Relocate for Relocated<T> {
    fn offset(&self) -> usize {
        self.offset.0
    }

    fn relocate(&mut self, offset: AlignedOffset) {
        self.offset = offset;
    }
}

impl<T: Layout> Decay<T> for Relocated<T> {
    fn decay(inner: T) -> Relocated<T> {
        Relocated::new(inner)
    }
}
