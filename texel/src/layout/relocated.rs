use crate::layout::{AlignedOffset, Decay, Layout, PlaneOf, Relocate, SliceLayout};
use crate::texels::TexelRange;

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

    /// Get an index addressing all samples covered by the range of this relocated layout.
    pub fn texel_range(&self) -> TexelRange<T::Sample>
    where
        T: SliceLayout,
    {
        TexelRange::from_byte_range(self.inner.sample(), self.offset.get()..self.byte_len())
            .unwrap()
    }
}

impl<T: Layout> Layout for Relocated<T> {
    fn byte_len(&self) -> usize {
        self.inner.byte_len() + self.offset.0
    }
}

impl<T: Layout> Relocate for Relocated<T> {
    fn byte_offset(&self) -> usize {
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

impl<Idx, L> PlaneOf<Relocated<L>> for Idx
where
    Idx::Plane: Relocate,
    Idx: PlaneOf<L>,
{
    type Plane = Idx::Plane;

    fn get_plane(self, layout: &Relocated<L>) -> Option<Self::Plane> {
        let mut inner = self.get_plane(&layout.inner)?;
        let mut inner_offset = inner.byte_offset();
        // This addition preserves the alignment up to MAX_ALIGN.
        inner_offset += layout.offset.get();
        // As an approximation this should succeed based on alignment requirements. Otherwise this
        // is a best attempt.
        if inner.relocate_to_byte(inner_offset) {
            Some(inner)
        } else {
            None
        }
    }
}
