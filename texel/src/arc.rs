

#[derive(Clone, Default)]
pub(crate) struct ArcBuffer {
    inner: Arc<[AlignedAtomic]>,
}
