use core::marker::PhantomData;

use zerocopy::{AsBytes, FromBytes};

/// A simple 2D canvas for pixels.
///
/// It allows efficient conversion to all other specific canvas representations. This includes
/// conversion that change the memory layout and effective reinterpretation casts.
pub struct Canvas<P: AsBytes + FromBytes> {
    _type: PhantomData<P>,
}
