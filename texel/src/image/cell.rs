//! Defines the containers operating on `!Sync` shared bytes.
//!
//! Re-exported at its super `image` module.
use crate::buf::CellBuffer;
use crate::image::raw::RawImage;
use crate::layout::Bytes;

/// A container of allocated bytes, parameterized over the layout.
///
/// ## Differences to owned [`Image`]
///
/// The implementations for [`PartialEq`] and [`Eq`] are not provided. In many containers and
/// contexts these two traits are required to rule out absence of interior mutability.
#[derive(Clone)]
pub struct CellImage<Layout = Bytes> {
    inner: RawImage<CellBuffer, Layout>,
}
