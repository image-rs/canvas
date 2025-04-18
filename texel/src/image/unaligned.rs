//! Buffers that can work with unaligned underlying data.
//!
//! These are mostly for IO purposes since most compute algorithms are not expected to support
//! interactions with these buffers. However, these buffers support more generalized layouts with
//! the goal of admitting a description of arbitrary external resources. In consequence, these
//! buffers interact by references only.
use core::cell::Cell;

use crate::buf::buf;
use crate::image::{
    AtomicImage, AtomicImageRef, CellImage, CellImageRef, Image, ImageMut, ImageRef,
};
use crate::layout::{Bytes, Layout};

pub struct DataRef<'lt, Layout = Bytes> {
    data: &'lt [u8],
    layout: Layout,
}

pub struct DataMut<'lt, Layout = Bytes> {
    data: &'lt mut [u8],
    layout: Layout,
}

pub struct DataCells<'lt, Layout = Bytes> {
    data: &'lt [Cell<u8>],
    layout: Layout,
}

macro_rules! transfer_methods {
    (
        derive {
            $(write_to_buf: $write_to_buf:path,)?
            $(read_from_buf: $read_from_buf:path,)?
        }
    ) => {
        $(
            /// Write to an image, changing the layout in the process.
            ///
            /// See [`Self::write_to_image`] but works on a borrowed buffer, only when it has the same
            /// layout type. Clones the layout to the image buffer.
            ///
            /// Reallocates the image buffer when necessary to ensure that the allocated buffer fits the
            /// new data's layout.
            ///
            /// Consider [`Self::write_to_buf`] with the [`Image::as_capacity_buf_mut`] when you want to
            /// ignore the layout value of the target buffer.
            pub fn write_to(&self, buffer: &mut Image<L>)
            where
                L: Clone + Layout,
            {
                buffer.layout_mut_unguarded().clone_from(&self.layout);
                buffer.ensure_layout();
                $write_to_buf(self, buffer.as_capacity_buf_mut());
            }

            /// Write to an image, changing the layout in the process.
            ///
            /// Reallocates the image buffer when necessary to ensure that the allocated buffer fits the
            /// new data's layout.
            pub fn write_to_image(&self, buffer: Image<impl Layout>) -> Image<L>
            where
                L: Clone + Layout,
            {
                let mut buffer = buffer.with_layout(self.layout.clone());
                $write_to_buf(self, buffer.as_capacity_buf_mut());
                buffer
            }

            /// Write to a mutable borrowed buffer with layout.
            ///
            /// First verifies that the data will fit into the target. Then returns `Some` with a new
            /// reference to the target buffer that is using the data's layout. Otherwise, returns `None`.
            pub fn write_to_mut<'data>(
                &self,
                buffer: ImageMut<'data, impl Layout>,
            ) -> Option<ImageMut<'data, L>>
            where
                L: Clone + Layout,
            {
                let mut buffer = buffer.with_layout(self.layout.clone())?;
                $write_to_buf(self, buffer.as_mut_buf());
                Some(buffer)
            }
        )* /*write_to_buf */
    }
}

impl<'lt, L> DataRef<'lt, L> {
    /// Verifies the data against the layout before construction.
    ///
    /// Note that the type has no hard invariants.
    pub fn checked_new(data: &'lt [u8], layout: L) -> Option<Self>
    where
        L: Layout,
    {
        Some(data)
            .filter(|data| <dyn Layout>::fits_data(&layout, data))
            .map(|data| DataRef { data, layout })
    }

    /// Copy the data bytes of the layout to the byte buffer.
    pub fn write_to_buf<'data>(&self, buffer: &mut buf) {
        let len = buffer.len().min(core::mem::size_of_val(self.data));
        buffer[..len].copy_from_slice(self.data);
    }

    transfer_methods! {
        derive {
            write_to_buf: Self::write_to_buf,
        }
    }
}

impl<'lt, L> DataMut<'lt, L> {
    /// Verifies the data against the layout before construction.
    ///
    /// Note that the type has no hard invariants.
    pub fn checked_new(data: &'lt mut [u8], layout: L) -> Option<Self>
    where
        L: Layout,
    {
        Some(data)
            .filter(|data| <dyn Layout>::fits_data(&layout, data))
            .map(|data| DataMut { data, layout })
    }

    /// Copy the data bytes of the layout to the byte buffer.
    pub fn write_to_buf(&self, buffer: &mut buf) {
        let len = buffer.len().min(core::mem::size_of_val(self.data));
        buffer[..len].copy_from_slice(self.data);
    }

    /// Copy the data bytes of the layout to the byte buffer.
    pub fn read_from_buf(&mut self, buffer: &buf) {
        let len = buffer.len().min(core::mem::size_of_val(self.data));
        self.data[..len].copy_from_slice(buffer);
    }

    transfer_methods! {
        derive {
            write_to_buf: Self::write_to_buf,
            read_from_buf: Self::read_from_buf,
        }
    }
}
