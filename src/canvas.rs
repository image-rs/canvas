// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
use bytemuck::Pod;

use crate::buf::{Buffer, Cog};
use crate::layout::{DynLayout, Layout, SampleSlice};

/// A canvas, parameterized over the layout.
///
/// It is possible to convert the layout to a less strictly typed one without reallocating the
/// buffer. For example, all standard layouts such as `Matrix` can be weakened to `DynLayout`. The
/// reverse can not be done unchecked but is possible with fallible conversions.
///
/// A `Canvas` allows can not unsafely rely on the layout behaving correctly. Hence, direct
/// accessors may have suboptimal behaviour and perform a few (seemingly) redundant checks. More
/// optimal, but much more specialized, wrappers are provided in other types such as `Matrix`.
///
/// ```
/// ```
pub struct Canvas<'buf, Layout = DynLayout> {
    buffer: Cog<'buf>,
    layout: Layout,
}

/// Layout oblivious methods.
impl<'buf, L> Canvas<'buf, L> {
    /// Get a reference to the unstructured bytes of the canvas.
    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_bytes()
    }

    /// Get a mutable reference to the unstructured bytes of the canvas.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.buffer.as_bytes_mut()
    }

    /// Get a reference to the layout.
    pub fn layout(&self) -> &L {
        &self.layout
    }

    /// Convert the inner layout.
    ///
    /// This method expects that the converted layout is compatible with the current layout. Note
    /// that this permits user defined layouts of any kind and does not unsafely depend on the
    /// validity of the conversion.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub fn into_layout<Other>(mut self) -> Canvas<'buf, Other>
    where
        L: Into<Other>,
        Other: Layout,
    {
        let layout = self.layout.into();
        Cog::grow_to(&mut self.buffer, layout.byte_len());
        Canvas {
            buffer: self.buffer,
            layout,
        }
    }

    pub fn into_dynamic(self) -> Canvas<'buf>
    where
        L: Into<DynLayout>,
    {
        self.into_layout()
    }

    pub fn into_owned(self) -> Canvas<'static, L> {
        Canvas {
            buffer: Cog::Owned(Cog::into_owned(self.buffer)),
            layout: self.layout,
        }
    }
}

/// Methods for all `Layouts` (the trait).
impl<'buf, L: Layout> Canvas<'buf, L> {
    /// Allocate a buffer for a particular layout.
    pub fn new(layout: L) -> Self {
        let bytes = layout.byte_len();
        Canvas {
            buffer: Cog::Owned(Buffer::new(bytes)),
            layout,
        }
    }

    /// Create a canvas from a byte slice specifying the contents.
    ///
    /// If the layout requires more bytes then the remaining bytes are zero initialized.
    pub fn with_contents(buffer: &[u8], layout: L) -> Self {
        let mut buffer = Buffer::from(buffer);
        buffer.grow_to(layout.byte_len());
        Canvas {
            buffer: Cog::Owned(buffer),
            layout,
        }
    }

    /// Mutably borrow this canvas with another layout.
    ///
    /// The other layout could be completely incompatible and perform arbitrary mutations. This
    /// seems counter intuitive at first, but recall that these mutations are not unsound as they
    /// can not invalidate the bytes themselves and only write unexpected values. This provides
    /// more flexibility for 'transmutes' than easily expressible in the type system.
    ///
    /// # Panics
    /// This method panic if the other layout requires more bytes than allocated for this layout.
    pub fn as_layout<Other>(&mut self, other: Other) -> Canvas<'_, Other>
    where
        Other: Layout,
    {
        assert!(other.byte_len() <= self.buffer.len());
        Canvas {
            buffer: Cog::Borrowed(&mut self.buffer),
            layout: other,
        }
    }
}

/// Methods for layouts that are slices of individual samples.
impl<'buf, L: SampleSlice> Canvas<'buf, L>
where
    L::Sample: Pod,
{
    pub fn as_slice(&self) -> &[L::Sample] {
        self.buffer.as_pixels(L::sample())
    }
}

impl<Layout: Default> Default for Canvas<'_, Layout> {
    fn default() -> Self {
        Canvas {
            buffer: Cog::Owned(Buffer::default()),
            layout: Layout::default(),
        }
    }
}
