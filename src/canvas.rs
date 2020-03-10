// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
use core::fmt;

use bytemuck::Pod;

use crate::buf::{Buffer, Cog};
use crate::layout::{DynLayout, Layout, SampleSlice};
use crate::{Rec, ReuseError};

/// A canvas, parameterized over the layout.
///
/// The buffer is either owned or _mutably_ borrowed from another `Canvas`. Some allocating methods
/// may lead to an implicit change from a borrowed to an owned buffer. These methods are documented
/// as performing a fallible allocation. Other method calls on the previously borrowing canvas will
/// afterwards no longer change the bytes of the canvas it was borrowed from.
///
/// This type permits user defined layouts of any kind and does not unsafely depend on the validity
/// of the layouts. Correctness is achieved in the common case by discouraging methods that would
/// lead to a diverging size of the memory buffer and the layout. Hence, access to the image pixels
/// should not lead to panic unless an incorrectly implemented layout is used.
///
/// Since a `Canvas` can not unsafely rely on the layout behaving correctly, direct accessors may
/// have suboptimal behaviour and perform a few (seemingly) redundant checks. More optimal, but
/// much more specialized, wrappers are provided in other types such as `Matrix`.
///
/// Note also that the borrowing canvas can arbitrarily change its own layout and thus overwrite
/// the content with completely different types and layouts. This is intended to maximize the
/// flexibility for users. In complicated cases it could be hard for the type system to reflect the
/// compatibility of a custom pixel layout and a standard one. It is solely the user's
/// responsibility to use the interface sensibly. The _soundness_ of standard channel types (e.g.
/// `u8` or `u32`) is not impacted by this as any byte content is valid for them.
///
/// It is possible to convert the layout to a less strictly typed one without reallocating the
/// buffer. For example, all standard layouts such as `Matrix` can be weakened to `DynLayout`. The
/// reverse can not be done unchecked but is possible with fallible conversions.
///
/// ## Examples
///
/// ```
/// ```
#[derive(PartialEq, Eq)]
pub struct Canvas<'buf, Layout> {
    buffer: Cog<'buf>,
    layout: Layout,
}

/// Layout oblivious methods that can allocate and change to another buffer.
impl<'buf, L> Canvas<'buf, L> {
    /// Grow the buffer, preparing for another layout.
    ///
    /// This may allocate a new buffer and thus disassociate the image from the currently borrowed
    /// underlying buffer.
    ///
    /// # Panics
    /// This function will panic if an allocation is necessary but fails.
    pub fn grow(&mut self, layout: &impl Layout) {
        Cog::grow_to(&mut self.buffer, layout.byte_len());
    }

    /// Convert the inner layout.
    ///
    /// This method expects that the converted layout is compatible with the current layout.
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

    /// Convert the inner layout to a dynamic one.
    ///
    /// This is mostly convenience. Also not that `DynLayout` is of course not _completely_ generic
    /// but tries to emulate a large number of known layouts.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub fn into_dynamic(self) -> Canvas<'buf, DynLayout>
    where
        L: Into<DynLayout>,
    {
        self.into_layout()
    }

    /// Change the layout, reusing and growing the buffer.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub fn into_reinterpreted<Other: Layout>(mut self, layout: Other) -> Canvas<'buf, Other> {
        Cog::grow_to(&mut self.buffer, layout.byte_len());
        Canvas {
            buffer: self.buffer,
            layout,
        }
    }

    /// Mutably borrow this canvas with another arbitrary layout.
    ///
    /// The other layout could be completely incompatible and perform arbitrary mutations. This
    /// seems counter intuitive at first, but recall that these mutations are not unsound as they
    /// can not invalidate the bytes themselves and only write unexpected values. This provides
    /// more flexibility for 'transmutes' than easily expressible in the type system.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub fn as_reinterpreted<Other>(&mut self, other: Other) -> Canvas<'_, Other>
    where
        Other: Layout,
    {
        self.grow(&other);
        Canvas {
            buffer: Cog::Borrowed(&mut self.buffer),
            layout: other,
        }
    }

    /// Take ownership of the image's bytes.
    ///
    /// # Panics
    /// This method panics if allocation fails.
    pub fn into_owned(self) -> Canvas<'static, L> {
        Canvas {
            buffer: Cog::Owned(Cog::into_owned(self.buffer)),
            layout: self.layout,
        }
    }
}

/// Layout oblivious methods, these also never allocate or panic.
impl<'buf, L> Canvas<'buf, L> {
    /// Get a reference to the unstructured bytes of the canvas.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes`].
    ///
    /// [`as_layout_bytes`]: #method.as_layout_bytes
    pub fn as_capacity_bytes(&self) -> &[u8] {
        self.buffer.as_bytes()
    }

    /// Get a mutable reference to the unstructured bytes of the canvas.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes_mut`].
    ///
    /// [`as_layout_bytes_mut`]: #method.as_layout_bytes_mut
    pub fn as_capacity_bytes_mut(&mut self) -> &mut [u8] {
        self.buffer.as_bytes_mut()
    }

    /// Get a reference to the layout.
    pub fn layout(&self) -> &L {
        &self.layout
    }

    /// Get a mutable reference to the layout.
    ///
    /// Be mindful not to modify the layout to exceed the allocated size.
    pub fn layout_mut_unguarded(&mut self) -> &mut L {
        &mut self.layout
    }

    /// Reinterpret the bits in another layout.
    ///
    /// This method fails if the layout requires more bytes than are currently allocated.
    pub fn try_reinterpret<Other>(self, layout: Other) -> Result<Canvas<'buf, Other>, Self>
    where
        Other: Layout,
    {
        if self.buffer.len() > layout.byte_len() {
            Err(self)
        } else {
            Ok(Canvas {
                buffer: self.buffer,
                layout,
            })
        }
    }

    /// Change the layout without checking the buffer.
    pub fn reinterpret_unguarded<Other: Layout>(self, layout: Other) -> Canvas<'buf, Other> {
        Canvas {
            buffer: self.buffer,
            layout,
        }
    }

    /// Borrow the buffer with the same layout.
    pub fn borrow_mut(&mut self) -> Canvas<'_, L>
    where
        L: Clone,
    {
        Canvas {
            buffer: Cog::Borrowed(&mut self.buffer),
            layout: self.layout.clone(),
        }
    }
}

/// Methods specifically with a dynamic layout.
impl<'buf> Canvas<'buf, DynLayout> {
    pub fn try_from_dynamic<Other>(self, layout: Other) -> Result<Canvas<'buf, Other>, Self>
    where
        Other: Into<DynLayout> + Clone,
    {
        let reference = layout.clone().into();
        if self.layout == reference {
            Ok(Canvas {
                buffer: self.buffer,
                layout,
            })
        } else {
            Err(self)
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

    /// Get a reference to those bytes used by the layout.
    pub fn as_bytes(&self) -> &[u8] {
        &self.as_capacity_bytes()[..self.layout.byte_len()]
    }

    /// Get a mutable reference to those bytes used by the layout.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let len = self.layout.byte_len();
        &mut self.as_capacity_bytes_mut()[..len]
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

    /// Reuse the buffer for a new image layout of the same type.
    pub fn try_reuse(&mut self, layout: L) -> Result<(), ReuseError> {
        if self.as_capacity_bytes().len() >= layout.byte_len() {
            self.layout = layout;
            Ok(())
        } else {
            Err(ReuseError {
                capacity: self.as_capacity_bytes().len(),
                requested: Some(layout.byte_len()),
            })
        }
    }
}

/// Methods for layouts that are slices of individual samples.
impl<'buf, L: SampleSlice> Canvas<'buf, L>
where
    L::Sample: Pod,
{
    /// Interpret an existing buffer as a pixel canvas.
    ///
    /// The data already contained within the buffer is not modified so that prior initialization
    /// can be performed or one array of samples reinterpreted for an image of other sample type.
    /// However, the `Rec` will be logically resized which will zero-initialize missing elements if
    /// the current buffer is too short.
    ///
    /// # Panics
    ///
    /// This function will panic if resizing causes a reallocation that fails.
    pub fn from_rec(buffer: Rec<L::Sample>, layout: L) -> Self {
        let mut buffer = buffer.into_inner();
        buffer.grow_to(layout.byte_len());
        Self {
            buffer: Cog::Owned(buffer),
            layout,
        }
    }

    pub fn as_slice(&self) -> &[L::Sample] {
        self.buffer.as_pixels(self.layout.sample())
    }

    pub fn as_mut_slice(&mut self) -> &mut [L::Sample] {
        self.buffer.as_mut_pixels(self.layout.sample())
    }

    /// Convert back into an vector-like of sample types.
    pub fn into_rec(self) -> Rec<L::Sample> {
        let sample = self.layout.sample();
        // Avoid calling any method of `Layout` after this. Not relevant for safety but might be in
        // the future, if we want to avoid the extra check in `resize`.
        let count = self.as_slice().len();
        let buffer = Cog::into_owned(self.buffer);
        let mut rec = Rec::from_buffer(buffer, sample);
        // This should never reallocate at this point but we don't really know or care.
        rec.resize(count);
        rec
    }
}

impl<Layout: Clone> Clone for Canvas<'_, Layout> {
    fn clone(&self) -> Self {
        use alloc::borrow::ToOwned;
        Canvas {
            buffer: Cog::Owned(self.buffer.to_owned()),
            layout: self.layout.clone(),
        }
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

impl<L> fmt::Debug for Canvas<'_, L>
where
    L: SampleSlice + fmt::Debug,
    L::Sample: Pod + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Canvas")
            .field("layout", &self.layout)
            .field("content", &self.as_slice())
            .finish()
    }
}
