use core::ops;

use crate::buf::{atomic_buf, buf, cell_buf, AtomicBuffer, Buffer, CellBuffer};
use crate::layout::{Decay, DynLayout, Layout, SliceLayout, Take};
use crate::{BufferReuseError, TexelBuffer};

/// Inner buffer implementation.
///
/// Not exposed to avoid leaking the implementation detail of the `Buf` type parameter. This allows
/// a single implementation for borrowed and owned buffers while keeping `buf`, `Cog` etc. private.
#[derive(Default, Clone, PartialEq, Eq)]
pub(crate) struct RawImage<Buf, Layout> {
    buffer: Buf,
    layout: Layout,
}

pub(crate) trait BufferLike {
    /// Get the length of the buffer in bytes.
    fn byte_len(&self) -> usize;
    /// Convert the bytes into a normalized buffer allocation.
    fn into_owned(self) -> Buffer;
    /// Transfer all bytes to a new instance.
    fn take(&mut self) -> Self
    where
        Self: Sized;
}

pub(crate) trait BufferRef: BufferLike + ops::Deref<Target = buf> {}

pub(crate) trait BufferMut: BufferRef + ops::DerefMut {}

pub(crate) trait Growable: BufferLike {
    fn grow_to(&mut self, _: usize);
}

/// Layout oblivious methods that can allocate and change to another buffer.
impl<B: Growable, L> RawImage<B, L> {
    /// Grow the buffer, preparing for another layout.
    ///
    /// This may allocate a new buffer and thus disassociate the image from the currently borrowed
    /// underlying buffer.
    ///
    /// # Panics
    /// This function will panic if an allocation is necessary but fails.
    pub(crate) fn grow(&mut self, layout: &impl Layout) {
        Growable::grow_to(&mut self.buffer, layout.byte_len());
    }

    /// Convert the inner layout.
    ///
    /// This method expects that the converted layout is compatible with the current layout.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn decay<Other>(mut self) -> RawImage<B, Other>
    where
        Other: Decay<L>,
    {
        let layout = Other::decay(self.layout);
        Growable::grow_to(&mut self.buffer, layout.byte_len());
        RawImage {
            buffer: self.buffer,
            layout,
        }
    }

    /// Change the layout, reusing and growing the buffer.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn with_layout<Other: Layout>(mut self, layout: Other) -> RawImage<B, Other> {
        Growable::grow_to(&mut self.buffer, layout.byte_len());
        RawImage {
            buffer: self.buffer,
            layout,
        }
    }

    /// Mutably borrow this image with another arbitrary layout.
    ///
    /// The other layout could be completely incompatible and perform arbitrary mutations. This
    /// seems counter intuitive at first, but recall that these mutations are not unsound as they
    /// can not invalidate the bytes themselves and only write unexpected values. This provides
    /// more flexibility for 'transmutes' than easily expressible in the type system.
    ///
    /// # Panics
    /// This method panics if the new layout requires more bytes and allocation fails.
    pub(crate) fn as_reinterpreted<Other>(&mut self, other: Other) -> RawImage<&'_ mut buf, Other>
    where
        B: BufferMut,
        Other: Layout,
    {
        self.grow(&other);
        RawImage {
            buffer: &mut self.buffer,
            layout: other,
        }
    }

    /// Change the layout and then resize the buffer so that it still fits.
    pub(crate) fn mutate_layout<T>(&mut self, f: impl FnOnce(&mut L) -> T) -> T
    where
        L: Layout,
    {
        let t = f(&mut self.layout);
        self.buffer.grow_to(self.layout.byte_len());
        t
    }
}

/// Layout oblivious methods, these also never allocate or panic.
impl<B: BufferLike, L> RawImage<B, L> {
    /// Get a mutable reference to the unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes_mut`].
    ///
    /// [`as_layout_bytes_mut`]: #method.as_layout_bytes_mut
    pub(crate) fn as_capacity_bytes_mut(&mut self) -> &mut [u8]
    where
        B: BufferMut,
    {
        self.buffer.as_bytes_mut()
    }

    /// Get a mutable reference to the unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes_mut`].
    ///
    /// [`as_layout_bytes_mut`]: #method.as_layout_bytes_mut
    pub(crate) fn as_capacity_buf_mut(&mut self) -> &mut buf
    where
        B: BufferMut,
    {
        &mut self.buffer
    }

    /// Take ownership of the image's bytes.
    ///
    /// # Panics
    /// This method panics if allocation fails.
    pub(crate) fn into_owned(self) -> RawImage<Buffer, L> {
        RawImage {
            buffer: BufferLike::into_owned(self.buffer),
            layout: self.layout,
        }
    }
}

/// Methods specifically with a dynamic layout.
impl<B> RawImage<B, DynLayout> {
    pub(crate) fn try_from_dynamic<Other>(self, layout: Other) -> Result<RawImage<B, Other>, Self>
    where
        Other: Into<DynLayout> + Clone,
    {
        let reference = layout.clone().into();
        if self.layout == reference {
            Ok(RawImage {
                buffer: self.buffer,
                layout,
            })
        } else {
            Err(self)
        }
    }
}

impl<B, L> RawImage<B, L> {
    /// Allocate a buffer for a particular layout.
    pub(crate) fn new(layout: L) -> Self
    where
        L: Layout,
        B: From<Buffer>,
    {
        let bytes = layout.byte_len();
        RawImage {
            buffer: Buffer::new(bytes).into(),
            layout,
        }
    }

    /// Create a image from a byte slice specifying the contents.
    ///
    /// If the layout requires more bytes then the remaining bytes are zero initialized.
    pub(crate) fn with_contents(buffer: &[u8], layout: L) -> Self
    where
        L: Layout,
        B: From<Buffer>,
    {
        let mut buffer = Buffer::from(buffer);
        buffer.grow_to(layout.byte_len());
        RawImage {
            buffer: buffer.into(),
            layout,
        }
    }

    /// Create a new raw image from layout and buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer does not have enough space for the layout.
    pub fn from_buffer(layout: L, buffer: B) -> Self
    where
        B: BufferLike,
        L: Layout,
    {
        assert!(<dyn Layout>::fits_buffer(&layout, &buffer));
        RawImage { buffer, layout }
    }

    pub(crate) fn with_buffer_unchecked(layout: L, buffer: B) -> Self
    where
        B: ops::Deref<Target = buf>,
    {
        RawImage { buffer, layout }
    }

    pub(crate) fn into_parts(self) -> (B, L) {
        (self.buffer, self.layout)
    }

    /// Get a reference to the layout.
    pub(crate) fn layout(&self) -> &L {
        &self.layout
    }

    /// Get a mutable reference to the layout.
    ///
    /// Be mindful not to modify the layout to exceed the allocated size.
    pub(crate) fn layout_mut_unguarded(&mut self) -> &mut L {
        &mut self.layout
    }

    /// Get a reference to the unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes`].
    ///
    /// [`as_layout_bytes`]: #method.as_layout_bytes
    pub(crate) fn as_capacity_bytes(&self) -> &[u8]
    where
        B: ops::Deref<Target = buf>,
    {
        self.buffer.as_bytes()
    }

    /// Get a reference to the aligned unstructured bytes of the image.
    ///
    /// Note that this may return more bytes than required for the specific layout for various
    /// reasons. See also [`as_layout_bytes`].
    ///
    /// [`as_layout_bytes`]: #method.as_layout_bytes
    pub(crate) fn as_capacity_buf(&self) -> &buf
    where
        B: ops::Deref<Target = buf>,
    {
        &self.buffer
    }

    /// Get a reference to those bytes used by the layout.
    pub(crate) fn as_bytes(&self) -> &[u8]
    where
        B: ops::Deref<Target = buf>,
        L: Layout,
    {
        &self.as_capacity_bytes()[..self.layout.byte_len()]
    }

    pub fn as_buf(&self) -> &buf
    where
        B: ops::Deref<Target = buf>,
        L: Layout,
    {
        let byte_len = self.layout.byte_len();
        self.buffer.truncate(byte_len)
    }

    pub fn as_mut_buf(&mut self) -> &mut buf
    where
        B: ops::DerefMut<Target = buf>,
        L: Layout,
    {
        let byte_len = self.layout.byte_len();
        self.buffer.truncate_mut(byte_len)
    }

    pub(crate) fn as_slice(&self) -> &[L::Sample]
    where
        B: ops::Deref<Target = buf>,
        L: SliceLayout,
    {
        let texel = self.layout.sample();
        texel.cast_buf(self.as_buf())
    }

    /// Get a mutable reference to the buffer. It is inadvisible to modify the buffer in a way that
    /// it can no longer hold the layout.
    pub(crate) fn get_mut(&mut self) -> &mut B {
        &mut self.buffer
    }

    /// Borrow the buffer with the same layout.
    pub(crate) fn as_borrow(&self) -> RawImage<&'_ buf, &'_ L>
    where
        B: ops::Deref<Target = buf>,
    {
        RawImage {
            buffer: &self.buffer,
            layout: &self.layout,
        }
    }

    /// Borrow the buffer mutably with the same layout.
    pub(crate) fn as_borrow_mut(&mut self) -> RawImage<&'_ mut buf, &'_ mut L>
    where
        B: ops::DerefMut<Target = buf>,
    {
        RawImage {
            buffer: &mut self.buffer,
            layout: &mut self.layout,
        }
    }

    /// Check if the buffer is enough for another layout.
    pub(crate) fn fits(&self, other: &impl Layout) -> bool
    where
        B: ops::Deref<Target = buf>,
    {
        <dyn Layout>::fits_buf(other, &self.buffer)
    }

    /// Convert the inner layout.
    ///
    /// This method drops the image if the new layout requires more bytes than the current buffer.
    /// It's recommended you call this only on reference-type buffers.
    pub(crate) fn checked_decay<Other>(self) -> Option<RawImage<B, Other>>
    where
        B: ops::Deref<Target = buf>,
        Other: Decay<L>,
    {
        let layout = Other::decay(self.layout);
        if <dyn Layout>::fits_buf(&layout, &self.buffer) {
            Some(RawImage {
                buffer: self.buffer,
                layout,
            })
        } else {
            None
        }
    }

    /// Change the layout without checking the buffer.
    pub(crate) fn mogrify_layout<Other: Layout>(
        self,
        layout: impl FnOnce(L) -> Other,
    ) -> RawImage<B, Other> {
        let layout = layout(self.layout);

        RawImage {
            buffer: self.buffer,
            layout,
        }
    }

    /// Reinterpret the bits in another layout.
    ///
    /// This method fails if the layout requires more bytes than are currently allocated.
    pub(crate) fn try_reinterpret<Other>(self, layout: Other) -> Result<RawImage<B, Other>, Self>
    where
        B: ops::Deref<Target = buf>,
        Other: Layout,
    {
        if self.buffer.len() < layout.byte_len() {
            Err(self)
        } else {
            Ok(RawImage {
                buffer: self.buffer,
                layout,
            })
        }
    }
}

/// Methods for all `Layouts` (the trait).
impl<B: BufferLike, L: Layout> RawImage<B, L> {
    /// Get a mutable reference to those bytes used by the layout.
    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        B: BufferMut,
    {
        let len = self.layout.byte_len();
        &mut self.as_capacity_bytes_mut()[..len]
    }

    /// Reuse the buffer for a new image layout of the same type.
    pub(crate) fn try_reuse(&mut self, layout: L) -> Result<(), BufferReuseError> {
        if self.buffer.byte_len() >= layout.byte_len() {
            self.layout = layout;
            Ok(())
        } else {
            Err(BufferReuseError {
                capacity: self.buffer.byte_len(),
                requested: Some(layout.byte_len()),
            })
        }
    }

    /// Change the layout but require that the new layout fits the buffer, never reallocate.
    pub(crate) fn mutate_inplace<T>(&mut self, f: impl FnOnce(&mut L) -> T) -> T
    where
        L: Layout,
    {
        let t = f(&mut self.layout);
        assert!(
            self.layout.byte_len() <= self.buffer.byte_len(),
            "Modification required buffer allocation, was not in-place"
        );
        t
    }

    /// Take the buffer and layout from this image, moving content into a new instance.
    ///
    /// Asserts that the moved-from container can hold the emptied layout.
    pub(crate) fn take(&mut self) -> Self
    where
        L: Take,
    {
        let buffer = self.buffer.take();
        let layout = self.mutate_inplace(Take::take);
        RawImage::from_buffer(layout, buffer)
    }
}

/// Methods for layouts that are slices of individual samples.
impl<B: BufferLike, L: SliceLayout> RawImage<B, L> {
    /// Interpret an existing buffer as a pixel image.
    ///
    /// The data already contained within the buffer is not modified so that prior initialization
    /// can be performed or one array of samples reinterpreted for an image of other sample type.
    /// However, the `TexelBuffer` will be logically resized which will zero-initialize missing elements if
    /// the current buffer is too short.
    ///
    /// # Panics
    ///
    /// This function will panic if resizing causes a reallocation that fails.
    pub(crate) fn from_texel_buffer(buffer: TexelBuffer<L::Sample>, layout: L) -> Self
    where
        B: From<Buffer>,
    {
        let buffer = buffer.into_inner();
        assert!(buffer.len() >= layout.byte_len());
        Self {
            buffer: buffer.into(),
            layout,
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [L::Sample]
    where
        B: BufferMut,
    {
        self.layout.sample().cast_mut_buf(self.as_mut_buf())
    }

    /// Convert back into an vector-like of sample types.
    pub(crate) fn into_buffer(self) -> TexelBuffer<L::Sample>
    where
        B: BufferRef,
    {
        let sample = self.layout.sample();
        // Avoid calling any method of `Layout` after this. Not relevant for safety but might be in
        // the future, if we want to avoid the extra check in `resize`.
        let count = self.as_slice().len();
        let buffer = self.buffer.into_owned();
        let mut rec = TexelBuffer::from_buffer(buffer, sample);
        // This should never reallocate at this point but we don't really know or care.
        rec.resize(count);
        rec
    }
}

impl BufferLike for Buffer {
    fn byte_len(&self) -> usize {
        self.as_bytes().len()
    }

    fn into_owned(self) -> Self {
        self
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl BufferLike for CellBuffer {
    fn byte_len(&self) -> usize {
        core::mem::size_of_val(&**self)
    }

    fn into_owned(self) -> Buffer {
        self.to_owned()
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl BufferLike for AtomicBuffer {
    fn byte_len(&self) -> usize {
        core::mem::size_of_val(&**self)
    }

    fn into_owned(self) -> Buffer {
        self.to_owned()
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl BufferLike for &'_ buf {
    fn byte_len(&self) -> usize {
        self.as_bytes().len()
    }

    fn into_owned(self) -> Buffer {
        Buffer::from(self.as_bytes())
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl BufferLike for &'_ mut buf {
    fn byte_len(&self) -> usize {
        self.as_bytes().len()
    }

    fn into_owned(self) -> Buffer {
        Buffer::from(self.as_bytes())
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl BufferLike for &'_ cell_buf {
    fn byte_len(&self) -> usize {
        self.as_texels(crate::texels::U8).as_slice_of_cells().len()
    }

    fn into_owned(self) -> Buffer {
        let mut target = Buffer::new(self.byte_len());
        crate::texels::U8.load_cell_slice(self.as_texels(crate::texels::U8), target.as_bytes_mut());
        target
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl BufferLike for &'_ atomic_buf {
    fn byte_len(&self) -> usize {
        self.as_texels(crate::texels::U8).len()
    }

    fn into_owned(self) -> Buffer {
        let source = self.as_texels(crate::texels::U8);
        let mut target = Buffer::new(source.len());
        source.write_to_slice(target.as_bytes_mut());
        target
    }

    fn take(&mut self) -> Self {
        core::mem::take(self)
    }
}

impl dyn Layout + '_ {
    fn fits_buffer(&self, buffer: &dyn BufferLike) -> bool {
        self.byte_len() <= buffer.byte_len()
    }
}

impl Growable for Buffer {
    fn grow_to(&mut self, bytes: usize) {
        Buffer::grow_to(self, bytes);
    }
}

impl BufferRef for Buffer {}
impl BufferMut for Buffer {}

impl BufferRef for &'_ mut buf {}
impl BufferMut for &'_ mut buf {}
