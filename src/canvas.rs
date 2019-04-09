// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use zerocopy::{AsBytes, FromBytes};
use crate::{AsPixel, Rec, Pixel};

/// A 2d matrix of pixels.
pub struct Canvas<P: AsBytes + FromBytes> {
    inner: Rec<P>,
    layout: Layout<P>,
}

/// Describes the memory region used for the image.
///
/// The underlying buffer may have more data allocated than this region and cause the overhead to
/// be reused when resizing the image. All ways to construct this already check that all pixels
/// within the resulting image can be addressed via an index.
pub struct Layout<P> {
    width: usize,
    height: usize,
    pixel: Pixel<P>,
}

impl<P: AsBytes + FromBytes> Canvas<P> {
    /// Allocate a canvas with specified layout.
    ///
    /// # Panics
    /// When allocation of memory fails.
    pub fn with_layout(layout: Layout<P>) -> Self {
        Canvas {
            inner: Rec::bytes_for_pixel(layout.pixel, layout.byte_len()),
            layout,
        }
    }

    /// Directly try to allocate a canvas from width and height.
    ///
    /// # Panics
    /// This panics when the layout described by `width` and `height` can not be allocated, for
    /// example due to it being an invalid layout. If you want to handle the layout being invalid,
    /// consider using `Layout::from_width_and_height` and `Canvas::with_layout`.
    pub fn with_width_and_height(width: usize, height: usize) -> Self
        where P: AsPixel
    {
        let layout = Layout::width_and_height(width, height)
            .expect("Pixel layout can not fit into memory");
        Self::with_layout(layout)
    }

    pub fn as_slice(&self) -> &[P] {
        &self.inner.as_slice()[..self.layout.len()]
    }

    pub fn as_mut_slice(&mut self) -> &mut [P] {
        &mut self.inner.as_mut_slice()[..self.layout.len()]
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// See `transmute_to` for details.
    pub fn transmute<Q: AsPixel + AsBytes + FromBytes>(self) -> Canvas<Q> {
        self.transmute_to(Q::pixel())
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// See `transmute_to` for details.
    pub fn transmute_to<Q: AsBytes + FromBytes>(self, pixel: Pixel<Q>) -> Canvas<Q> {
        let layout = self.layout.transmute_to(pixel);
        let inner = self.inner.reinterpret_to(pixel);

        Canvas {
            layout,
            inner,
        }
    }
}

impl<P> Layout<P> {
    pub fn width_and_height_for_pixel(pixel: Pixel<P>, width: usize, height: usize)
        -> Option<Self> 
    {
        let max_index = Self::max_index(width, height)?;
        let _ = max_index.checked_mul(pixel.size())?;

        Some(Layout {
            width,
            height,
            pixel,
        })
    }

    pub fn width_and_height(width: usize, height: usize) -> Option<Self>
        where P: AsPixel
    {
        Self::width_and_height_for_pixel(P::pixel(), width, height)
    }

    /// Get the required bytes for this layout.
    pub fn byte_len(self) -> usize {
        // Exactly this does not overflow due to construction.
        self.pixel.size() * self.width * self.height
    }

    /// The number of pixels in this layout
    pub fn len(self) -> usize {
        self.width * self.height
    }

    pub fn width(self) -> usize {
        self.width
    }

    pub fn height(self) -> usize {
        self.height
    }

    pub fn pixel(self) -> Pixel<P> {
        self.pixel
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// See `transmute_to` for details.
    pub fn transmute<Q: AsPixel>(self) -> Layout<Q> {
        self.transmute_to(Q::pixel())
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// # Panics
    /// Like `std::mem::transmute`, the size of the two types need to be equal. This ensures that
    /// all indices are valid in both directions.
    pub fn transmute_to<Q>(self, pixel: Pixel<Q>) -> Layout<Q> {
        assert!(self.pixel.size() == pixel.size());
        Layout {
            width: self.width,
            height: self.height,
            pixel,
        }
    }

    fn max_index(width: usize, height: usize) -> Option<usize> {
        width.checked_mul(height)
    }
}

impl<P> Clone for Layout<P> { 
    fn clone(&self) -> Self {
        Layout {
            .. *self // This is, apparently, legal.
        }
    }
}

impl<P> Copy for Layout<P> { }

impl<P: AsPixel> Default for Layout<P> {
    fn default() -> Self {
        Layout {
            width: 0,
            height: 0,
            pixel: P::pixel(),
        }
    }
}
