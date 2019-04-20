// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::ops::{Index, IndexMut};

use zerocopy::{AsBytes, FromBytes};
use crate::{AsPixel, Rec, Pixel};

/// A 2d matrix of pixels.
///
/// The layout describes placement of samples within the memory buffer. An abstraction layer that
/// provides strided access to such pixel data is not intended to be baked into this struct.
/// Instead, it will always store the data in a row-major layout without holes.
///
/// There are two levels of control over the allocation behaviour of a `Canvas`. The direct
/// methods, currently `with_width_and_height` only, lead to a canvas without intermediate steps
/// but may panic due to an invalid layout. Manually using the intermediate [`Layout`] gives custom
/// error handling options and additional offers inspection of the details of the to-be-allocated
/// buffer. A third option is currently not available and depends on support from the Rust standard
/// library, which could also handle allocation failures.
///
/// ## Usage for trusted inputs
///
/// Directly allocate your desired layout with `with_width_and_height`. This may panic when the
/// allocation itself fails or when the allocation for the layout could not described, as the
/// layout would not fit inside the available memory space (i.e. the indices would overflow a
/// `usize`).
///
/// ## Usage for untrusted inputs
///
/// In some cases, for untrusted input such as in image parsing libraries, more control is desired.
/// There is no way to currently catch an allocation failure in stable Rust. Thus, even reasonable
/// bounds can lead to a `panic`, and this is unpreventable (note: when the `try_*` methods of
/// `Vec` become stable this will change).  But one still may want to check the required size
/// before allocation.
///
/// Firstly, no method will implicitely try to allocate memory and methods that will note the
/// potential panic from allocation failure.
///
/// Secondly, an instance of [`Layout`] can be constructed in a panic free manner without any
/// allocation and independently from the `Canvas` instance. By providing it to the `with_layout`
/// constructor ensures that all potential intermediate failures–except as mentioned before–can be
/// explicitely handled by the caller. Furthermore, some utility methods allow inspection of the
/// eventual allocation size before the reservation of memory.
///
/// ## Restrictions
///
/// As previously mentioned, the samples in the internal buffer layout always appear without any
/// holes. Therefore a fast `crop` operation requires wrapping the abstraction layer provided here
/// into another layer describing the *accessible image*, independent from the layout of the actual
/// *pixel data*. This separation of concern–layout vs. acess logic–simplifies the implementation
/// and keeps it agnostic of the desired low-cost operations. Consider that other use cases may
/// require operatios other than `crop` with constant time. Instead of choosing some consistent by
/// limited set here, the mechanism to achieve it is deferred to an upper layer for further
/// freedom. Other structs may, in the future, provide other pixel layouts.
///
/// [`Layout`]: ./struct.Layout.html
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

    pub fn as_bytes(&self) -> &[u8] {
        &self.inner.as_bytes()[..self.layout.byte_len()]
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.inner.as_bytes_mut()[..self.layout.byte_len()]
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// See `transmute_to` for details.
    pub fn transmute<Q: AsPixel + AsBytes + FromBytes>(self) -> Canvas<Q> {
        self.transmute_to(Q::pixel())
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// # Panics
    ///
    /// Like `std::mem::transmute`, the size of the two types need to be equal. This ensures that
    /// all indices are valid in both directions.
    pub fn transmute_to<Q: AsBytes + FromBytes>(self, pixel: Pixel<Q>) -> Canvas<Q> {
        let layout = self.layout.transmute_to(pixel);
        let inner = self.inner.reinterpret_to(pixel);

        Canvas {
            layout,
            inner,
        }
    }

    fn index_of(&self, x: usize, y: usize) -> usize {
        assert!(self.layout.in_bounds(x, y));

        // Can't overflow, surely smaller than `layout.max_index()`.
        y*self.layout.width() + x
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

    fn in_bounds(self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
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

impl<P: AsBytes + FromBytes> Index<(usize, usize)> for Canvas<P> {
    type Output = P;

    fn index(&self, (x, y): (usize, usize)) -> &P {
        &self.as_slice()[self.index_of(x, y)]
    }
}

impl<P: AsBytes + FromBytes> IndexMut<(usize, usize)> for Canvas<P> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut P {
        let index = self.index_of(x, y);
        &mut self.as_mut_slice()[index]
    }
}
