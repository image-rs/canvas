// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019 The `image-rs` developers
use core::ops::{Index, IndexMut};
use core::{cmp, fmt};

use crate::buf::Buffer;
use crate::canvas::{Canvas, RawCanvas};
use crate::layout::Matrix as Layout;
use crate::{layout, AsTexel, BufferReuseError, Texel, TexelBuffer};

/// A 2d, width-major matrix of pixels.
///
/// The layout describes placement of samples within the memory buffer. An abstraction layer that
/// provides strided access to such pixel data is not intended to be baked into this struct.
/// Instead, it will always store the data in a row-major layout without holes.
///
/// There are two levels of control over the allocation behaviour of a `Matrix`. The direct
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
/// Firstly, no method will implicitly try to allocate memory and methods that will note the
/// potential panic from allocation failure.
///
/// Secondly, an instance of [`Layout`] can be constructed in a panic free manner without any
/// allocation and independently from the `Matrix` instance. By providing it to the `with_layout`
/// constructor ensures that all potential intermediate failures–except as mentioned before–can be
/// explicitly handled by the caller. Furthermore, some utility methods allow inspection of the
/// eventual allocation size before the reservation of memory.
///
/// ## Restrictions
///
/// As previously mentioned, the samples in the internal buffer layout always appear without any
/// holes. Therefore a fast `crop` operation requires wrapping the abstraction layer provided here
/// into another layer describing the *accessible image*, independent from the layout of the actual
/// *pixel data*. This separation of concern–layout versus access logic–simplifies the implementation
/// and keeps it agnostic of the desired low-cost operations. Consider that other use cases may
/// require operations other than `crop` with constant time. Instead of choosing some consistent by
/// limited set here, the mechanism to achieve it is deferred to an upper layer for further
/// freedom. Other structs may, in the future, provide other pixel layouts.
///
/// [`Layout`]: ./struct.Layout.html
#[derive(Clone, PartialEq, Eq)]
pub struct Matrix<P> {
    inner: RawCanvas<Buffer, Layout<P>>,
}

/// Error representation for a failed buffer reuse for a canvas.
///
/// Emitted as a result of [`Matrix::from_buffer`] when the buffer capacity is not large enough to
/// serve as an image of requested layout with causing a reallocation.
///
/// It is possible to retrieve the buffer that cause the failure with `into_buffer`. This allows one
/// to manually try to correct the error with additional checks, or implement a fallback strategy
/// which does not require the interpretation as a full image.
///
/// ```
/// # use canvas::{Matrix, layout, TexelBuffer};
/// let buffer = TexelBuffer::<u8>::new(16);
/// let allocation = buffer.as_bytes().as_ptr();
///
/// let bad_layout = layout::Matrix::width_and_height(buffer.capacity() + 1, 1).unwrap();
/// let error = match Matrix::from_reused_buffer(buffer, bad_layout) {
///     Ok(_) => unreachable!("The layout requires one too many pixels"),
///     Err(error) => error,
/// };
///
/// // Get back the original buffer.
/// let buffer = error.into_buffer();
/// assert_eq!(buffer.as_bytes().as_ptr(), allocation);
/// ```
///
/// [`Matrix::from_buffer`]: ./struct.Matrix.html#method.from_buffer
#[derive(PartialEq, Eq)]
pub struct MatrixReuseError<P> {
    buffer: TexelBuffer<P>,
    layout: Layout<P>,
}

/// The canvas could not be mapped to another pixel type without reuse.
///
/// This may be caused since the layout would be invalid or due to the layout being too large for
/// the current buffer allocation.
///
/// # Examples
///
/// Use the error type to conveniently enforce a custom policy for allowed and prohibited
/// allocations.
///
/// ```
/// # use canvas::Matrix;
/// # let canvas = Matrix::<u8>::with_width_and_height(2, 2);
/// # struct RequiredAllocationTooLarge;
///
/// match canvas.map_reuse(f32::from) {
///     // Everything worked fine.
///     Ok(canvas) => Ok(canvas),
///     Err(error) => {
///         // Manually validate if this reallocation should be allowed?
///         match error.layout() {
///             // Accept an allocation only if its smaller than a page
///             Some(layout) if layout.byte_len() <= (1 << 12)
///                 => Ok(error.into_canvas().map(f32::from)),
///             _ => Err(RequiredAllocationTooLarge),
///         }
///     },
/// }
///
/// # ;
/// ```
#[derive(PartialEq, Eq)]
pub struct MapReuseError<P, Q> {
    buffer: Matrix<P>,
    layout: Option<Layout<Q>>,
}

impl<P> Matrix<P> {
    /// Allocate a canvas with specified layout.
    ///
    /// # Panics
    /// When allocation of memory fails.
    pub fn with_layout(layout: Layout<P>) -> Self {
        let rec = TexelBuffer::bytes_for_texel(layout.pixel, layout.byte_len());
        Self::new_raw(rec, layout)
    }

    /// Directly try to allocate a canvas from width and height.
    ///
    /// # Panics
    /// This panics when the layout described by `width` and `height` can not be allocated, for
    /// example due to it being an invalid layout. If you want to handle the layout being invalid,
    /// consider using `Layout::from_width_and_height` and `Matrix::with_layout`.
    pub fn with_width_and_height(width: usize, height: usize) -> Self
    where
        P: AsTexel,
    {
        let layout =
            Layout::width_and_height(width, height).expect("Texel layout can not fit into memory");
        Self::with_layout(layout)
    }

    /// Interpret an existing buffer as a pixel canvas.
    ///
    /// The data already contained within the buffer is not modified so that prior initialization
    /// can be performed or one array of samples reinterpreted for an image of other sample type.
    /// However, the `TexelBuffer` will be logically resized which will zero-initialize missing elements if
    /// the current buffer is too short.
    ///
    /// # Panics
    ///
    /// This function will panic if resizing causes a reallocation that fails.
    pub fn from_buffer(mut buffer: TexelBuffer<P>, layout: Layout<P>) -> Self {
        buffer.resize_bytes(layout.byte_len());
        Self::new_raw(buffer, layout)
    }

    /// Reuse an existing buffer for a pixel canvas.
    ///
    /// Similar to `from_buffer` but this function will never reallocate the inner buffer. Instead, it
    /// will return the `TexelBuffer` unmodified if the creation fails. See [`MatrixReuseError`] for
    /// further information on the error and retrieving the buffer.
    ///
    /// [`MatrixReuseError`]: ./struct.CanvasReuseError.html
    pub fn from_reused_buffer(
        mut buffer: TexelBuffer<P>,
        layout: Layout<P>,
    ) -> Result<Self, MatrixReuseError<P>> {
        match buffer.reuse_bytes(layout.byte_len()) {
            Ok(_) => (),
            Err(_) => return Err(MatrixReuseError { buffer, layout }),
        }
        Ok(Self::new_raw(buffer, layout))
    }

    fn new_raw(inner: TexelBuffer<P>, layout: Layout<P>) -> Self {
        assert_eq!(inner.len(), layout.len(), "Texel count agrees with buffer");
        Matrix {
            inner: RawCanvas::from_buffer(inner, layout),
        }
    }

    pub fn as_slice(&self) -> &[P] {
        self.inner.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [P] {
        self.inner.as_mut_slice()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }

    /// Resize the buffer for a new image.
    ///
    /// # Panics
    ///
    /// This function will panic if an allocation is necessary but fails.
    pub fn resize(&mut self, layout: Layout<P>) {
        self.inner.grow(&layout);
        *self.inner.layout_mut_unguarded() = layout;
    }

    /// Reuse the buffer for a new image layout.
    pub fn reuse(&mut self, layout: Layout<P>) -> Result<(), BufferReuseError> {
        self.inner.try_reuse(layout)
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// See [`Matrix::transmute_to`] for details.
    pub fn transmute<Q: AsTexel>(self) -> Matrix<Q> {
        self.transmute_to(Q::texel())
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// # Panics
    ///
    /// Like [`core::mem::transmute`], the size of the two types need to be equal. This ensures that
    /// all indices are valid in both directions.
    pub fn transmute_to<Q: AsTexel>(self, pixel: Texel<Q>) -> Matrix<Q> {
        let layout = self.layout().transmute_to(pixel);
        let inner = self.inner.reinterpret_unguarded(|_| layout);
        Matrix { inner }
    }

    /// Get the layout of the matrix.
    fn layout(&self) -> Layout<P> {
        *self.inner.layout()
    }

    pub fn into_buffer(self) -> TexelBuffer<P> {
        self.inner.into_buffer()
    }

    fn index_of(&self, x: usize, y: usize) -> usize {
        self.layout().index_of(x, y)
    }

    /// Apply a function to all pixel values.
    ///
    /// See [`Matrix::map_to`] for the details.
    ///
    /// # Panics
    ///
    /// This function will panic if the new layout would be invalid (because the new pixel type
    /// requires a larger buffer than can be allocate) or if the reallocation fails.
    pub fn map<F, Q>(self, map: F) -> Matrix<Q>
    where
        F: Fn(P) -> Q,
        Q: AsTexel,
    {
        self.map_to(map, Q::texel())
    }

    /// Apply a function to all pixel values.
    ///
    /// Unlike [`Matrix::transmute_to`] there are no restrictions on the pixel types. This will
    /// reuse the underlying buffer or resize it if that is not possible.
    ///
    /// # Panics
    ///
    /// This function will panic if the new layout would be invalid (because the new pixel type
    /// requires a larger buffer than can be allocate) or if the reallocation fails.
    pub fn map_to<F, Q>(self, map: F, pixel: Texel<Q>) -> Matrix<Q>
    where
        F: Fn(P) -> Q,
    {
        // First compute the new layout ..
        let layout = self
            .layout()
            .map_to(pixel)
            .expect("Texel layout can not fit into memory");
        // .. then do the actual pixel mapping.
        let inner = self.into_buffer().map_to(map, pixel);
        Matrix::from_buffer(inner, layout)
    }

    pub fn map_reuse<F, Q>(self, map: F) -> Result<Matrix<Q>, MapReuseError<P, Q>>
    where
        F: Fn(P) -> Q,
        Q: AsTexel,
    {
        self.map_reuse_to(map, Q::texel())
    }

    pub fn map_reuse_to<F, Q>(
        self,
        map: F,
        pixel: Texel<Q>,
    ) -> Result<Matrix<Q>, MapReuseError<P, Q>>
    where
        F: Fn(P) -> Q,
    {
        let layout = match self.layout().map_to(pixel) {
            Some(layout) => layout,
            None => {
                return Err(MapReuseError {
                    buffer: self,
                    layout: None,
                })
            }
        };

        if self.inner.as_bytes().len() < layout.byte_len() {
            return Err(MapReuseError {
                buffer: self,
                layout: Some(layout),
            });
        }

        let inner = self.into_buffer().map_to(map, pixel);

        Ok(Matrix::from_buffer(inner, layout))
    }
}

impl<P> Layout<P> {
    pub fn width_and_height_for_texel(
        pixel: Texel<P>,
        width: usize,
        height: usize,
    ) -> Option<Self> {
        let max_index = Self::max_index(width, height)?;
        let _ = max_index.checked_mul(pixel.size())?;

        Some(Layout {
            width,
            height,
            pixel,
        })
    }

    pub fn width_and_height(width: usize, height: usize) -> Option<Self>
    where
        P: AsTexel,
    {
        Self::width_and_height_for_texel(P::texel(), width, height)
    }

    pub const fn empty(pixel: Texel<P>) -> Self {
        Layout {
            pixel,
            width: 0,
            height: 0,
        }
    }

    pub fn with_matrix(pixel: Texel<P>, matrix: layout::MatrixBytes) -> Option<Self> {
        if pixel.size() == matrix.element.size() {
            Some(Layout {
                pixel,
                width: matrix.first_dim,
                height: matrix.second_dim,
            })
        } else {
            None
        }
    }

    pub fn into_matrix_bytes(self) -> layout::MatrixBytes {
        layout::MatrixBytes {
            element: self.pixel.into(),
            first_dim: self.width,
            second_dim: self.height,
        }
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

    pub fn pixel(self) -> Texel<P> {
        self.pixel
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// See `transmute_to` for details.
    pub fn transmute<Q: AsTexel>(self) -> Layout<Q> {
        self.transmute_to(Q::texel())
    }

    /// Reinterpret to another, same size pixel type.
    ///
    /// # Panics
    /// Like `std::mem::transmute`, the size of the two types need to be equal. This ensures that
    /// all indices are valid in both directions.
    pub fn transmute_to<Q>(self, pixel: Texel<Q>) -> Layout<Q> {
        assert!(
            self.pixel.size() == pixel.size(),
            "{} vs {}",
            self.pixel.size(),
            pixel.size()
        );
        Layout {
            width: self.width,
            height: self.height,
            pixel,
        }
    }

    /// Utility method to change the pixel type without changing the dimensions.
    pub fn map<Q: AsTexel>(self) -> Option<Layout<Q>> {
        self.map_to(Q::texel())
    }

    /// Utility method to change the pixel type without changing the dimensions.
    pub fn map_to<Q>(self, pixel: Texel<Q>) -> Option<Layout<Q>> {
        Layout::width_and_height_for_texel(pixel, self.width, self.height)
    }

    pub(crate) fn index_of(self, x: usize, y: usize) -> usize {
        assert!(self.in_bounds(x, y));

        // Can't overflow, surely smaller than `layout.max_index()`.
        y * self.width() + x
    }

    pub(crate) fn in_bounds(self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    fn max_index(width: usize, height: usize) -> Option<usize> {
        width.checked_mul(height)
    }
}

impl<P> MatrixReuseError<P> {
    /// Unwrap the original buffer.
    pub fn into_buffer(self) -> TexelBuffer<P> {
        self.buffer
    }
}

impl<P, Q> MapReuseError<P, Q> {
    /// Unwrap the original buffer.
    pub fn into_canvas(self) -> Matrix<P> {
        self.buffer
    }

    /// The layout that would be required to perform the map operation.
    ///
    /// Returns `Some(_)` if such a layout can be constructed in theory and return `None` if it
    /// would exceed the platform address space.
    pub fn layout(&self) -> Option<Layout<Q>> {
        self.layout
    }
}

impl<P> From<Canvas<Layout<P>>> for Matrix<P> {
    fn from(canvas: Canvas<Layout<P>>) -> Self {
        let layout = *canvas.layout();
        let rec = canvas.into_buffer();
        Self::new_raw(rec, layout)
    }
}

impl<P> From<Matrix<P>> for Canvas<Layout<P>> {
    fn from(matrix: Matrix<P>) -> Self {
        let layout = matrix.layout();
        let rec = matrix.into_buffer();
        Canvas::from_buffer(rec, layout)
    }
}

impl<P> layout::Layout for Layout<P> {
    fn byte_len(&self) -> usize {
        Layout::byte_len(*self)
    }
}

impl<P> layout::SliceLayout for Layout<P> {
    type Sample = P;

    fn sample(&self) -> Texel<P> {
        self.pixel
    }
}

impl<P: AsTexel> Default for Layout<P> {
    fn default() -> Self {
        Self::empty(P::texel())
    }
}

impl<P> layout::Take for Layout<P> {
    fn take(&mut self) -> Self {
        core::mem::replace(self, Self::empty(self.pixel))
    }
}

impl<P> fmt::Debug for Layout<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Layout")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pixel", &self.pixel)
            .finish()
    }
}

impl<P> Clone for Layout<P> {
    fn clone(&self) -> Self {
        Layout { ..*self }
    }
}

impl<P> Copy for Layout<P> {}

impl<P> cmp::PartialEq for Layout<P> {
    fn eq(&self, other: &Self) -> bool {
        (self.width, self.height) == (other.width, other.height)
    }
}

impl<P> cmp::Eq for Layout<P> {}

impl<P> cmp::PartialOrd for Layout<P> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        if self.width < other.width && self.height < other.height {
            Some(cmp::Ordering::Less)
        } else if self.width > other.width && self.height > other.height {
            Some(cmp::Ordering::Greater)
        } else if self.width == other.width && self.height == other.height {
            Some(cmp::Ordering::Equal)
        } else {
            None
        }
    }
}

impl<P: AsTexel> Default for Matrix<P> {
    fn default() -> Self {
        Matrix::from_buffer(TexelBuffer::default(), Layout::default())
    }
}

impl<P> Index<(usize, usize)> for Matrix<P> {
    type Output = P;

    fn index(&self, (x, y): (usize, usize)) -> &P {
        &self.as_slice()[self.index_of(x, y)]
    }
}

impl<P> IndexMut<(usize, usize)> for Matrix<P> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut P {
        let index = self.index_of(x, y);
        &mut self.as_mut_slice()[index]
    }
}

impl<P: fmt::Debug> fmt::Debug for Matrix<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("layout", self.inner.layout())
            .field("content", &self.inner.as_slice())
            .finish()
    }
}

impl<P> fmt::Debug for MatrixReuseError<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Matrix requires {} elements but buffer has capacity for only {}",
            self.layout.len(),
            self.buffer.capacity()
        )
    }
}

impl<P, Q> fmt::Debug for MapReuseError<P, Q> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.layout {
            Some(layout) => write!(
                f,
                "Mapping canvas requires {} bytes but current buffer has a capacity of {}",
                layout.byte_len(),
                self.buffer.inner.as_capacity_bytes().len(),
            ),
            None => write!(f, "Mapped canvas can not be allocated"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_reuse() {
        let rec = TexelBuffer::<u8>::new(4);
        assert!(rec.capacity() >= 4);
        let layout = Layout::width_and_height(2, 2).unwrap();
        let mut canvas =
            Matrix::from_reused_buffer(rec, layout).expect("Rec is surely large enough");
        canvas
            .reuse(Layout::width_and_height(1, 1).unwrap())
            .expect("Can scale down the image");
        canvas.resize(Layout::width_and_height(0, 0).unwrap());
        canvas
            .reuse(layout)
            .expect("Can still reuse original allocation");
    }
}
