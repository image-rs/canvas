// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019-2025 The `image-rs` developers
//! An in-memory buffer for image data.
//!
//! It acknowledges that in a typical image pipeline there exist multiple valid but competing
//! representations:
//!
//! - A reader that decodes bytes into pixel representations loads a texel type from raw bytes,
//!   decodes it, and stores it into a matrix as another type. For this it may primarily have to
//!   work with unaligned data.
//! - Library transformation typically consume data as typed slices: `&[Rgb<u16>]`.
//! - SIMD usage as well as GPU buffers may require that data is passed as slice of highly aligned
//!   data: `&[Simd<u16, 16>]`.
//! - The workload of transformations must be shared between workers that need to access portions
//!   of the image that do not overlap in a _logical_ sense, eg. squares of 8x8 pixels each, but
//!   that do overlap in the sense of the physical layout where references to contiguous bytes
//!   would alias.
//! - Planar layouts split channels to different aligned portions in a larger allocation, each such
//!   plane having its own internal layout (e.g. with subsampling or heterogeneous channel types).
//!
//! This crate offers the vocabulary types for buffers, layouts, and texels to *safely* cover many
//! of the above use cases.
//!
//! Table of contents:
//!
//! 1. [General Usage](#general-usage)
//!     1. [Planar layouts](#planar-layouts)
//!     2. [Concurrent editing](#concurrent-editing)
//! 2. [Image data transfer](#image-data-transfer)
//!
//! # General Usage
//!
//! In a simple example we will allocate a matrix representing 16-bit rgb pixels, then transform
//! those pixels into network endian byte order in-place, and encode that as a PNM image according
//! to the `netpbm` implementation. Note how we can convert quite freely between different ways of
//! viewing the data.
//!
#![cfg_attr(not(miri), doc = "```")]
#![cfg_attr(miri, doc = "```no_run")] // too expensive and pointless
//! use image_texel::{Matrix, Image, texels::U16};
//! // Editing is simpler with the `Matrix` type.
//! let mut matrix = Matrix::<[u16; 4]>::with_width_and_height(400, 400);
//!
//! // Draw a bright red line.
//! for i in 0..400 {
//!     // Assign color as u8-RGBA
//!     matrix[(i, i)] = [0xFFFF, 0x00, 0x00, 0xFFFF];
//! }
//!
//! // General operations are simpler with the `Image` type.
//! let mut image = Image::from(matrix);
//!
//! // Encode components to network endian by iterating one long slice.
//! image
//!     .as_mut_texels(U16)
//!     .iter_mut()
//!     .for_each(|p| *p = p.to_be());
//!
//! let pam_header = format!("P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 65535\nTUPLETYPE
//! RGB_ALPHA\nENDHDR\n", 400, 400);
//!
//! let mut network_data = vec![];
//! network_data.extend_from_slice(pam_header.as_bytes());
//! network_data.extend_from_slice(image.as_bytes());
//! ```
//!
//! It is your responsibility to ensure the different layouts are compatible, although the library
//! tries to help by a few standardized layout traits and aiming to make correct methods simpler
//! to use than methods with preconditions. You are allowed to mix it up, in fact this is
//! encouraged for reusing buffers. However, of course when you do you may be a little surprised
//! with the data you find—the data is not zeroed.
//!
#![cfg_attr(not(miri), doc = "```")]
#![cfg_attr(miri, doc = "```no_run")] // too expensive and pointless
//! use image_texel::{Image, layout, texels};
//! let matrix_u8 = layout::Matrix::from_width_height(texels::U8, 32, 32).unwrap();
//! let mut image = Image::new(matrix_u8);
//! // Fill with some arbitrary data..
//! image.shade(|x, y, pix| *pix = (x * y) as u8);
//!
//! let matrix_u32 = layout::Matrix::from_width_height(texels::U32, 32, 32).unwrap();
//! let reuse = image.with_layout(matrix_u32);
//! // Huh? The 9th element now aliases the start of the second row before.
//! assert_eq!(reuse.as_slice()[8], u32::to_be(0x00010203));
//! ```
//!
//! ## Planar layouts
//!
//! The contiguous image allocation of `image_texel`'s buffers can be split into _planes_ of data.
//! A plane starts at any arbitrary byte boundary that is aligned to the maximum alignment of
//! texels, which is to say that algorithms that can be applied to a buffer as a whole may can also
//! interact with a plane. A more precise meaning of a plane depends entirely on the layout
//! containing it.
//!
//! The `layout` module defines a some containers that represent planar layouts built from their
//! individual components. This relationship is expressed as [`PlaneOf`][`layout::PlaneOf`].
//!
//! ```
//! use image_texel::{Image, layout, texels};
//!
//! let matrix_rgb = layout::Matrix::from_width_height(texels::U16.array::<3>(), 32, 32).unwrap();
//! let matrix_a = layout::Matrix::from_width_height(texels::U8, 32, 32).unwrap();
//!
//! // An RGB plane, followed by its alpha mask.
//! let image = Image::new(layout::PlaneBytes::new([matrix_rgb.into(), matrix_a.into()]));
//!
//! // Split them into planes via `impl PlaneOf<PlaneBytes> for usize`
//! let [rgb, a] = image.as_ref().into_planes([0, 1]).unwrap();
//! ```
//!
//! ## Concurrent editing
//!
//! This becomes especially important for shared buffers. We have two kinds of buffers that can be
//! edited by multiple owners. A [`CellImage`][`image::CellImage`] is shared between owners but can
//! not be sent across threads. Crucially, however, each duplicate owner may use any layout type
//! and layout value that is chooses. This buffer behaves very similar to an [`Image`] except that
//! its operations do not reallocate the buffer as this would remove the sharing.
//!
#![cfg_attr(not(miri), doc = "```")]
#![cfg_attr(miri, doc = "```no_run")] // too expensive and pointless
//! use image_texel::{image::CellImage, layout, texels};
//!
//! let matrix = layout::Matrix::from_width_height(texels::U32, 32, 32).unwrap();
//! // An RGB plane, followed by its alpha mask.
//! let image_u32 = CellImage::new(matrix);
//!
//! // Another way to refer to that image may be interpreting each u32 as 4 channels.
//! let matrix = layout::Matrix::from_width_height(texels::U8.array::<4>(), 32, 32).unwrap();
//! let image_rgba = image_u32.clone().try_with_layout(matrix)?;
//!
//! // Let's pretend we have some async thread pool that reads into the image and works on it:
//! # fn spawn_local<T>(_: T) {} // hey this is just for show. The types are not accurate
//!
//! // We do not care about the component type in this function.
//! async fn fill_image(matrix: CellImage<layout::MatrixBytes>) {
//!     loop {
//!       // .. refill the buffer by reads whenever we are signalled.
//!     }
//! }
//!
//! async fn consume_buffer(matrix: CellImage<layout::Matrix<[u8; 4]>>) {
//!  // do some work on each new image.
//! }
//!
//! spawn_local(fill_image(image_u32.decay()));
//! spawn_local(consume_buffer(image_rgba));
//!
//! # Ok::<_, image_texel::BufferReuseError>(())
//! ```
//!
//! An [`AtomicImage`][`image::AtomicImage`] can be shared between threads but its buffer
//! modifications are not straightforward. In simplistic terms, it allows modifying disjunct parts
//! of images concurrently but you should synchronize all modifications on the same part, e.g. via
//! a lock, when the result values of those modifications is important. It always maintains
//! soundness guarantees with such modifications.
//!
#![cfg_attr(not(miri), doc = "```")]
#![cfg_attr(miri, doc = "```no_run")] // too expensive and pointless
//! use image_texel::{image::AtomicImage, layout, texels};
//!
//! let matrix = layout::Matrix::from_width_height(texels::U8.array::<4>(), 1920, 1080).unwrap();
//! let image = AtomicImage::new(matrix);
//!
//! # struct PretendThisDefinesABlock;
//! # fn matrix_tiles(_: &impl layout::MatrixLayout) -> Vec<PretendThisDefinesABlock> { vec![] }
//! # fn fill_block(_: AtomicImage<layout::Matrix<[u8; 4]>>, _: PretendThisDefinesABlock) {}
//!
//! std::thread::scope(|s| {
//!     // Decouple our work into tiles of the original image.
//!     for tile in matrix_tiles(image.layout()) {
//!         let work_image = image.clone();
//!         s.spawn(move || {
//!             fill_block(work_image, tile);
//!         });
//!     }
//! });
//!
//! # Ok::<_, image_texel::BufferReuseError>(())
//! ```
//!
//! # Image data transfer
//!
//! By data transfer we mean writing and reading semantically meaningful parts of an image into
//! *unaligned* external byte buffers. In particular, when that part is one plane out of a larger
//! number of image planes. A common case might be the initialization of an alpha mask. If you do
//! not care about layouts of your data at all then you may get a slice or mutable slice to your
//! data from any [`Image`]—or a `Cell<u8>` and [`texels::AtomicSliceRef`] for the shared data
//! equivalents—and interact through those.
//!
//! Otherwise, the primary module for interacting with data is [`image::data`][`image::data`] as
//! well as the [`as_source`][`Image::as_source`] and [`as_target`][`Image::as_target`] methods
//! that view any applicable image container as a byte data container. Let's see the example before
//! where an image contains two planes and we write the alpha mask:
//!
//! ```
//! # fn testing() -> Option<()> {
//! use image_texel::{Image, image::data, layout, texels};
//!
//! let matrix_rgb = layout::Matrix::from_width_height(texels::U16.array::<3>(), 32, 32)?;
//! let matrix_a = layout::Matrix::from_width_height(texels::U8, 32, 32)?;
//! // An RGB plane, followed by its alpha mask.
//! let mut image = Image::new(layout::PlaneBytes::new([matrix_rgb.into(), matrix_a.into()]));
//!
//! // Grab a mutable reference to its alpha plane.
//! let [alpha] = image.as_mut().into_planes([1]).unwrap();
//! # let ref your_32x32_byte_mask = [0u8; 32 * 32];
//! let bytes = data::DataRef::new(your_32x32_byte_mask);
//!
//! // Ignore the alpha layout, just write our bits in there. This gives
//! // us back an image with a `Bytes` layout which we do not need.
//! let _ = bytes.as_source().write_to_mut(alpha.decay())?;
//! # Some::<()>(())
//! # }
//! ```
//!
//! If you want to modify the layout of the image you're assigning to, then you would instead use
//! [`assign`][`Image::assign`] which will modify the layout in-place instead of returning a
//! container with a new type. For this you interpret the data source with the layout you want to
//! apply:
//!
//! ```
//! # fn testing() -> Option<()> {
//! use image_texel::{Image, image::data, layout, texels};
//! // An initially empty image.
//! let mut image = Image::new(layout::Matrix::empty(texels::U8));
//!
//! let matrix_a = layout::Matrix::from_width_height(texels::U8, 16, 16)?;
//! # let ref your_16x16_ico = [0u8; 16 * 16];
//! let bytes = data::DataRef::with_layout_at(your_16x16_ico, matrix_a, 0)?;
//!
//! image.assign(bytes.as_source());
//! assert_eq!(image.layout().byte_len(), 256);
//! # Some::<()>(())
//! # }
//! ```
// Be std for doctests, avoids a weird warning about missing allocator.
#![cfg_attr(not(doctest), no_std)]
// The only module allowed to be `unsafe` is `texel`. We need it however, as we have a custom
// dynamically sized type with an unsafe alignment invariant.
#![deny(unsafe_code)]
extern crate alloc;

mod buf;
pub mod image;
pub mod layout;
mod matrix;
mod rec;
mod stride;
mod texel;

pub use self::image::Image;
pub use self::matrix::{Matrix, MatrixReuseError};
pub use self::rec::{BufferReuseError, TexelBuffer};
pub use self::texel::{AsTexel, Texel};

/// Constants for predefined texel types.
///
/// Holding an instance of `Texel<T>` certifies that the type `T` is compatible with the texel
/// concept, that is: its alignment requirement is *small* enough, its size is non-zero, it does
/// not contain any padding, and it is a plain old data type without any inner invariants
/// (including sharing predicates). These assertions allow a number of operations such as
/// reinterpreting aligned byte slices, writing to and read from byte buffers, fallible cast to
/// other texels, etc.
///
/// For types that guarantee the property, see [`AsTexel`] and its impls.
///
/// # Extending
///
/// The recommended method of extending this with a custom type is by implementing `bytemuck::Pod`
/// for this type. This applies a number of consistency checks.
///
/// ```rust
/// use bytemuck::{Pod, Zeroable};
/// use image_texel::{AsTexel, Texel};
///
/// #[derive(Clone, Copy, Pod, Zeroable)]
/// #[repr(C)]
/// struct Rgb(pub [u8; 3]);
///
/// impl AsTexel for Rgb {
///     fn texel() -> Texel<Rgb> {
///         Texel::for_type().expect("verified by bytemuck and image_texel")
///     }
/// }
///
/// impl Rgb {
///     const TEXEL: Texel<Self> = match Texel::for_type() {
///         Some(texel) => texel,
///         None => panic!("compilation error"),
///     };
/// }
/// ```
pub mod texels {
    pub use crate::texel::constants::*;
    pub use crate::texel::IsTransparentWrapper;
    pub use crate::texel::MaxAligned;
    pub use crate::texel::MaxAtomic;
    pub use crate::texel::MaxCell;

    pub use crate::buf::{
        atomic_buf, buf, cell_buf, AtomicBuffer, AtomicRef, AtomicSliceRef, Buffer, CellBuffer,
        TexelRange,
    };
}
