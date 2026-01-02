# Endianess

When we're composing memory for use by a memory-interfaced secondary processor
(e.g. a GPU or the VGA framebuffer) we must match bit and byte order of data to
their requirement. Other libraries we can treat a lot of data as host ordered
by their nominal Rust type but for generality this representation is only a
temporary view when processing such data.

We treat any necessary transformation as part of a texel fetch and store
operation. Recall that [`SampleBits`][`crate::layout::SampleBits`] lets us
demux the bits of a texel into individual typed components. The endianess
defines an algorithm by which those component bits in our host interpretation
map to the bits of the underlying texel. Now our main problem is in ensuring
that the data structure representation is simple—in particular: consistent with
the possible `bits` and `parts` of a texel, avoiding duplicated and
contradictory data—while allowing the flexibility of real world scenarios.

The Linux framebuffer for instance describes itself with information where each
individual channel may have flipped bit order and have differing bit widths.
Also the order of color fields within a pixel is described in a host-integer
layout, so in a of a 4-byte RGBA8 texel the bit numbered 0 may be the least
significant bit when represented as a `u32`.

```rust
#[repr(C)]
pub struct fb_bitfield {
    pub offset: u32,    /* beginning of bitfield        */
    pub length: u32,    /* length of bitfield           */
    pub msb_right: u32, /* != 0 : Most significant bit is */
}

#[repr(C)]
#[non_exhaustive]
pub struct fb_var_screeninfo {
    // [… elided]
    pub bits_per_pixel: u32,
    // [… elided]
    pub red: fb_bitfield,   /* bitfield in fb mem if true color, */
    pub green: fb_bitfield, /* else only length is significant */
    pub blue: fb_bitfield,
    pub transp: fb_bitfield, /* transparency                 */
    // [… elided]
}
```

We design this with two orthogonal fields, for one the byte-for-byte order when
loading a texel from the backing storage into the byte array (indexed as 0 as
the highest bit of the left-most byte) of a processing buffer and secondly a
per-field bit order that should be adjusted when extracting the bits into their
own type.
