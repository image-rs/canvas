// Distributed under The MIT License (MIT)
//
// Copyright (c) 2019, 2020 The `image-rs` developers
use crate::buf::{Buffer, Cog};
use crate::layout::{Element, Matrix};

pub struct Canvas<'buf, Layout> {
    buffer: Cog<'buf>,
    layout: Layout,
}

impl<'buf, Layout> Canvas<'buf, Layout> {
    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.buffer.as_bytes_mut()
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
