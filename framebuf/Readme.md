# image-framebuf

Provides a planar, colored, convertible frame buffer based on `image-canvas`.

## Goals

The buffer should be free of IO, yet provide a rich API for a color image
buffer: that is, a buffer with _correct_ color spaces, color and sample
representations at various bit depths, that can be effectively and efficiently
semantically manipulated on a CPU. Provide a lingua franca for passing portions
of an image to other routines.

## Non-Goals

Do not solve IO, bindings for reading into portions can be written on top of
the exposed methods for mutating and referencing parts of the frame.

Do not provide GPU bindings.
