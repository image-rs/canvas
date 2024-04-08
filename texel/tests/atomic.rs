use image_texel::texels::{AtomicBuffer, U32};
use std::{mem, thread};

#[test]
fn mapping_atomics_parallel() {
    const LEN: usize = 128;
    let buffer = AtomicBuffer::new(LEN * mem::size_of::<u32>());
    // And receive all the results in this shared copy of our buffer.
    let output_tap = buffer.clone();

    const SPLIT_MAX: usize = 1 << 6;
    // Proxy for whether we run with optimization. Makes execution time bearable.
    #[cfg(debug_assertions)]
    const REPEAT: usize = 1 << 6;
    #[cfg(not(debug_assertions))]
    const REPEAT: usize = 1 << 12;

    for split in 0..SPLIT_MAX {
        // We want the modifying loops to overlap as much as possible for the strongest test, so
        // ensure they do not run early.
        let barrier = &std::sync::Barrier::new(2);

        // Concurrently and repeatedly increment non-overlapping parts of the image.
        thread::scope(|join| {
            let img_a = buffer.clone();
            let img_b = buffer.clone();

            join.spawn(move || {
                let _ = barrier.wait();
                for _ in 0..REPEAT {
                    img_a.map_within(..split, 0, |n: u32| n + 1, U32, U32);
                }
            });

            join.spawn(move || {
                let _ = barrier.wait();
                for _ in 0..REPEAT {
                    img_b.map_within(split.., split, |n: u32| n + 1, U32, U32);
                }
            });
        });
    }

    // Each individual `u32` has been incremented precisely as often as each other. Since the
    // individual transforms are synchronized with thread-scope and within they do not overlap with
    // each other, we must expect that the values have each been touched precisely how we intended
    // them to.
    let expected = (SPLIT_MAX * REPEAT) as u32;
    assert_eq!(
        output_tap.to_owned().as_texels(U32)[..LEN].to_vec(),
        (0..LEN as u32).map(|_| expected).collect::<Vec<_>>()
    );
}
