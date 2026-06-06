## 2025-02-20 - Unrolling 3x3 Convolution Kernels (Sobel) in Java
**Learning:** In Java, flattening small kernel loops like 3x3 Sobel filters directly into unrolled arithmetic yields a dramatic speedup (often 2.5x to 4x). The overhead from a nested 3x3 loop, including bounds-checking and arithmetic for computing array indexes (e.g. `(r+kr)*pW+(c+kc)` vs pre-calculated row offsets `row0 + c`), is surprisingly large relative to the actual math being performed.
**Action:** When identifying mathematical kernels like Sobel filters in image or heightmap processing loops, unroll the 3x3 array accesses directly, use local row offset variables, and multiply by the inverse (e.g., `* 0.125f`) instead of dividing.

## 2025-02-20 - Optimizing Bilinear Interpolation
**Learning:** In Java image processing, pre-calculating interpolation coordinates/weights outside of the horizontal (inner) loop and caching the arrays saves huge amounts of repetitive float arithmetic and Math.floor calls per pixel.
**Action:** When resizing arrays/images or doing interpolation (e.g., bilinearResize), allocate small 1D arrays for the inner loop coordinates (`c0`, `c1`, `wc`, etc.) beforehand.
