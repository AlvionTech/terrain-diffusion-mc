package com.github.xandergos.terraindiffusionmc.infinitetensor;

import java.util.Arrays;

/**
 * An N-dimensional float array with row-major (C-order) layout.
 * Used as the data container for InfiniteTensor computations.
 */
public class FloatTensor {
    public final int[] shape;
    public final float[] data;
    final int[] strides;

    public FloatTensor(int[] shape) {
        this.shape = shape.clone();
        int total = 1;
        for (int d : shape) total *= d;
        this.data = new float[total];
        this.strides = computeStrides(shape);
    }

    public FloatTensor(int[] shape, float[] data) {
        this.shape = shape.clone();
        this.data = data;
        this.strides = computeStrides(shape);
    }

    static int[] computeStrides(int[] shape) {
        int n = shape.length;
        int[] s = new int[n];
        int stride = 1;
        for (int i = n - 1; i >= 0; i--) {
            s[i] = stride;
            stride *= shape[i];
        }
        return s;
    }

    public int ndim() {
        return shape.length;
    }

    public long byteSize() {
        return (long) data.length * Float.BYTES;
    }

    /**
     * Add values from src into this tensor at a sub-region.
     * dstRegion[d] = {start, stop}, srcRegion[d] = {start, stop}.
     * The region sizes must match in every dimension.
     */
    public void addFrom(FloatTensor src, int[][] dstRegion, int[][] srcRegion) {
        int n = shape.length;
        int[] count = new int[n];
        int total = 1;
        for (int d = 0; d < n; d++) {
            count[d] = dstRegion[d][1] - dstRegion[d][0];
            total *= count[d];
        }
        if (total == 0) return;

        int dstBase = 0;
        int srcBase = 0;
        for (int d = 0; d < n; d++) {
            dstBase += dstRegion[d][0] * strides[d];
            srcBase += srcRegion[d][0] * src.strides[d];
        }

        if (n == 3) {
            int d0Count = count[0], d1Count = count[1], d2Count = count[2];
            int d0DstStride = strides[0], d1DstStride = strides[1], d2DstStride = strides[2];
            int d0SrcStride = src.strides[0], d1SrcStride = src.strides[1], d2SrcStride = src.strides[2];
            for (int i0 = 0; i0 < d0Count; i0++) {
                int dstFlat0 = dstBase + i0 * d0DstStride;
                int srcFlat0 = srcBase + i0 * d0SrcStride;
                for (int i1 = 0; i1 < d1Count; i1++) {
                    int dstFlat1 = dstFlat0 + i1 * d1DstStride;
                    int srcFlat1 = srcFlat0 + i1 * d1SrcStride;
                    for (int i2 = 0; i2 < d2Count; i2++) {
                        data[dstFlat1 + i2 * d2DstStride] += src.data[srcFlat1 + i2 * d2SrcStride];
                    }
                }
            }
        } else if (n == 2) {
            int d0Count = count[0], d1Count = count[1];
            int d0DstStride = strides[0], d1DstStride = strides[1];
            int d0SrcStride = src.strides[0], d1SrcStride = src.strides[1];
            for (int i0 = 0; i0 < d0Count; i0++) {
                int dstFlat0 = dstBase + i0 * d0DstStride;
                int srcFlat0 = srcBase + i0 * d0SrcStride;
                for (int i1 = 0; i1 < d1Count; i1++) {
                    data[dstFlat0 + i1 * d1DstStride] += src.data[srcFlat0 + i1 * d1SrcStride];
                }
            }
        } else if (n == 1) {
            int d0Count = count[0];
            int d0DstStride = strides[0];
            int d0SrcStride = src.strides[0];
            for (int i0 = 0; i0 < d0Count; i0++) {
                data[dstBase + i0 * d0DstStride] += src.data[srcBase + i0 * d0SrcStride];
            }
        } else {
            // N-dimensional fallback avoiding division and modulo per pixel
            int[] current = new int[n];
            int dstFlat = dstBase;
            int srcFlat = srcBase;

            for (int flat = 0; flat < total; flat++) {
                data[dstFlat] += src.data[srcFlat];

                for (int d = n - 1; d >= 0; d--) {
                    current[d]++;
                    dstFlat += strides[d];
                    srcFlat += src.strides[d];
                    if (current[d] < count[d]) {
                        break;
                    }
                    current[d] = 0;
                    dstFlat -= count[d] * strides[d];
                    srcFlat -= count[d] * src.strides[d];
                }
            }
        }
    }

    /**
     * Extract a contiguous sub-region as a new zero-based tensor.
     * region[d] = {start, stop}.
     */
    public FloatTensor slice(int[][] region) {
        int n = shape.length;
        int[] newShape = new int[n];
        for (int d = 0; d < n; d++) {
            newShape[d] = region[d][1] - region[d][0];
        }
        FloatTensor result = new FloatTensor(newShape);
        int[][] dstRegion = new int[n][2];
        for (int d = 0; d < n; d++) {
            dstRegion[d][0] = 0;
            dstRegion[d][1] = newShape[d];
        }
        result.addFrom(this, dstRegion, region);
        return result;
    }

    @Override
    public String toString() {
        return "FloatTensor(shape=" + Arrays.toString(shape) + ")";
    }
}
