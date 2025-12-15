# Exercise 04: 2D Kernels (Image Inversion)

## Goal
Learn how to map 2D data (images) to the CUDA grid and block hierarchy. You will implement a kernel that inverts the colors of an image.

## Learning Objectives
1.  Understand `dim3` for defining 2D grid and block dimensions.
2.  Calculate 2D global indices (`x`, `y`) from thread and block IDs.
3.  Map 2D indices to linear memory addresses (row-major order).
4.  Handle image strides (pitch) if necessary (simplified here to continuous arrays).

## Practical Motivation
Images are 2D arrays. While 1D vector addition is simple, processing images requires understanding how to cover a 2D plane with 2D blocks of threads. This pattern is the foundation for almost all image processing on GPU (filtering, resizing, warping).

## Theory: 2D Indexing
CUDA allows `gridDim` and `blockDim` to be 3D. For images, we use `.x` (width) and `.y` (height).
-   **x index:** `col = blockIdx.x * blockDim.x + threadIdx.x`
-   **y index:** `row = blockIdx.y * blockDim.y + threadIdx.y`
-   **Linear Index:** `idx = row * width + col` (assuming 1 channel). For 3 channels (RGB), it's often `idx = (row * width + col) * 3 + channel`.

## Step-by-Step Instructions

### Task 1: The Kernel (`src/image_invert.cu`)
1.  Define a kernel `invertImageKernel(unsigned char* input, unsigned char* output, int width, int height, int channels)`.
2.  Calculate `x` and `y`.
3.  Check bounds: `if (x < width && y < height)`.
4.  Compute linear index. Note that each pixel has `channels` bytes.
    -   Option A: One thread per pixel, loop over channels inside.
    -   Option B: One thread per channel (treat image as `width * channels` wide).
    -   **Let's use Option A for conceptual clarity:**
        ```cpp
        int pixelIdx = (y * width + x) * channels;
        for(int c=0; c<channels; ++c) {
            output[pixelIdx + c] = 255 - input[pixelIdx + c];
        }
        ```

### Task 2: Host Code
1.  Load an image using OpenCV (`cv::imread`).
2.  Allocate device memory (`cudaMalloc`) for input and output.
3.  Copy image data to device.
4.  Define dimensions:
    ```cpp
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    ```
5.  Launch kernel.
6.  Copy result back and save/display.

## Common Pitfalls
-   **Integer Division:** Remember ceiling division for grid size.
-   **Channels:** Don't forget RGB images have 3 bytes per pixel. If you calculate `x` up to `width`, ensure your memory access accounts for the stride of 3.
-   **Memory Alignment:** OpenCV `Mat` rows are often aligned (padded). For simplicity, use `isContinuous()` check or clone the matrix to ensure it is continuous before copying.

## Code Hints
```cpp
// Kernel
__global__ void invert(unsigned char* img, int w, int h, int c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < w && y < h) {
        int idx = (y * w + x) * c;
        for (int k = 0; k < c; ++k) {
            img[idx + k] = 255 - img[idx + k];
        }
    }
}
```

## Verification
The program should produce an "inverted.jpg" that looks like a photo negative of the input.
