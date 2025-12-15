# Exercise 09: NVIDIA Performance Primitives (NPP)

## Goal
Use the NVIDIA Performance Primitives (NPP) library for optimized image processing functions without writing kernels.

## Learning Objectives
1.  Understand the NPP library structure (`nppc`, `nppi`, `npps`).
2.  Use `nppiMalloc` and `nppiFree` for image memory.
3.  Perform an image resizing operation using `nppiResize`.
4.  Understand NPP strides (pitch) and ROI (Region of Interest).

## Practical Motivation
NVIDIA provides a vast library of hand-tuned image processing functions (resizing, filtering, color conversion, statistics). For standard operations, NPP is often faster and much easier to maintain than custom kernels. It's the "OpenCV of GPU".

## Theory: Strides and ROI
-   **Step/Stride:** Row alignment in bytes. NPP functions require the stride.
-   **NppiSize:** Struct `{width, height}` defining image dimensions.
-   **NppiRect:** Struct `{x, y, width, height}` defining ROI.

## Step-by-Step Instructions

### Task 1: Setup (`src/npp_resize.cpp`)
1.  Create a synthetic host image (single channel 8-bit, e.g., gradient).
2.  Allocate device memory using `nppiMalloc_8u_C1` (returns pointer and step).

### Task 2: Resize
1.  Define source size and ROI.
2.  Define destination size (e.g., 0.5x scale).
3.  Allocate destination memory.
4.  Call `nppiResize_8u_C1R`.
    -   Needs: src ptr, src step, src ROI, dst ptr, dst step, dst ROI, interpolation mode (`NPPI_INTER_LINEAR`).

### Task 3: Verification
1.  Copy result back to host.
2.  Check dimensions and pixel values (basic check).

## Common Pitfalls
-   **Linking:** NPP is split into many libraries (`nppc`, `nppi`, `nppig`, etc.). You often need to link multiple.
-   **Step Size:** Do not assume `step == width`. Always use the step returned by `nppiMalloc`.
-   **Pointer Arithmetic:** If using raw `cudaMalloc`, ensure you calculate pitch correctly. `nppiMalloc` is safer for NPP functions.

## Code Hints
```cpp
#include <npp.h>

Npp8u *d_src, *d_dst;
int srcStep, dstStep;
d_src = nppiMalloc_8u_C1(srcW, srcH, &srcStep);
d_dst = nppiMalloc_8u_C1(dstW, dstH, &dstStep);

NppiSize srcSize = {srcW, srcH};
NppiRect srcROI = {0, 0, srcW, srcH};
NppiSize dstSize = {dstW, dstH};
NppiRect dstROI = {0, 0, dstW, dstH};

nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                  d_dst, dstStep, dstSize, dstROI,
                  NPPI_INTER_LINEAR);
```

## Verification
Output "PASS" if the operation completes without error and output image has correct dimensions.
