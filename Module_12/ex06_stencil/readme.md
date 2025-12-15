# Exercise 06: Stencil Operations (Box Filter)

## Goal
Implement a 2D convolution (Box Filter) using shared memory to handle the "halo" (boundary) pixels efficiently.

## Learning Objectives
1.  Understand the Stencil pattern (accessing neighbors).
2.  Handle the "halo" problem: threads need data from neighboring blocks.
3.  Load a larger tile into shared memory than the compute tile.
4.  Compare Global Memory vs. Shared Memory implementation performance (conceptually).

## Practical Motivation
Convolution is the core of CNNs and image filtering (blur, sharpen, edge detect). A 3x3 filter requires reading 9 pixels for every 1 pixel output. In Global Memory, this means huge bandwidth overlap. By loading the image block + border (halo) into Shared Memory, we read each pixel from Global Memory only once (ideally).

## Theory: Halo Handling
If a block computes output for a $16 \times 16$ tile, and we use a $3 \times 3$ filter (radius 1), we need input data of size $(16+2) \times (16+2) = 18 \times 18$.
Strategies:
1.  **Thread expansion:** Launch $18 \times 18$ threads, but only inner $16 \times 16$ write output.
2.  **Collaborative loading:** Launch $16 \times 16$ threads. Some threads load multiple pixels (center + boundary).

We will use **Strategy 1** (simpler logic) or **Strategy 2** (classic optimization). Let's stick to a simpler version of Strategy 2 where we load the corresponding pixel, and then handle boundary loading. Actually, the easiest efficient way is:
-   Define Block Size: $16 \times 16$.
-   Define Shared Memory Size: $(16+2) \times (16+2)$.
-   Map thread $(tx, ty)$ to output $(col, row)$.
-   Load center pixel into shared memory at $(tx+1, ty+1)$.
-   If thread is at boundary of block, load the "halo" pixels into shared memory.
-   Sync.
-   Compute.

## Step-by-Step Instructions

### Task 1: The Kernel (`src/box_filter.cu`)
1.  Define `BLOCK_SIZE` 16 and `RADIUS` 1 (3x3 filter).
2.  Shared memory: `__shared__ unsigned char smem[BLOCK_SIZE + 2*RADIUS][BLOCK_SIZE + 2*RADIUS]`.
3.  Global indices: `gx, gy`.
4.  Local indices within shared memory: `lx = threadIdx.x + RADIUS`, `ly = threadIdx.y + RADIUS`.
5.  **Load Data:**
    -   Load central pixel `input[gy][gx]` to `smem[ly][lx]`.
    -   Check if thread is on left/right/top/bottom edge of block, and if so, load corresponding halo pixels from global memory into `smem`.
    -   Handle image boundaries (clamp or zero).
6.  `__syncthreads()`.
7.  **Compute:**
    -   Sum `smem[ly+dy][lx+dx]` for dy, dx in -1 to 1.
    -   Average (divide by 9).
    -   Write to `output[gy][gx]`.

### Task 2: Host Code
1.  Load image (or generate synthetic).
2.  Launch kernel.
3.  Verify.

## Common Pitfalls
-   **Halo Loading Logic:** It's easy to miss a corner case. Ensure corners (diagonal neighbors) are loaded.
-   **Indexing:** Mixing up local (shared mem) and global indices.

## Code Hints
```cpp
// Simplified loading: 
// Map linear thread index to linear shared mem index loop?
// Or just let threads load their own + neighbors if needed.

// Easiest approach for learning:
// Just load the main pixel. Then have specific `if` blocks:
if (threadIdx.x < RADIUS) { /* Load left halo */ }
if (threadIdx.x >= BLOCK_SIZE - RADIUS) { /* Load right halo */ }
// ... same for Y ...
// ... AND corners ...
```

## Verification
Output image should be blurred.
