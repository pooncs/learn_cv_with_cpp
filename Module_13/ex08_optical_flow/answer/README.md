# Module 13 Exercise 08: Optical Flow

## Overview

This exercise implements the **Lucas-Kanade Optical Flow** algorithm for sparse feature tracking.
We implement the algorithm from scratch to understand the underlying mathematics (spatial gradients, temporal derivatives, iterative refinement) and compare it with OpenCV's production-ready `calcOpticalFlowPyrLK`.

## Key Concepts

- **Optical Flow Equation**: $I_x u + I_y v + I_t = 0$
- **Lucas-Kanade Method**: Solves the aperture problem by assuming constant flow in a local window.
  - Minimizes $\sum (I(x) - J(x+u))^2$
  - Uses spatial gradients of the template (previous image) for efficiency (Inverse Compositional approximation).
- **Iterative Refinement**: Newton-Raphson iteration to handle displacements larger than 1 pixel (though limited without pyramids).
- **Pyramidal Implementation**: OpenCV's implementation uses image pyramids to handle large motions.

## Implementation Details

- **`OpticalFlowTracker` Class**:
  - `computeFlowCustom`: Single-level iterative Lucas-Kanade implementation.
    - Computes gradients $I_x, I_y$ using Sobel.
    - Builds structure tensor $G = \sum \nabla I \nabla I^T$.
    - Iteratively solves for displacement $\delta u = G^{-1} \sum \nabla I (I - J)$.
  - `computeFlowOpenCV`: Wrapper for `cv::calcOpticalFlowPyrLK`.

## Build Instructions

This project uses **Conan** for dependency management (OpenCV).

1. Install dependencies:
   ```bash
   conan install . --output-folder=build --build=missing
   ```

2. Build the project:
   ```bash
   cd build
   cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```

3. Run the example:
   ```bash
   ./optical_flow
   ```

## Note on Dependencies

If you encounter version conflicts with `libtiff` or `libjpeg` when installing OpenCV via Conan, you may need to adjust the `conanfile.txt` or force specific versions. A common resolution is to ensure consistent versions of `libtiff` and `libjpeg` are used across the dependency graph.
