# Comprehensive C++17 Computer Vision Curriculum

## Overview
This curriculum is designed to take a Computer Vision engineer from Intern to Principal level, focusing on modern C++17 standards, performance optimization, and industry-standard libraries.

## Prerequisites
- Basic C++ knowledge
- Calculus and Linear Algebra fundamentals
- Linux/Windows development environment

## Module Structure
Each module contains **10 independent exercises** progressing from simple to complex. Each independent exercise should include the following:
* todo/ with complete project scaffolding:
- include/ for headers
- src/ for implementation stubs
- test/ for verification
- CMakeLists.txt for build configuration
* answer/ with complete solution code
- All code files include:
* Detailed step-by-step comments
* Proper C++17 syntax and best practices
---

## Exercise Components
- Detailed readme.md for each exercise containing:
* Clear exercise goal and learning objectives (each exercise should not take more than 1-2hr to complete at the most advance level)
* Practical motivation and real-world applications
* Technical background with step-by-step algorithm explanations
* Common pitfalls and debugging strategies
* Specific implementation tasks with milestones
* Code hints including recommended functions and usage examples

## Prerequisites
Use conan as the package manager. Use this build command to install the dependencies for MSVC, otherwise clang can also be used (installed in C:\Program Files\LLVM):
conan install . -s compiler.cppstd=17 --output-folder=build --build=missing --settings=build_type=Release
cmake --preset conan-default
cmake --build build --config Release 
Some external lib can be found here: C:\Users\hmgics\projects\midas\3rdParty

## Phase 1: Foundations (Intern - Associate)

### Module 01: Modern C++17 Core for CV
**Goal:** Master memory management and modern syntax essential for high-performance CV.
1.  **Auto & Type Inference:** Refactor legacy code using `auto` and `decltype` to understand type deduction rules.
2.  **STL Containers Performance:** Benchmark `std::vector` vs `std::list` vs `std::deque` for pixel buffer storage.
3.  **Structured Bindings:** Use structured bindings to unpack `std::tuple` and `struct` return types from a mock API.
4.  **Modern Error Handling:** Replace error codes with `std::optional` and `std::expected` (or variant-based result).
5.  **Type-Safe Unions:** Implement a shape container using `std::variant` and `std::visit`.
6.  **Lambda Expressions:** Write complex sorting comparators and filters for a list of detected objects.
7.  **Smart Pointers I (Unique):** Implement a factory function returning `std::unique_ptr` to manage sensor drivers.
8.  **Smart Pointers II (Shared):** Simulate a scene graph where nodes share resources using `std::shared_ptr` and `std::weak_ptr`.
9.  **Move Semantics:** Optimize a heavy `Matrix` class to support move construction and assignment.
10. **RAII Wrapper:** Create a `FileHandle` or `CudaStream` wrapper that automatically cleans up resources (The "Rule of Five").
11.  **Unit Testing Basics:** Write tests for a math utility using GoogleTest.
12.  **Scoped Timer:** Implement RAII timing with `std::chrono` for profiling.
13.  **Minimal Logger:** Replace `std::cerr` with a simple leveled logger.
14.  **File I/O & Serialization:** Read/write binary buffers and JSON configs using `nlohmann_json`.

### Module 02: Linear Algebra & Geometry with Eigen
**Goal:** Understand mathematical transformations programmatically.
1.  **Matrix Initialization:** Initialize dynamic and fixed-size Eigen matrices from arrays and files.
2.  **Basic Arithmetic:** Implement element-wise operations vs. matrix multiplication.
3.  **Block Operations:** Extract and manipulate Regions of Interest (ROI) using `.block()`.
4.  **Linear Solvers:** Solve $Ax=b$ for camera calibration using LLT and LDLT decomposition.
5.  **Eigen Decomposition:** Compute principal axes of a 2D point cloud using Eigenvalues.
6.  **Rotation Representations:** Convert between Rotation Matrices, Euler Angles, and Axis-Angle.
7.  **Quaternions:** Implement robust rotation interpolation (SLERP) using Eigen Quaternions.
8.  **Rigid Body Transforms:** Construct $4 \times 4$ transformation matrices (SE3) to chain robot joint poses.
9.  **Least Squares:** Fit a line/plane to noisy 3D points using SVD.
10. **Geometry Project:** Implement a full 3D coordinate transformer class for camera-to-world projection.

### Module 03: Image Processing Essentials (OpenCV)
**Goal:** Deep dive into image memory layout and basic algorithms.
1.  **Mat Internals:** Create a `cv::Mat` manually and inspect `step`, `data`, and reference counting.
2.  **Pixel Access:** Benchmark `.at<>`, `ptr<>`, and iterator access methods.
3.  **Color Spaces:** Implement manual RGB-to-Grayscale and RGB-to-HSV conversion kernels.
4.  **Convolution:** Implement a 3x3 convolution from scratch and verify against `cv::filter2D`.
5.  **Morphology:** Implement Dilation and Erosion manually on binary images.
6.  **Histograms:** Compute and visualize image histograms; implement histogram equalization.
7.  **Thresholding:** Implement global and adaptive (local mean) thresholding.
8.  **Gradients:** Compute Sobel X and Y derivatives and gradient magnitude/orientation.
9.  **Canny Edge:** Step-by-step implementation of non-maximum suppression.
10. **Custom Filter:** Build an optimized separable Gaussian filter.

### Module 04: Camera Models & Calibration
**Goal:** Connect the 3D world to 2D images.
1.  **Pinhole Projection:** Implement a function mapping $(X,Y,Z) \to (u,v)$ given intrinsics $K$.
2.  **Inverse Projection:** Map pixel $(u,v)$ to a 3D ray.
3.  **Distortion Models:** Implement Radial (k1, k2) and Tangential (p1, p2) distortion functions.
4.  **Undistortion:** Create a lookup table (map) to undistort an image.
5.  **Homography:** Compute the homography matrix between two planes (4 point correspondence).
6.  **Perspective Warp:** Apply perspective transformation to "rectify" a slanted document image.
7.  **Chessboard Detection:** Use OpenCV to detect corners in a calibration pattern.
8.  **PnP Solver:** Solve Perspective-n-Point to find camera pose from 3D-2D matches.
9.  **Stereo Rectification:** Compute rectification transforms given two camera matrices.
10. **Calibration Tool:** Develop a complete application that calibrates a camera from a video stream.

### Module 05: Basic 3D Vision (Open3D)
**Goal:** Working with depth and point clouds.
1.  **Point Cloud IO:** Read/Write PLY and XYZ files manually.
2.  **Depth to Cloud:** Convert a depth image + intrinsics into a 3D point cloud.
3.  **Downsampling:** Implement Voxel Grid downsampling.
4.  **Normal Estimation:** Compute surface normals using k-nearest neighbors and PCA.
5.  **Outlier Removal:** Implement Statistical Outlier Removal (SOR).
6.  **ICP Alignment:** Use Open3D to align two point clouds (Point-to-Point).
7.  **Colored Clouds:** Map RGB texture onto a generated point cloud.
8.  **Mesh Generation:** Create a mesh from points using Poisson Surface Reconstruction.
9.  **Octrees:** Build an Octree from a point cloud for efficient searching.
10. **3D Viewer:** Create a visualization tool supporting multiple clouds and camera viewpoints.

---

## Phase 2: Intermediate (Associate - Senior)
### Module 06: Feature Detection & Matching
**Goal:** Understanding sparse features for tracking and recognition.
1.  **Harris Corners:** Implement Harris Corner Response calculation manually.
2.  **FAST Keypoints:** Implement the FAST corner detection test.
3.  **Descriptor Extraction:** Extract ORB descriptors for detected keypoints.
4.  **Brute Force Matching:** Implement Nearest Neighbor matching with Hamming distance.
5.  **Ratio Test:** Filter matches using Lowe's Ratio Test.
6.  **FLANN Matching:** Use KD-Trees for approximate nearest neighbor search.
7.  **RANSAC Homography:** Implement RANSAC to robustly estimate homography from noisy matches.
8.  **Epipolar Geometry:** Compute the Fundamental Matrix $F$ from matches.
9.  **Triangulation:** Triangulate 3D points from stereo matches.
10. **Panorama Stitcher:** Pipeline to detect, match, warp, and blend two images into a mosaic.

### Module 07: Multithreading & Real-time Systems
**Goal:** Building responsive and high-throughput pipelines.
1.  **std::thread:** Launch parallel tasks to process image blocks.
2.  **Data Races:** Create a race condition and fix it using `std::mutex`.
3.  **Deadlocks:** Simulate a deadlock scenario and resolve it using `std::lock` / `std::scoped_lock`.
4.  **Condition Variables:** Implement a thread-safe Queue for frame buffering.
5.  **Producer-Consumer:** Build a pipeline where one thread captures and another processes.
6.  **std::async & Futures:** Run independent algorithms (e.g., Face Detect + Color Hist) in parallel.
7.  **Thread Pools:** Implement a fixed-size thread pool to manage worker tasks.
8.  **Atomics:** Use `std::atomic` for lock-free counters and flags.
9.  **Parallel Algorithms:** Use C++17 `std::execution::par` with `std::for_each`.
10. **Frame Grabber:** Architect a low-latency grabber class that never blocks the main UI thread.
11.  **Thread-Safe Buffer Pool:** Implement a pool using `std::mutex` and `std::condition_variable`.

### Module 08: Modern CMake & Build Systems
**Goal:** Managing complex dependencies and cross-platform builds.
1.  **Basic Target:** Define a library and executable with `add_library` and `add_executable`.
2.  **Target Properties:** Set C++ standards and compile options via `target_compile_features`.
3.  **Include Directories:** Manage public/private headers with `target_include_directories`.
4.  **Finding Packages:** Use `find_package(OpenCV)` and link properly.
5.  **Subdirectories:** Structure a project with `add_subdirectory` for modularity.
6.  **FetchContent:** Automatically download and build a dependency (e.g., `fmt` or `json`).
7.  **Custom Commands:** Add a build step to generate code or copy assets.
8.  **GTest Integration:** Set up Google Test within CMake.
9.  **Packaging:** Use CPack to generate a `.deb` or `.msi` installer.
10. **Modular Library:** Create a proper `Config.cmake` export for your CV library so others can `find_package` it.
11. **Google Benchmark — Project Integration:** Add `benchmark` via vcpkg/FetchContent; create microbenchmark targets and wire them into CTest (JSON output).
12. **Benchmark Design — Reproducible Microbenchmarks:** Use fixtures, input size sweeps via `State.range()`, custom counters (pixels/s, MB/s), complexity annotations. Benchmark pixel access and feature extraction.
13. **Real vs Synthetic Data Benchmarks:** Compare with real data under `data/`; export JSON to `run/perf/`.
14. **spdlog — Structured & Async Logging:** Rotating file/console sinks, async mode, thread-safe initialization, per-module levels from YAML/JSON.
15. **glog — Compatibility Layer:** Use `LOG(INFO)`, `LOG(ERROR)`, `CHECK` macros; add compile-time switch to pick backend.
16. **Unified Logging Interface:** Provide `core/logger.hpp` facade; backend selected with compile definitions; no global state leakage.

### Module 09: GUI & Visualization (Qt6)
**Goal:** Building tools for visualization and debugging.
1.  **Qt Setup:** Create a basic "Hello World" window with CMake and Qt6.
2.  **Signals & Slots:** Connect a slider to a value label.
3.  **Image Display:** Convert `cv::Mat` to `QImage` and display it on a `QLabel`.
4.  **Custom Widget:** Subclass `QWidget` to draw overlays (boxes, text) on an image.
5.  **Event Handling:** Capture mouse clicks to select pixels/regions.
6.  **OpenGL Widget:** Render a simple 3D triangle using `QOpenGLWidget`.
7.  **File Dialogs:** Implement "Open Image" and "Save Result" functionality.
8.  **Threading in Qt:** Use `QThread` or `QtConcurrent` to run processing without freezing UI.
9.  **Charts:** Plot a live histogram using Qt Charts.
10. **Annotation Tool:** Build a tool to draw bounding boxes and save them to JSON.

### Module 10: Deep Learning Deployment (TensorRT/ONNX)
**Goal:** Running neural networks in C++ production environments.
1.  **ONNX Export:** (Python) Export a simple PyTorch model to ONNX.
2.  **ONNX Runtime:** Load and run the ONNX model in C++.
3.  **TensorRT Builder:** Convert an ONNX model to a TensorRT Engine.
4.  **Engine Serialization:** Save and load the TensorRT plan file.
5.  **Input Buffers:** Manage GPU memory for model inputs/outputs.
6.  **Inference Execution:** Run synchronous inference using TensorRT `enqueue`.
7.  **Pre-processing:** Implement image resizing and normalization on CPU (OpenCV).
8.  **Post-processing:** Parse raw output tensors into bounding boxes/classes.
9.  **NMS:** Implement Non-Maximum Suppression for object detection.
10. **YOLOv8 Engine:** End-to-end wrapper class for YOLOv8 inference.


### Module 11: System Architecture
**Goal:** Expand intermediate skills with documentation, deployment practices, and prepare for large-scale deployment and data engineering challenges. 
1.  **Documentation Management:** Generate API documentation using **Doxygen** for C++ and **Sphinx** for Python bindings.
2.  **Containerization Basics:** Create Dockerfiles for building and running CV applications in isolated environments.
3.  **CI/CD Pipeline:** Implement a GitHub Actions or GitLab CI pipeline to build, test, and deploy your CV project automatically.
4.  **Large-Scale Containerization:** Deploy CV services using **Kubernetes** clusters for scalability.
5.  **Deployment Monitoring:** Design and integrate monitoring dashboards using **Jenkins** or Prometheus/Grafana.
6.  **Big Data Handling:** Implement pipelines for **SQL**, **Apache Spark**, and **Hadoop** to process large datasets.
7.  **Storage Paradigms:** Compare and implement **Data Lake** vs **Data Warehouse** architectures for CV data.

---

## Phase 3: Advanced (Senior - Principal)

### Module 12: CUDA & Heterogeneous Computing
**Goal:** GPU acceleration for massive parallelism.
1.  **Device Query:** Query GPU properties (cores, memory, compute capability).
2.  **Memory Management:** Practice `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
3.  **Vector Add:** Write a simple 1D kernel to add arrays.
4.  **2D Kernels:** Write a kernel to invert image colors (pixel-wise).
5.  **Shared Memory:** Implement 1D matrix multiplication using shared memory tiling.
6.  **Stencil Operations:** Implement a Box Filter using global memory vs shared memory.
7.  **Streams:** Overlap memory transfer and kernel execution.
8.  **Thrust:** Use `thrust::sort` and `thrust::transform` for rapid development.
9.  **NPP Library:** Use NVIDIA Performance Primitives for resizing/conversion.
10. **Bilateral Filter:** Implement an optimized Bilateral Filter kernel.

### Module 13: State Estimation & Tracking
**Goal:** Robust tracking in noisy environments.
1.  **Probability Basics:** Implement 1D convolution for probability updates.
2.  **Kalman Filter 1D:** Track a constant voltage signal.
3.  **Linear Kalman Filter (CV):** Track a 2D point moving with constant velocity.
4.  **Extended Kalman Filter (EKF):** Track a robot moving in a circle (nonlinear model).
5.  **Unscented Kalman Filter (UKF):** Compare UKF vs EKF for highly nonlinear systems.
6.  **Mahalanobis Distance:** Implement gating to reject outliers.
7.  **Hungarian Algorithm:** Solve the data association assignment problem.
8.  **Optical Flow:** Implement Lucas-Kanade sparse optical flow.
9.  **Mean Shift:** Implement the Mean Shift tracking algorithm.
10. **SORT Tracker:** Combine Kalman Filter + Hungarian Algorithm + IoU for Multi-Object Tracking.

### Module 14: SLAM & Optimization
**Goal:** Simultaneous Localization and Mapping.
1.  **Least Squares Optimization:** Solve a simple curve fitting problem using Gauss-Newton.
2.  **G2O / Ceres Basics:** Set up a simple "Pose Graph" with 3 nodes.
3.  **Visual Odometry (Feature):** Estimate motion between two frames using Essential Matrix.
4.  **Visual Odometry (Direct):** Estimate motion by minimizing photometric error.
5.  **Bundle Adjustment:** Refine camera poses and 3D points simultaneously.
6.  **Loop Closure:** Detect when the camera revisits a location (BoW).
7.  **Map Data Structure:** Implement a Keyframe management system.
8.  **Lie Algebra:** Implement perturbation on SE(3) for optimization.
9.  **Stereo VO:** Implement a simple frame-to-frame Stereo Odometry.
10. **Mini-SLAM:** Integrate VO + Local Map + Optimization into a pipeline.

### Module 15: Motion Planning & Robotics
**Goal:** Acting upon perception.
1.  **Configuration Space:** Visualize C-Space for a 2D robot.
2.  **A* Search:** Implement A* on a grid map.
3.  **Dijkstra:** Compare Dijkstra vs A*.
4.  **RRT:** Implement Rapidly-exploring Random Trees.
5.  **RRT*:** Implement the optimizing variant of RRT.
6.  **Potential Fields:** Implement obstacle avoidance using artificial potential fields.
7.  **Trajectory Smoothing:** Use B-Splines to smooth a path.
8.  **Collision Checking:** Implement circle/rectangle collision detection.
9.  **Costmaps:** Generate an inflation layer around obstacles.
10. **Obstacle Avoidance:** Build a local planner that reacts to dynamic obstacles.

### Module 16: Advanced System Architecture
**Goal:** Designing scalable and maintainable CV platforms.
1.  **Factory Pattern:** Implement a `ModuleFactory` to create algorithms by string name.
2.  **Strategy Pattern:** Switch between different feature detectors at runtime.
3.  **Observer Pattern:** Implement an Event Bus for inter-module communication.
4.  **Pipeline Pattern:** Design a `Pipeline` class that chains `Processors`.
5.  **Plugin System:** Load `.so` / `.dll` files dynamically at runtime.
6.  **SIMD Intrinsics:** Optimize a dot product using AVX2 intrinsics.
7.  **Memory Pooling:** Implement a pool to reuse large image buffers.
8.  **Profiling:** Use `valgrind` / `perf` to identify hotspots in the pipeline.
9.  **Logging System:** Create a thread-safe, leveled logger.
10. **Graph Pipeline:** Implement a DAG (Directed Acyclic Graph) executor for complex flows.

### Module 17: Software Quality & Tooling
**Goal:** Improve code reliability and maintainability.
1.  **Config Validation:** Validate YAML configs against schema before execution.
2.  **Continuous Integration:** Add CMake + CTest pipeline with GitHub Actions.
3.  **Performance Regression Guards:** Use benchmark JSON output to assert minimum throughput in CI.
4.  **Static Analysis:** Integrate clang-tidy checks into CMake build.
5.  **Sanitizers:** Use ASan/UBSan for runtime error detection.
6.  **Code Coverage:** Generate coverage reports with gcov/lcov.
7.  **CI Integration:** Automate quality checks in CI pipeline.
8.  **Documentation:** Generate API docs using Doxygen.

### Module 18: Data Management for CV
**Goal:** Ensure reproducibility and efficient data handling.
1.  **Dataset Versioning:** Use DVC or Git LFS for large datasets.
2.  **Caching Strategies:** Implement intermediate result caching for pipelines.
3.  **Data Integrity:** Validate checksums for input/output files.
4.  **Reproducibility:** Create run manifests with configs and hashes.
5.  **Storage Optimization:** Compress and shard large datasets for distributed training.
