# Exercise 06: FLANN Matching

## Goal
Use the Fast Library for Approximate Nearest Neighbors (FLANN) to match descriptors efficiently.

## Learning Objectives
1.  **Approximate vs Exact:** Understand that for large datasets (millions of points), exact search is too slow. We trade a small loss in accuracy for a massive gain in speed.
2.  **KD-Trees:** How KD-Trees partition space for floating-point descriptors (SIFT/SURF).
3.  **LSH (Locality Sensitive Hashing):** The appropriate index for *binary* descriptors (ORB/BRIEF).
4.  **OpenCV Interface:** Master `cv::FlannBasedMatcher`.

## Practical Motivation
Brute force matching is $O(N \cdot M)$. For real-time SLAM or large-scale retrieval, this is unacceptable. FLANN chooses the best indexing structure (KD-Tree, K-Means Tree, or LSH) to speed up queries to logarithmic time $O(M \log N)$.

## Theory: Indexing Types
*   **KD-Tree:** Good for low-dimensional float data (SIFT 128D).
*   **LSH (Locality Sensitive Hashing):** Good for binary data (ORB 256 bits). It hashes similar items to the same bucket with high probability.

**Important:** Standard `cv::FlannBasedMatcher` defaults to KD-Tree and expects `CV_32F`. For ORB (`CV_8U`), you must either:
1.  Convert ORB to `CV_32F` (bad, loses binary efficiency).
2.  Use `cv::FlannBasedMatcher` with an LSH index configuration.

## Step-by-Step Instructions

### Task 1: Setup FLANN for ORB
Create a `cv::FlannBasedMatcher` using the LSH index parameters.
```cpp
cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
```

### Task 2: Match and Measure Time
1.  Generate a large number of keypoints (e.g., 2000+) to notice the speed difference (though on modern CPUs 2000 is still fast for BF).
2.  Match using FLANN.
3.  Measure the time taken using `std::chrono`.

### Task 3: Compare with Brute Force
Run BFMatcher on the same data and compare execution time.

## Common Pitfalls
1.  **Crashing on ORB:** If you use the default `FlannBasedMatcher()` with ORB, it *will* crash or throw an error because it expects floats.
2.  **Index Building:** FLANN builds an index (tree/hash table). This takes time. For single-shot matching, BF might be faster. FLANN wins when reusing the same train set.

## Verification
1.  Ensure matches are roughly the same as BFMatcher (visually).
2.  Verify code does not crash with binary descriptors.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
