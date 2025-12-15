# Exercise 04: Brute Force Matching

## Goal
Implement a Brute Force Matcher for binary descriptors (ORB) using Hamming Distance.

## Learning Objectives
1.  **Descriptor Distance:** Understand the difference between Euclidean ($L_2$) and Hamming distance (XOR count).
2.  **Nearest Neighbor Search:** Implement the $O(N \cdot M)$ search algorithm to find the closest descriptor in a database.
3.  **Cross-Check:** Implement bidirectional matching (A->B and B->A) to filter inconsistent matches.

## Practical Motivation
Once we have descriptors for two images, we need to find which points correspond to the same physical 3D point. The simplest method is "Brute Force": compare every descriptor in Image A with every descriptor in Image B.

## Theory: Hamming Distance
For binary descriptors like ORB, BRIEF, or BRISK, the descriptor is a string of bits.
The "distance" between two descriptors is the number of bits that differ.
This can be computed efficiently using the XOR operation (`^`) followed by a population count (counting set bits).
In C++, `std::bitset::count()` or `__builtin_popcount` (compiler intrinsic) is very fast.

$$ D(d_1, d_2) = \text{popcount}(d_1 \oplus d_2) $$

## Step-by-Step Instructions

### Task 1: Hamming Distance Function
Implement a function that takes two `cv::Mat` rows (descriptors) and returns the integer Hamming distance.
-   **Note:** ORB descriptors are `CV_8U`. You need to process them byte-by-byte or cast to larger chunks if careful.

### Task 2: Nearest Neighbor Search
For each query descriptor in Image A:
1.  Iterate through all target descriptors in Image B.
2.  Compute the distance.
3.  Keep the index of the target with the *minimum* distance.

### Task 3: Visualization
Use `cv::drawMatches` to visualize the result. Since we haven't filtered outliers yet, expect many wrong matches (diagonal lines are good, chaotic lines are bad).

## Common Pitfalls
1.  **Data Types:** Ensure you are reading the descriptors as `uchar` or `uint8_t`.
2.  **Empty Descriptors:** Handle cases where no keypoints were detected.
3.  **Efficiency:** While "Brute Force" implies slow, ensure you aren't re-allocating memory inside loops.

## Verification
1.  Match an image against a slightly rotated version of itself.
2.  Compare your matches with `cv::BFMatcher(cv::NORM_HAMMING)`.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
