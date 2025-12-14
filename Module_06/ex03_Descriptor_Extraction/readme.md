# Exercise 03: ORB Descriptor Extraction

## Goal
Implement the descriptor extraction phase of ORB (Oriented FAST and Rotated BRIEF). You will manually compute the orientation of keypoints and extract binary descriptors.

## Learning Objectives
1.  **Intensity Centroid:** Calculate the angle of a keypoint using moments.
2.  **BRIEF Descriptor:** Understand Binary Robust Independent Elementary Features.
3.  **Steering:** Rotate the BRIEF test pattern according to the keypoint orientation to achieve rotation invariance.
4.  **Hamming Distance:** Understand why binary descriptors use Hamming distance (XOR and bit count) instead of Euclidean distance.

## Practical Motivation
Detecting points (Harris/FAST) is only half the battle. To match them across images, we need a "fingerprint" or descriptor for each point. SIFT and SURF are patented (or were) and slow. ORB is a free, efficient alternative built on simple binary tests, making it ideal for real-time applications on mobile devices.

## Theory: ORB Components

### 1. Orientation (Intensity Centroid)
To make FAST rotation invariant, we compute the angle of the patch. We use the "intensity centroid":
$$ m_{pq} = \sum_{x,y} x^p y^q I(x,y) $$
The centroid is $C = (\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}})$.
The angle is $\theta = \text{atan2}(m_{01}, m_{10})$.

### 2. Rotated BRIEF
BRIEF compares intensity of pairs of points $(p_1, p_2)$ in a smoothed patch.
$$ \tau(p; x, y) = \begin{cases} 1 & \text{if } I(p+x) < I(p+y) \\ 0 & \text{otherwise} \end{cases} $$
A descriptor is a bitstring of length $n$ (e.g., 256).

To be rotation invariant, we rotate the coordinates $(x, y)$ by $\theta$:
$$ \begin{bmatrix} x' \\ y' \end{bmatrix} = R_\theta \begin{bmatrix} x \\ y \end{bmatrix} $$

## Step-by-Step Instructions

### Task 1: Compute Orientation
For each keypoint:
1.  Extract a patch (e.g., $31 \times 31$).
2.  Compute moments $m_{01}$ and $m_{10}$.
3.  Compute angle $\theta$ (in degrees or radians).
4.  Store it in `keypoint.angle`.

### Task 2: Define Pattern
Create a fixed pattern of 256 pairs of points (Gaussian distribution around center). You can use a random number generator to create this once.

### Task 3: Compute Descriptor
For each keypoint:
1.  Compute $\sin \theta$ and $\cos \theta$.
2.  For each of the 256 pairs $(a, b)$:
    - Rotate $a$ to $a'$ and $b$ to $b'$.
    - Sample image intensity at $a'$ and $b'$.
    - If $I(a') < I(b')$, set bit to 1, else 0.
3.  Store the 256 bits (32 bytes) in a `cv::Mat` row.

## Common Pitfalls
1.  **Sampling Bounds:** Rotating points might push them outside the image. Check bounds or pad the image.
2.  **Smoothing:** BRIEF is sensitive to noise. Always Gaussian blur the image before extraction (ORB uses an integral image or pre-smoothing).
3.  **Coordinate System:** Ensure your rotation matrix matches the image coordinate system (y-down).

## Verification
1.  Match descriptors between a rotated version of the same image.
2.  Compare with `cv::ORB`.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
