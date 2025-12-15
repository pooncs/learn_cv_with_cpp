# Exercise 05: Lowe's Ratio Test

## Goal
Implement Lowe's Ratio Test to filter ambiguous matches from the Brute Force matching results.

## Learning Objectives
1.  **Ambiguity in Matching:** Understand that sometimes a feature is similar to *many* features in the other image (e.g., repeating patterns).
2.  **Ratio Test:** Implement the heuristic: keep a match only if the distance to the nearest neighbor is significantly smaller than the distance to the *second* nearest neighbor.
3.  **KNN Search:** Upgrade your matcher to find the $K=2$ nearest neighbors.

## Practical Motivation
In a repetitive texture (like a brick wall), a corner on one brick looks like a corner on another brick. Simple Nearest Neighbor will just pick one, often wrong. 
David Lowe (SIFT creator) observed that for correct matches, the first neighbor is much closer than the second. For ambiguous matches, they are close to each other.
$$ \frac{\text{distance}(1st)}{\text{distance}(2nd)} < \text{ratio} $$
A typical ratio is 0.7 or 0.8.

## Step-by-Step Instructions

### Task 1: KNN Matcher
Modify your Brute Force matcher to return the top $K$ matches for each query, instead of just 1.
-   Store the best $K$ distances and indices.

### Task 2: Implement Ratio Test
1.  For each query descriptor, retrieve the 2 nearest neighbors ($m_1, m_2$).
2.  Check condition: $m_1.distance < \text{ratio} \cdot m_2.distance$.
3.  If true, keep $m_1$ as a "good" match.
4.  If false, discard.

### Task 3: Visualization
Visualize the matches before and after the ratio test. You should see a significant reduction in diagonal outliers.

## Common Pitfalls
1.  **Not enough neighbors:** If `trainDescriptors` has fewer than 2 items, you cannot compute the ratio.
2.  **Float vs Int:** Distances are integers (Hamming), but the ratio is float. Cast before division.
3.  **Strict Inequality:** Use `<` strictly.

## Verification
1.  Match an image with repetitive patterns (e.g., a chessboard) against a rotated version.
2.  Observe that the Ratio Test removes many false positives on the grid.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
