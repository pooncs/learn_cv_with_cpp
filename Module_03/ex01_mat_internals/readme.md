# Exercise 01: cv::Mat Internals

## Goal
Deep dive into `cv::Mat` memory layout, reference counting, and basic properties.

## Learning Objectives
1.  **Memory Model:** Header (metadata) vs Data (pixels).
2.  **Shallow vs Deep Copy:** When does copying `Mat` copy the pixels? (Spoiler: Almost never by default).
3.  **Strides:** Understanding `step` and padding.

## Analogy: The Library Card
*   **`cv::Mat` (Header):** A **Library Card**.
    *   It contains info: "Title: Mona Lisa", "Size: 100x100", "Location: Shelf 5".
    *   It is very small and light.
*   **The Data (Pixels):** The actual **Book**.
    *   It is heavy (Megabytes).
*   **Shallow Copy (`Mat B = A`):** Photocopying the Library Card.
    *   You have two cards. They both point to the *same* Book.
    *   If you draw a mustache on the Mona Lisa using Card B, the person holding Card A sees it too.
*   **Deep Copy (`Mat C = A.clone()`):** Buying a brand new copy of the Book.
    *   Now you have two separate books. Modifying one doesn't affect the other.

## Practical Motivation
*   **Performance:** Passing `Mat` by value is cheap (it just copies the header).
*   **Bug Source:** "I modified the image in a function, why did my original image change?" (Because you passed by value, which is a shallow copy!).

## Step-by-Step Instructions

### Task 1: Create Mat Manually
Open `src/main.cpp`.
*   Create a `cv::Mat` named `image` of size $480 \times 640$, type `CV_8UC3` (8-bit Unsigned, 3 Channels), initialized to green.
    *   *Hint:* `cv::Scalar(blue, green, red)`.

### Task 2: Inspect Properties
*   Print:
    *   `rows`, `cols`
    *   `channels()`
    *   `elemSize()` (Bytes per pixel: 3 for RGB)
    *   `step` (Bytes per row: usually cols * elemSize, but can have padding).

### Task 3: Reference Counting (Shallow Copy)
*   Create `cv::Mat shallow_copy = image;`
*   Draw a white rectangle on `shallow_copy`.
*   Check if `image` has the rectangle. (It should).

### Task 4: Deep Copy
*   Create `cv::Mat deep_copy = image.clone();`
*   Draw a black circle on `deep_copy`.
*   Check if `image` has the circle. (It should NOT).

## Verification
Compile and run.
```bash
cd todo
mkdir build && cd build
cmake ..
cmake --build .
./main
```
