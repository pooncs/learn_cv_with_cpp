# Exercise 03: Image Display

## Goal
Load an image using OpenCV (`cv::Mat`) and display it in a Qt application by converting it to `QImage` and showing it on a `QLabel`.

## Learning Objectives
1.  Understand the memory layout differences between `cv::Mat` (BGR) and `QImage` (RGB).
2.  Perform efficient data conversion.
3.  Scale images for display.

## Practical Motivation
OpenCV is the standard for processing, but Qt is better for GUI. We need a bridge. `cv::imshow` is good for debugging but limited. Integrating into a Qt window allows custom UI controls.

## Theory: Color Spaces & Stride
*   **OpenCV:** Defaults to BGR. Data is often contiguous, but may have padding (stride).
*   **Qt:** `QImage` supports many formats, usually RGB888 or ARGB32.
*   **Conversion:** We must swap channels (BGR -> RGB) and ensure the data pointer is valid for the `QImage` lifetime.

## Step-by-Step Instructions

### Task 1: Load Image
Open `todo/src/main.cpp`.
1.  Use `cv::imread` to load an image (path: `data/lenna.png` - provided).
2.  Check if empty.

### Task 2: Convert to QImage
1.  Convert BGR to RGB: `cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB)`.
2.  Create `QImage`:
    ```cpp
    QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
    ```
    *   **Warning:** `QImage` does *not* copy data by default. The `cv::Mat` must stay alive!

### Task 3: Display
1.  Create a `QLabel`.
2.  Convert `QImage` to `QPixmap`: `QPixmap::fromImage(qimg)`.
3.  Set pixmap on label: `label->setPixmap(...)`.
4.  Resize label to fit.

## Verification
Run the app. You should see the image.
