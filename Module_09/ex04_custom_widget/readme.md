# Exercise 04: Custom Widget

## Goal
Subclass `QWidget` to create a custom widget that displays an image and draws overlays (rectangles, text) on top of it.

## Learning Objectives
1.  Understand the `paintEvent` method in Qt.
2.  Use `QPainter` to draw primitives (lines, rects, text).
3.  Coordinate systems in Qt.

## Practical Motivation
Standard widgets (`QLabel`, `QPushButton`) are limited. For a CV tool (e.g., drawing bounding boxes), you need a custom widget that handles rendering manually.

## Theory: QPainter
*   **paintEvent:** Called whenever the widget needs to be updated.
*   **QPainter:** The object used to draw. Must be initialized with the target device (usually `this`).
*   **Coordinate System:** (0,0) is top-left.

## Step-by-Step Instructions

### Task 1: Create ImageWidget Class
Open `todo/src/main.cpp`.
1.  Define a class `ImageWidget` inheriting from `QWidget`.
2.  Add a member `QImage m_image` and a method `setImage(const QImage &img)`.
3.  Implement `paintEvent(QPaintEvent *event)`.

### Task 2: Implement paintEvent
1.  Create `QPainter painter(this)`.
2.  Draw the image: `painter.drawImage(0, 0, m_image)`.
3.  Set pen color/width: `painter.setPen(QPen(Qt::red, 3))`.
4.  Draw a rectangle: `painter.drawRect(50, 50, 100, 100)`.

### Task 3: Use in Main
1.  Load an image (or create a dummy one).
2.  Create `ImageWidget`.
3.  Set the image.
4.  Show the widget.

## Verification
Run the app. You should see the image with a red box overlay.
