# Exercise 05: Event Handling

## Goal
Capture mouse events to select pixels or regions on an image. This is essential for interactive tools like annotators or seed point selection for segmentation.

## Learning Objectives
1.  Override virtual event handlers (`mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent`).
2.  Work with `QMouseEvent` to get coordinates and button states.
3.  Map widget coordinates to image coordinates.

## Practical Motivation
A user clicks on a GUI widget, but we need to know which pixel in the underlying image corresponds to that click, handling scaling and aspect ratios correctly.

## Theory: Event Propagation
When a user clicks, Qt sends a `QEvent` to the widget under the cursor. If the widget ignores it, it propagates up the parent chain. To handle it, we override the specific handler.
*   `event->pos()`: Coordinates relative to the widget.
*   `event->buttons()`: Which buttons are pressed (Left, Right, Middle).

## Step-by-Step Instructions

### Task 1: Subclass QLabel
Open `todo/src/main.cpp`.
1.  Create `ClickableLabel` inheriting from `QLabel`.
2.  Override `void mousePressEvent(QMouseEvent *event)`.

### Task 2: Handle Mouse Press
1.  In `mousePressEvent`, check if `event->button() == Qt::LeftButton`.
2.  Get position: `event->pos()`.
3.  Print coordinates to `std::cout` or emit a signal `clicked(QPoint)`.

### Task 3: Coordinate Mapping (Bonus)
If the image is scaled (e.g., `setScaledContents(true)`), `event->pos()` is the widget coordinate, not the image pixel.
1.  Calculate scaling factor: `scaleX = imageWidth / widgetWidth`.
2.  `imageX = widgetX * scaleX`.

## Verification
Run the app. Click on the image. The console should print "Clicked at: x, y".
