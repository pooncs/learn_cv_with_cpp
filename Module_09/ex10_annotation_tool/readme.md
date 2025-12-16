# Exercise 10: Annotation Tool

## Goal
Build a complete tool to draw bounding boxes on an image and save them to a JSON file.

## Learning Objectives
1.  Combine Custom Widget, Event Handling, and File I/O.
2.  Serialize data to JSON using `nlohmann/json`.
3.  Manage state (list of boxes).

## Practical Motivation
Data labeling is the fuel for Deep Learning. Building your own light-weight labeler is a common task when off-the-shelf tools don't fit the workflow.

## Theory: State Management
We need to store a list of rectangles (`QRect`).
*   **Mouse Press:** Start a new rect.
*   **Mouse Move:** Update the current rect size (drag).
*   **Mouse Release:** Finalize the rect.
*   **Paint Event:** Draw all stored rects + current rect.

## Step-by-Step Instructions

### Task 1: AnnotatorWidget
Open `todo/src/main.cpp`.
1.  Inherit `QWidget`.
2.  Members: `QImage`, `QVector<QRect> boxes`, `QRect currentBox`, `bool drawing`.
3.  Implement mouse events to update `boxes`.
4.  Implement `paintEvent` to draw image and boxes.

### Task 2: Main Window
1.  Add "Open Image" button (load image).
2.  Add "Save JSON" button.

### Task 3: Save Logic
1.  Iterate `boxes`.
2.  Create a JSON object: `[{"x": 10, "y": 10, "w": 100, "h": 100}, ...]`.
3.  Save to file.

## Verification
1.  Open image.
2.  Draw a few boxes.
3.  Save to `labels.json`.
4.  Verify JSON content.
