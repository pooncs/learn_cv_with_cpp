# Exercise 02: Strategy Pattern

## Goal
Implement a mechanism to switch between different algorithms (Strategies) at runtime without changing the code that uses them.

## Learning Objectives
1.  **Strategy Interface:** Define a common interface for a family of algorithms.
2.  **Context Class:** Maintain a reference to a Strategy object.
3.  **Runtime Switching:** Swap the strategy object dynamically.

## Practical Motivation
You have a `FaceDetector`. Sometimes you want `HaarCascade` (fast), other times `CNN` (accurate). The Strategy pattern lets the `FaceDetector` class remain unchanged while the underlying math changes.

## Step-by-Step Instructions
1.  Define `IFeatureDetector` with `detect()`.
2.  Implement `SiftDetector` and `OrbDetector`.
3.  Create `VisualOdometry` context class that holds a `unique_ptr<IFeatureDetector>`.
4.  Add `set_detector(unique_ptr<IFeatureDetector>)` to switch strategies.

## Verification
*   Initialize VO with SIFT. Run it.
*   Switch to ORB. Run it.
*   Verify different output messages/behavior.
