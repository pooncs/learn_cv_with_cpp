# Exercise 09: Mean Shift Tracking

## Goal
Implement object tracking using the Mean Shift algorithm, which tracks objects based on their color distribution (histogram).

## Learning Objectives
1.  **Color Histograms:** Compute a Hue-Saturation histogram of the target object.
2.  **Back Projection:** Create a probability map where pixels are replaced by the probability of belonging to the target.
3.  **Mean Shift Algorithm:** Iteratively find the peak of the probability density (centroid of the back projection).
4.  **CamShift:** An extension that also adapts the window size and orientation.

## Practical Motivation
If you are tracking a red ball, Optical Flow might fail if the ball spins (texture changes) but stays in place. Mean Shift works because the *color* distribution remains constant.

**Analogy:**
*   **Back Projection:** A heat map showing "Redness".
*   **Mean Shift:** A hill climber. You are standing on a hill (probability map) in the fog. You blindly take steps upwards until you reach the peak (centroid). As the ball moves, the peak moves, and you climb towards the new peak.

## Step-by-Step Instructions

### Task 1: Initialize
1.  Read first frame.
2.  User selects a ROI (Region of Interest) containing the object.
3.  Convert ROI to HSV.
4.  Calculate Histogram (`calcHist`) of the Hue channel. Normalize it to 0-255.

### Task 2: Tracking Loop
1.  Read next frame.
2.  Convert to HSV.
3.  Calculate **Back Projection** (`calcBackProject`) using the target histogram.
4.  Call `meanShift` (or `CamShift`) with the back projection and the previous window.
5.  Draw the new window.

## Code Hints
```cpp
// 1. Histogram
Mat roi_hsv; cvtColor(roi, roi_hsv, COLOR_BGR2HSV);
calcHist(&roi_hsv, 1, channels, mask, roi_hist, 1, histSize, ranges);
normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

// 2. Loop
calcBackProject(&hsv, 1, channels, roi_hist, backproj, ranges);
meanShift(backproj, track_window, TermCriteria(...));
rectangle(frame, track_window, Scalar(0,0,255), 2);
```

## Verification
Select a distinctively colored object (e.g., a bright mug or face). The rectangle should stick to it as you move it around.
