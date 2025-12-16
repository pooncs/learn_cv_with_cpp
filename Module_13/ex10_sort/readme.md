# Exercise 10: SORT (Simple Online and Realtime Tracking)

## Goal
Implement a basic version of the SORT algorithm, which associates detections from an object detector (like YOLO) across frames using Kalman Filters and IoU (Intersection over Union).

## Learning Objectives
1.  **Data Association:** Matching detections in Frame $t$ to tracks from Frame $t-1$.
2.  **Hungarian Algorithm:** Optimally solving the assignment problem (minimizing cost).
3.  **Kalman Filter Tracking:** Using KF (from Ex04) to predict where a track will be in the next frame.
4.  **Track Lifecycle:** managing when to create a new track and when to delete a lost track.

## Practical Motivation
YOLO gives you boxes for every frame, but it doesn't know that "Car #1" in Frame 10 is the same as "Car #1" in Frame 11. SORT bridges this gap.

**Analogy:**
*   **YOLO:** A security guard who yells "There's a person!" every second but has amnesia.
*   **SORT:** A manager with a clipboard (Kalman Filter).
    *   Guard: "Person at (10, 10)!"
    *   Manager: "Okay, that's close to where Bob was (predicted). That must be Bob."
    *   Guard: "Person at (100, 100)!"
    *   Manager: "I don't know anyone near there. I'll start a file for a New Person."

## Step-by-Step Instructions

### Task 1: KalmanTracker Class
1.  Wrap the OpenCV `cv::KalmanFilter`.
2.  State: `[cx, cy, s, r, vx, vy, vs]` (Center, Scale/Area, Aspect Ratio, Velocities).
3.  Measurement: `[cx, cy, s, r]`.

### Task 2: Associate Detections to Tracks
1.  Get current tracks (Predicted locations).
2.  Get current detections (YOLO output).
3.  Compute IoU Matrix between all Tracks and Detections.
4.  Solve assignment (Hungarian Algo or Greedy match).
    *   If IoU < Threshold (e.g., 0.3), reject match.

### Task 3: Update Tracks
1.  **Matched:** Update Kalman Filter with new detection.
2.  **Unmatched Detections:** Create new Tracks.
3.  **Unmatched Tracks:** Increment "lost" counter. Delete if lost for too long (e.g., 30 frames).

## Code Hints
```cpp
// 1. Predict
for (auto& track : tracks) track.predict();

// 2. Associate
// ... compute IoU matrix ...
// ... assign ...

// 3. Update
for (auto match : matches) {
    tracks[match.trackIdx].update(detections[match.detIdx]);
}
```

## Verification
Simulate two boxes moving across a screen. The tracker should maintain the same ID for each box even if they cross paths (if using KF correctly) or at least follow them robustly.
