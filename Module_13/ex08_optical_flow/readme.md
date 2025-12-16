# Exercise 08: Optical Flow (Lucas-Kanade)

## Goal
Implement Sparse Optical Flow using the Lucas-Kanade algorithm to track feature points across video frames.

## Learning Objectives
1.  **Optical Flow Assumption:** Brightness constancy (pixel doesn't change color as it moves).
2.  **Lucas-Kanade Method:** Solving for motion $(u, v)$ in a small window.
3.  **OpenCV Implementation:** `cv::calcOpticalFlowPyrLK`.
4.  **Feature Detection:** Using `cv::goodFeaturesToTrack` to find trackable points.

## Practical Motivation
If you want to know *how fast* a car is moving or stabilize a shaky video, you need to track how pixels move from one frame to the next.

**Analogy:**
*   **Feature Points:** Distinctive landmarks (corners of a building).
*   **Optical Flow:** Watching those landmarks move as you drive by. By seeing how much they shift in your vision, you know your speed and direction.

## Theory: The Math
Equation: $I(x, y, t) = I(x+dx, y+dy, t+dt)$
Taylor Expansion leads to: $I_x u + I_y v + I_t = 0$
*   $I_x, I_y$: Spatial gradients (Edges).
*   $I_t$: Temporal gradient (Difference between frames).
*   $u, v$: Velocity vector (Flow).

## Step-by-Step Instructions

### Task 1: Setup Video Capture
1.  Open webcam or video file.
2.  Read first frame (`old_frame`), convert to grayscale.

### Task 2: Detect Features
1.  Call `cv::goodFeaturesToTrack` on `old_frame` to find initial points (`p0`).

### Task 3: Track Loop
1.  Read next frame (`frame`), convert to gray (`frame_gray`).
2.  Call `cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err)`.
3.  Select good points (where `status == 1`).
4.  Draw tracks.
5.  Update `old_gray = frame_gray` and `p0 = good_new`.

## Code Hints
```cpp
vector<Point2f> p0, p1;
goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7);

while(true) {
    // ... read frame ...
    calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err);
    
    // Filter bad points
    vector<Point2f> good_new;
    for(uint i = 0; i < p0.size(); i++) {
        if(status[i] == 1) {
            good_new.push_back(p1[i]);
            line(mask, p1[i], p0[i], Scalar(0,255,0), 2);
        }
    }
    p0 = good_new;
}
```

## Verification
Run the program and move your hand or camera. Green lines should trace the movement of corners.
