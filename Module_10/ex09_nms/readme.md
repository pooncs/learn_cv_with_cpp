# Exercise 09: Non-Maximum Suppression (NMS)

## Goal
Implement NMS to remove overlapping duplicate detections.

## Learning Objectives
1.  Understand Intersection over Union (IoU).
2.  Implement the NMS algorithm.
3.  Use `cv::dnn::NMSBoxes` (or manual implementation) to filter results.

## Practical Motivation
Object detectors often output multiple boxes for the same object (e.g., slightly shifted). NMS keeps only the best one.

**Analogy:** Imagine a group of paparazzi taking photos of the same celebrity. You have 50 photos, but you only want the best one for the front page. NMS is like sorting the photos by quality (confidence), picking the best one, and then discarding all other photos that look almost identical (high overlap) to the best one. You repeat this for every celebrity found.

## Theory: NMS Algorithm
1.  Sort detections by confidence (descending).
2.  Pick the highest confidence box (A).
3.  Compare A with all other boxes (B).
4.  If IoU(A, B) > threshold, remove B.
5.  Repeat until no boxes remain.

## Step-by-Step Instructions

### Task 1: IoU Function
Open `todo/src/main.cpp`.
1.  Implement `float iou(const cv::Rect& a, const cv::Rect& b)`.
    *   Intersection Area / Union Area.
    *   Union = AreaA + AreaB - Intersection.

### Task 2: NMS Loop
1.  Sort detections.
2.  Iterate and keep track of suppressed indices.

### Task 3: Use OpenCV NMS
1.  `cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices)`.
2.  Extract kept detections.

## Verification
Create two overlapping boxes. NMS should remove the one with lower score.
