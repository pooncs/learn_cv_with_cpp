# Exercise 06: Perspective Warp

## Goal
Apply perspective transformation to "rectify" a slanted document image.

## Learning Objectives
1.  Identify 4 corners of a document in an image.
2.  Define the target "canonical" view (e.g., A4 aspect ratio).
3.  Compute Homography and warp.

## Practical Motivation
Document scanning apps (CamScanner) use this to un-distort photos of receipts/papers taken at an angle.

## Theory & Background

### Canonical View
If we know the physical aspect ratio of the document (e.g., A4 is $210 \times 297$ mm), we can define the destination points as:
- $(0, 0)$
- $(W, 0)$
- $(W, H)$
- $(0, H)$
where $W/H \approx 210/297$.

## Implementation Tasks

### Task 1: Rectify
Implement `rectify_document(img, corners, aspect_ratio)` which:
1.  Sorts corners (TL, TR, BR, BL).
2.  Estimates width/height.
3.  Computes H.
4.  Warps.

## Common Pitfalls
- Ordering of corners is crucial.
- Output image size estimation.
