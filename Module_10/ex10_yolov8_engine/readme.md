# Exercise 10: YOLOv8 Engine

## Goal
Create an end-to-end `YoloEngine` class that wraps TensorRT inference, pre-processing, and post-processing.

## Learning Objectives
1.  Design a clean C++ API for object detection (`detect(image) -> detections`).
2.  Manage resources (RAII) for TensorRT objects.
3.  Integrate all previous steps into a production-ready class.

## Practical Motivation
In a real application, you don't want raw CUDA calls in your business logic. You want a simple `detector.infer(frame)` interface.

## Step-by-Step Instructions

### Task 1: Class Definition
Open `todo/src/main.cpp`.
1.  Define `class YoloEngine`.
2.  Constructor: Load engine, create context, allocate buffers.
3.  Destructor: Free buffers.
4.  Method: `std::vector<Detection> infer(const cv::Mat& img)`.

### Task 2: Implement Inference
1.  Pre-process image (Ex 07).
2.  Copy to GPU.
3.  Execute (Ex 06).
4.  Copy to CPU.
5.  Post-process & NMS (Ex 08, 09).

### Task 3: Run
1.  Instantiate `YoloEngine`.
2.  Load image.
3.  Run inference.
4.  Draw bounding boxes on image using `cv::rectangle`.

## Verification
Save/Show the result image with bounding boxes drawn.
