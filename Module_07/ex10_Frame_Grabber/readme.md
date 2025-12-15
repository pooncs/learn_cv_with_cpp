# Exercise 10: Low-Latency Frame Grabber

## Goal
Implement a `FrameGrabber` class that runs acquisition in a background thread and allows the main thread to poll the latest frame without blocking.

## Learning Objectives
1.  **Decoupling:** Separating the acquisition rate (e.g., 30 FPS) from the processing/display rate (e.g., 60 FPS or variable).
2.  **Thread Safety:** Swapping the "latest" frame pointer atomically or using a mutex.
3.  **Triple Buffering:** (Concept) One buffer being written, one ready to read, one being read.

## Practical Motivation
If your UI thread calls `camera.read()`, it might block for 33ms. This freezes the GUI. A frame grabber runs `read()` in a loop and updates a shared `latestFrame` variable. The UI just copies `latestFrame` instantly.

## Step-by-Step Instructions

### Task 1: FrameGrabber Class
-   Members: `std::thread grabThread`, `std::mutex mtx`, `cv::Mat latestFrame`, `bool running`.
-   Method `start()`: Launches thread.
-   Method `stop()`: Stops thread.
-   Method `getLatest(cv::Mat& out)`: Locks mutex, copies `latestFrame` to `out`, returns true if new frame.

### Task 2: Background Loop
-   Capture frames continuously (simulate with `sleep` or `VideoCapture`).
-   Lock mutex, update `latestFrame`, unlock.

### Task 3: Consumer
-   Main thread loops and calls `getLatest()`.
-   It should never block for the duration of a frame capture.

## Verification
1.  Set producer to 10 FPS (100ms).
2.  Set consumer to 100 FPS (10ms).
3.  Consumer should see the same frame 10 times (or get same frame quickly) without waiting 100ms each time.

```bash
cd todo
mkdir build && cd build
conan install .. -s compiler.cppstd=17 --output-folder=. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
ctest
```
