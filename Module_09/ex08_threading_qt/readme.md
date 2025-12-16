# Exercise 08: Threading in Qt

## Goal
Perform a long-running task without freezing the UI using `QtConcurrent` and `QFutureWatcher`.

## Learning Objectives
1.  Understand why the UI freezes if you run loops in the main thread.
2.  Use `QtConcurrent::run` to execute a function in a thread pool.
3.  Monitor progress/completion using `QFutureWatcher`.

## Practical Motivation
Image processing (e.g., training, heavy filtering) takes time. If you run it in the main event loop, the window becomes unresponsive ("Not Responding").

## Theory: QtConcurrent
*   `QtConcurrent::run`: High-level API to run a function in a background thread.
*   `QFuture`: Represents the result of the asynchronous computation.
*   `QFutureWatcher`: Emits signals (`finished`, `progressValueChanged`) when the future updates.

## Step-by-Step Instructions

### Task 1: Setup UI
Open `todo/src/main.cpp`.
1.  Create `QPushButton` ("Start Processing") and `QProgressBar`.
2.  Set progress bar range (0-100).

### Task 2: Define Heavy Task
1.  Create a standalone function `void heavyTask()` (or one that returns a value).
2.  Simulate work with `QThread::sleep` or a loop.

### Task 3: Connect and Run
1.  On button click:
    *   Disable button.
    *   Call `QtConcurrent::run(heavyTask)`.
    *   Setup `QFutureWatcher` to watch the result.
2.  Connect `QFutureWatcher::finished` signal to re-enable button and show "Done".

## Verification
Run the app. Click "Start". The UI should remain responsive (you can move the window) while the task runs.
