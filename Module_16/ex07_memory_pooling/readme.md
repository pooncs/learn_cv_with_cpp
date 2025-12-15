# Exercise 07: Memory Pooling

## Goal
Implement an Object Pool to reuse large memory buffers (Images) and avoid allocation overhead.

## Learning Objectives
1.  **Allocation Cost:** `malloc`/`new` is slow and causes fragmentation.
2.  **Pool Logic:** `acquire()` (get from pool or create) and `release()` (return to pool).
3.  **Custom Deleters:** Using `std::shared_ptr` with a custom deleter to automatically return objects to the pool.

## Practical Motivation
At 60 FPS, allocating a 1080p image (6MB) every frame is a disaster. Reusing the same 3-4 buffers is much better.

## Step-by-Step Instructions
1.  Create `ImagePool`.
2.  Maintain a `std::stack<Image*>`.
3.  `acquire()`: Pop from stack or `new`.
4.  `release()`: Push back to stack.
5.  Wrap in `shared_ptr` so `release()` is called automatically when the user is done.

## Verification
*   Loop 100 times acquiring and releasing images.
*   Check pointer addresses. They should be the same (reuse) rather than new ones.
