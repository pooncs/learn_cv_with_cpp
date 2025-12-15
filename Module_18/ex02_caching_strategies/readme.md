# Exercise 02: Caching Strategies

## Goal
Implement a caching system to avoid re-computing expensive operations (e.g., feature extraction) when inputs haven't changed.

## Learning Objectives
1.  **Hashing:** Generate unique signatures for inputs (file content + parameters).
2.  **Serialization:** Save/Load results to disk.
3.  **Cache Logic:** Implement check-compute-save flow.

## Practical Motivation
If preprocessing 10,000 images takes 2 hours, you don't want to restart from zero if you change a downstream training parameter. Caching the preprocessed data saves time.

## Step-by-Step Instructions

### Task 1: Hash Function
Implement a function that computes SHA-256 (or simpler) of a file or config struct.

### Task 2: Cache Manager
Create a class `CacheManager` that checks if `cache/{hash}.dat` exists.

### Task 3: Integrate
Wrap your `process_image` function:
```cpp
std::string hash = compute_hash(image_path, params);
if (cache.exists(hash)) {
    return cache.load(hash);
} else {
    auto result = process_image(image_path, params);
    cache.save(hash, result);
    return result;
}
```

## Verification
Run the pipeline twice. The second run should be near-instant and report "Loaded from cache".
