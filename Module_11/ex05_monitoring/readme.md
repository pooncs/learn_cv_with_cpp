# Exercise 05: Monitoring and Metrics

## Goal
Instrument a C++ application to expose metrics (counters, gauges) for Prometheus.

## Learning Objectives
1.  **Observability:** Understand why logging isn't enough.
2.  **Metric Types:** Counters (always go up) vs Gauges (can go up/down) vs Histograms (latency buckets).
3.  **Exposition:** How to expose metrics via HTTP (`/metrics`).
4.  **Prometheus:** How Prometheus scrapes targets.

## Practical Motivation
If your server crashes at 3 AM, how do you know? If it's slow, is it the database or the CPU? Logs tell you *what* happened. Metrics tell you *how much* and *how fast*.

**Analogy:**
*   **Logs:** A diary. "Dear Diary, today I felt sick." (Detailed, text-based, hard to aggregate).
*   **Metrics:** A dashboard in a car. Speedometer (Gauge), Odometer (Counter), RPM. (Numerical, easy to graph, instant status).

## Theory: Prometheus Format
Text-based exposition:
```
# HELP requests_total Total number of requests
# TYPE requests_total counter
requests_total 42
```

## Step-by-Step Instructions

### Task 1: Create a Metric Class
1.  Create a simple `Counter` class.
2.  `void inc()`: increments value.
3.  `int get()`: returns value.

### Task 2: Simulate Work
1.  In a loop, simulate processing requests.
2.  Increment the counter.

### Task 3: Expose (Simulation)
1.  Print the metrics in Prometheus format to `std::cout` every few seconds.
2.  (Optional) In a real app, you'd use `prometheus-cpp` library to serve this on `http://localhost:8080/metrics`.

## Code Hints
```cpp
class Counter {
    std::atomic<int> val{0};
public:
    void inc() { val++; }
    int get() { return val; }
};

// Output
std::cout << "requests_total " << counter.get() << std::endl;
```

## Verification
Run the program. It should output lines looking like Prometheus metrics.
