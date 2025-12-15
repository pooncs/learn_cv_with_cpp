# Exercise 05: Deployment Monitoring

## Goal
Expose metrics from your C++ application for Prometheus to scrape.

## Learning Objectives
1.  **Metrics:** Counters (requests processed) and Gauges (memory usage).
2.  **Prometheus Client:** Use a C++ library (like `prometheus-cpp`) to expose an HTTP endpoint `/metrics`.
3.  **Grafana:** Visualize the data.

## Practical Motivation
You need to know if your service is healthy or if latency is spiking.

## Step-by-Step Instructions
1.  Integrate `prometheus-cpp`.
2.  Increment a counter every time `process_image()` is called.
3.  Start an HTTP server on port 8080.

## Verification
`curl localhost:8080/metrics` should show your custom counter.
