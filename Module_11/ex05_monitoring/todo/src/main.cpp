#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

// TODO: Implement Counter class (atomic)

// TODO: Implement Gauge class (atomic)

int main() {
    // TODO: Instantiate metrics

    std::cout << "Starting Metric Simulation..." << std::endl;

    for (int i = 0; i < 5; ++i) {
        // TODO: Simulate work (inc/dec metrics)

        // TODO: Expose Metrics in Prometheus format
        // # HELP ...
        // # TYPE ...
        // name value
    }

    return 0;
}
