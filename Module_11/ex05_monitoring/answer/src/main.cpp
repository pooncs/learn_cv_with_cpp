#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

class Counter {
    std::atomic<int> val{0};
public:
    void inc() { val++; }
    int get() const { return val.load(); }
};

class Gauge {
    std::atomic<int> val{0};
public:
    void set(int v) { val.store(v); }
    void inc() { val++; }
    void dec() { val--; }
    int get() const { return val.load(); }
};

int main() {
    Counter requests;
    Gauge active_workers;

    std::cout << "Starting Metric Simulation..." << std::endl;

    for (int i = 0; i < 5; ++i) {
        // Simulate work
        active_workers.inc();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        requests.inc();
        active_workers.dec();

        // Expose Metrics (Simulated HTTP response)
        std::cout << "--- /metrics ---" << std::endl;
        std::cout << "# HELP requests_total Total number of requests" << std::endl;
        std::cout << "# TYPE requests_total counter" << std::endl;
        std::cout << "requests_total " << requests.get() << std::endl;
        
        std::cout << "# HELP active_workers Number of workers currently processing" << std::endl;
        std::cout << "# TYPE active_workers gauge" << std::endl;
        std::cout << "active_workers " << active_workers.get() << std::endl;
        std::cout << "----------------" << std::endl;
    }

    return 0;
}
