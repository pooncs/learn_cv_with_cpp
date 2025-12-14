#include <iostream>
#include <vector>
#include <list>
#include <deque>
#include <chrono>
#include <numeric>

int main() {
    const int N = 100000;
    
    // Benchmark Vector
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> vec;
    for(int i=0; i<N; ++i) vec.push_back(i);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Vector push_back: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us\n";

    // Benchmark List
    start = std::chrono::high_resolution_clock::now();
    std::list<int> lst;
    for(int i=0; i<N; ++i) lst.push_back(i);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "List push_back: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us\n";

    // Random Access Benchmark
    start = std::chrono::high_resolution_clock::now();
    long long sum = 0;
    for(int i=0; i<N; ++i) sum += vec[i];
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Vector access: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us\n";

    // List access (simulation)
    start = std::chrono::high_resolution_clock::now();
    sum = 0;
    for(auto it = lst.begin(); it != lst.end(); ++it) sum += *it;
    end = std::chrono::high_resolution_clock::now();
    std::cout << "List traversal: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us\n";

    return 0;
}
