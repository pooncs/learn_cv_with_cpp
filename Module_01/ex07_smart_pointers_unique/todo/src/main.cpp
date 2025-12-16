#include <iostream>
#include <memory> // Required for smart pointers

// Task 1: Create a Sensor Class
class Sensor {
public:
    Sensor() { std::cout << "Sensor initialized.\n"; }
    ~Sensor() { std::cout << "Sensor shutdown.\n"; }
    
    void read() { std::cout << "Reading data...\n"; }
};

// Task 2: Factory Function
// TODO: Return a std::unique_ptr<Sensor>
// std::unique_ptr<Sensor> create_sensor() { ... }
Sensor* create_sensor_legacy() {
    return new Sensor();
}

// Task 4: Function taking ownership
void process_sensor(std::unique_ptr<Sensor> s) {
    if (s) {
        std::cout << "Processing sensor...\n";
        s->read();
    }
    std::cout << "process_sensor finishing. Sensor will be destroyed here.\n";
}

int main() {
    std::cout << "--- Legacy Way ---\n";
    Sensor* raw_s = create_sensor_legacy();
    raw_s->read();
    delete raw_s; // Don't forget this!

    std::cout << "\n--- Modern Way ---\n";
    // Task 2: Call your factory
    // auto unique_s = create_sensor();

    // Task 3: Try to copy (uncomment to see error)
    // auto s2 = unique_s; 

    // Task 3: Move ownership
    // auto s2 = std::move(unique_s);
    
    // Task 4: Pass to function
    // process_sensor(std::move(s2));
    
    return 0;
}
