#include <iostream>
#include <memory>

class Sensor {
public:
    Sensor() { std::cout << "Sensor initialized.\n"; }
    ~Sensor() { std::cout << "Sensor shutdown.\n"; }
    
    void read() { std::cout << "Reading data...\n"; }
};

// Factory function returning unique_ptr
std::unique_ptr<Sensor> create_sensor() {
    // std::make_unique is safer and cleaner than 'new'
    return std::make_unique<Sensor>();
}

// Function that takes ownership of the sensor
// The sensor dies when this function returns
void process_sensor(std::unique_ptr<Sensor> s) {
    if (s) {
        std::cout << "Processing sensor...\n";
        s->read();
    }
    std::cout << "process_sensor finishing. Sensor will be destroyed here.\n";
}

int main() {
    std::cout << "--- Creating Sensor ---\n";
    auto my_sensor = create_sensor();
    
    if (my_sensor) {
        my_sensor->read();
    }

    // std::unique_ptr<Sensor> copy = my_sensor; // ERROR: Call to implicitly-deleted copy constructor

    std::cout << "--- Moving Sensor ---\n";
    // Transfer ownership to 'owner2'
    auto owner2 = std::move(my_sensor);

    if (!my_sensor) {
        std::cout << "my_sensor is now empty (nullptr).\n";
    }
    if (owner2) {
        std::cout << "owner2 now has the sensor.\n";
    }

    std::cout << "--- Passing to Function ---\n";
    // Transfer ownership to the function
    process_sensor(std::move(owner2));

    std::cout << "--- End of Main ---\n";
    return 0;
}
