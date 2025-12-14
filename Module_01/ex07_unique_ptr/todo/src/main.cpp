#include <iostream>
#include <memory>
#include <string>

class Sensor {
public:
    virtual void read() = 0;
    virtual ~Sensor() { std::cout << "Sensor destroyed\n"; }
};

class Camera : public Sensor {
public:
    void read() override { std::cout << "Reading frame\n"; }
    ~Camera() { std::cout << "Camera destroyed\n"; }
};

// TODO: Implement createSensor returning unique_ptr
// std::unique_ptr<Sensor> createSensor(...)

int main() {
    {
        // TODO: Create a sensor and use it
    }
    std::cout << "Scope ended\n";
    return 0;
}
