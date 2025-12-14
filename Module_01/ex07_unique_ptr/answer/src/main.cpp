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

class Lidar : public Sensor {
public:
    void read() override { std::cout << "Reading point cloud\n"; }
    ~Lidar() { std::cout << "Lidar destroyed\n"; }
};

std::unique_ptr<Sensor> createSensor(const std::string& type) {
    if (type == "camera") return std::make_unique<Camera>();
    if (type == "lidar") return std::make_unique<Lidar>();
    return nullptr;
}

int main() {
    {
        auto sensor = createSensor("camera");
        if(sensor) sensor->read();
        // Automatically destroyed here
    }
    std::cout << "Scope ended\n";
    return 0;
}
