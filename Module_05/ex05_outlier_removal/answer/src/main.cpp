#include <iostream>
#include "sor.hpp"

int main() {
    std::vector<Point3D> cloud;
    // Dense cluster
    for(int i=0; i<50; ++i) cloud.push_back({(float)i*0.1f, 0, 0});
    
    // Outliers
    cloud.push_back({100, 100, 100});
    cloud.push_back({-50, -50, -50});
    
    auto clean = remove_outliers(cloud, 5, 1.0f);
    
    std::cout << "Original: " << cloud.size() << "\n";
    std::cout << "Filtered: " << clean.size() << "\n"; // Should be 50

    return 0;
}
