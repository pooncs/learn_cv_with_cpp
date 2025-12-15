#include <iostream>
#include "viewer.hpp"

int main() {
    // Generate a colorful cube
    std::vector<Point3D> cloud;
    for(float x=0; x<1.0; x+=0.05) {
        for(float y=0; y<1.0; y+=0.05) {
            for(float z=0; z<1.0; z+=0.05) {
                cloud.push_back({x, y, z, (uint8_t)(x*255), (uint8_t)(y*255), (uint8_t)(z*255)});
            }
        }
    }
    
    Camera cam(800, 600);
    cam.lookAt({2, 2, 2}, {0.5, 0.5, 0.5}, {0, 1, 0});
    
    cv::Mat img = render_point_cloud(cloud, cam);
    
    std::cout << "Rendered " << cloud.size() << " points.\n";
    // cv::imshow("Viewer", img);
    // cv::waitKey(0);

    return 0;
}
