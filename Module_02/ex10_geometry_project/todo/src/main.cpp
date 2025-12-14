#include <iostream>
#include "transformer.hpp"

int main() {
    Transformer transformer;
    
    // Intrinsics: 640x480 camera, focal length 500
    transformer.set_intrinsics(500, 500, 320, 240);

    // Extrinsics: Camera at (0, 0, -2) looking at World Origin
    // T_wc: Trans = (0,0,-2), Rot = Identity
    // T_cw = T_wc.inv()
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    T_wc(2, 3) = -2.0;
    transformer.set_extrinsics(T_wc.inverse());

    // Test Point at World Origin (0,0,0)
    // Camera is at -2 Z, so point is at +2 Z in camera frame.
    // Project: (320, 240) center of image
    Eigen::Vector3d p_world(0, 0, 0);
    Eigen::Vector2d p_pix = transformer.project(p_world);

    std::cout << "Project (0,0,0): " << p_pix.transpose() << "\n";

    // Back project center pixel with depth 2.0
    Eigen::Vector3d p_world_rec = transformer.back_project(p_pix, 2.0);
    std::cout << "Back Project (center, 2.0): " << p_world_rec.transpose() << "\n";

    return 0;
}
