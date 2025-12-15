#include <iostream>
#include "Calibrator.hpp"

int main() {
    // This example simulates a calibration loop. 
    // In a real app, use cv::VideoCapture(0)
    
    cv::Size board_size(9, 6);
    Calibrator calib(board_size, 0.025f); // 25mm squares

    std::cout << "Calibration Tool Simulation.\n";
    std::cout << "In real usage, press 'c' to capture, 's' to save and exit.\n";

    // Simulate capturing a few frames
    // Since we don't have a camera or images, we just show the logic structure.
    
    /*
    cv::VideoCapture cap(0);
    while(true) {
        cv::Mat frame;
        cap >> frame;
        if(frame.empty()) break;
        
        cv::Mat display;
        calib.process_frame(frame, display);
        
        cv::imshow("Calibration", display);
        char key = (char)cv::waitKey(30);
        
        if (key == 'c') {
            calib.add_sample();
            std::cout << "Samples: " << calib.get_sample_count() << "\n";
        }
        if (key == 's') {
            double rms = calib.calibrate();
            std::cout << "RMS: " << rms << "\n";
            calib.save("calib.yml");
            break;
        }
    }
    */

    return 0;
}
