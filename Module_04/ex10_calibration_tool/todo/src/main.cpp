#include <iostream>
#include "Calibrator.hpp"

int main() {
    cv::Size board_size(9, 6);
    Calibrator calib(board_size, 0.025f);

    std::cout << "Calibration Tool.\n";
    // TODO: Implement main loop

    return 0;
}
