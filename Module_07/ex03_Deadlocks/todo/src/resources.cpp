#include "resources.hpp"
#include <thread>
#include <chrono>

namespace cv_curriculum {

void unsafeSwap(Resource& r1, Resource& r2) {
    // TODO: Implement naive locking (lock r1, sleep, lock r2)
}

void safeSwap(Resource& r1, Resource& r2) {
    // TODO: Implement safe locking using std::scoped_lock or std::lock
}

} // namespace cv_curriculum
