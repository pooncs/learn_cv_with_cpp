#include <glog/logging.h>

int main(int argc, char* argv[]) {
    // Init glog
    google::InitGoogleLogging(argv[0]);
    
    // Log to stderr
    FLAGS_logtostderr = 1;

    LOG(INFO) << "Starting application...";
    LOG(WARNING) << "This is a warning";
    
    int a = 10;
    int b = 5;
    CHECK_GT(a, b) << "a should be greater than b";

    LOG(INFO) << "Done.";
    
    return 0;
}
