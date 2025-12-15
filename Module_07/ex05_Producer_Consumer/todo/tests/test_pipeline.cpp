#include <gtest/gtest.h>
#include <thread>
#include "pipeline.hpp"

TEST(PipelineTest, StartStop) {
    cv_curriculum::Pipeline pipeline;
    
    // Just verify it doesn't crash or hang
    pipeline.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    pipeline.stop();
    
    SUCCEED();
}
