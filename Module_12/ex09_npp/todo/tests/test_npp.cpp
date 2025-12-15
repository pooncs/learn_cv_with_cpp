#include <gtest/gtest.h>
#include <npp.h>
#include <cuda_runtime.h>

TEST(NPPTest, AllocatesAndFrees) {
    int w = 64;
    int h = 64;
    int step = 0;
    Npp8u* d_ptr = nppiMalloc_8u_C1(w, h, &step);
    
    EXPECT_NE(d_ptr, nullptr);
    EXPECT_GE(step, w); // Step should be at least width
    
    nppiFree(d_ptr);
}
