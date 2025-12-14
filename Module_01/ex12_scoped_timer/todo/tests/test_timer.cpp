#include <gtest/gtest.h>
#include "ScopedTimer.hpp"
#include <sstream>

// Since ScopedTimer prints to stdout, we can capture stdout to verify?
// Or we can just ensure it compiles and runs without crashing.
// Proper testing might require dependency injection of the output stream.

TEST(ScopedTimerTest, CompilesAndRuns) {
    {
        ScopedTimer t("TestTimer");
    }
    SUCCEED();
}
