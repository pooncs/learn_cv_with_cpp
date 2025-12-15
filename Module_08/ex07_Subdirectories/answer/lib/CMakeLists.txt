add_library(my_lib my_lib.cpp)
# Note: CMAKE_CURRENT_SOURCE_DIR is implicitly used for source files
# But for include directories, we need to specify where headers are.
# Since my_lib.hpp is in the same folder as CMakeLists.txt (lib/), we use .
target_include_directories(my_lib PUBLIC .)
