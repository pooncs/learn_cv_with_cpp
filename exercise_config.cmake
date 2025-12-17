# exercise_config.cmake
# Common configuration for all exercises to find 3rdParty libraries

# 3rdParty Libraries (Local)
set(MIDAS_3RDPARTY_DIR "C:/Users/hmgics/projects/midas/3rdParty")

# Add paths to CMAKE_PREFIX_PATH so find_package() can locate them
# Use PREPEND to ensure these are found BEFORE Conan or system packages
list(PREPEND CMAKE_PREFIX_PATH "${MIDAS_3RDPARTY_DIR}/opencv")
list(PREPEND CMAKE_PREFIX_PATH "${MIDAS_3RDPARTY_DIR}/open3d")
list(PREPEND CMAKE_PREFIX_PATH "${MIDAS_3RDPARTY_DIR}/qt6")
list(PREPEND CMAKE_PREFIX_PATH "${MIDAS_3RDPARTY_DIR}/TensorRT")

message(STATUS "3rdParty Prefix Path: ${CMAKE_PREFIX_PATH}")
