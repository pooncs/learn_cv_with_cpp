# Sample Toolchain File
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake ..

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# In a real scenario, you would point to arm-linux-gnueabihf-g++
# Here we just use the host compiler for demonstration
if(WIN32)
    # Just use cl or g++ if available
    # set(CMAKE_CXX_COMPILER "cl.exe")
else()
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()

# Don't look for programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Look for libraries in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
