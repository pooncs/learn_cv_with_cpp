cmake_minimum_required(VERSION 3.14) # FetchContent needs 3.11+, 3.14 recommended
project(ex15_fetch LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.1.1 # Use a specific tag/commit for stability
)

FetchContent_MakeAvailable(fmt)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE fmt::fmt)
