cmake_minimum_required(VERSION 3.15)
project(ex02_cache CXX)

find_package(fmt REQUIRED)
find_package(OpenSSL REQUIRED)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE src)
target_link_libraries(main PRIVATE fmt::fmt OpenSSL::SSL OpenSSL::Crypto)
target_compile_features(main PRIVATE cxx_std_17)
