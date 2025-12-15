cmake_minimum_required(VERSION 3.10)
project(ex12_install LANGUAGES CXX)

add_executable(my_app src/main.cpp)

# Install executable to bin/
install(TARGETS my_app DESTINATION bin)

# Install readme to share/my_app/
install(FILES src/readme.txt DESTINATION share/my_app)
