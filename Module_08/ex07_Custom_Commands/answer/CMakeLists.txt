cmake_minimum_required(VERSION 3.15)
project(ex07_custom_commands_answer)

set(CMAKE_CXX_STANDARD 17)

set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
set(VERSION_HEADER "${GENERATED_DIR}/version.h")

file(MAKE_DIRECTORY ${GENERATED_DIR})

# Generate version.h
# Note: On Windows 'echo' might behave differently in shell vs command.
# Using cmake -E echo is safer cross-platform.
add_custom_command(
    OUTPUT ${VERSION_HEADER}
    COMMAND ${CMAKE_COMMAND} -E echo "#define PROJECT_VERSION \"1.0.0\"" > ${VERSION_HEADER}
    COMMENT "Generating version.h"
    VERBATIM
)

add_executable(app src/main.cpp ${VERSION_HEADER})

target_include_directories(app PRIVATE ${GENERATED_DIR})
