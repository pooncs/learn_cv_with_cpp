cmake_minimum_required(VERSION 3.15)
project(CV_CPP_Curriculum)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 1. Dependency Management
# -----------------------------------------------------------------------------

# A. 3rdParty Libraries (Local)
# -----------------------------------------------------------------------------
set(MIDAS_3RDPARTY_DIR "C:/Users/hmgics/projects/midas/3rdParty")

# Add paths to CMAKE_PREFIX_PATH so find_package() can locate them
list(APPEND CMAKE_PREFIX_PATH "${MIDAS_3RDPARTY_DIR}/opencv")
list(APPEND CMAKE_PREFIX_PATH "${MIDAS_3RDPARTY_DIR}/open3d")
list(APPEND CMAKE_PREFIX_PATH "${MIDAS_3RDPARTY_DIR}/qt6")
list(APPEND CMAKE_PREFIX_PATH "${MIDAS_3RDPARTY_DIR}/TensorRT")

# B. Conan Dependencies
# -----------------------------------------------------------------------------
# Assuming 'conan install .' has been run and generated 'conan_toolchain.cmake'
# Users should invoke CMake with: -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake

# If not using toolchain, try to find packages standardly (assuming Conan generator adds them to path)
find_package(Eigen3 REQUIRED)
find_package(gtest REQUIRED)
find_package(benchmark QUIET)
find_package(spdlog QUIET)
find_package(nlohmann_json QUIET)

# -----------------------------------------------------------------------------
# 2. Global Options
# -----------------------------------------------------------------------------
enable_testing()

# -----------------------------------------------------------------------------
# 3. Add Modules
# -----------------------------------------------------------------------------
# We can optionally add all modules here if we want a monolithic build.
# For this curriculum, exercises are often standalone, but we can aggregate them.

# Helper macro to add exercises
macro(add_module_exercises MODULE_NAME EXERCISE_LIST)
    foreach(EX ${EXERCISE_LIST})
        add_subdirectory(${MODULE_NAME}/${EX}/answer)
        # We don't add 'todo' to the build as it might have compilation errors by design
        # or conflict with 'answer' targets if names aren't unique.
    endforeach()
endmacro()

# Define exercises for Module 01
set(MODULE_01_EXERCISES
    ex01_auto ex02_stl_perf ex03_structured_bindings ex04_error_handling
    ex05_variant ex06_lambdas ex07_unique_ptr ex08_shared_ptr
    ex09_move_semantics ex10_raii_wrapper
)
# add_module_exercises(Module_01 "${MODULE_01_EXERCISES}")

# Define exercises for Module 02
set(MODULE_02_EXERCISES
    ex01_matrix_init ex02_arithmetic ex03_block_ops ex04_linear_solvers
    ex05_eigen_decomp ex06_rotations ex07_quaternions ex08_se3_transforms
    ex09_least_squares ex10_geometry_project
)
# add_module_exercises(Module_02 "${MODULE_02_EXERCISES}")

# Define exercises for Module 03
set(MODULE_03_EXERCISES
    ex01_mat_internals ex02_pixel_access ex03_color_spaces ex04_convolution
    ex05_morphology ex06_histograms ex07_thresholding ex08_gradients
    ex09_canny ex10_custom_filter
)
# add_module_exercises(Module_03 "${MODULE_03_EXERCISES}")
