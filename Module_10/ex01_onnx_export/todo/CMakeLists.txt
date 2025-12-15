cmake_minimum_required(VERSION 3.15)
project(ex01_onnx_export NONE)

# Just copy the python script to the build directory
configure_file(src/export_model.py export_model.py COPYONLY)

add_custom_target(run_export
    COMMAND python export_model.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ONNX export script"
)
