cmake_minimum_required(VERSION 3.15)
project(ex01_onnx_export_answer NONE)

configure_file(src/export_model.py export_model.py COPYONLY)

add_custom_target(run_export_answer
    COMMAND python export_model.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ONNX export script (Answer)"
)
