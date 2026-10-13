add_library(flexeva_maya src/capture.cpp src/engine.cpp src/paper_cost.cpp src/paper_runtime.cpp src/paper_partitions.cpp src/paper_dependencies.cpp src/paper_dependencies_index.cpp src/paper_scheduler.cpp src/paper_feedback.cpp)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_library(FlexEva::maya ALIAS flexeva_maya)
target_include_directories(flexeva_maya PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/maya/cpp/fake_cuda/include/common)
target_link_libraries(flexeva_maya PUBLIC FlexEva::core FlexEva::code_layer fake_cuda_core OpenSSL::Crypto)
target_compile_features(flexeva_maya PUBLIC cxx_std_20)
set_target_properties(flexeva_maya PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(flexeva_maya PRIVATE -Wall -Wextra -Wpedantic)

option(FLEXEVA_MAYA_BUILD_PYTHON "Build the in-process Maya workload binding" OFF)
if(FLEXEVA_MAYA_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    Python3_add_library(_flexeva_maya_paper MODULE src/python_bindings.cpp WITH_SOABI)
    target_link_libraries(_flexeva_maya_paper PRIVATE FlexEva::maya)
    target_compile_features(_flexeva_maya_paper PRIVATE cxx_std_20)
    target_compile_options(_flexeva_maya_paper PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(_flexeva_maya_paper PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
endif()
