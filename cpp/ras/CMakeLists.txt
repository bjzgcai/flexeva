cmake_minimum_required(VERSION 3.20)
project(flexeva_ras VERSION 0.1.0 LANGUAGES CXX)

add_library(flexeva_ras src/ras.cpp)
add_library(FlexEva::ras ALIAS flexeva_ras)
set_target_properties(flexeva_ras PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(flexeva_ras PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(flexeva_ras PUBLIC cxx_std_20)
set_target_properties(flexeva_ras PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(flexeva_ras PRIVATE -Wall -Wextra -Wpedantic)

option(FLEXEVA_RAS_BUILD_TESTS "Build the two RAS acceptance entrypoints" OFF)
if(FLEXEVA_RAS_BUILD_TESTS)
    enable_testing()
    foreach(test_name IN ITEMS partition_refresh_contract refresh_storage_scaling)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE FlexEva::ras)
        target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
    add_test(NAME partition_refresh_contract COMMAND partition_refresh_contract)
    # The scaling executable is opt-in: it allocates a million-partition state.
endif()
