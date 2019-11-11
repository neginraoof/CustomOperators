cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.h)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files
include_directories(<PATH_TO_ONNXRUNTIME_INCLUDE_DIR>)
include_directories(<PATH_TO_ONNXRUNTIME_DIR>)
include_directories(<PATH_TO_EIGEN_DIR>)
include_directories(<PATH_TO_ONNXRUNTIME_TEST_UTIL_INCLUDE_DIR>)

#User libraires path could be: /usr/lib/x86_64-linux-gnu
find_library(GTEST_LIBRARY gtest HINTS <PATH_TO_USER_LIB>)
target_link_libraries(customop PUBLIC ${GTEST_LIBRARY})

find_library(PROTOBUF_LIBRARY protobuf HINTS <PATH_TO_USER_LIB>)
target_link_libraries(customop PUBLIC ${PROTOBUF_LIBRARY})

find_library(THREAD_LIBRARY pthread HINTS <PATH_TO_USER_LIB>)
target_link_libraries(customop PUBLIC ${THREAD_LIBRARY})

find_library(ONNXRUNTIME_MOCKED_ALLOCATOR_LIBRARY onnxruntime_mocked_allocator HINTS <PATH_TO_ONNXRUNTIME_LIB>)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_MOCKED_ALLOCATOR_LIBRARY})

find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS <PATH_TO_ONNXRUNTIME_LIB>)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
