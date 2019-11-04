cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)

set(SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
add_executable(customop ${SOURCE} ${HEADER})

#Include path to header files
include_directories(<PATH_TO_ONNXRUNTIME_EIGEN_HEADER_FILE>)

#Include path to header files for unit test
include_directories(<PATH_TO_ONNXRUNTIME_INCLUDE_DIR>)
include_directories(<PATH_TO_ONNXRUNTIME_TEST_UTIL_INCLUDE_DIR>)
include_directories(<PATH_TO_ONNXRUNTIME_INCLUDE_DIR>)


#Link libraries needed for unit test
find_library(GTEST_LIBRARY gtest HINTS /usr/lib/x86_64-linux-gnu)
find_library(PROTOBUF_LIBRARY protobuf HINTS /usr/lib/x86_64-linux-gnu)
find_library(THREAD_LIBRARY pthread HINTS /usr/lib/x86_64-linux-gnu)
find_library(ONNXRUNTIME_MOCKED_ALLOCATOR_LIBRARY onnxruntime_mocked_allocator HINTS <PATH_TO_ONNXRUNTIME_BUILD_DIR>)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS <PATH_TO_ONNXRUNTIME_BUILD_DIR>)


target_link_libraries(customop PUBLIC ${ONNXRUNTIME_MOCKED_ALLOCATOR_LIBRARY})
target_link_libraries(customop PUBLIC ${GTEST_LIBRARY})
target_link_libraries(customop PUBLIC ${PROTOBUF_LIBRARY})
target_link_libraries(customop PUBLIC ${GTEST_LIBRARY})
target_link_libraries(customop PUBLIC ${THREAD_LIBRARY})
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})