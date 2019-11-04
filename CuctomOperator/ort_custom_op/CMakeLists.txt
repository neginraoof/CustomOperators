cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)

#Include path to header files
include_directories(/home/neraoof/onnxruntime/include/onnxruntime)
include_directories(/home/neraoof/onnxruntime/cmake/external/eigen)
include_directories(/home/neraoof/onnxruntime/onnxruntime/test/util/include)
include_directories(/home/neraoof/onnxruntime/onnxruntime)


set(SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
add_executable(customop ${SOURCE} ${HEADER})


find_library(GTEST_LIBRARY gtest HINTS /usr/lib/x86_64-linux-gnu)
find_library(PROTOBUF_LIBRARY protobuf HINTS /usr/lib/x86_64-linux-gnu)
find_library(THREAD_LIBRARY pthread HINTS /usr/lib/x86_64-linux-gnu)
find_library(ONNXRUNTIME_MOCKED_ALLOCATOR_LIBRARY onnxruntime_mocked_allocator HINTS /home/neraoof/onnxruntime/build/Linux/RelWithDebInfo)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/neraoof/onnxruntime/build/Linux/RelWithDebInfo)


target_link_libraries(customop PUBLIC ${ONNXRUNTIME_MOCKED_ALLOCATOR_LIBRARY})
target_link_libraries(customop PUBLIC ${GTEST_LIBRARY})
target_link_libraries(customop PUBLIC ${PROTOBUF_LIBRARY})
target_link_libraries(customop PUBLIC ${GTEST_LIBRARY})
target_link_libraries(customop PUBLIC ${THREAD_LIBRARY})
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})