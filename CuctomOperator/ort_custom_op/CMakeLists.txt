cmake_minimum_required(VERSION 2.8.9)
project (customop)

#Include path to header files
include_directories(/home/neraoof/onnxruntime/include/onnxruntime)
include_directories(/home/neraoof/onnxruntime/onnxruntime/test/util/include)
include_directories(/home/neraoof/onnxruntime/onnxruntime)
include_directories(/home/neraoof/onnxruntime/cmake/external/eigen)

add_executable(customop custom_op_test.cc)

find_library(GTEST_LIBRARY gtest HINTS /usr/lib/x86_64-linux-gnu)
find_library(PROTOBUF_LIBRARY protobuf HINTS /usr/lib/x86_64-linux-gnu)
find_library(THREAD_LIBRARY pthread HINTS /usr/lib/x86_64-linux-gnu)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/neraoof/onnxruntime/build/Linux/RelWithDebInfo)
find_library(ONNXRUNTIME_MOCKED_ALLOCATOR_LIBRARY onnxruntime_mocked_allocator HINTS /home/neraoof/onnxruntime/build/Linux/RelWithDebInfo)


target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_MOCKED_ALLOCATOR_LIBRARY})
target_link_libraries(customop PUBLIC ${GTEST_LIBRARY})
target_link_libraries(customop PUBLIC ${PROTOBUF_LIBRARY})
target_link_libraries(customop PUBLIC ${GTEST_LIBRARY})
target_link_libraries(customop PUBLIC ${THREAD_LIBRARY})

