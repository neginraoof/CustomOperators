# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/anaconda3/envs/pytorchonnx/bin/cmake

# The command to remove a file.
RM = /opt/anaconda3/envs/pytorchonnx/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/build

# Include any dependencies generated for this target.
include CMakeFiles/customop.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/customop.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/customop.dir/flags.make

CMakeFiles/customop.dir/custom_op_test.cc.o: CMakeFiles/customop.dir/flags.make
CMakeFiles/customop.dir/custom_op_test.cc.o: ../custom_op_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/customop.dir/custom_op_test.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/customop.dir/custom_op_test.cc.o -c /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/custom_op_test.cc

CMakeFiles/customop.dir/custom_op_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/customop.dir/custom_op_test.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/custom_op_test.cc > CMakeFiles/customop.dir/custom_op_test.cc.i

CMakeFiles/customop.dir/custom_op_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/customop.dir/custom_op_test.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/custom_op_test.cc -o CMakeFiles/customop.dir/custom_op_test.cc.s

# Object files for target customop
customop_OBJECTS = \
"CMakeFiles/customop.dir/custom_op_test.cc.o"

# External object files for target customop
customop_EXTERNAL_OBJECTS =

customop: CMakeFiles/customop.dir/custom_op_test.cc.o
customop: CMakeFiles/customop.dir/build.make
customop: /home/neraoof/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime_mocked_allocator.a
customop: /usr/lib/libgtest.a
customop: /usr/lib/x86_64-linux-gnu/libprotobuf.so
customop: /usr/lib/libgtest.a
customop: /usr/lib/x86_64-linux-gnu/libpthread.so
customop: /home/neraoof/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so
customop: /usr/lib/x86_64-linux-gnu/libprotobuf.so
customop: /usr/lib/x86_64-linux-gnu/libpthread.so
customop: /home/neraoof/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so
customop: CMakeFiles/customop.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable customop"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/customop.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/customop.dir/build: customop

.PHONY : CMakeFiles/customop.dir/build

CMakeFiles/customop.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/customop.dir/cmake_clean.cmake
.PHONY : CMakeFiles/customop.dir/clean

CMakeFiles/customop.dir/depend:
	cd /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/build /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/build /home/neraoof/CustomOperators/CuctomOperator/ort_custom_op/build/CMakeFiles/customop.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/customop.dir/depend
