# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ls/slambook_self_practice/ch2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ls/slambook_self_practice/ch2/build

# Include any dependencies generated for this target.
include CMakeFiles/hellslam.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hellslam.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hellslam.dir/flags.make

CMakeFiles/hellslam.dir/helloslam.cpp.o: CMakeFiles/hellslam.dir/flags.make
CMakeFiles/hellslam.dir/helloslam.cpp.o: ../helloslam.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ls/slambook_self_practice/ch2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hellslam.dir/helloslam.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hellslam.dir/helloslam.cpp.o -c /home/ls/slambook_self_practice/ch2/helloslam.cpp

CMakeFiles/hellslam.dir/helloslam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hellslam.dir/helloslam.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ls/slambook_self_practice/ch2/helloslam.cpp > CMakeFiles/hellslam.dir/helloslam.cpp.i

CMakeFiles/hellslam.dir/helloslam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hellslam.dir/helloslam.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ls/slambook_self_practice/ch2/helloslam.cpp -o CMakeFiles/hellslam.dir/helloslam.cpp.s

CMakeFiles/hellslam.dir/helloslam.cpp.o.requires:

.PHONY : CMakeFiles/hellslam.dir/helloslam.cpp.o.requires

CMakeFiles/hellslam.dir/helloslam.cpp.o.provides: CMakeFiles/hellslam.dir/helloslam.cpp.o.requires
	$(MAKE) -f CMakeFiles/hellslam.dir/build.make CMakeFiles/hellslam.dir/helloslam.cpp.o.provides.build
.PHONY : CMakeFiles/hellslam.dir/helloslam.cpp.o.provides

CMakeFiles/hellslam.dir/helloslam.cpp.o.provides.build: CMakeFiles/hellslam.dir/helloslam.cpp.o


# Object files for target hellslam
hellslam_OBJECTS = \
"CMakeFiles/hellslam.dir/helloslam.cpp.o"

# External object files for target hellslam
hellslam_EXTERNAL_OBJECTS =

hellslam: CMakeFiles/hellslam.dir/helloslam.cpp.o
hellslam: CMakeFiles/hellslam.dir/build.make
hellslam: CMakeFiles/hellslam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ls/slambook_self_practice/ch2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hellslam"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hellslam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hellslam.dir/build: hellslam

.PHONY : CMakeFiles/hellslam.dir/build

CMakeFiles/hellslam.dir/requires: CMakeFiles/hellslam.dir/helloslam.cpp.o.requires

.PHONY : CMakeFiles/hellslam.dir/requires

CMakeFiles/hellslam.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hellslam.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hellslam.dir/clean

CMakeFiles/hellslam.dir/depend:
	cd /home/ls/slambook_self_practice/ch2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ls/slambook_self_practice/ch2 /home/ls/slambook_self_practice/ch2 /home/ls/slambook_self_practice/ch2/build /home/ls/slambook_self_practice/ch2/build /home/ls/slambook_self_practice/ch2/build/CMakeFiles/hellslam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hellslam.dir/depend

