# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build

# Include any dependencies generated for this target.
include CMakeFiles/subscriber.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/subscriber.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/subscriber.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/subscriber.dir/flags.make

statekey.hpp: /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/statekey.idl
statekey.hpp: /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/lib/libcycloneddsidlcxx.0.11.0.dylib
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating statekey.hpp, statekey.cpp"
	/Users/aidankealey/Documents/fifth_year/ELEC_477/team25/bin/idlc -l/Users/aidankealey/Documents/fifth_year/ELEC_477/team25/lib/libcycloneddsidlcxx.0.11.0.dylib -Wno-implicit-extensibility -o/Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/statekey.idl

statekey.cpp: statekey.hpp
	@$(CMAKE_COMMAND) -E touch_nocreate statekey.cpp

CMakeFiles/subscriber.dir/subscriber.cpp.o: CMakeFiles/subscriber.dir/flags.make
CMakeFiles/subscriber.dir/subscriber.cpp.o: /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/subscriber.cpp
CMakeFiles/subscriber.dir/subscriber.cpp.o: CMakeFiles/subscriber.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/subscriber.dir/subscriber.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/subscriber.dir/subscriber.cpp.o -MF CMakeFiles/subscriber.dir/subscriber.cpp.o.d -o CMakeFiles/subscriber.dir/subscriber.cpp.o -c /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/subscriber.cpp

CMakeFiles/subscriber.dir/subscriber.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/subscriber.dir/subscriber.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/subscriber.cpp > CMakeFiles/subscriber.dir/subscriber.cpp.i

CMakeFiles/subscriber.dir/subscriber.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/subscriber.dir/subscriber.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/subscriber.cpp -o CMakeFiles/subscriber.dir/subscriber.cpp.s

CMakeFiles/subscriber.dir/statekey.cpp.o: CMakeFiles/subscriber.dir/flags.make
CMakeFiles/subscriber.dir/statekey.cpp.o: statekey.cpp
CMakeFiles/subscriber.dir/statekey.cpp.o: CMakeFiles/subscriber.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/subscriber.dir/statekey.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/subscriber.dir/statekey.cpp.o -MF CMakeFiles/subscriber.dir/statekey.cpp.o.d -o CMakeFiles/subscriber.dir/statekey.cpp.o -c /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build/statekey.cpp

CMakeFiles/subscriber.dir/statekey.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/subscriber.dir/statekey.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build/statekey.cpp > CMakeFiles/subscriber.dir/statekey.cpp.i

CMakeFiles/subscriber.dir/statekey.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/subscriber.dir/statekey.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build/statekey.cpp -o CMakeFiles/subscriber.dir/statekey.cpp.s

# Object files for target subscriber
subscriber_OBJECTS = \
"CMakeFiles/subscriber.dir/subscriber.cpp.o" \
"CMakeFiles/subscriber.dir/statekey.cpp.o"

# External object files for target subscriber
subscriber_EXTERNAL_OBJECTS =

subscriber: CMakeFiles/subscriber.dir/subscriber.cpp.o
subscriber: CMakeFiles/subscriber.dir/statekey.cpp.o
subscriber: CMakeFiles/subscriber.dir/build.make
subscriber: /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/lib/libddscxx.0.11.0.dylib
subscriber: /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/lib/libddsc.0.11.0.dylib
subscriber: CMakeFiles/subscriber.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable subscriber"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/subscriber.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/subscriber.dir/build: subscriber
.PHONY : CMakeFiles/subscriber.dir/build

CMakeFiles/subscriber.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/subscriber.dir/cmake_clean.cmake
.PHONY : CMakeFiles/subscriber.dir/clean

CMakeFiles/subscriber.dir/depend: statekey.cpp
CMakeFiles/subscriber.dir/depend: statekey.hpp
	cd /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4 /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4 /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/a4/build/CMakeFiles/subscriber.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/subscriber.dir/depend
