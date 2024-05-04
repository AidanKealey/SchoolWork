#
#

project(assign4 LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.16)

# specify location of custom installed cyclone location
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(LINUX)
    message(STATUS ">>> Linux")
    set (CMAKE_PREFIX_PATH $ENV{HOME}/DDS/cyclonedds/install;$ENV{HOME}/DDS/cyclonedds-cxx/install)
endif()
if(APPLE)
    message(STATUS ">>> Mac OS")
    set(CMAKE_PREFIX_PATH /Users/aidankealey/Documents/fifth_year/ELEC_477/team25/lib/cmake) # /Users/dean/Teaching/477/DDS/install/lib/cmake)
    LIST(APPEND CMAKE_PROGRAM_PATH  "/Users/aidankealey/Documents/fifth_year/ELEC_477/team25/bin") # "/Users/dean/Teaching/477/DDS/install/bin")
endif()

# CXX standard at least 
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

if(NOT TARGET CycloneDDS-CXX::ddscxx)
  find_package(CycloneDDS-CXX REQUIRED)
endif()

# generation of code from idl files. One line per type
idlcxx_generate(TARGET statedata FILES statekey.idl WARNINGS no-implicit-extensibility)
idlcxx_generate(TARGET zonedata FILES zoneTransfer.idl WARNINGS no-implicit-extensibility)

add_executable(flights flights.cpp aircraft.cpp)
# add_executable(subscriber subscriber.cpp)
add_executable(torontoAD torontoAD.cpp)
add_executable(torontoCentre torontoCentre.cpp)
add_executable(query query.cpp)
# add_executable(genData genData.cpp)

# Link both executables to idl data type library and ddscxx.
target_link_libraries(flights CycloneDDS-CXX::ddscxx statedata)
# target_link_libraries(subscriber CycloneDDS-CXX::ddscxx statedata)
target_link_libraries(torontoAD CycloneDDS-CXX::ddscxx statedata zonedata)
target_link_libraries(torontoCentre CycloneDDS-CXX::ddscxx statedata zonedata)
target_link_libraries(query CycloneDDS-CXX::ddscxx zonedata)

set_property(TARGET flights PROPERTY CXX_STANDARD ${cyclonedds_cpp_std_to_use})
# set_property(TARGET subscriber PROPERTY CXX_STANDARD ${cyclonedds_cpp_std_to_use})
set_property(TARGET torontoAD PROPERTY CXX_STANDARD ${cyclonedds_cpp_std_to_use})
set_property(TARGET torontoCentre PROPERTY CXX_STANDARD ${cyclonedds_cpp_std_to_use})
set_property(TARGET query PROPERTY CXX_STANDARD ${cyclonedds_cpp_std_to_use})
