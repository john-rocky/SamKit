cmake_minimum_required(VERSION 3.20)
project(SAMKit VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ENABLE_OPENCV "Enable OpenCV for image processing" OFF)

# Find packages
find_package(Threads REQUIRED)

if(ENABLE_OPENCV)
    find_package(OpenCV QUIET)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    else()
        message(WARNING "OpenCV not found, some features will be disabled")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/include)

# Core library
file(GLOB_RECURSE SAMKIT_CORE_SOURCES 
    core/src/*.cpp
    core/src/*.cc
)

file(GLOB_RECURSE SAMKIT_CORE_HEADERS
    core/include/*.h
    core/include/*.hpp
)

add_library(samkit_core ${SAMKIT_CORE_SOURCES})
target_include_directories(samkit_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(samkit_core PUBLIC Threads::Threads)

if(OpenCV_FOUND AND ENABLE_OPENCV)
    target_link_libraries(samkit_core PUBLIC ${OpenCV_LIBS})
    target_compile_definitions(samkit_core PUBLIC SAMKIT_USE_OPENCV)
endif()

# Platform-specific settings
if(APPLE)
    target_compile_options(samkit_core PRIVATE
        -Wall -Wextra -Wpedantic
        -fvisibility=hidden
    )
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        target_compile_options(samkit_core PRIVATE -march=armv8.2-a+fp16)
    endif()
elseif(ANDROID)
    target_compile_options(samkit_core PRIVATE
        -Wall -Wextra
        -fvisibility=hidden
        -ffunction-sections
        -fdata-sections
    )
    target_link_options(samkit_core PRIVATE
        -Wl,--gc-sections
        -Wl,--exclude-libs,ALL
    )
else()
    target_compile_options(samkit_core PRIVATE
        -Wall -Wextra -Wpedantic
        -fvisibility=hidden
        -march=native
    )
endif()

# Optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(samkit_core PRIVATE -O3 -DNDEBUG)
    if(NOT MSVC)
        target_compile_options(samkit_core PRIVATE -ffast-math)
    endif()
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    # Download and configure Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Add test subdirectory
    add_subdirectory(core/tests)
endif()

# Installation
install(TARGETS samkit_core
    EXPORT SAMKitTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY core/include/samkit
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(EXPORT SAMKitTargets
    FILE SAMKitTargets.cmake
    NAMESPACE SAMKit::
    DESTINATION lib/cmake/SAMKit
)

# Config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SAMKitConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/SAMKitConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SAMKitConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SAMKitConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SAMKitConfigVersion.cmake"
    DESTINATION lib/cmake/SAMKit
)

# Export compile commands for development
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Summary
message(STATUS "")
message(STATUS "SAMKit Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "  OpenCV support: ${ENABLE_OPENCV}")
if(OpenCV_FOUND AND ENABLE_OPENCV)
    message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
endif()
message(STATUS "")