cmake_minimum_required(VERSION 3.22.1)
project(samkit-android)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include core library from root
set(SAMKIT_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../core)

# Core sources
file(GLOB_RECURSE SAMKIT_CORE_SOURCES
    ${SAMKIT_CORE_DIR}/src/*.cpp
    ${SAMKIT_CORE_DIR}/src/*.cc
)

# JNI sources
file(GLOB_RECURSE JNI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

# Create shared library
add_library(samkit-jni SHARED
    ${SAMKIT_CORE_SOURCES}
    ${JNI_SOURCES}
)

# Include directories
target_include_directories(samkit-jni PRIVATE
    ${SAMKIT_CORE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Find required libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Link libraries
target_link_libraries(samkit-jni
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
)

# Compiler flags
target_compile_options(samkit-jni PRIVATE
    -Wall
    -Wextra
    -fvisibility=hidden
    -ffunction-sections
    -fdata-sections
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-ffast-math>
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
)

# Linker flags
target_link_options(samkit-jni PRIVATE
    -Wl,--gc-sections
    -Wl,--exclude-libs,ALL
)

# ARM NEON optimizations
if(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    target_compile_options(samkit-jni PRIVATE -mfpu=neon -mfloat-abi=softfp)
elseif(${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(samkit-jni PRIVATE -march=armv8-a+fp16)
endif()