cmake_minimum_required(VERSION 3.22.1)

project(
    ImageFilter_CUDA
    VERSION 1.0.0
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_WARNINGS_AS_ERRORS "Enable warnings as errors" ON)
option(ENABLE_SANITIZE_ADDRESS "Enable address sanitizers" ON)
option(ENABLE_SANITIZE_UNDEFINED "Enable undefined sanitizers" ON)
option(ENABLE_LTO "Enable LTO" OFF)
option(ENABLE_CLANG_TIDY "Enable to add clang tidy." ON)
option(ENABLE_CLANG_FORMAT "Enable to add clang-format." ON)
option(ENABLE_CMAKE_FORMAT "Enable to add cmake-format." ON)

# Project / Lib names
set(LIB_IMAGE_FILTER_NAME imagefilter)
set(LIB_IMAGE_FILTER_TEST_NAME imagefilterTest)
set(EXE_NAME ImageFilter_CUDA)

# CMake Modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)
include(ConfigSafeGuards)
include(Tools)

if(ENABLE_WARNINGS)
    include(Warnings)
endif()

add_cmake_format_target()
add_clang_format_target()

if(ENABLE_SANITIZE_ADDRESS OR ENABLE_SANITIZE_UNDEFINED)
    include(Sanitizers)
endif()

if(ENABLE_LTO)
    include(LTO)
endif()


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)


# Subdirectories
add_subdirectory(configured)
add_subdirectory(libs)
add_subdirectory(app)
