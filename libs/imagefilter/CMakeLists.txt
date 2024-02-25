set(LIBRARY_SOURCES "imagefilter.cpp")
set(LIBRARY_HEADERS "imagefilter.hpp")
set(LIBRARY_INCLUDES "./" "${CMAKE_BINARY_DIR}/configured_files/include")

add_library(${LIB_IMAGE_FILTER_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_include_directories(${LIB_IMAGE_FILTER_NAME} PUBLIC ${LIBRARY_INCLUDES})

if(${ENABLE_WARNINGS})
    target_set_warnings(
        TARGET
        ${LIB_IMAGE_FILTER_NAME}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})
endif()

if(${ENABLE_LTO})
    target_enable_lto(
        TARGET
        ${LIB_IMAGE_FILTER_NAME}
        ENABLE
        ON)
endif()

if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${LIB_IMAGE_FILTER_NAME})
endif()

add_subdirectory(test)