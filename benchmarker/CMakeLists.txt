set(BENCHMARKER_SOURCES "main.cpp")
set(BENCHMARKER_INCLUDES 
"${CMAKE_BINARY_DIR}/configured_files/include")

add_executable(${BENCHMARKER_NAME} ${BENCHMARKER_SOURCES})
target_include_directories(${BENCHMARKER_NAME} PRIVATE ${BENCHMARKER_INCLUDES})
target_link_libraries(${BENCHMARKER_NAME} PRIVATE ${LIB_IMAGE_FILTER_NAME} ${LIB_BENCHMARKER_NAME} CLI11::CLI11)

if(${ENABLE_WARNINGS})
    target_set_warnings(
        TARGET
        ${BENCHMARKER_NAME}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})
endif()

if(${ENABLE_LTO})
    target_enable_lto(
        TARGET
        ${BENCHMARKER_NAME}
        ENABLE
        ON)
endif()

if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${BENCHMARKER_NAME})
endif()
