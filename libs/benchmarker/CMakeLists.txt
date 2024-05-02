set(LIB_BENCHMARKER_SOURCES "Benchmarker.cpp")
set(LIB_BENCHMARKER_HEADERS "Benchmarker.hpp")
set(LIB_BENCHMARKER_INCLUDES "./")


add_library(${LIB_BENCHMARKER_NAME} SHARED ${LIB_BENCHMARKER_SOURCES} ${LIB_BENCHMARKER_HEADERS})
target_include_directories(${LIB_BENCHMARKER_NAME} PUBLIC ${LIB_BENCHMARKER_INCLUDES})
target_link_libraries(${LIB_BENCHMARKER_NAME} PRIVATE ${LIB_IMAGE_FILTER_NAME})
target_link_directories(${LIB_BENCHMARKER_NAME} PUBLIC ${LIB_BENCHMARKER_INCLUDES} ${LIB_IMAGE_FILTER_NAME})

if(${ENABLE_WARNINGS})
    target_set_warnings(
        TARGET
        ${LIB_BENCHMARKER_NAME}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})
endif()

if(${ENABLE_LTO})
    target_enable_lto(
        TARGET
        ${LIB_BENCHMARKER_NAME}
        ENABLE
        ON)
endif()

if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${LIB_BENCHMARKER_NAME})
endif()