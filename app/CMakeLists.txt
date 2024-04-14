set(EXE_SOURCES "main.cpp")
set(INCLUDES 
"${CMAKE_BINARY_DIR}/configured_files/include")

add_executable(${EXE_NAME} ${EXE_SOURCES})
target_include_directories(${EXE_NAME} PUBLIC ${INCLUDES})
target_link_libraries(${EXE_NAME} PUBLIC ${LIB_IMAGE_FILTER_NAME} CLI11::CLI11)

if(${ENABLE_WARNINGS})
    target_set_warnings(
        TARGET
        ${EXE_NAME}
        ENABLE
        ${ENABLE_WARNINGS}
        AS_ERRORS
        ${ENABLE_WARNINGS_AS_ERRORS})
endif()

if(${ENABLE_LTO})
    target_enable_lto(
        TARGET
        ${EXE_NAME}
        ENABLE
        ON)
endif()

if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${EXE_NAME})
endif()
