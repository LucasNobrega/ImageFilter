set(LIB_TEST_SOURCES "test.cpp")

add_executable(${LIB_IMAGE_FILTER_TEST_NAME} ${LIB_TEST_SOURCES})
target_link_libraries(${LIB_IMAGE_FILTER_TEST_NAME} PUBLIC ${LIB_IMAGE_FILTER_NAME} gtest_main gmock_main)