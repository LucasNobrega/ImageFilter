#include <gtest/gtest.h>
#include "imagefilter.hpp"

TEST(LibraryTestSuite, TestSample)
{
    int sum = sumNumbers(2, 4);
    ASSERT_EQ(6, sum);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}