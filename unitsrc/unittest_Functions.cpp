/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.11.081102
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2018/11/02 | 1.4.11.081102 | new file
 *
 *
 */

#include <gtest/gtest.h>

#include "Macro.h"

#include "Functions.h"

TEST(TestAround, Test01)
{
    EXPECT_EQ(AROUND(1.1), 1);
    EXPECT_EQ(AROUND(1.5), 2);
    EXPECT_EQ(AROUND(1.9), 2);
    EXPECT_EQ(AROUND(1e10), 10000000000);
    EXPECT_EQ(AROUND(GIGABYTE), GIGABYTE);
    EXPECT_EQ(AROUND(2.0 * GIGABYTE), 2.0 * GIGABYTE);
    EXPECT_EQ(AROUND(10.0 * GIGABYTE), 10.0 * GIGABYTE);
    EXPECT_EQ(AROUND(100.0 * GIGABYTE), 100.0 * GIGABYTE);

    // std::cout << 100 * GIGABYTE << std::endl;
    // std::cout << GIGABYTE << std::endl;
    // std::cout << 10.0 * GIGABYTE << std::endl;
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);


    return RUN_ALL_TESTS();
}
