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

#include "Precision.h"

TEST(Test_GSL_MAX_RFLOAT, Test01)
{
    EXPECT_EQ(TSGSL_MAX_RFLOAT(GIGABYTE, GIGABYTE), GIGABYTE);
    EXPECT_EQ(TSGSL_MAX_RFLOAT(10.0 * GIGABYTE, GIGABYTE), 10.0 * GIGABYTE);
    EXPECT_EQ(TSGSL_MAX_RFLOAT(100.0 * GIGABYTE, GIGABYTE), 100.0 * GIGABYTE);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
