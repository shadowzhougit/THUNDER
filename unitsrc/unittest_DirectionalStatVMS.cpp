/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.14.090629
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2019/06/29 | 1.4.14.090709 | new file
 */

#include <gtest/gtest.h>
//#include <gmock/gmock.h>

#include <DirectionalStat.h>

INITIALIZE_EASYLOGGINGPP

//class DirectionalStatVMSTest : public :: testing:: Test
class DirectionalStatVMSTest : public :: testing:: TestWithParam<int>
{
};

TEST_P(DirectionalStatVMSTest, SampleVMSNorm)
{
    dmat2 m = dmat2::Zero(1, 2);

    sampleVMS(m, dvec2(1, 0), 1e-3, 1);

    EXPECT_EQ(m.row(0).norm(), 1) << "NOT NORMALIZED, NORM OF THIS VECTOR IS " << m.row(0).norm();
}

INSTANTIATE_TEST_CASE_P(DirectionalStatVMSTestRepeat, DirectionalStatVMSTest, ::testing::Range(0, 10000));

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
