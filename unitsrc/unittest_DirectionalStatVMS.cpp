/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.14.090629
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2019/07/09 | 1.4.14.090709 | new file
 *  Mingxu Hu   | 2019/08/21 | 1.4.14.090821 | DirectionalStatVMSPdfVMSKappaTest added
 */

#include <gtest/gtest.h>

#include <DirectionalStat.h>

INITIALIZE_EASYLOGGINGPP

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

TEST(DirectionalStatVMSPdfVMSKappaTest, DirectionalStatVMSPdfVMSKappaTestKappaEqualToZero)
{
    EXPECT_EQ(pdfVMSKappa(dvec2(1, 0), dvec2(1, 0), 0), 1.0 / (2 * M_PI));
    EXPECT_EQ(pdfVMSKappa(dvec2(1, 0), dvec2(0, 1), 0), 1.0 / (2 * M_PI));
    EXPECT_EQ(pdfVMSKappa(dvec2(0, 1), dvec2(1, 0), 0), 1.0 / (2 * M_PI));
    EXPECT_EQ(pdfVMSKappa(dvec2(0, 1), dvec2(0, 1), 0), 1.0 / (2 * M_PI));
    EXPECT_EQ(pdfVMSKappa(dvec2(sqrt(2) / 2, sqrt(2)), dvec2(0, 1), 0), 1.0 / (2 * M_PI));
    EXPECT_EQ(pdfVMSKappa(dvec2(sqrt(2) / 2, sqrt(2)), dvec2(1, 0), 0), 1.0 / (2 * M_PI));
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
