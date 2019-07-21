/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.11.081102
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2019/06/29 | 1.4.14.090629 | new file
 */

#include <gtest/gtest.h>

#include <LinearAlgebra.h>

class LinearAlgebraTest : public :: testing:: Test
{
    protected:

        void SetUp()
        {
            _v1 = dvec4(1, 0, 0, 0);
            _v2 = dvec4(1, 0, 0, 0);
            _v3 = dvec4(0, 0, 0, 1);
            _v4 = dvec4(1 + EQUAL_ACCURACY / 2, 0, 0, 0);

            _m1 << 1, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0;

            _m2 << 1, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0;

            _m3 << 0, 0, 0, 1,
                   0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0;

            _m4 << 0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 1;

            _m5 << 1 + EQUAL_ACCURACY / 2, 0, 0, 0,
                   0                     , 0, 0, 0,
                   0                     , 0, 0, 0,
                   0                     , 0, 0, 0;
        }

        dvec4 _v1;
        dvec4 _v2;
        dvec4 _v3;
        dvec4 _v4;

        dmat44 _m1;
        dmat44 _m2;
        dmat44 _m3;
        dmat44 _m4;
        dmat44 _m5;
};

TEST_F(LinearAlgebraTest, VectorEquivalence)
{
    EXPECT_TRUE(_v1 == _v1);
    EXPECT_TRUE(_v1 == _v2);
    EXPECT_FALSE(_v1 == _v3);
    EXPECT_FALSE(_v2 == _v3);
    EXPECT_TRUE(_v1 == _v4);
    EXPECT_FALSE(_v3 == _v4);
}

TEST_F(LinearAlgebraTest, VectorDot)
{
    EXPECT_EQ(dot(_v1, _v1), 1);
    EXPECT_EQ(dot(_v1, _v2), 1);
    EXPECT_EQ(dot(_v1, _v3), 0);
    EXPECT_NEAR(dot(_v1, _v4), 1, EQUAL_ACCURACY);
}

TEST_F(LinearAlgebraTest, MatrixEquivalence)
{
    EXPECT_TRUE(_m1 == _m1);
    EXPECT_TRUE(_m1 == _m2);
    EXPECT_FALSE(_m1 == _m3);
    EXPECT_FALSE(_m2 == _m3);
    EXPECT_TRUE(_m1 == _m5);
    EXPECT_FALSE(_m3 == _m5);
}

TEST_F(LinearAlgebraTest, FunctionTensor)
{
    EXPECT_TRUE(tensor(_v1, _v1) == _m1);
    EXPECT_TRUE(tensor(_v1, _v3) == _m3);
    EXPECT_TRUE(tensor(_v3, _v3) == _m4);
    EXPECT_FALSE(tensor(_v1, _v1) == _m3);
    EXPECT_FALSE(tensor(_v1, _v3) == _m1);
    EXPECT_TRUE(tensor(_v1, _v1) == _m5);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
