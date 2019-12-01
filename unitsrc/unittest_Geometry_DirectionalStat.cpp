/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.14.090629
 *  @copyright GPLv2
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2019/06/29 | 1.4.14.090629 | new file
 */

#include <gtest/gtest.h>

#include <DirectionalStat.h>

INITIALIZE_EASYLOGGINGPP

class DirectionalStatTest : public :: testing:: Test
{
    protected:

        void SetUp()
        {
            _set0 = dmat4::Zero(100, 4);
            _set0.col(0) = dvec::Ones(100);

            SetUpSet1();

            _set2 = dmat4::Zero(100, 4);
            _set2.col(1) = dvec::Ones(100);

            _set3 = dmat4::Zero(100, 4);
            _set3.col(1).head(50) = dvec::Ones(50);
            _set3.col(1).tail(50) = -dvec::Ones(50);

            _m0 << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

            _m1 << gsl_pow_2(10), 0, 0, 0,
                   0            , 1, 0, 0,
                   0            , 0, 1, 0,
                   0            , 0, 0, 1;

            SetUpSet4();
        }

        void SetUpSet1()
        {
            int n = 10000;

            gsl_rng* engine = get_random_engine();

            _set1 = dmat4::Zero(n, 4);

            for (int i = 0; i < n; i++)
            {
                dvec4 v;
                for (int j = 0; j < 4; j++)
                    v(j) = gsl_ran_gaussian(engine, 1);

                v /= v.norm();

                _set1.row(i) = v.transpose();
            }
        }

        void SetUpSet4()
        {
            int n = 10000;

            _set4 = dmat4::Zero(n, 4);

            sampleACG(_set4, _m1, n);
        }

        dmat4 _set0;
        dmat4 _set1;
        dmat4 _set2;
        dmat4 _set3;
        dmat4 _set4;

        dmat44 _m0;
        dmat44 _m1;
};

TEST_F(DirectionalStatTest, MeanOfStillRotations)
{
    EXPECT_NEAR(fabs(dot(mean(_set0), dvec4(1, 0, 0, 0))), 1, EQUAL_ACCURACY);
}

TEST_F(DirectionalStatTest, MeanOfSameRotations)
{
    EXPECT_NEAR(fabs(dot(mean(_set2), dvec4(0, 1, 0, 0))), 1, EQUAL_ACCURACY);
}

TEST_F(DirectionalStatTest, MeanOfSameRotationsButAxialQuaternion)
{
    EXPECT_NEAR(fabs(dot(mean(_set3), dvec4(0, 1, 0, 0))), 1, EQUAL_ACCURACY);
}

TEST_F(DirectionalStatTest, InferACGIdentity)
{
    dmat44 A;
    inferACG(A, _set1);

    EXPECT_NEAR(sqrt((abs((A -_m0).array())).sum()), 0, 0.5);
}

TEST_F(DirectionalStatTest, InferACGStillCentral)
{
    EXPECT_NEAR(inferACGStillCentral(_set4), 10, 0.5);
}

/***
TEST(gsl_sf_bessel_I0, gsl_sf_bessel_I0_AvoidOverflow)
{
    std::cout << gsl_sf_bessel_I0(0) << std::endl;
    std::cout << gsl_sf_bessel_I0(5) << std::endl;
    std::cout << gsl_sf_bessel_I0(10) << std::endl;
    std::cout << gsl_sf_bessel_I0(15) << std::endl;
    std::cout << gsl_sf_bessel_I0(20) << std::endl;
}

class DirectionalStatTestP : public :: testing:: TestWithParam<int>
{
};

TEST_P(DirectionalStatTestP, TEST)
{
}
***/

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
