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

#include <core/equivalence.h>

INITIALIZE_EASYLOGGINGPP

class ImageEquivalenceTest : public :: testing:: Test
{
    protected:

        void SetUp()
        {
            _i1 = Image(100, 100, RL_SPACE);
            SET_0_RL(_i1);
            _i1(0) = 1;

            _i2 = Image(100, 100, RL_SPACE);
            SET_0_RL(_i2);
            _i2(1) = 1;

            _i3 = Image(200, 200, RL_SPACE);
            SET_0_RL(_i3);

            _i4 = _i1.copyImage();

            _i5 = Image(100, 100, FT_SPACE);
            SET_0_FT(_i5);
            _i5[0] = COMPLEX(1, 0);

            _i6 = Image(100, 100, FT_SPACE);
            SET_0_FT(_i6);
            _i6[1] = COMPLEX(1, 0);

            _i7 = _i5.copyImage();
        }

        Image _i1;
        Image _i2;
        Image _i3;
        Image _i4;
        Image _i5;
        Image _i6;
        Image _i7;
};

TEST_F(ImageEquivalenceTest, Equivalence_1)
{
    EXPECT_TRUE((_i1 == _i1));
    EXPECT_FALSE((_i1 == _i2));
    EXPECT_FALSE((_i1 == _i3));
    EXPECT_TRUE((_i1 == _i1));
    EXPECT_FALSE((_i1 == _i5));
    EXPECT_FALSE((_i1 == _i6));
    EXPECT_TRUE((_i5 == _i5));
    EXPECT_FALSE((_i5 == _i6));
    EXPECT_TRUE((_i5 == _i7));
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
