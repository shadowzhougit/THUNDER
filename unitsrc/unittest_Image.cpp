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

#include <iostream>

#include <gtest/gtest.h>

#include "Image.h"

INITIALIZE_EASYLOGGINGPP

class ImageTest : public :: testing:: TestWithParam<int>
{
    protected:

        void SetUp()
        {
            _i2 = Image(100, 100, RL_SPACE);
            _i3 = Image(100, 100, FT_SPACE);
        }

        Image _i1;
        Image _i2;
        Image _i3;
};

TEST_P(ImageTest, ImageTestDefaultInitializer)
{
    EXPECT_TRUE(_i1.isEmptyRL());
    EXPECT_TRUE(_i1.isEmptyFT());

    EXPECT_FALSE(_i2.isEmptyRL());
    EXPECT_TRUE(_i2.isEmptyFT());

    EXPECT_TRUE(_i3.isEmptyRL());
    EXPECT_FALSE(_i3.isEmptyFT());

    _i1.swap(_i3);

    EXPECT_TRUE(_i1.isEmptyRL());
    EXPECT_FALSE(_i1.isEmptyFT());

    EXPECT_TRUE(_i3.isEmptyRL());
    EXPECT_TRUE(_i3.isEmptyFT());
}

INSTANTIATE_TEST_CASE_P(UUIDTestRepeat, ImageTest, ::testing::Range(0, 10000));

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
