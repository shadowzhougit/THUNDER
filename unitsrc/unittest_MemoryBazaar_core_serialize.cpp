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

#include "core/serialize.h"
#include "core/serializeImage.h"

#include "Image.h"

INITIALIZE_EASYLOGGINGPP

/***
class MemoryBazaarTest : public :: testing:: Test
{
    protected:

        void SetUp()
        {
        }

        MemoryBazaar<Image> _mb;
};

TEST_F(MemoryBazaarTest, Test_1)
{
    _mb.foo();
}
***/

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
