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

#include "core/UUID.h"

class UUIDTest : public :: testing:: TestWithParam<int>
{
};

TEST_P(UUIDTest, MemoryBazaarTestWithVariableComibiation)
{
    std::cout << "UUID = " << generateUUID() << std::endl;
}

INSTANTIATE_TEST_CASE_P(UUIDTestRepeat, UUIDTest, ::testing::Range(0, 100));

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
