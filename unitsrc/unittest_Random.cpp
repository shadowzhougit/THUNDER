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

#include <Random.h>

INITIALIZE_EASYLOGGINGPP

TEST(shuffledIndexTest, Test01)
{
    std::cout << shuffledIndex(100) << std::endl;
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
