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

TEST(drawWithWeightIndex, Test02)
{
    vec v = vec::Zero(5);

    v(1) = 1;
    v(2) = 2;
    v(3) = 3;
    v(4) = 4;

    vec w = vec::Zero(5);

    // std::cout << v << std::endl;
    int n = 100000;

    for (int i = 0; i < n; i++)
    {
        w(drawWithWeightIndex(v)) += 1;
        // std::cout << drawWithWeightIndex(v) << std::endl;
    }

    w.array() /= n;

    std::cout << w << std::endl;
}

TEST(drawWithWeightIndex, Test03)
{
    vec v = vec::Zero(5);

    v(1) = 1;
    v(2) = 1;
    v(3) = 1;
    v(4) = 1;

    vec w = vec::Zero(5);

    // std::cout << v << std::endl;
    int n = 100000;

    for (int i = 0; i < n; i++)
    {
        w(drawWithWeightIndex(v)) += 1;
        // std::cout << drawWithWeightIndex(v) << std::endl;
    }

    w.array() /= n;

    std::cout << w << std::endl;
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
