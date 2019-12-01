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

class MemoryBazaarSerializeTest : public :: testing:: Test
{
    protected:

        void SetUp()
        {
            _i1 = Image(100, 100, RL_SPACE);
            _i2 = _i1.copyImage();

            _i3 = Image(100, 100, FT_SPACE);
            _i4 = _i3.copyImage();

            _i5 = Image(200, 200, RL_SPACE);
            _i6 = _i5.copyImage();

            _i7 = Image(200, 200, FT_SPACE);
            _i8 = _i7.copyImage();
        }

        Image _i1, _i2, _i3, _i4, _i5, _i6, _i7, _i8;
};

TEST_F(MemoryBazaarSerializeTest, SizeTest)
{
    EXPECT_EQ(serializeSize(_i1), serializeSize(_i2));
    EXPECT_EQ(serializeSize(_i3), serializeSize(_i4));
    EXPECT_EQ(serializeSize(_i1), serializeSize(_i3));

    EXPECT_NE(serializeSize(_i1), serializeSize(_i5));
    EXPECT_NE(serializeSize(_i1), serializeSize(_i7));

    EXPECT_EQ(serializeSize(_i5), serializeSize(_i6));
    EXPECT_EQ(serializeSize(_i7), serializeSize(_i8));
    EXPECT_EQ(serializeSize(_i5), serializeSize(_i7));
}

TEST_F(MemoryBazaarSerializeTest, SerializeAndDeserializeTest)
{
    size_t size = serializeSize(_i1);

    void* m = malloc(size);

    serialize(m, _i1);

    deserialize(_i2, m, size);
    deserialize(_i3, m, size);
    deserialize(_i5, m, size);
    deserialize(_i7, m, size);

    EXPECT_TRUE((_i1 == _i2));
    EXPECT_TRUE((_i1 == _i3));
    EXPECT_TRUE((_i1 == _i5));
    EXPECT_TRUE((_i1 == _i7));
    EXPECT_FALSE((_i2 == _i6));
    EXPECT_FALSE((_i2 == _i8));

    free(m);
}

class MemoryBazaarDeserializeTest : public :: testing :: TestWithParam<int>
{
    protected:

        void SetUp()
        {
            _i1 = Image(100, 100, RL_SPACE);
            _i2 = Image(100, 100, FT_SPACE);
        }

        Image _i1;
        Image _i2;
};

TEST_P(MemoryBazaarDeserializeTest, MemoryBazaarDeserializeTest_1)
{
    size_t size = serializeSize(_i1);

    // std::cout << "size = " << size << std::endl;

    void* m = malloc(size);

    deserialize(_i1, m, size);
    deserialize(_i2, m, size);

    free(m);
}

INSTANTIATE_TEST_CASE_P(MemoryBazaarDeserializeTestRepeat, MemoryBazaarDeserializeTest, ::testing::Range(0, 100000));

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
