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

#include "MemoryBazaar.h"
#include "Image.h"

#include "core/serializeImage.h"

INITIALIZE_EASYLOGGINGPP

class MemoryBazaarTestPFloat : public :: testing:: TestWithParam<::testing::tuple<int , int>>
{
};

class MemoryBazaarTestPDouble : public :: testing:: TestWithParam<::testing::tuple<int , int>>
{
};

class MemoryBazaarTestPInt : public :: testing:: TestWithParam<::testing::tuple<int , int>>
{
};

class MemoryBazaarTestPImage : public :: testing:: TestWithParam<::testing::tuple<int , int>>
{
};

class MemoryBazaarTestPImageCopy : public :: testing:: TestWithParam<::testing::tuple<int , int>>
{
};

class MemoryBazaarTestP_Image_Publication_MemoryBazaar : public :: testing:: TestWithParam<::testing::tuple<int, int>>
{
};

class MemoryBazaarTestP_Debug_MemoryBazaar : public :: testing:: TestWithParam<int>
{
};

class MemoryBazaarTestP_Image_Publication_Vector : public :: testing:: TestWithParam<int>
{
};

TEST_P(MemoryBazaarTestPFloat, MemoryBazaarTestWithVariableComibiation)
{
    int nItem = testing::get<0>(GetParam());
    int nt = testing::get<1>(GetParam());

    MemoryBazaar<float, BaseType> mb(nt, nItem, 10, sizeof(float), 4);

    MemoryBazaarDustman<float, BaseType> mbd(&mb);

    #pragma omp parallel for schedule(dynamic) num_threads(nt) firstprivate(mbd)
    for (int i = 0; i < nItem; i++)
    {
        mb[i] = i;
    }

    #pragma omp parallel for schedule(dynamic) num_threads(nt) firstprivate(mbd)
    for (int i = nItem - 1; i >= 0; i--)
    {
        // mb[i];

        float mbi = mb[i];

        EXPECT_EQ(mbi, i);
        EXPECT_NE(mbi, -1);
    }

    mb.cleanUp();
}

TEST_P(MemoryBazaarTestPDouble, MemoryBazaarTestWithVariableComibiation)
{
    int nItem = testing::get<0>(GetParam());
    int nt = testing::get<1>(GetParam());

    MemoryBazaar<double, BaseType> mb(nt, nItem, 10, sizeof(double), 4);

    MemoryBazaarDustman<double, BaseType> mbd(&mb);

    #pragma omp parallel for schedule(dynamic, 2) num_threads(nt) firstprivate(mbd)
    for (int i = 0; i < nItem; i++)
    {
        mb[i] = i;
    }

    #pragma omp parallel for schedule(dynamic, 2) num_threads(nt) firstprivate(mbd)
    for (int i = nItem - 1; i >= 0; i--)
    {
        double mbi = mb[i];

        EXPECT_EQ(mbi, i);
        EXPECT_NE(mbi, -1);
    }

    mb.cleanUp();
}

TEST_P(MemoryBazaarTestPInt, MemoryBazaarTestWithVariableComibiation)
{
    int nItem = testing::get<0>(GetParam());
    int nt = testing::get<1>(GetParam());

    MemoryBazaar<int, BaseType> mb(nt, nItem, 10, sizeof(int), 4);

    MemoryBazaarDustman<int, BaseType> mbd(&mb);

    #pragma omp parallel for schedule(dynamic, 2) num_threads(nt) firstprivate(mbd)
    for (int i = 0; i < nItem; i++)
    {
        mb[i] = i;
    }

    #pragma omp parallel for schedule(dynamic, 2) num_threads(nt) firstprivate(mbd)
    for (int i = nItem - 1; i >= 0; i--)
    {
        int mbi = mb[i];

        EXPECT_EQ(mbi, i);
        EXPECT_NE(mbi, -1);
    }

    mb.cleanUp();
}

TEST_P(MemoryBazaarTestPImage, MemoryBazaarTestWithVariableComibiation)
{
    int nItem = testing::get<0>(GetParam());
    int nt = testing::get<1>(GetParam());

    MemoryBazaar<Image, DerivedType> mb(nt, nItem, 10, serializeSize(Image(100, 100, RL_SPACE)), 4);

    MemoryBazaarDustman<Image, DerivedType> mbd(&mb);

    #pragma omp parallel for schedule(dynamic, 2) num_threads(nt) firstprivate(mbd)
    for (int i = 0; i < nItem; i++)
    {
        Image img(100, 100, RL_SPACE);
        SET_0_RL(img);
        img(i % 100) = 1;

        mb[i] = img.copyImage();
    }

    #pragma omp parallel for schedule(dynamic, 2) num_threads(nt) firstprivate(mbd)
    for (int i = nItem - 1; i >= 0; i--)
    {
        Image img(100, 100, RL_SPACE);
        SET_0_RL(img);
        img(i % 100) = 1;

        EXPECT_EQ(mb[i], img);
    }

    mb.cleanUp();
}

TEST_P(MemoryBazaarTestPImageCopy, MemoryBazaarTestWithVariableComibiation)
{
    int nItem = testing::get<0>(GetParam());
    int nt = testing::get<1>(GetParam());

    MemoryBazaar<Image, DerivedType> mb(nt, nItem, 10, serializeSize(Image(100, 100, RL_SPACE)), 4);

    MemoryBazaarDustman<Image, DerivedType> mbd(&mb);

    vector<Image> vi;
    vi.resize(nItem);

    MemoryBazaarDustman<Image, DerivedType> mbd1(&mb);
    #pragma omp parallel for num_threads(nt) firstprivate(mbd1)
    for (int i = 0; i < nItem; i++)
    {
        Image img(100, 100, RL_SPACE);
        SET_0_RL(img);
        img(i % 100) = 1;

        mb[i] = img.copyImage();
    }

    MemoryBazaarDustman<Image, DerivedType> mbd2(&mb);
    #pragma omp parallel for num_threads(nt) firstprivate(mbd2)
    for (int i = 0; i < nItem; i++)
    {
        vi[i] = mb[i].copyImage();
    }

    mb.cleanUp();
}

TEST_P(MemoryBazaarTestP_Image_Publication_MemoryBazaar, MemoryBazaarTestWithVariableComibiation)
{
    size_t nItem = pow(2, testing::get<0>(GetParam()));
    int packSize = testing::get<1>(GetParam());

    std::cout << "nItem = 2 ** " << testing::get<0>(GetParam()) << std::endl;
    std::cout << "packSize = " << packSize << std::endl;

    MemoryBazaar<Image, DerivedType, 4> mb(24, nItem, nItem / 4, serializeSize(Image(200, 200, RL_SPACE)), packSize);

    MemoryBazaarDustman<Image, DerivedType> mbd(&mb);

    MemoryBazaarDustman<Image, DerivedType> mbd1(&mb);
    #pragma omp parallel for num_threads(24) firstprivate(mbd1)
    for (int i = 0; i < nItem; i++)
    {
        mb[i];
    }

    mb.cleanUp();
}

TEST_P(MemoryBazaarTestP_Debug_MemoryBazaar, MemoryBazaarTestWithVariable)
{
    size_t _IDSize = 56257;
    size_t _nPxl = 38977;
    size_t _nStall = 14064;

    MemoryBazaar<RFLOAT, BaseType, 4> mb(24, _IDSize * _nPxl, _nStall, sizeof(RFLOAT), _nPxl);

    MemoryBazaarDustman<RFLOAT, BaseType> mbd(&mb);

    #pragma omp parallel for num_threads(24) firstprivate(mbd)
    for (size_t i = 0; i < _IDSize * _nPxl; i++)
    {
        mb.endLastVisit();

        mb[i] = 1;
    }

    mb.cleanUp();
}

TEST_P(MemoryBazaarTestP_Image_Publication_Vector, MemoryBazaarTestWithVariable)
{
    size_t nItem = pow(2, GetParam());

    std::cout << "nItem = 2 ** " << GetParam() << std::endl;

    vector<Image> vi;
    vi.resize(nItem);

    #pragma omp parallel for num_threads(24)
    for (int i = 0; i < nItem; i++)
    {
        vi[i];
    }
}

class MemoryBazaarTestT212 : public :: testing:: Test
{
    protected:

        void SetUp()
        {
            _nPxl = 34;

            _IDSize = 14122;

            _nStall = 256;

            _nThread = 256;

            _datPR.setUp(_nThread, _IDSize * _nPxl, _nStall, sizeof(RFLOAT), _nPxl);
            _datPI.setUp(_nThread, _IDSize * _nPxl, _nStall, sizeof(RFLOAT), _nPxl);
        }

        void TearDown()
        {
            _datPR.cleanUp();

            _datPI.cleanUp();
        }

        MemoryBazaar<RFLOAT, BaseType, 4> _datPR;

        MemoryBazaar<RFLOAT, BaseType, 4> _datPI;

        size_t _nPxl;

        size_t _IDSize;

        size_t _nStall;

        size_t _nThread;
};


/***
TEST_F(MemoryBazaarTestT212, Test01)
{
    std::cout << "nd = " << _IDSize << std::endl;
    std::cout << "m = " << _nPxl << std::endl;
    std::cout << "nThread = " << _nThread << std::endl;
    MemoryBazaarDustman<RFLOAT, BaseType, 4> datPRDustman(&_datPR);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> datPIDustman(&_datPI);
    #pragma omp parallel for schedule(dynamic) num_threads(_nThread) firstprivate(datPRDustman, datPIDustman)
    for (size_t id = 0; id < _IDSize; id++)
    {
        _datPR.endLastVisit(_nPxl * id);
        _datPI.endLastVisit(_nPxl * id);

        RFLOAT* ptr_datPR = &_datPR[_nPxl * id];
        RFLOAT* ptr_datPI = &_datPI[_nPxl * id];
    }
}
***/

// INSTANTIATE_TEST_CASE_P(MemoryBazaarTestWithVariableComibiation_1, MemoryBazaarTestPFloat, ::testing::Combine(::testing::Values(1, 10, 12, 15, 100, 1000, 10000, 100000), ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256)));

// INSTANTIATE_TEST_CASE_P(MemoryBazaarTestWithVariableComibiation_2, MemoryBazaarTestPDouble, ::testing::Combine(::testing::Values(1, 10, 12, 15, 100, 1000, 10000, 100000), ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256)));

// INSTANTIATE_TEST_CASE_P(MemoryBazaarTestWithVariableComibiation_3, MemoryBazaarTestPInt, ::testing::Combine(::testing::Values(1, 10, 12, 15, 100, 1000, 10000, 100000), ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256)));

// INSTANTIATE_TEST_CASE_P(MemoryBazaarTestWithVariableComibiation_4, MemoryBazaarTestPImage, ::testing::Combine(::testing::Values(1, 10, 12, 15, 100, 1000), ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256)));

// INSTANTIATE_TEST_CASE_P(MemoryBazaarTestWithVariableComibiation_5, MemoryBazaarTestPImageCopy, ::testing::Combine(::testing::Values(1, 10, 12, 15, 100, 1000, 10000), ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128, 256)));

// INSTANTIATE_TEST_CASE_P(MemoryBazaarTestWithVariableComibiation_Publication_MemoryBazzar, MemoryBazaarTestP_Image_Publication_MemoryBazaar, ::testing::Combine(::testing::Range(10, 20), ::testing::Values(1, 4, 16, 64, 256)));

// INSTANTIATE_TEST_CASE_P(MemoryBazaarTestWithVariableComibiation_Publication_Vector, MemoryBazaarTestP_Image_Publication_Vector, ::testing::Range(10, 20));

INSTANTIATE_TEST_CASE_P(MemoryBazaarTestWithVariableComibiation_Debug_MemoryBazaar, MemoryBazaarTestP_Debug_MemoryBazaar, ::testing::Range(10, 11));

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    loggerInit(argc, argv);

    return RUN_ALL_TESTS();
}
