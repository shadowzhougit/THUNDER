/*******************************************************************
 *       Filename:  test_simd.cpp                                     
 *                                                                 
 *    Description:                                        
 *                                                                 
 *        Version:  1.0                                            
 *        Created:  07/22/2019 05:41:29 PM                                 
 *       Revision:  none                                           
 *       Compiler:  gcc                                           
 *                                                                 
 *         Author:  Ruan Huabin                                      
 *          Email:  ruanhuabin@tsinghua.edu.cn                                        
 *        Company:  Dep. of CS, Tsinghua Unversity                                      
 *                                                                 
 *******************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "core/logDataVSPrior.h"
//using namespace std;

void printResult(RFLOAT *resultBuffer, RFLOAT *simd256MNResult, size_t start, size_t end )
{
    printf("value in result buffer and simd result buffer: \n");
    for(size_t i = start; i < end; i ++)
    {
        printf("resultBuffer[%lu] = %f, simdResultBuffer[%lu] = %f\n", i, resultBuffer[i], i, simd256MNResult[i]);
    }

}


bool isSame(RFLOAT a, RFLOAT b)
{
#ifdef SINGLE_PRECISION
    RFLOAT error = fabsf(a - b);
#else
    RFLOAT error = fabs(a - b);
#endif

    if(error/a < 1e-5)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void checkResult(RFLOAT *resultBuffer, RFLOAT *simd256MNResult, size_t n)
{
    size_t cnt = 0;
    for(size_t i = 0; i < n; i ++)
    {
        if(isSame(resultBuffer[i], simd256MNResult[i]) == false)
        {
            printf("Not equal, [expect, real] = [%f, %f]\n", resultBuffer[i], simd256MNResult[i]);
            cnt ++;
        }

        if(cnt > 5)
        {
            printf("At least five result is not the same\n");
            abort();
        }
        
    }

    if(cnt == 0)
    {
        printf("[%ld] result is the same\n", n);
    }

}


int main ( int argc, char *argv[] )
{ 
    const size_t m = 211;
    const size_t n = 33;
    const size_t itemNum = m * n;
    MemoryBazaar<RFLOAT, BaseType, 4> datR(1, itemNum, itemNum, sizeof(float), 1);
    MemoryBazaar<RFLOAT, BaseType, 4> datI(1, itemNum, itemNum, sizeof(float), 1);
    MemoryBazaar<RFLOAT, BaseType, 4> ctf(1, itemNum, itemNum, sizeof(float), 1);
    MemoryBazaar<RFLOAT, BaseType, 4> sigRcp(1, itemNum, itemNum, sizeof(float), 1);

    RFLOAT *priR   = new RFLOAT[m];
    RFLOAT *priI   = new RFLOAT[m];

    for(size_t i = 0; i < itemNum; i ++ )
    {
        datR[i]   = (i + 1) * 1.4;
        datI[i]   = (i + 1) * 1.5;
        ctf[i]    = (i + 1) * 1.8;
        sigRcp[i] = (i + 1) * 1.9;

    }

    for(size_t i = 0; i < m; i ++)
    {
        priR[i]   = (i + 1) * 1.6;
        priI[i]   = (i + 1) * 1.7;
    }

    RFLOAT *resultBuffer = new RFLOAT[n];
    RFLOAT *simdResultBuffer = new RFLOAT[n];
    memset(resultBuffer, '\0', n * sizeof(RFLOAT));
    memset(simdResultBuffer, '\0', n * sizeof(RFLOAT));

    resultBuffer = logDataVSPrior(datR, datI, priR, priI, ctf, sigRcp, n, m, resultBuffer);

    printResult(resultBuffer, simdResultBuffer, 0, n);


    delete []priR;
    delete []priI;
    delete []resultBuffer;
    delete [] simdResultBuffer;


    return 0;
#ifdef PIXEL_MAJOR
    const size_t m = 2121;
    const size_t n = 353535;

    size_t itemNum = m > n ? m : n;

    RFLOAT *datR   = new RFLOAT[itemNum];
    RFLOAT *datI   = new RFLOAT[itemNum];
    RFLOAT *priR   = new RFLOAT[itemNum];
    RFLOAT *priI   = new RFLOAT[itemNum];
    RFLOAT *ctf    = new RFLOAT[itemNum];
    RFLOAT *sigRcp = new RFLOAT[itemNum];

    for(size_t i = 0;i < itemNum; i ++)
    {
        datR[i]   = (i + 1) * 1.4;
        datI[i]   = (i + 1) * 1.5;
        priR[i]   = (i + 1) * 1.6;
        priI[i]   = (i + 1) * 1.7;
        ctf[i]    = (i + 1) * 1.8;
        sigRcp[i] = (i + 1) * 1.9;
    }

    //RFLOAT origMResult = logDataVSPrior_Orig_M(datR, datI, priR, priI, ctf, sigRcp, m);
    RFLOAT origMResult = logDataVSPrior(datR, datI, priR, priI, ctf, sigRcp, m);
    printf("orig M Result = %f\n", origMResult);
#ifdef ENABLE_SIMD_256
#ifdef SINGLE_PRECISION
    RFLOAT simd256FloatResult = logDataVSPrior(datR, datI, priR, priI, ctf, sigRcp, m);
    printf("simd256FloatResult = %f\n", simd256FloatResult);
#endif
#endif

#ifdef ENABLE_SIMD_256
#ifndef SINGLE_PRECISION
    RFLOAT simd256DoubleResult = logDataVSPrior(datR, datI, priR, priI, ctf, sigRcp, m);
    printf("simd256DoubleResult = %f\n", simd256DoubleResult);
#endif
#endif

    RFLOAT *resultBuffer = new RFLOAT[itemNum];
    RFLOAT *simdResultBuffer = new RFLOAT[itemNum];
    for(size_t i = 0; i < itemNum; i ++)
    {
        resultBuffer[i] = 0.0f;
        simdResultBuffer[i] = 0.0f;
    }

    size_t itemNumMN = m * n;
    RFLOAT *datRMN = new RFLOAT[itemNumMN];
    RFLOAT *datIMN = new RFLOAT[itemNumMN];
    RFLOAT *ctfMN = new RFLOAT[itemNumMN];
    RFLOAT *sigRcpMN = new RFLOAT[itemNumMN];
    for(size_t i = 0;i < itemNumMN; i ++)
    {
        datRMN[i]   = (i + 1) * 1.4;
        datIMN[i]   = (i + 1) * 1.5;
        ctfMN[i]    = (i + 1) * 1.8;
        sigRcpMN[i] = (i + 1) * 1.9;
    }

    RFLOAT *origMNResult = logDataVSPrior(datRMN, datIMN, priR, priI, ctfMN, sigRcpMN, n, m, resultBuffer);
#ifdef ENABLE_SIMD_256
#ifdef SINGLE_PRECISION
    RFLOAT* simd256MNFloatResult = logDataVSPrior(datRMN, datIMN, priR, priI, ctfMN, sigRcpMN, n, m, simdResultBuffer);
    checkResult(resultBuffer, simd256MNFloatResult, n);
    printResult(resultBuffer, simd256MNFloatResult, 0, 5);
    printResult(resultBuffer, simd256MNFloatResult, n -5, n);

    //printf("simd256MNFloatResult = %f\n", simd256MNFloatResult);
#endif
#endif

#ifdef ENABLE_SIMD_256
#ifndef SINGLE_PRECISION
    RFLOAT* simd256MNDoubleResult = logDataVSPrior(datRMN, datIMN, priR, priI, ctfMN, sigRcpMN, n, m, simdResultBuffer);
    checkResult(resultBuffer, simd256MNDoubleResult,n);
    printResult(resultBuffer, simd256MNDoubleResult, 0, 5);
    printResult(resultBuffer, simd256MNDoubleResult,n - 5, n);
    //printf("simd256MNDoubleResult = %f\n", simd256MNDoubleResult);
#endif
#endif


    delete[] datR;
    delete[] datI;
    delete[] priR;
    delete[] priI;
    delete[] ctf;
    delete[] sigRcp;
    delete[] resultBuffer;
    delete[] simdResultBuffer;

    delete[] datRMN;
    delete[] datIMN;
    delete[] ctfMN;
    delete[] sigRcpMN;
    return EXIT_SUCCESS;
#endif
}


