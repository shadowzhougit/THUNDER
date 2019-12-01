#include "core/logDataVSPrior.h"

#ifndef ENABLE_SIMD_256
#ifndef ENABLE_SIMD_512

RFLOAT logDataVSPrior(const RFLOAT* datR,
        const RFLOAT* datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        const RFLOAT* ctf,
        const RFLOAT* sigRcp,
        const size_t m)
{

    RFLOAT result = 0;

    for (size_t i = 0; i < m; i++)
    {
        RFLOAT xR = datR[i] - ctf[i] * priR[i];
        RFLOAT xI = datI[i] - ctf[i] * priI[i];

        result += (xR * xR + xI * xI) * sigRcp[i];
    }

    return result;
}

/***
RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
        MemoryBazaar<RFLOAT, BaseType, 4>& datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
        MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
        const size_t n,
        const size_t m,
        RFLOAT* result)
{
    for (size_t i = 0; i < n; i++)
    {
        result[i] += logDataVSPrior(&datR[m * i],
                &datI[m * i],
                priR,
                priI,
                &ctf[m * i],
                &sigRcp[m * i],
                m);
    }

    return result;
}
***/

#endif
#endif

#ifdef ENABLE_SIMD_256
#ifdef SINGLE_PRECISION

RFLOAT logDataVSPrior_float_SIMD256(const RFLOAT* datR,
        const RFLOAT* datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        const RFLOAT* ctf,
        const RFLOAT* sigRcp,
        const size_t m)
{
    __m256 ymm1, ymm2, ymm3, ymm4, ymm5,ymm6;
    ymm6 = _mm256_setzero_ps();
    int i = 0;
    for(i = 0; i <= (m - 8); i +=8)
    {
        //ymm1 = _mm256_set_ps(ctf[i+7], ctf[i+6], ctf[i+5], ctf[i + 4], ctf[i+3], ctf[i+2], ctf[i+1], ctf[i]); //ctf[i]
        //ymm2 = _mm256_set_ps(dat[i+7].dat[0], dat[i+6].dat[0], dat[i+5].dat[0],dat[i + 4].dat[0], dat[i+3].dat[0], dat[i+2].dat[0], dat[i+1].dat[0],dat[i].dat[0]);//dat[i].dat[0]
        //ymm3 = _mm256_set_ps(dat[i+7].dat[1], dat[i+6].dat[1], dat[i+5].dat[1],dat[i + 4].dat[1], dat[i+3].dat[1], dat[i+2].dat[1], dat[i+1].dat[1],dat[i].dat[1]);//dat[i].dat[1]
        //ymm4 = _mm256_set_ps(pri[i+7].dat[0], pri[i+6].dat[0], pri[i+5].dat[0],pri[i + 4].dat[0], pri[i+3].dat[0], pri[i+2].dat[0], pri[i+1].dat[0],pri[i].dat[0]);//pri[i].dat[0]
        //ymm5 = _mm256_set_ps(pri[i+7].dat[1], pri[i+6].dat[1], pri[i+5].dat[1],pri[i + 4].dat[1], pri[i+3].dat[1], pri[i+2].dat[1], pri[i+1].dat[1],pri[i].dat[1]);//pri[i].dat[1]

        //ymm1 = _mm256_set_ps(ctf[i+7], ctf[i+6], ctf[i+5], ctf[i + 4], ctf[i+3], ctf[i+2], ctf[i+1], ctf[i]); //ctf[i]
        //ymm2 = _mm256_set_ps(datR[i+7], datR[i+6], datR[i+5],datR[i + 4], datR[i+3], datR[i+2], datR[i+1],datR[i]);//datR[i]
        //ymm3 = _mm256_set_ps(datI[i+7], datI[i+6], datI[i+5],datI[i + 4], datI[i+3], datI[i+2], datI[i+1],datI[i]);//datI[i]
        //ymm4 = _mm256_set_ps(priR[i+7], priR[i+6], priR[i+5],priR[i + 4], priR[i+3], priR[i+2], priR[i+1],priR[i]);//priR[i]
        //ymm5 = _mm256_set_ps(priI[i+7], priI[i+6], priI[i+5],priI[i + 4], priI[i+3], priI[i+2], priI[i+1],priI[i]);//priI[i]


        ymm1 = _mm256_loadu_ps(ctf + i);
        ymm2 = _mm256_loadu_ps(datR + i);
        ymm3 = _mm256_loadu_ps(datI + i);
        ymm4 = _mm256_loadu_ps(priR + i);
        ymm5 = _mm256_loadu_ps(priI + i);

        ymm4 = _mm256_mul_ps(ymm1, ymm4); //tmpReal
        ymm5 = _mm256_mul_ps(ymm1, ymm5);//tmpImag

        ymm4 = _mm256_sub_ps(ymm2, ymm4);//tmp1Real
        ymm5 = _mm256_sub_ps(ymm3, ymm5); //tmp1Imag

        ymm4 = _mm256_mul_ps(ymm4, ymm4);
        ymm5 = _mm256_mul_ps(ymm5, ymm5);

        ymm4 = _mm256_add_ps(ymm4, ymm5); //tmp2
        //ymm5 = _mm256_set_ps(sigRcp[i+7], sigRcp[i+6], sigRcp[i+5], sigRcp[i + 4], sigRcp[i+3], sigRcp[i+2], sigRcp[i+1], sigRcp[i]); //sigRcp
        ymm5 = _mm256_loadu_ps(sigRcp + i);

        ymm4 = _mm256_mul_ps(ymm4, ymm5);//tmp3
        ymm6 = _mm256_add_ps(ymm6, ymm4);//result2

    }

    float tmp[8] __attribute__((aligned(64)));
    _mm256_store_ps(tmp, ymm6);

    RFLOAT result = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    RFLOAT tmpReal = 0.0;
    RFLOAT tmpImag = 0.0;
    RFLOAT tmp1Real = 0.0;
    RFLOAT tmp1Imag = 0.0;
    RFLOAT tmp2;
    RFLOAT tmp3;

    for (; i < m; i++)
    {
        tmpReal = ctf[i] * priR[i];
        tmpImag = ctf[i] * priI[i];
        tmp1Real = datR[i] - tmpReal;
        tmp1Imag = datI[i] - tmpImag;
        tmp2 = tmp1Real * tmp1Real + tmp1Imag * tmp1Imag;
        tmp3 = tmp2 * sigRcp[i];

        result += tmp3;

    }

    return result;
}

#endif
#endif

#ifdef ENABLE_SIMD_256
#ifndef SINGLE_PRECISION
RFLOAT logDataVSPrior_double_SIMD256(const RFLOAT* datR,
        const RFLOAT* datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        const RFLOAT* ctf,
        const RFLOAT* sigRcp,
        const size_t m)
{
    __m256d ymm1, ymm2, ymm3, ymm4, ymm5,ymm6;
    ymm6 = _mm256_setzero_pd();
    int i = 0;
    for(i = 0; i <= (m - 4); i +=4)
    {
        //ymm1 = _mm256_set_pd(ctf[i+3], ctf[i+2], ctf[i+1], ctf[i]); //ctf[i]
        //ymm2 = _mm256_set_pd(dat[i+3].dat[0], dat[i+2].dat[0], dat[i+1].dat[0],dat[i].dat[0]);//dat[i].dat[0]
        //ymm3 = _mm256_set_pd(dat[i+3].dat[1], dat[i+2].dat[1], dat[i+1].dat[1],dat[i].dat[1]);//dat[i].dat[1]
        //ymm4 = _mm256_set_pd(pri[i+3].dat[0], pri[i+2].dat[0], pri[i+1].dat[0],pri[i].dat[0]);//pri[i].dat[0]
        //ymm5 = _mm256_set_pd(pri[i+3].dat[1], pri[i+2].dat[1], pri[i+1].dat[1],pri[i].dat[1]);//pri[i].dat[1]

        //ymm1 = _mm256_set_pd(ctf[i+3], ctf[i+2], ctf[i+1], ctf[i]); //ctf[i]
        //ymm2 = _mm256_set_pd(datR[i+3], datR[i+2], datR[i+1],datR[i]);//datR[i]
        //ymm3 = _mm256_set_pd(datI[i+3], datI[i+2], datI[i+1],datI[i]);//datI[i]
        //ymm4 = _mm256_set_pd(priR[i+3], priR[i+2], priR[i+1],priR[i]);//priR[i]
        //ymm5 = _mm256_set_pd(priI[i+3], priI[i+2], priI[i+1],priI[i]);//priI[i]


        ymm1 = _mm256_loadu_pd(ctf + i);
        ymm2 = _mm256_loadu_pd(datR + i);
        ymm3 = _mm256_loadu_pd(datI + i);
        ymm4 = _mm256_loadu_pd(priR + i);
        ymm5 = _mm256_loadu_pd(priI + i);

        ymm4 = _mm256_mul_pd(ymm1, ymm4); //tmpReal
        ymm5 = _mm256_mul_pd(ymm1, ymm5);//tmpImag

        ymm4 = _mm256_sub_pd(ymm2, ymm4);//tmp1Real
        ymm5 = _mm256_sub_pd(ymm3, ymm5); //tmp1Imag

        ymm4 = _mm256_mul_pd(ymm4, ymm4);
        ymm5 = _mm256_mul_pd(ymm5, ymm5);

        ymm4 = _mm256_add_pd(ymm4, ymm5); //tmp2

        //ymm5 = _mm256_set_pd(sigRcp[i+3], sigRcp[i+2], sigRcp[i+1], sigRcp[i]); //sigRcp

        ymm5 = _mm256_loadu_pd(sigRcp + i);
        ymm4 = _mm256_mul_pd(ymm4, ymm5);//tmp3

        ymm6 = _mm256_add_pd(ymm6, ymm4);//result2

    }


    double  tmp[4] __attribute__((aligned(64)));
    _mm256_store_pd(tmp, ymm6);

    RFLOAT result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    RFLOAT tmpReal = 0.0;
    RFLOAT tmpImag = 0.0;
    RFLOAT tmp1Real = 0.0;
    RFLOAT tmp1Imag = 0.0;
    RFLOAT tmp2;
    RFLOAT tmp3;

    for (; i < m; i++)
    {

        tmpReal = ctf[i] * priR[i];
        tmpImag = ctf[i] * priI[i];
        tmp1Real = datR[i] - tmpReal;
        tmp1Imag = datI[i] - tmpImag;

        tmp2 = tmp1Real * tmp1Real + tmp1Imag * tmp1Imag;
        tmp3 = tmp2 * sigRcp[i];


        result += tmp3;
    }

    return result;
}
#endif
#endif

#ifdef ENABLE_SIMD_512
#ifdef SINGLE_PRECISION
RFLOAT logDataVSPrior_float_SIMD512(const RFLOAT* datR,
        const RFLOAT* datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        const RFLOAT* ctf,
        const RFLOAT* sigRcp,
        const size_t m)
{
    __m512 ymm1, ymm2, ymm3, ymm4, ymm5,ymm6;
    ymm6 = _mm512_setzero_ps();
    int i = 0;
    for(i = 0; i <= (m - 16); i +=16)
    {
        //ymm1 = _mm512_set_ps(ctf[i+15], ctf[i+14], ctf[i+13], ctf[i+12], ctf[i+11], ctf[i+10], ctf[i+9], ctf[i+8],ctf[i+7], ctf[i+6], ctf[i+5], ctf[i+4], ctf[i+3], ctf[i+2], ctf[i+1], ctf[i]); //ctf[i]

        //ymm2 = _mm512_set_ps(dat[i+15].dat[0], dat[i+14].dat[0], dat[i+13].dat[0],dat[i+12].dat[0], dat[i+11].dat[0], dat[i+10].dat[0], dat[i+9].dat[0],dat[i+8].dat[0], dat[i+7].dat[0],  dat[i+6].dat[0],  dat[i+5].dat[0], dat[i+4].dat[0],  dat[i +3].dat[0], dat[i +2].dat[0], dat[i+1].dat[0],dat[i].dat[0]);//dat[i].dat[0]

        //ymm3 = _mm512_set_ps(dat[i+15].dat[1], dat[i+14].dat[1], dat[i+13].dat[1],dat[i+12].dat[1], dat[i+11].dat[1], dat[i+10].dat[1], dat[i+9].dat[1],dat[i+8].dat[1], dat[i+7].dat[1],  dat[i+6].dat[1],  dat[i+5].dat[1], dat[i+4].dat[1],  dat[i+3].dat[1],  dat[i+2].dat[1],  dat[i+1].dat[1],dat[i].dat[1]);//dat[i].dat[1]

        //ymm4 = _mm512_set_ps(pri[i+15].dat[0], pri[i+14].dat[0], pri[i+13].dat[0],pri[i+12].dat[0], pri[i+11].dat[0], pri[i+10].dat[0], pri[i+9].dat[0],pri[i+8].dat[0], pri[i+7].dat[0],  pri[i+6].dat[0],  pri[i+5].dat[0], pri[i+4].dat[0],  pri[i+3].dat[0],  pri[i+2].dat[0],  pri[i+1].dat[0],pri[i].dat[0]);//pri[i].dat[0]

        //ymm5 = _mm512_set_ps(pri[i+15].dat[1], pri[i+14].dat[1], pri[i+13].dat[1],pri[i+12].dat[1], pri[i+11].dat[1], pri[i+10].dat[1], pri[i+9].dat[1],pri[i+8].dat[1], pri[i+7].dat[1],  pri[i+6].dat[1],  pri[i+5].dat[1], pri[i+4].dat[1],  pri[i+3].dat[1],  pri[i+2].dat[1],  pri[i+1].dat[1],pri[i].dat[1]);//pri[i].dat[1]



        ymm1 = _mm512_loadu_ps(ctf + i);
        ymm2 = _mm512_loadu_ps(datR + i);
        ymm3 = _mm512_loadu_ps(datI + i);
        ymm4 = _mm512_loadu_ps(priR + i);
        ymm5 = _mm512_loadu_ps(priI + i);


        ymm4 = _mm512_mul_ps(ymm1, ymm4); //tmpReal
        ymm5 = _mm512_mul_ps(ymm1, ymm5);//tmpImag

        ymm4 = _mm512_sub_ps(ymm2, ymm4);//tmp1Real
        ymm5 = _mm512_sub_ps(ymm3, ymm5); //tmp1Imag

        ymm4 = _mm512_mul_ps(ymm4, ymm4);
        ymm5 = _mm512_mul_ps(ymm5, ymm5);

        ymm4 = _mm512_add_ps(ymm4, ymm5); //tmp2
        //ymm5 = _mm512_set_ps(sigRcp[i+15], sigRcp[i+14], sigRcp[i+13], sigRcp[i+12], sigRcp[i+11], sigRcp[i+10], sigRcp[i+9], sigRcp[i+8], sigRcp[i+7],  sigRcp[i+6],  sigRcp[i+5],  sigRcp[i + 4], sigRcp[i+3], sigRcp[i+2], sigRcp[i+1], sigRcp[i]); //sigRcp
        ymm5 = _mm512_loadu_ps(sigRcp + i);

        ymm4 = _mm512_mul_ps(ymm4, ymm5);//tmp3
        ymm6 = _mm512_add_ps(ymm6, ymm4);//result2

    }


    float tmp[16] __attribute__((aligned(64)));
    _mm512_store_ps(tmp, ymm6);

    RFLOAT result = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] + \
                    tmp[8] + tmp[9] + tmp[10] + tmp[11] + tmp[12] + tmp[13] + tmp[14] + tmp[15];
    RFLOAT tmpReal = 0.0;
    RFLOAT tmpImag = 0.0;
    RFLOAT tmp1Real = 0.0;
    RFLOAT tmp1Imag = 0.0;
    RFLOAT tmp2;
    RFLOAT tmp3;

    for (; i < m; i++)
    {
        tmpReal = ctf[i] * priR[i];
        tmpImag = ctf[i] * priI[i];
        tmp1Real = datR[i] - tmpReal;
        tmp1Imag = datI[i] - tmpImag;

        tmp2 = tmp1Real * tmp1Real + tmp1Imag * tmp1Imag;
        tmp3 = tmp2 * sigRcp[i];

        result += tmp3;

    }

    return result;
}

#endif
#endif

#ifdef ENABLE_SIMD_512
#ifndef SINGLE_PRECISION

RFLOAT logDataVSPrior_double_SIMD512(const RFLOAT* datR,
        const RFLOAT* datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        const RFLOAT* ctf,
        const RFLOAT* sigRcp,
        const size_t m)
{
    __m512d ymm1, ymm2, ymm3, ymm4, ymm5,ymm6;
    ymm6 = _mm512_setzero_pd();
    int i = 0;
    for(i = 0; i <= (m - 8); i +=8)
    {
        //ymm1 = _mm512_set_pd(ctf[i+7], ctf[i+6], ctf[i+5], ctf[i + 4], ctf[i+3], ctf[i+2], ctf[i+1], ctf[i]); //ctf[i]
        //ymm2 = _mm512_set_pd(dat[i+7].dat[0], dat[i+6].dat[0], dat[i+5].dat[0],dat[i + 4].dat[0],dat[i+3].dat[0], dat[i+2].dat[0], dat[i+1].dat[0],dat[i].dat[0]);//dat[i].dat[0]
        //ymm3 = _mm512_set_pd(dat[i+7].dat[1], dat[i+6].dat[1], dat[i+5].dat[1],dat[i + 4].dat[1],dat[i+3].dat[1], dat[i+2].dat[1], dat[i+1].dat[1],dat[i].dat[1]);//dat[i].dat[1]
        //ymm4 = _mm512_set_pd(pri[i+7].dat[0], pri[i+6].dat[0], pri[i+5].dat[0],pri[i + 4].dat[0],pri[i+3].dat[0], pri[i+2].dat[0], pri[i+1].dat[0],pri[i].dat[0]);//pri[i].dat[0]
        //ymm5 = _mm512_set_pd(pri[i+7].dat[1], pri[i+6].dat[1], pri[i+5].dat[1],pri[i + 4].dat[1],pri[i+3].dat[1], pri[i+2].dat[1], pri[i+1].dat[1],pri[i].dat[1]);//pri[i].dat[1]

        //ymm1 = _mm512_set_pd(ctf[i+7], ctf[i+6], ctf[i+5], ctf[i + 4], ctf[i+3], ctf[i+2], ctf[i+1], ctf[i]); //ctf[i]
        //ymm2 = _mm512_set_pd(datR[i+7], datR[i+6], datR[i+5],datR[i + 4],datR[i+3], datR[i+2], datR[i+1], datR[i]);//
        //ymm3 = _mm512_set_pd(datI[i+7], datI[i+6], datI[i+5],datI[i + 4],datI[i+3], datI[i+2], datI[i+1], datI[i]);//
        //ymm4 = _mm512_set_pd(priR[i+7], priR[i+6], priR[i+5],priR[i + 4],priR[i+3], priR[i+2], priR[i+1],priR[i]);//priR[i]
        //ymm5 = _mm512_set_pd(priI[i+7], priI[i+6], priI[i+5],priI[i + 4],priI[i+3], priI[i+2], priI[i+1],priI[i]);//priI[i]


        ymm1 = _mm512_loadu_pd(ctf + i);
        ymm2 = _mm512_loadu_pd(datR + i);
        ymm3 = _mm512_loadu_pd(datI + i);
        ymm4 = _mm512_loadu_pd(priR + i);
        ymm5 = _mm512_loadu_pd(priI + i);

        ymm4 = _mm512_mul_pd(ymm1, ymm4); //tmpReal
        ymm5 = _mm512_mul_pd(ymm1, ymm5);//tmpImag

        ymm4 = _mm512_sub_pd(ymm2, ymm4);//tmp1Real
        ymm5 = _mm512_sub_pd(ymm3, ymm5); //tmp1Imag

        ymm4 = _mm512_mul_pd(ymm4, ymm4);
        ymm5 = _mm512_mul_pd(ymm5, ymm5);

        ymm4 = _mm512_add_pd(ymm4, ymm5); //tmp2

        //ymm5 = _mm512_set_pd(sigRcp[i+7], sigRcp[i+6], sigRcp[i+5], sigRcp[i+4], sigRcp[i+3], sigRcp[i+2], sigRcp[i+1], sigRcp[i]); //sigRcp
        ymm5 = _mm512_loadu_pd(sigRcp + i);

        ymm4 = _mm512_mul_pd(ymm4, ymm5);//tmp3

        ymm6 = _mm512_add_pd(ymm6, ymm4);//result2

    }

    double  tmp[8] __attribute__((aligned(64)));
    _mm512_store_pd(tmp, ymm6);

    RFLOAT result = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    RFLOAT tmpReal = 0.0;
    RFLOAT tmpImag = 0.0;
    RFLOAT tmp1Real = 0.0;
    RFLOAT tmp1Imag = 0.0;
    RFLOAT tmp2;
    RFLOAT tmp3;

    for (; i < m; i++)
    {
        tmpReal = ctf[i] * priR[i];
        tmpImag = ctf[i] * priI[i];
        tmp1Real = datR[i] - tmpReal;
        tmp1Imag = datI[i] - tmpImag;

        tmp2 = tmp1Real * tmp1Real + tmp1Imag * tmp1Imag;
        tmp3 = tmp2 * sigRcp[i];

        result += tmp3;
    }

    return result;
}
#endif
#endif

/***
#ifdef ENABLE_SIMD_256
#ifdef SINGLE_PRECISION

RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
        MemoryBazaar<RFLOAT, BaseType, 4>& datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
        MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
        const size_t n,
        const size_t m,
        RFLOAT* SIMDResult)

{
    RFLOAT *result = SIMDResult;
    for(size_t i = 0; i < n; i ++)
    {

        result[i] += logDataVSPrior(&datR[m * i], 
                &datI[m * i], 
                priR, 
                priI, 
                &ctf[m * i], 
                &sigRcp[m * i], 
                m);
    }
    return result;
}

#endif
#endif

#ifdef ENABLE_SIMD_256
#ifndef SINGLE_PRECISION

RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
        MemoryBazaar<RFLOAT, BaseType, 4>& datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
        MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
        const size_t n,
        const size_t m,
        RFLOAT* SIMDResult)

{
    RFLOAT *result = SIMDResult;
    for(size_t i = 0; i < n; i ++)
    {

        result[i] += logDataVSPrior(&datR[m * i], 
                &datI[m * i], 
                priR, 
                priI, 
                &ctf[m * i], 
                &sigRcp[m * i], 
                m);
    }
    return result;
}

#endif
#endif

#ifdef ENABLE_SIMD_512
#ifdef SINGLE_PRECISION

RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
        MemoryBazaar<RFLOAT, BaseType, 4>& datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
        MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
        const size_t n,
        const size_t m,
        RFLOAT* SIMDResult)

{
    RFLOAT *result = SIMDResult;
    for(size_t i = 0; i < n; i ++)
    {

        result[i] += logDataVSPrior(&datR[m * i], 
                &datI[m * i], 
                priR, 
                priI, 
                &ctf[m * i], 
                &sigRcp[m * i], 
                m);
    }
    return result;
}

#endif
#endif

#ifdef ENABLE_SIMD_512
#ifndef SINGLE_PRECISION

RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
        MemoryBazaar<RFLOAT, BaseType, 4>& datI,
        const RFLOAT* priR,
        const RFLOAT* priI,
        MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
        MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
        const size_t n,
        const size_t m,
        RFLOAT* SIMDResult)

{
    RFLOAT *result = SIMDResult;
    for(size_t i = 0; i < n; i ++)
    {

        result[i] += logDataVSPrior(&datR[m * i], 
                &datI[m * i], 
                priR, 
                priI, 
                &ctf[m * i], 
                &sigRcp[m * i], 
                m);
    }

    return result;
}

#endif
#endif

***/

RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
                       MemoryBazaar<RFLOAT, BaseType, 4>& datI,
                       const RFLOAT* priR,
                       const RFLOAT* priI,
                       MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
                       MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
                       const size_t n,
                       const size_t m,
                       RFLOAT* SIMDResult)

{
    RFLOAT *result = SIMDResult;
    for(size_t i = 0; i < n; i ++)
    {

        result[i] += logDataVSPrior(&datR[m * i], 
                                    &datI[m * i], 
                                    priR, 
                                    priI, 
                                    &ctf[m * i], 
                                    &sigRcp[m * i], 
                                    m);
    }

    return result;
}

RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
                       MemoryBazaar<RFLOAT, BaseType, 4>& datI,
                       const RFLOAT* priR,
                       const RFLOAT* priI,
                       MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
                       MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
                       const size_t nd,
                       const size_t np,
                       const size_t m,
                       RFLOAT* SIMDResult,
                       const size_t nThread)

{
    // std::cout << "nd = " << nd << std::endl;
    // std::cout << "np = " << np << std::endl;
    // std::cout << "m = " << m << std::endl;
    // std::cout << "nThread = " << nThread;

    RFLOAT *result = SIMDResult;

    MemoryBazaarDustman<RFLOAT, BaseType, 4> datRDustman(&datR);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> datIDustman(&datI);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> ctfDustman(&ctf);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> sigRcpDustman(&sigRcp);
    #pragma omp parallel for schedule(dynamic) num_threads(nThread) firstprivate(datRDustman, datIDustman, ctfDustman, sigRcpDustman)
    // #pragma omp parallel for schedule(dynamic) num_threads(nThread) firstprivate(datRDustman, datIDustman)
    for(size_t id = 0; id < nd; id++) // loop over data
    {
        datR.endLastVisit(m * id);
        datI.endLastVisit(m * id);
        ctf.endLastVisit(m * id);
        sigRcp.endLastVisit(m * id);

        RFLOAT* ptr_datR = &datR[m * id];
        RFLOAT* ptr_datI = &datI[m * id];
        RFLOAT* ptr_ctf = &ctf[m * id];
        RFLOAT* ptr_sigRcp = &sigRcp[m * id];

        for (size_t ip = 0; ip < np; ip++) // loop over prior
        {
            result[ip * nd + id] += logDataVSPrior(ptr_datR, 
                                                   ptr_datI, 
                                                   &priR[m * ip], 
                                                   &priI[m * ip], 
                                                   ptr_ctf, 
                                                   ptr_sigRcp, 
                                                   m);
        }
    }

    return result;
}
