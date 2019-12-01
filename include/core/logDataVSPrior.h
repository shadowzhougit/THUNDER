/** @file
 *  @author Huabin Ruan
 *  @version 1.4.14.190716
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Huabin Ruan | 2019/07/16 | 1.4.14.190716 | Refactor SIMD functions
 *
 * @brief This file defines simd implementation for logDataVSPrior.
 *
 */

#ifndef LOG_DATA_VS_PRIOR_H
#define LOG_DATA_VS_PRIOR_H

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Precision.h"
#include "MemoryBazaar.h"

//#include <immintrin.h>
//#include <unistd.h>
//#include <stdio.h>
//typedef float RFLOAT;
////typedef double RFLOAT;

//#define  ENABLE_SIMD_256
//#define  SINGLE_PRECISION


#ifndef ENABLE_SIMD_256
#ifndef ENABLE_SIMD_512

/**
 * @brief This function calculates the logarithm of the likelihood an image (data) and a projection (prior).
 *
 * @return the logarithm of the likelihood
 */
RFLOAT logDataVSPrior(const RFLOAT* datR,   /**< [in] real part of the serialized image */
                      const RFLOAT* datI,   /**< [in] imaginary part of the serialized image */
                      const RFLOAT* priR,   /**< [in] real part of the serialized rojection */
                      const RFLOAT* priI,   /**< [in] imaginary part of the projection */
                      const RFLOAT* ctf,    /**< [in] the serialized CTF */
                      const RFLOAT* sigRcp, /**< [in] the serialized reciprocal of squared standard deviation of image noise */
                      const size_t m        /**< [in] the number of elements in this serialization */
                      );

/***
RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
                       MemoryBazaar<RFLOAT, BaseType, 4>& datI,
                       const RFLOAT* priR,
                       const RFLOAT* priI,
                       MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
                       MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
                       const size_t n,
                       const size_t m,
                       RFLOAT* resultBuffer);
***/
#endif
#endif

#ifdef ENABLE_SIMD_256
#ifdef SINGLE_PRECISION

RFLOAT logDataVSPrior_float_SIMD256(const RFLOAT* datR,   /**< [in] real part of the serialized image */
                                    const RFLOAT* datI,   /**< [in] imaginary part of the serialized image */
                                    const RFLOAT* priR,   /**< [in] real part of the serialized rojection */
                                    const RFLOAT* priI,   /**< [in] imaginary part of the projection */
                                    const RFLOAT* ctf,    /**< [in] the serialized CTF */
                                    const RFLOAT* sigRcp, /**< [in] the serialized reciprocal of squared standard deviation of image noise */
                                    const size_t m        /**< [in] the number of elements in this serialization */
                                 );

inline RFLOAT logDataVSPrior(const RFLOAT* datR,
                             const RFLOAT* datI,
                             const RFLOAT* priR,
                             const RFLOAT* priI,
                             const RFLOAT* ctf,
                             const RFLOAT* sigRcp,
                             const size_t m)
{
    return logDataVSPrior_float_SIMD256(datR, datI, priR, priI, ctf, sigRcp, m);
}

#endif
#endif

#ifdef ENABLE_SIMD_256
#ifndef SINGLE_PRECISION

RFLOAT logDataVSPrior_double_SIMD256(const RFLOAT* datR,   /**< [in] real part of the serialized image */
                                     const RFLOAT* datI,   /**< [in] imaginary part of the serialized image */
                                     const RFLOAT* priR,   /**< [in] real part of the serialized rojection */
                                     const RFLOAT* priI,   /**< [in] imaginary part of the projection */
                                     const RFLOAT* ctf,    /**< [in] the serialized CTF */
                                     const RFLOAT* sigRcp, /**< [in] the serialized reciprocal of squared standard deviation of image noise */
                                     const size_t m        /**< [in] the number of elements in this serialization */
                                 );

inline RFLOAT logDataVSPrior(const RFLOAT* datR,
                             const RFLOAT* datI,
                             const RFLOAT* priR,
                             const RFLOAT* priI,
                             const RFLOAT* ctf,
                             const RFLOAT* sigRcp,
                             const size_t m)
{
    return logDataVSPrior_double_SIMD256(datR, datI, priR, priI, ctf, sigRcp, m);
}

#endif
#endif

#ifdef ENABLE_SIMD_512
#ifdef SINGLE_PRECISION

RFLOAT logDataVSPrior_float_SIMD512(const RFLOAT* datR,   /**< [in] real part of the serialized image */
                                    const RFLOAT* datI,   /**< [in] imaginary part of the serialized image */
                                    const RFLOAT* priR,   /**< [in] real part of the serialized rojection */
                                    const RFLOAT* priI,   /**< [in] imaginary part of the projection */
                                    const RFLOAT* ctf,    /**< [in] the serialized CTF */
                                    const RFLOAT* sigRcp, /**< [in] the serialized reciprocal of squared standard deviation of image noise */
                                    const size_t m        /**< [in] the number of elements in this serialization */
                                    );

inline RFLOAT logDataVSPrior(const RFLOAT* datR,
                             const RFLOAT* datI,
                             const RFLOAT* priR,
                             const RFLOAT* priI,
                             const RFLOAT* ctf,
                             const RFLOAT* sigRcp,
                             const size_t m)
{
    return logDataVSPrior_float_SIMD512(datR, datI, priR, priI, ctf, sigRcp, m);
}

#endif
#endif

#ifdef ENABLE_SIMD_512
#ifndef SINGLE_PRECISION

RFLOAT logDataVSPrior_double_SIMD512(const RFLOAT* datR,   /**< [in] real part of the serialized image */
                                     const RFLOAT* datI,   /**< [in] imaginary part of the serialized image */
                                     const RFLOAT* priR,   /**< [in] real part of the serialized rojection */
                                     const RFLOAT* priI,   /**< [in] imaginary part of the projection */
                                     const RFLOAT* ctf,    /**< [in] the serialized CTF */
                                     const RFLOAT* sigRcp, /**< [in] the serialized reciprocal of squared standard deviation of image noise */
                                     const size_t m        /**< [in] the number of elements in this serialization */
                                     );

inline RFLOAT logDataVSPrior(const RFLOAT* datR,
                             const RFLOAT* datI,
                             const RFLOAT* priR,
                             const RFLOAT* priI,
                             const RFLOAT* ctf,
                             const RFLOAT* sigRcp,
                             const size_t m)
{
    return logDataVSPrior_double_SIMD512(datR, datI, priR, priI, ctf, sigRcp, m);
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
                       RFLOAT* resultBuffer);

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
                       RFLOAT* SIMDResult);
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
                       RFLOAT* SIMDResult);
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
                       RFLOAT* SIMDResult);
#endif
#endif
***/

RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
                       MemoryBazaar<RFLOAT, BaseType, 4>& datI,
                       const RFLOAT* priR,
                       const RFLOAT* priI,
                       MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
                       MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
                       const size_t n,        /**< [in] the number of images in data */
                       const size_t m,        /**< [in] the number of elements in this serialization */
                       RFLOAT* SIMDResult);

/**
 * @return a matrix, with data major, prior minor indexing, a.k.a., ip * nd + id
 */
RFLOAT* logDataVSPrior(MemoryBazaar<RFLOAT, BaseType, 4>& datR,
                       MemoryBazaar<RFLOAT, BaseType, 4>& datI,
                       const RFLOAT* priR,
                       const RFLOAT* priI,
                       MemoryBazaar<RFLOAT, BaseType, 4>& ctf,
                       MemoryBazaar<RFLOAT, BaseType, 4>& sigRcp,
                       const size_t nd,       /**< [in] the number of images in data */
                       const size_t np,       /**< [in] the number of images in prior */
                       const size_t m,        /**< [in] the number of elements in this serialization */
                       RFLOAT* SIMDResult,
                       const size_t nThread);

#endif // LOG_DATA_VS_PRIOR_H
