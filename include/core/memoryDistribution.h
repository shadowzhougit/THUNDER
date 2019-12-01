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

#ifndef MEMORY_DISTRIBUTION_H
#define MEMORY_DISTRIBUTION_H

#include "Optimiser.h"

void assignMemoryDistribution(MemoryDistribution& md,
                              const RFLOAT totalMemorySize,
                              const size_t imgSize,
                              const size_t imgPackSize,
                              const size_t nImg,
                              const size_t minNStall)
{
    // two order of priority

    // img, imgOri, low priority
    // datPR, datPI, CtfP, SigRcpP, high priority

    RFLOAT portionImg = 1;
    RFLOAT portionImgOri = 1;
    RFLOAT portionDatPR = M_PI / 8;
    RFLOAT portionDatPI = M_PI / 8;
    RFLOAT portionCtfP = M_PI / 8;
    RFLOAT portionSigRcpP = M_PI / 8;

    RFLOAT total = portionImg
                 + portionImgOri
                 + portionDatPR
                 + portionDatPI
                 + portionCtfP
                 + portionSigRcpP;

    RFLOAT totalLowPriority = portionImg + portionImgOri;

    RFLOAT totalHighPriority = portionDatPR
                             + portionDatPI
                             + portionCtfP
                             + portionSigRcpP;

    // assign with minimal number of stalls

    md.nStallImg = minNStall;
    md.nStallImgOri = minNStall;
    md.nStallDatPR = minNStall;
    md.nStallDatPI = minNStall;
    md.nStallCtfP = minNStall;
    md.nStallSigRcpP = minNStall;

    RFLOAT sizeAllocated = imgSize * imgPackSize * total * minNStall * 4;

    if (sizeAllocated >= totalMemorySize)
    {
        // there is no room to assign
        return;
    }

    if (minNStall >= nImg)
    {
        // there is no need to assign
        return;
    }

    size_t nImgForHighPriority = CEIL(TSGSL_MIN_RFLOAT((totalMemorySize - sizeAllocated) / (totalHighPriority * imgSize), (RFLOAT)(nImg - minNStall)) / imgPackSize / 4);

    md.nStallDatPR += nImgForHighPriority;
    md.nStallDatPI += nImgForHighPriority;
    md.nStallCtfP += nImgForHighPriority;
    md.nStallSigRcpP += nImgForHighPriority;

    sizeAllocated += imgSize * imgPackSize * totalHighPriority * nImgForHighPriority * 4;

    if (sizeAllocated >= totalMemorySize)
    {
        // there is no room to assign
        return;
    }

    size_t nImgForLowPriority = CEIL(TSGSL_MIN_RFLOAT((totalMemorySize - sizeAllocated) / (totalLowPriority * imgSize), (RFLOAT)(nImg - minNStall)) / imgPackSize / 4);

    md.nStallImg += nImgForLowPriority;
    md.nStallImgOri += nImgForLowPriority;


    // std::cout << "CEIL((RFLOAT)nImg / 4) = " << CEIL((RFLOAT)nImg / 4) << std::endl;

    // size_t nImgInMemory = nImg;

    // size_t nImgInMemory = CEIL((RFLOAT)nImg / 4);

    // md.nStallImg = nImgInMemory;
    // md.nStallImgOri = nImgInMemory;
    // md.nStallDatPR = nImgInMemory;
    // md.nStallDatPI = nImgInMemory;
    // md.nStallCtfP = nImgInMemory;
    // md.nStallSigRcpP = nImgInMemory;
}

size_t referenceMemorySize(const size_t boxsize,
                           const size_t pf,
                           const int mode,
                           const size_t nReference)
{
    if (mode == MODE_2D)
    {
        return 4 * (boxsize * pf) * (boxsize * pf) * nReference * sizeof(RFLOAT);
    }
    else
    {
        return 4 * (boxsize * pf) * (boxsize * pf) * (boxsize * pf) * nReference * sizeof(RFLOAT);
    }
}

#endif // MEMORY_DISTRIBUTION_H
