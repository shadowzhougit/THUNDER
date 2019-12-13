/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef IMAGE_FUNCTIONS_H
#define IMAGE_FUNCTIONS_H

#include <cmath>

#include <iostream>

#include <omp_compat.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics.h>

#include "Precision.h"

#include "Random.h"
#include "FFT.h"
#include "Image.h"
#include "Volume.h"

inline void IMG_EXTRACT_RL(Image& dst,
                           const Image& src,
                           const RFLOAT ef,
                           const unsigned int nThread)
{
    dst.alloc(AROUND(ef * src.nColRL()),
              AROUND(ef * src.nRowRL()),
              RL_SPACE);

    #pragma omp parallel for num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_RL(dst)
        dst.setRL(src.getRL(i, j), i, j);
}

/**
 * This function extracts the center block out of a volume in real space.
 *
 * @param dst the destination volume
 * @param src the source volume
 * @param ef  the extraction factor (0 < ef <= 1)
 */
inline void VOL_EXTRACT_RL(Volume& dst,
                           const Volume& src,
                           const RFLOAT ef,
                           const unsigned int nThread)
{
    dst.alloc(AROUND(ef * src.nColRL()),
              AROUND(ef * src.nRowRL()),
              AROUND(ef * src.nSlcRL()),
              RL_SPACE);

    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_RL(dst)
        dst.setRL(src.getRL(i, j, k), i, j, k);
}

inline void IMG_EXTRACT_FT(Image& dst,
                           const Image& src,
                           const RFLOAT ef,
                           const unsigned int nThread)
{
    dst.alloc(AROUND(ef * src.nColRL()),
              AROUND(ef * src.nRowRL()),
              FT_SPACE);

    #pragma omp parallel for num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_FT(dst)
        dst.setFTHalf(src.getFTHalf(i, j), i, j);
}

/**
 * This function extracts the center block out of a volume in Fourier space.
 *
 * @param dst the destination volume
 * @param src the source volume
 * @param ef  the extraction factor (0 < ef <= 1)
 */
inline void VOL_EXTRACT_FT(Volume& dst,
                           const Volume& src,
                           const RFLOAT ef,
                           const unsigned int nThread)
{
    dst.alloc(AROUND(ef * src.nColRL()),
              AROUND(ef * src.nRowRL()),
              AROUND(ef * src.nSlcRL()),
              FT_SPACE);

    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_FT(dst)
        dst.setFTHalf(src.getFTHalf(i, j, k), i, j, k);
}

inline void IMG_BOX_RL(Image& dst,
                       const Image& src,
                       const unsigned int nThread)
{
    #pragma omp parallel for num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_RL(dst)
        dst.setRL(src.getRL(i, j), i, j);
}

inline void VOL_BOX_RL(Volume& dst,
                       const Volume& src,
                       const unsigned int nThread)
{
    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_RL(dst)
        dst.setRL(src.getRL(i, j, k), i, j, k);
}

inline void IMG_BOX_FT(Image& dst,
                       const Image& src,
                       const unsigned int nThread)
{
    #pragma omp parallel for num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_FT(dst)
        dst.setFT(src.getFT(i, j), i, j);
}

inline void VOL_BOX_FT(Volume& dst,
                       const Volume& src,
                       const unsigned int nThread)
{
    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_FT(dst)
        dst.setFT(src.getFT(i, j, k), i, j, k);
}

inline void IMG_REPLACE_RL(Image& dst,
                           const Image& src,
                           const unsigned int nThread)
{
    #pragma omp parallel for num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_RL(src)
        dst.setRL(src.getRL(i, j), i, j);
}

/**
 * This function replaces the center block of a volume with another volume in
 * real space.
 *
 * @param dst the destination volume
 * @param src the source volume
 */
inline void VOL_REPLACE_RL(Volume& dst,
                           const Volume& src,
                           const unsigned int nThread)
{
    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_RL(src)
        dst.setRL(src.getRL(i, j, k), i, j, k);
}

inline void IMG_REPLACE_FT(Image& dst,
                           const Image& src,
                           const unsigned int nThread)
{
    #pragma omp parallel for num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_FT(src)
        dst.setFTHalf(src.getFTHalf(i, j), i, j);
}

/**
 * This function replaces the center block of a volume with another volume in
 * Fourier space.
 *
 * @param dst the destination volume
 * @param src the source volume
 */
inline void VOL_REPLACE_FT(Volume& dst,
                           const Volume& src,
                           const unsigned int nThread)
{
    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_FT(src)
        dst.setFTHalf(src.getFTHalf(i, j, k), i, j, k);
}

inline void IMG_PAD_RL(Image& dst,
                       const Image& src,
                       const int pf,
                       const unsigned int nThread)
{
    dst.alloc(pf * src.nColRL(),
              pf * src.nRowRL(),
              RL_SPACE);

    #pragma omp parallel for num_threads(nThread)
    SET_0_RL(dst);

    #pragma omp parallel for num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_RL(src)
        dst.setRL(src.getRL(i, j), i, j);
}

/**
 * This function pads a volume in real space.
 *
 * @param dst the destination volume
 * @param src the source volume
 * @param pf  the padding factor
 */
inline void VOL_PAD_RL(Volume& dst,
                       const Volume& src,
                       const int pf,
                       const unsigned int nThread)
{
    dst.alloc(pf * src.nColRL(),
              pf * src.nRowRL(),
              pf * src.nSlcRL(),
              RL_SPACE);

    #pragma omp parallel for num_threads(nThread)
    SET_0_RL(dst);

    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_RL(src)
        dst.setRL(src.getRL(i, j, k), i, j, k);
}

inline void IMG_PAD_FT(Image& dst,
                       const Image& src,
                       const int pf,
                       const unsigned int nThread)
{
    dst.alloc(pf * src.nColRL(),
              pf * src.nRowRL(),
              FT_SPACE);

    #pragma omp parallel for num_threads(nThread)
    SET_0_FT(dst);

    #pragma omp parallel for num_threads(nThread)
    IMAGE_FOR_EACH_PIXEL_FT(src)
        dst.setFTHalf(src.getFTHalf(i, j), i, j);
}

/**
 * This function pads a volume in Fourier space.
 *
 * @param dst the destination volume
 * @param src the source volume
 * @param pf  the padding factor
 */
inline void VOL_PAD_FT(Volume& dst,
                       const Volume& src,
                       const int pf,
                       const unsigned int nThread)
{
    dst.alloc(pf * src.nColRL(),
              pf * src.nRowRL(),
              pf * src.nSlcRL(),
              FT_SPACE);

    #pragma omp parallel for num_threads(nThread)
    SET_0_FT(dst);

    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_FT(src)
        dst.setFTHalf(src.getFTHalf(i, j, k), i, j, k);
}

/**
 * This function replaces a slice of a volume with an image given in real space.
 *
 * @param dst the destination volume
 * @param src the source image
 * @param k   the index of the slice
 */
inline void SLC_REPLACE_RL(Volume& dst, const Image& src, const int k)
{
    IMAGE_FOR_EACH_PIXEL_RL(src)
        dst.setRL(src.getRL(i, j), i, j, k);
}

/**
 * This function replaces a slice of a volume with an image given in Fourier
 * space.
 *
 * @param dst the destination volume
 * @param src the source image
 * @param k   the index of the slice
 */
inline void SLC_REPLACE_FT(Volume& dst, const Image& src, const int k)
{
    IMAGE_FOR_EACH_PIXEL_FT(src)
        dst.setFTHalf(src.getFTHalf(i, j), i, j, k);
}

/**
 * This macro extracts a slice out of a volume and stores it in an image in
 * real space.
 *
 * @param dst the destination image
 * @param src the source volume
 * @param k   the index of the slice
 */
inline void SLC_EXTRACT_RL(Image& dst, const Volume& src, const int k)
{
    IMAGE_FOR_EACH_PIXEL_RL(dst)
        dst.setRL(src.getRL(i, j, k), i, j);
}

/**
 * This macro extracts a slice out of a volume and stores it in an image in
 * Fourier space.
 *
 * @param dst the destination image
 * @param src the source volume
 * @param k   the index of the slice
 */
inline void SLC_EXTRACT_FT(Image& dst, const Volume& src, const int k)
{
    IMAGE_FOR_EACH_PIXEL_FT(dst)
        dst.setFTHalf(src.getFTHalf(i, j, k), i, j);
}

vec2 centroid(const Image& img,
              const unsigned int nThread);

vec3 centroid(const Volume& vol,
              const unsigned int nThread);

void mul(Image& dst,
         const Image& a,
         const Image& b,
         const int r);

void mul(Image& dst,
         const Image& a,
         const Image& b,
         const int* iPxl,
         const int nPxl);

/**
 * This function generates a "translation image" with a given vector indicating
 * the number of columns and the number of rows. An image can be translated by
 * this vector, just by multiplying this "translation image" in Fourier space.
 *
 * @param dst       the translation image
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
//void translate(Image& dst,
//               const RFLOAT nTransCol,
//               const RFLOAT nTransRow);

/**
 * This function generates a "translation image" with a given vector indicating
 * the number of columns and the number of rows using multiple threads. An
 * image can be translated by this vector, just by multiplying this "translation
 * image" in Fourier space.
 *
 * @param dst       the translation image
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const unsigned int nThread);

void translate(Volume& dst,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const RFLOAT nTransSlc,
               const unsigned int nThread);

/**
 * This function generates a "translation image" in a certain frequency
 * threshold with a given vector indicating the number of columns and the number
 * of rows. An image can be translated by this vector, just by multiplying this
 * "translation image" in Fourier space.
 *
 * @param dst       the translation image
 * @param r         the resolution threshold
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
//void translate(Image& dst,
//               const RFLOAT r,
//               const RFLOAT nTransCol,
//               const RFLOAT nTransRow);

/**
 * This function generates a "translation image" in a certain frequency
 * threshold with a given vector indicating the number of columns and the number
 * of rows using multiple threads. An image can be translated by this vector, just
 * by multiplying this "translation image" in Fourier space.
 *
 * @param dst       the translation image
 * @param r         the resolution threshold
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const RFLOAT r,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const unsigned int nThread);

void translate(Image& dst,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int* iCol,
               const int* iRow,
               const int* iPxl,
               const int nPxl,
               const unsigned int nThread);

void translate(Complex* dst,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl,
               const unsigned int nThread);

void translate(RFLOAT* dstR,
               RFLOAT* dstI,
               const RFLOAT* srcR,
               const RFLOAT* srcI,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl,
               const unsigned int nThread);

/**
 * This function translations an image with a given vector indicating by the number
 * of columns and the number of rows.
 *
 * @param dst       the destination image (Fourier space)
 * @param src       the source image (Fourier space)
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
//void translate(Image& dst,
//               const Image& src,
//               const RFLOAT nTransCol,
//               const RFLOAT nTransRow);

/**
 * This function translations an image with a given vector indicating by the number
 * of columns and the number of rows using multiple threads.
 *
 * @param dst       the destination image (Fourier space)
 * @param src       the source image (Fourier space)
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const Image& src,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const unsigned int nThread);

void translate(Volume& dst,
               const Volume& src,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const RFLOAT nTransSlc,
               const unsigned int nThread);

/**
 * This function translations an image in a certain frequency threshold with a
 * given vector indicating by the number of columns and the number of rows.
 *
 * @param dst       the destination image (Fourier space)
 * @param src       the source image (Fourier space)
 * @param r         the resolution threshold
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
//void translate(Image& dst,
//               const Image& src,
//               const RFLOAT r,
//               const RFLOAT nTransCol,
//               const RFLOAT nTransRow);

/**
 * This function translations an image in a certain frequency threshold with a
 * given vector indicating by the number of columns and the number of rows using
 * multiple threads.
 *
 * @param dst       the destination image (Fourier space)
 * @param src       the source image (Fourier space)
 * @param r         the resolution threshold
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const Image& src,
               const RFLOAT r,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const unsigned int nThread);

void translate(const int ip,
               Image& img,
               const RFLOAT r,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const unsigned int nThread);

void translate(Volume& dst,
               const Volume& src,
               const RFLOAT r,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const RFLOAT nTransSlc,
               const unsigned int nThread);

void tranlsate(const int ip,
               Volume& vol,
               const RFLOAT r,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const RFLOAT nTransSlc,
               const unsigned int nThread);

//void translate(Image& dst,
//               const Image& src,
//               const RFLOAT nTransCol,
//               const RFLOAT nTransRow,
//               const int* iCol,
//               const int* iRow,
//               const int* iPxl,
//               const int nPxl);

void translate(Image& dst,
               const Image& src,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int* iCol,
               const int* iRow,
               const int* iPxl,
               const int nPxl,
               const unsigned int nThread);

//void translate(Complex* dst,
//               const Complex* src,
//               const RFLOAT nTransCol,
//               const RFLOAT nTransRow,
//               const int nCol,
//               const int nRow,
//               const int* iCol,
//               const int* iRow,
//               const int nPxl);


void translate(Complex* dst,
               const Complex* src,
               const RFLOAT nTransCol,
               const RFLOAT nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl,
               const unsigned int nThread);

/**
 * This function calculates the cross-correlation image of two images in a
 * certain region.
 * @param dst the destination image
 * @param a the image A
 * @param b the image B
 * @param r the radius of the frequency
 */
void crossCorrelation(Image& dst,
                      const Image& a,
                      const Image& b,
                      const RFLOAT r);

/**
 * This function calculates the most likely translation between two images using
 * max cross correlation method.
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 * @param a Image A
 * @param b Image B
 * @param r the radius of the frequency
 * @param maxX the maximum column translation
 * @param maxY the maximum row translation
 */
void translate(int& nTranCol,
               int& nTranRow,
               const Image& a,
               const Image& b,
               const RFLOAT r,
               const int maxX,
               const int maxY,
               const unsigned int nThread);

/**
 * This function calculates the standard deviation of an image given when the
 * mean value is given.
 *
 * @param mean the mean value
 * @param the image to be calculated
 */
RFLOAT stddev(const RFLOAT mean,
              const Image& src);

/**
 * This function calculates the mean and standard deviation of an image.
 *
 * @param mean the mean value
 * @param stddev the standard deviation
 * @param src the image to be calculated
 */
void meanStddev(RFLOAT& mean,
                RFLOAT& stddev,
                const Image& src);

RFLOAT centreStddev(const RFLOAT mean,
                    const Image& src,
                    const RFLOAT r);

void centreMeanStddev(RFLOAT& mean,
                      RFLOAT& stddev,
                      const Image& src,
                      const RFLOAT r);

/**
 * This function calculates the standard deviation of the background when the
 * mean value is given.
 *
 * @param mean the mean value
 * @param src the image to be calculated
 * @param r the radius
 */
RFLOAT bgStddev(const RFLOAT mean,
                const Image& src,
                const RFLOAT r);

RFLOAT bgSttdev(const RFLOAT mean,
                const Volume& src,
                const RFLOAT r);

/**
 * This function calculates the mean and standard deviation of the background.
 * The background stands for the outer region beyond a certain radius.
 *
 * @param mean the mean value
 * @param stddev the standard deviation
 * @param src the image to be calculated
 * @param r the radius
 */
void bgMeanStddev(RFLOAT& mean,
                  RFLOAT& stddev,
                  const Image& src,
                  const RFLOAT r);

void bgMeanStddev(RFLOAT& mean,
                  RFLOAT& stddev,
                  const Volume& src,
                  const RFLOAT r);

void bgMeanStddev(RFLOAT& mean,
                  RFLOAT& stddev,
                  const Volume& src,
                  const RFLOAT rU,
                  const RFLOAT rL);


/**
 * This function removes white and black dust from the image. Any value out of
 * the range (mean - bDust * stddev, mean + wDust * stddev) will be replaced by
 * a draw from N(mean, stddev).
 * @param img the image to be processed
 * @param wDust the factor of white dust
 * @param bDust the factor of black dust
 * @param mean the mean value
 * @param stddev the standard deviation
 */
void removeDust(Image& img,
                const RFLOAT wDust,
                const RFLOAT bDust,
                const RFLOAT mean,
                const RFLOAT stddev);

/**
 * This function normalizes the image according to the mean and stddev of the
 * background dust points are removed according to wDust and bDust.
 *
 * @param img   the image to be processed
 * @param wDust the factor of white dust
 * @param bDust the factor of black dust
 * @param r     the radius
 */
void normalise(Image& img,
               const RFLOAT wDust,
               const RFLOAT bDust,
               const RFLOAT r);

/**
 * This function extracts a sub-image from an image.
 *
 * @param dst  the destination image
 * @param src  the source image
 * @param xOff the column shift
 * @param yOff the row shift
 */
void extract(Image& dst,
             const Image& src,
             const int xOff,
             const int yOff);

/**
 * This function performs binning on an image.
 *
 * @param dst the destination image
 * @param src the source image
 * @param bf  the binning factor
 */
void binning(Image& dst,
             const Image& src,
             const int bf);

#endif // IMAGE_FUNCTIONS_H
