/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Image.h"

Image::Image() : _nCol(0), _nRow(0) {}

Image::Image(const long nCol,
             const long nRow,
             const int space)
{
    alloc(nCol, nRow, space);
}

Image::~Image() {}

void Image::swap(Image& that)
{
    ImageBase::swap(that);

    std::swap(_nCol, that._nCol);
    std::swap(_nRow, that._nRow);

    std::swap(_nColFT, that._nColFT);

    FOR_CELL_DIM_2
        std::swap(_box[j][i], that._box[j][i]);
}

Image Image::copyImage() const
{
    Image out;
    
    copyBase(out);

    out._nCol = _nCol;
    out._nRow = _nRow;

    out._nColFT = _nColFT;

    FOR_CELL_DIM_2
        out._box[j][i] = _box[j][i];

    return out;
}

void Image::alloc(const int space)
{
    alloc(_nCol, _nRow, space);
}

void Image::alloc(const long nCol,
                  const long nRow,
                  const int space)
{
    _nCol = nCol;
    _nRow = nRow;

    if (space == RL_SPACE)
    {
        clearRL();

        _sizeRL = nCol * nRow;
        _sizeFT = (nCol / 2 + 1) * nRow;

#ifdef CXX11_PTR
        _dataRL.reset(new RFLOAT[_sizeRL]);
#endif

#ifdef FFTW_PTR
#ifdef FFTW_PTR_THREAD_SAFETY
        #pragma omp critical  (line81)
#endif
        _dataRL = (RFLOAT*)TSFFTW_malloc(_sizeRL * sizeof(RFLOAT));
#endif
    }
    else if (space == FT_SPACE)
    {
        clearFT();

        _sizeRL = nCol * nRow;
        _sizeFT = (nCol / 2 + 1) * nRow;

#ifdef CXX11_PTR
        _dataFT.reset(new Complex[_sizeFT]);
#endif

#ifdef FFTW_PTR
#ifdef FFTW_PTR_THREAD_SAFETY
        #pragma omp critical  (line99)
#endif
        _dataFT = (Complex*)TSFFTW_malloc(_sizeFT * sizeof(Complex));
#endif
    }

    initBox();
}

void Image::saveRLToBMP(const char* filename) const
{
    // size_t nRowBMP = _nRow / 4 * 4;
    // size_t nColBMP = _nCol / 4 * 4;

    long nRowBMP = _nRow / 4 * 4;
    long nColBMP = _nCol / 4 * 4;

    float* image = new float[nRowBMP * nColBMP];

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Calculating Values in RL_BMP";
#endif

    for (long i = -nRowBMP / 2; i < nRowBMP / 2; i++)
        for (long j = -nColBMP / 2; j < nColBMP / 2; j++)
            image[(i + nRowBMP / 2)
                * nColBMP
                + (j + nColBMP / 2)] = _dataRL[(i >= 0 ? i : i + _nRow)
                                             * _nCol
                                             + (j >= 0 ? j : j + _nCol)];

    BMP bmp;

    if (bmp.open(filename, "wb") == 0)
        REPORT_ERROR("FAILING TO OPEN BITCAMP FILE");

    if (bmp.createBMP(image, nColBMP, nRowBMP) == false)
        REPORT_ERROR("FAILING TO CREATE BMP FILE");

    bmp.close();

    delete[] image;
}

void Image::saveFTToBMP(const char* filename, RFLOAT c) const
{
    // size_t nRowBMP = _nRow / 4 * 4;
    // size_t nColBMP = _nCol / 4 * 4;

    long nRowBMP = _nRow / 4 * 4;
    long nColBMP = _nCol / 4 * 4;

    float* image = new float[nRowBMP * nColBMP];

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Calculating Values in FT_BMP";
#endif

    for (long i = 0; i < nRowBMP; i++)
        for (long j = 0; j <= nColBMP / 2; j++)
        {
            RFLOAT value = TSGSL_complex_abs2(_dataFT[(_nCol / 2 + 1) * i + j]);
            value = log(1 + value * c);

            long iImage = (i + nRowBMP / 2) % nRowBMP;
            long jImage = (j + nColBMP / 2) % nColBMP;
            image[nColBMP * iImage + jImage] = value;
        }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Performing Hermite Symmetry";
#endif

    for (long i = 1; i < nRowBMP; i++)
        for (long j = 1; j < nColBMP / 2; j++)
        {
            size_t iDst = i * nColBMP + j;
            size_t iSrc = (nRowBMP - i + 1) * nColBMP - j;
            image[iDst] = image[iSrc];
        }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Fixing Up the Missing Part";
#endif

    for (long j = 1; j < nColBMP / 2; j++)
    {
        long iDst = j;
        long iSrc = nColBMP - j;
        image[iDst] = image[iSrc];
    }

    BMP bmp;

    if (bmp.open(filename, "wb") == 0)
        REPORT_ERROR("FAILING TO OPEN BITCAMP FILE");
    if (bmp.createBMP(image, nColBMP, nRowBMP) == false)
        REPORT_ERROR("FAILING TO CREATE BMP FILE");

    bmp.close();

    delete[] image;
}

RFLOAT Image::getRL(const long iCol,
                    const long iRow) const
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryRL(iCol, iRow);
#endif

    return _dataRL[iRL(iCol, iRow)];
}

void Image::setRL(const RFLOAT value,
                  const long iCol,
                  const long iRow)
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryRL(iCol, iRow);
#endif

    _dataRL[iRL(iCol, iRow)] = value;
}

Complex Image::getFT(long iCol,
                     long iRow) const
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(iCol, iRow);
#endif
    
    bool conj;
    long index = iFT(conj, iCol, iRow);

    return conj ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}

Complex Image::getFTHalf(const long iCol,
                         const long iRow) const
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(iCol, iRow);
#endif

    return _dataFT[iFTHalf(iCol, iRow)];
}

void Image::setFT(const Complex value,
                  long iCol,
                  long iRow)
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(iCol, iRow);
#endif

    bool conj;
    size_t index = iFT(conj, iCol, iRow);

    _dataFT[index] = conj ? CONJUGATE(value) : value;
}

void Image::setFTHalf(const Complex value,
                      const long iCol,
                      const long iRow)
{
#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(iCol, iRow);
#endif

    _dataFT[iFTHalf(iCol, iRow)] = value;
}

void Image::addFT(const Complex value,
                  long iCol,
                  long iRow)
{
    bool conj;
    size_t index = iFT(conj, iCol, iRow);

    Complex val = conj ? CONJUGATE(value) : value;

    #pragma omp atomic
    _dataFT[index].dat[0] += val.dat[0];
    #pragma omp atomic
    _dataFT[index].dat[1] += val.dat[1];
}

void Image::addFTHalf(const Complex value,
                      const long iCol,
                      const long iRow)
{
    #pragma omp atomic
    _dataFT[iFTHalf(iCol, iRow)].dat[0] += value.dat[0];
    #pragma omp atomic
    _dataFT[iFTHalf(iCol, iRow)].dat[1] += value.dat[1];
}

void Image::addFT(const RFLOAT value,
                  long iCol,
                  long iRow)
{
    #pragma omp atomic
    _dataFT[iFT(iCol, iRow)].dat[0] += value;
}

void Image::addFTHalf(const RFLOAT value,
                      const long iCol,
                      const long iRow)
{
    #pragma omp atomic
    _dataFT[iFTHalf(iCol, iRow)].dat[0] += value;
}

/*
RFLOAT Image::getBiLinearRL(const RFLOAT iCol,
                            const RFLOAT iRow) const
{
    RFLOAT w[2][2];
    int x0[2];
    RFLOAT x[2] = {iCol, iRow};
    //WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_BI_INTERP_LINEAR(w, x0, x);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryRL(x0[0], x0[1]);
    coordinatesInBoundaryRL(x0[0] + 1, x0[1] + 1);
#endif

    RFLOAT result = 0;
    FOR_CELL_DIM_3 result += w[j][i] * getRL(x0[0] + i, x0[1] + j);
    return result;
}

Complex Image::getBiLinearFT(const RFLOAT iCol,
                             const RFLOAT iRow) const
{
    RFLOAT w[2][2];
    int x0[2];
    RFLOAT x[2] = {iCol, iRow};
    //WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_BI_INTERP_LINEAR(w, x0, x);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    coordinatesInBoundaryFT(x0[0], x0[1]);
    coordinatesInBoundaryFT(x0[0] + 1, x0[1] + 1);
#endif

    Complex result = COMPLEX(0, 0);
    FOR_CELL_DIM_2 result += w[j][i] * getFT(x0[0] + i , x0[1] + j);
    return result;
}
*/

Complex Image::getByInterpolationFT(RFLOAT iCol,
                                    RFLOAT iRow,
                                    const int interp) const
{
    bool conj = conjHalf(iCol, iRow);

    if (interp == NEAREST_INTERP)
    {
        Complex result = getFTHalf(AROUND(iCol), AROUND(iRow));

        return conj ? CONJUGATE(result) : result;
    }

    RFLOAT w[2][2];
    long x0[2];
    RFLOAT x[2] = {iCol, iRow};

    //WG_BI_INTERP(w, x0, x, interp);
    WG_BI_INTERP_LINEAR(w, x0, x);

    Complex result = getFTHalf(w, x0);

    return conj ? CONJUGATE(result) : result;
}

void Image::addFT(const Complex value,
                  RFLOAT iCol,
                  RFLOAT iRow)
{
    bool conj = conjHalf(iCol, iRow);

    RFLOAT w[2][2];
    long x0[2];
    RFLOAT x[2] = {iCol, iRow};

    //WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_BI_INTERP_LINEAR(w, x0, x);

    addFTHalf(conj ? CONJUGATE(value) : value,
              w,
              x0);
}

void Image::addFT(const RFLOAT value,
                  RFLOAT iCol,
                  RFLOAT iRow)
{
    conjHalf(iCol, iRow);

    RFLOAT w[2][2];
    long x0[2];
    RFLOAT x[2] = {iCol, iRow};

    //WG_BI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_BI_INTERP_LINEAR(w, x0, x);

    addFTHalf(value, w, x0);
}

void Image::clear()
{
    ImageBase::clear();

    _nCol = 0;
    _nRow = 0;

    _nColFT = 0;
}

void Image::initBox()
{
    _nColFT = _nCol / 2 + 1;

    FOR_CELL_DIM_2
        _box[j][i] = j * _nColFT + i;
}

void Image::coordinatesInBoundaryRL(const long iCol,
                                    const long iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol >= _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
    {
        REPORT_ERROR("ACCESSING VALUE OUT OF BOUNDARY");
        abort();
    }
}

void Image::coordinatesInBoundaryFT(const long iCol,
                                    const long iRow) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2))
    {
        REPORT_ERROR("ACCESSING VALUE OUT OF BOUNDARY");
        abort();
    }
}

Complex Image::getFTHalf(const RFLOAT w[2][2],
                         const long x0[2]) const
{
    Complex result = COMPLEX(0, 0);

    if (x0[1] != -1)
    {
        //Following codes are commented out by huabin
//        size_t index0 = iFTHalf(x0[0], x0[1]);

//        for (int i = 0; i < 4; i++)
//        {
//            size_t index = index0 + ((size_t*)_box)[i];

//#ifndef IMG_VOL_BOUNDARY_NO_CHECK
//            BOUNDARY_CHECK_FT(index);
//#endif

//            result += _dataFT[index] * ((RFLOAT*)w)[i];
//        }


//Follwing codes are added by huabin
        size_t index0 = (x0[1] >= 0 ? x0[1] : x0[1] + _nRow) * _nColFT + x0[0];
        size_t index = index0 + _box[0][0];
        result.dat[0] += _dataFT[index].dat[0] * w[0][0];
        result.dat[1] += _dataFT[index].dat[1] * w[0][0];

        index = index0 + _box[0][1];
        result.dat[0] += _dataFT[index].dat[0] * w[0][1];
        result.dat[1] += _dataFT[index].dat[1] * w[0][1];

        index = index0 + _box[1][0];
        result.dat[0] += _dataFT[index].dat[0] * w[1][0];
        result.dat[1] += _dataFT[index].dat[1] * w[1][0];

        index = index0 + _box[1][1];
        result.dat[0] += _dataFT[index].dat[0] * w[1][1];
        result.dat[1] += _dataFT[index].dat[1] * w[1][1];




    }
    else
    {
        FOR_CELL_DIM_2 result += getFTHalf(x0[0] + i,
                                           x0[1] + j)
                               * w[j][i];
    }
    return result;
}

void Image::addFTHalf(const Complex value,
                      const RFLOAT w[2][2],
                      const long x0[2])
{
    if (x0[1] != -1)
    {
#ifdef IMG_VOL_BOX_UNFOLD

        size_t index0 = iFTHalf(x0[0], x0[1]);

        for (long i = 0; i < 4; i++)
        {
            size_t index = index0 + ((size_t*)_box)[i];

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(index);
#endif

            #pragma omp atomic
            _dataFT[index].dat[0] += value.dat[0] * ((RFLOAT*)w)[i];
            #pragma omp atomic
            _dataFT[index].dat[1] += value.dat[1] * ((RFLOAT*)w)[i];
        }

#else

        size_t index0 = (x0[1] >= 0 ? x0[1] : x0[1] + _nRow) * _nColFT + x0[0];

        #pragma omp atomic
        _dataFT[index0 + _box[0][0]].dat[0] += value.dat[0] * w[0][0];
        #pragma omp atomic
        _dataFT[index0 + _box[0][0]].dat[1] += value.dat[1] * w[0][0];

        #pragma omp atomic
        _dataFT[index0 + _box[0][1]].dat[0] += value.dat[0] * w[0][1];
        #pragma omp atomic
        _dataFT[index0 + _box[0][1]].dat[1] += value.dat[1] * w[0][1];

        #pragma omp atomic
        _dataFT[index0 + _box[1][0]].dat[0] += value.dat[0] * w[1][0];
        #pragma omp atomic
        _dataFT[index0 + _box[1][0]].dat[1] += value.dat[1] * w[1][0];

        #pragma omp atomic
        _dataFT[index0 + _box[1][1]].dat[0] += value.dat[0] * w[1][1];
        #pragma omp atomic
        _dataFT[index0 + _box[1][1]].dat[1] += value.dat[1] * w[1][1];

#endif
    }
    else
    {
        FOR_CELL_DIM_2 addFTHalf(value * w[j][i],
                                 x0[0] + i,
                                 x0[1] + j);
    }
}

void Image::addFTHalf(const RFLOAT value,
                      const RFLOAT w[2][2],
                      const long x0[2])
{
    if (x0[1] != -1)
    {
#ifdef IMG_VOL_BOX_UNFOLD
        size_t index0 = iFTHalf(x0[0], x0[1]);

        for (long i = 0; i < 4; i++)
        {
            size_t index = index0 + ((size_t*)_box)[i];

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(index);
#endif

            #pragma omp atomic
            _dataFT[index].dat[0] += value * ((RFLOAT*)w)[i];
        }

#else

        size_t index0 = (x0[1] >= 0 ? x0[1] : x0[1] + _nRow) * _nColFT + x0[0];

        #pragma omp atomic
        _dataFT[index0 + _box[0][0]].dat[0] += value * w[0][0];
        #pragma omp atomic
        _dataFT[index0 + _box[0][1]].dat[0] += value * w[0][1];
        #pragma omp atomic
        _dataFT[index0 + _box[1][0]].dat[0] += value * w[1][0];
        #pragma omp atomic
        _dataFT[index0 + _box[1][1]].dat[0] += value * w[1][1];

#endif
    }
    else
    {
        FOR_CELL_DIM_2 addFTHalf(value * w[j][i],
                                 x0[0] + i,
                                 x0[1] + j);
    }
}
