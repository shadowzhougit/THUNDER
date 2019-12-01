/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Volume.h"

Volume::Volume() : ImageBase(), _nCol(0), _nRow(0), _nSlc(0) {}

Volume::Volume(const long nCol,
               const long nRow,
               const long nSlc,
               const int space)
{
    alloc(nCol, nRow, nSlc, space);
}

Volume::~Volume() {}

void Volume::swap(Volume& that)
{
    ImageBase::swap(that);

    std::swap(_nCol, that._nCol);
    std::swap(_nRow, that._nRow);
    std::swap(_nSlc, that._nSlc);

    std::swap(_nColFT, that._nColFT);

    FOR_CELL_DIM_3
        std::swap(_box[k][j][i], that._box[k][j][i]);
}

Volume Volume::copyVolume() const
{
    Volume out;

    copyBase(out);

    out._nCol = _nCol;
    out._nRow = _nRow;
    out._nSlc = _nSlc;

    out._nColFT = _nColFT;

    FOR_CELL_DIM_3
        out._box[k][j][i] = _box[k][j][i];

    return out;
}

void Volume::alloc(int space)
{
    alloc(_nCol, _nRow, _nSlc, space);
}

void Volume::alloc(const long nCol,
                   const long nRow,
                   const long nSlc,
                   const int space)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;

    if (space == RL_SPACE)
    {
        clearRL();

        // _sizeRL = (size_t)nCol * (size_t)nRow * (size_t)nSlc;
        // _sizeFT = ((size_t)nCol / 2 + 1) * (size_t)nRow * (size_t)nSlc;

        _sizeRL = nCol * nRow * nSlc;
        _sizeFT = (nCol / 2 + 1) * nRow * nSlc;

#ifdef CXX11_PTR
        _dataRL.reset(new RFLOAT[_sizeRL]);
#endif

#ifdef FFTW_PTR
#ifdef FFTW_PTR_THREAD_SAFETY
        #pragma omp critical  (line86)
#endif
        _dataRL = (RFLOAT*)TSFFTW_malloc(_sizeRL * sizeof(RFLOAT));

        if (_dataRL == NULL)
        {
            REPORT_ERROR("FAIL TO ALLOCATE SPACE");

            abort();
        }
#endif
    }
    else if (space == FT_SPACE)
    {
        clearFT();

        // _sizeRL = (size_t)nCol * (size_t)nRow * (size_t)nSlc;
        // _sizeFT = (nCol / 2 + 1) * (size_t)nRow * (size_t)nSlc;

        _sizeRL = nCol * nRow * nSlc;
        _sizeFT = (nCol / 2 + 1) * nRow * nSlc;

#ifdef CXX11_PTR
        _dataFT.reset(new Complex[_sizeFT]);
#endif

#ifdef FFTW_PTR
#ifdef FFTW_PTR_THREAD_SAFETY
        #pragma omp critical  (line111)
#endif
        _dataFT = (Complex*)TSFFTW_malloc(_sizeFT * sizeof(Complex));

        if (_dataFT == NULL)
        {
            REPORT_ERROR("FAIL TO ALLOCATE SPACE");

            abort();
        }
#endif
    }

    initBox();
}

RFLOAT Volume::getRL(const long iCol,
                     const long iRow,
                     const long iSlc) const
{
    size_t index = iRL(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_RL(index);
#endif

    return _dataRL[index];
}

void Volume::setRL(const RFLOAT value,
                   const long iCol,
                   const long iRow,
                   const long iSlc)
{
    size_t index = iRL(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_RL(index);
#endif

    _dataRL[index] = value;
}

void Volume::addRL(const RFLOAT value,
                   const long iCol,
                   const long iRow,
                   const long iSlc)
{
    size_t index = iRL(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_RL(index);
#endif

    #pragma omp atomic
    _dataRL[index] += value;
}

Complex Volume::getFT(long iCol,
                      long iRow,
                      long iSlc) const
{
    bool conj;
    size_t index = iFT(conj, iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    return conj ? CONJUGATE(_dataFT[index]) : _dataFT[index];
}

Complex Volume::getFTHalf(const long iCol,
                          const long iRow,
                          const long iSlc) const
{
    size_t index = iFTHalf(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    return _dataFT[index];
}

void Volume::setFT(const Complex value,
                   long iCol,
                   long iRow,
                   long iSlc)
{
    bool conj;
    size_t index = iFT(conj, iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    _dataFT[index] = conj ? CONJUGATE(value) : value;
}

void Volume::setFTHalf(const Complex value,
                       const long iCol,
                       const long iRow,
                       const long iSlc)
{
    size_t index = iFTHalf(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    _dataFT[index] = value;
}

void Volume::addFT(const Complex value,
                   long iCol,
                   long iRow,
                   long iSlc)
{
    bool conj;
    size_t index = iFT(conj, iCol, iRow, iSlc);

    Complex val = conj ? CONJUGATE(value) : value;

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    #pragma omp atomic
    _dataFT[index].dat[0] += val.dat[0];
    #pragma omp atomic
    _dataFT[index].dat[1] += val.dat[1];
}

void Volume::addFTHalf(const Complex value,
                       const long iCol,
                       const long iRow,
                       const long iSlc)
{
    size_t index = iFTHalf(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    #pragma omp atomic
    _dataFT[index].dat[0] += value.dat[0];
    #pragma omp atomic
    _dataFT[index].dat[1] += value.dat[1];
}

void Volume::addFT(const RFLOAT value,
                   long iCol,
                   long iRow,
                   long iSlc)
{
    size_t index = iFT(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    #pragma omp atomic
    _dataFT[index].dat[0] += value;
}

void Volume::addFTHalf(const RFLOAT value,
                       const long iCol,
                       const long iRow,
                       const long iSlc)
{
    size_t index = iFTHalf(iCol, iRow, iSlc);

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
    BOUNDARY_CHECK_FT(index);
#endif

    #pragma omp atomic
    _dataFT[index].dat[0] += value;
}

RFLOAT Volume::getByInterpolationRL(const RFLOAT iCol,
                                    const RFLOAT iRow,
                                    const RFLOAT iSlc,
                                    const int interp) const
{
    if (interp == NEAREST_INTERP)
        return getRL(AROUND(iCol), AROUND(iRow), AROUND(iSlc));

    RFLOAT w[2][2][2];
    long x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};

    //WG_TRI_INTERP(w, x0, x, interp);
    WG_TRI_INTERP_LINEAR(w, x0, x);

    return getRL(w, x0);
}

Complex Volume::getByInterpolationFT(RFLOAT iCol,
                                     RFLOAT iRow,
                                     RFLOAT iSlc,
                                     const int interp) const
{
    bool conj = conjHalf(iCol, iRow, iSlc);

    if (interp == NEAREST_INTERP)
    {
        Complex result = getFTHalf(AROUND(iCol), AROUND(iRow), AROUND(iSlc));

        return conj ? CONJUGATE(result) : result;
    }

    RFLOAT w[2][2][2];
    long x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};

    //WG_TRI_INTERP(w, x0, x, interp);
    WG_TRI_INTERP_LINEAR(w, x0, x);

    Complex result = getFTHalf(w, x0);

    return conj ? CONJUGATE(result) : result;
}
//huabin
void Volume::addFT(const Complex value,
                   RFLOAT iCol,
                   RFLOAT iRow,
                   RFLOAT iSlc)
{
    bool conj = conjHalf(iCol, iRow, iSlc);

    RFLOAT w[2][2][2];
    long x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};

    //WG_TRI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_TRI_INTERP_LINEAR(w, x0, x);

    addFTHalf(conj ? CONJUGATE(value) : value,
              w,
              x0);
}

//huabin
void Volume::addFT(const RFLOAT value,
                   RFLOAT iCol,
                   RFLOAT iRow,
                   RFLOAT iSlc)
{
    conjHalf(iCol, iRow, iSlc);

    RFLOAT w[2][2][2];
    long x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};

    //WG_TRI_INTERP(w, x0, x, LINEAR_INTERP);
    WG_TRI_INTERP_LINEAR(w, x0, x);

    addFTHalf(value, w, x0);
}

void Volume::addFT(const Complex value,
                   const RFLOAT iCol,
                   const RFLOAT iRow,
                   const RFLOAT iSlc,
                   const RFLOAT a,
                   const RFLOAT alpha)
{
    VOLUME_SUB_SPHERE_FT(a)
    {
        RFLOAT r = NORM_3(iCol - i, iRow - j, iSlc - k);
        if (r < a) addFT(value * MKB_FT(r, a, alpha), i, j, k);
    }
}

void Volume::addFT(const RFLOAT value,
                   const RFLOAT iCol,
                   const RFLOAT iRow,
                   const RFLOAT iSlc,
                   const RFLOAT a,
                   const RFLOAT alpha)
{
    VOLUME_SUB_SPHERE_FT(a)
    {
        RFLOAT r = NORM_3(iCol - i, iRow - j, iSlc - k);
        if (r < a) addFT(value * MKB_FT(r, a, alpha), i, j, k);
    }
}

void Volume::addFT(const Complex value,
                   const RFLOAT iCol,
                   const RFLOAT iRow,
                   const RFLOAT iSlc,
                   const RFLOAT a,
                   const TabFunction& kernel)
{
    RFLOAT a2 = TSGSL_pow_2(a);

    VOLUME_SUB_SPHERE_FT(a)
    {
        RFLOAT r2 = QUAD_3(iCol - i, iRow - j, iSlc - k);
        if (r2 < a2) addFT(value * kernel(r2), i, j, k);
    }
}

void Volume::addFT(const RFLOAT value,
                   const RFLOAT iCol,
                   const RFLOAT iRow,
                   const RFLOAT iSlc,
                   const RFLOAT a,
                   const TabFunction& kernel)
{
    RFLOAT a2 = TSGSL_pow_2(a);

    VOLUME_SUB_SPHERE_FT(a)
    {
        RFLOAT r2 = QUAD_3(iCol - i, iRow - j, iSlc - k);
        if (r2 < a2) addFT(value * kernel(r2), i, j, k);
    }
}

void Volume::clear()
{
    ImageBase::clear();

    _nCol = 0;
    _nRow = 0;
    _nSlc = 0;

    _nColFT = 0;
}

void Volume::initBox()
{
    _nColFT = _nCol / 2 + 1;

    FOR_CELL_DIM_3
        _box[k][j][i] = k * _nColFT * _nRow
                      + j * _nColFT
                      + i;
}

void Volume::coordinatesInBoundaryRL(const long iCol,
                                     const long iRow,
                                     const long iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol >= _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
    {
        REPORT_ERROR("ACCESSING VALUE OUT OF BOUNDARY");
    }
}

void Volume::coordinatesInBoundaryFT(const long iCol,
                                     const long iRow,
                                     const long iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
    {
        REPORT_ERROR("ACCESSING VALUE OUT OF BOUNDARY");
    }
}

RFLOAT Volume::getRL(const RFLOAT w[2][2][2],
                     const long x0[3]) const
{
    RFLOAT result = 0;
    FOR_CELL_DIM_3 result += getRL(x0[0] + i, x0[1] + j, x0[2] + k)
                           * w[k][j][i];
    return result;
}

Complex Volume::getFTHalf(const RFLOAT w[2][2][2],
                          const long x0[3]) const
{
    Complex result = COMPLEX(0, 0);

    if ((x0[1] != -1) &&
        (x0[2] != -1))
    {
#ifndef IMG_VOL_BOX_UNFOLD

        size_t index0 = iFTHalf(x0[0], x0[1], x0[2]);

        for (long i = 0; i < 8; i++)
        {
            size_t index = index0 + ((size_t*)_box)[i];

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(index);
#endif

            result += _dataFT[index] * ((RFLOAT*)w)[i];
        }

#else

        size_t index0 = (x0[2] >= 0 ? x0[2] : x0[2] + _nSlc) * _nColFT * _nRow + (x0[1] >= 0 ? x0[1] : x0[1] + _nRow) * _nColFT + x0[0];

        size_t index;

        index = index0 + _box[0][0][0];
        result.dat[0] += _dataFT[index].dat[0] * w[0][0][0];
        result.dat[1] += _dataFT[index].dat[1] * w[0][0][0];

        index = index0 + _box[0][0][1];
        result.dat[0] += _dataFT[index].dat[0] * w[0][0][1];
        result.dat[1] += _dataFT[index].dat[1] * w[0][0][1];

        index = index0 + _box[0][1][0];
        result.dat[0] += _dataFT[index].dat[0] * w[0][1][0];
        result.dat[1] += _dataFT[index].dat[1] * w[0][1][0];

        index = index0 + _box[0][1][1];
        result.dat[0] += _dataFT[index].dat[0] * w[0][1][1];
        result.dat[1] += _dataFT[index].dat[1] * w[0][1][1];

        index = index0 + _box[1][0][0];
        result.dat[0] += _dataFT[index].dat[0] * w[1][0][0];
        result.dat[1] += _dataFT[index].dat[1] * w[1][0][0];
        
        index = index0 + _box[1][0][1];
        result.dat[0] += _dataFT[index].dat[0] * w[1][0][1];
        result.dat[1] += _dataFT[index].dat[1] * w[1][0][1];

        index = index0 + _box[1][1][0];
        result.dat[0] += _dataFT[index].dat[0] * w[1][1][0];
        result.dat[1] += _dataFT[index].dat[1] * w[1][1][0];

        index = index0 + _box[1][1][1];
        result.dat[0] += _dataFT[index].dat[0] * w[1][1][1];
        result.dat[1] += _dataFT[index].dat[1] * w[1][1][1];

#endif
    }
    else
    {
        FOR_CELL_DIM_3 result += getFTHalf(x0[0] + i,
                                           x0[1] + j,
                                           x0[2] + k)
                               * w[k][j][i];
    }

    return result;
}

void Volume::addFTHalf(const Complex value,
                       const RFLOAT w[2][2][2],
                       const long x0[3])
{
    if ((x0[1] != -1) &&
        (x0[2] != -1))
    {
#ifndef IMG_VOL_BOX_UNFOLD

        size_t index0 = iFTHalf(x0[0], x0[1], x0[2]);

        for (long i = 0; i < 8; i++)
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

        size_t index0 = (x0[2] >= 0 ? x0[2] : x0[2] + _nSlc) * _nColFT * _nRow
                      + (x0[1] >= 0 ? x0[1] : x0[1] + _nRow) * _nColFT
                      + x0[0];

        size_t index;

        index = index0 + _box[0][0][0];
        #pragma omp atomic
        _dataFT[index].dat[0] += value.dat[0] * w[0][0][0];
        #pragma omp atomic
        _dataFT[index].dat[1] += value.dat[1] * w[0][0][0];

        index = index0 + _box[0][0][1];
        #pragma omp atomic
        _dataFT[index].dat[0] += value.dat[0] * w[0][0][1];
        #pragma omp atomic
        _dataFT[index].dat[1] += value.dat[1] * w[0][0][1];

        index = index0 + _box[0][1][0];
        #pragma omp atomic
        _dataFT[index].dat[0] += value.dat[0] * w[0][1][0];
        #pragma omp atomic
        _dataFT[index].dat[1] += value.dat[1] * w[0][1][0];

        index = index0 + _box[0][1][1];
        #pragma omp atomic
        _dataFT[index].dat[0] += value.dat[0] * w[0][1][1];
        #pragma omp atomic
        _dataFT[index].dat[1] += value.dat[1] * w[0][1][1];

        index = index0 + _box[1][0][0];
        #pragma omp atomic
        _dataFT[index].dat[0] += value.dat[0] * w[1][0][0];
        #pragma omp atomic
        _dataFT[index].dat[1] += value.dat[1] * w[1][0][0];

        index = index0 + _box[1][0][1];
        #pragma omp atomic
        _dataFT[index].dat[0] += value.dat[0] * w[1][0][1];
        #pragma omp atomic
        _dataFT[index].dat[1] += value.dat[1] * w[1][0][1];

        index = index0 + _box[1][1][0];
        #pragma omp atomic
        _dataFT[index].dat[0] += value.dat[0] * w[1][1][0];
        #pragma omp atomic
        _dataFT[index].dat[1] += value.dat[1] * w[1][1][0];

        index = index0 + _box[1][1][1];
        #pragma omp atomic
        _dataFT[index].dat[0] += value.dat[0] * w[1][1][1];
        #pragma omp atomic
        _dataFT[index].dat[1] += value.dat[1] * w[1][1][1];

#endif
    }
    else
    {
        FOR_CELL_DIM_3 addFTHalf(value * w[k][j][i],
                                 x0[0] + i,
                                 x0[1] + j,
                                 x0[2] + k);
    }
}

void Volume::addFTHalf(const RFLOAT value,
                       const RFLOAT w[2][2][2],
                       const long x0[3])
{
    if ((x0[1] != -1) &&
        (x0[2] != -1))
    {
#ifndef IMG_VOL_BOX_UNFOLD

        size_t index0 = iFTHalf(x0[0], x0[1], x0[2]);

        for (long i = 0; i < 8; i++)
        {
            size_t index = index0 + ((size_t*)_box)[i];

#ifndef IMG_VOL_BOUNDARY_NO_CHECK
            BOUNDARY_CHECK_FT(index);
#endif

            #pragma omp atomic
            _dataFT[index].dat[0] += value * ((RFLOAT*)w)[i];
        }

#else

        size_t index0 = (x0[2] >= 0 ? x0[2] : x0[2] + _nSlc) * _nColFT * _nRow
                      + (x0[1] >= 0 ? x0[1] : x0[1] + _nRow) * _nColFT
                      + x0[0];

        #pragma omp atomic
        _dataFT[index0+_box[0][0][0]].dat[0] += value * w[0][0][0];
        #pragma omp atomic
        _dataFT[index0+_box[0][0][1]].dat[0] += value * w[0][0][1];
        #pragma omp atomic
        _dataFT[index0+_box[0][1][0]].dat[0] += value * w[0][1][0];
        #pragma omp atomic
        _dataFT[index0+_box[0][1][1]].dat[0] += value * w[0][1][1];
        #pragma omp atomic
        _dataFT[index0+_box[1][0][0]].dat[0] += value * w[1][0][0];
        #pragma omp atomic
        _dataFT[index0+_box[1][0][1]].dat[0] += value * w[1][0][1];
        #pragma omp atomic
        _dataFT[index0+_box[1][1][0]].dat[0] += value * w[1][1][0];
        #pragma omp atomic
        _dataFT[index0+_box[1][1][1]].dat[0] += value * w[1][1][1];

#endif
    }
    else
    {
        FOR_CELL_DIM_3 addFTHalf(value * w[k][j][i],
                                 x0[0] + i,
                                 x0[1] + j,
                                 x0[2] + k);
    }
}
