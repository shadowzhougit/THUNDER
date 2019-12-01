/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Projector.h"

Projector::Projector()
{
    _mode = MODE_3D;

    _maxRadius = -1;

    _interp = LINEAR_INTERP;

    _pf = 2;
}

Projector::~Projector() {}

void Projector::swap(Projector& that)
{
    std::swap(_mode, that._mode);
    std::swap(_maxRadius, that._maxRadius);
    std::swap(_interp, that._interp);
    std::swap(_pf, that._pf);
    
    _projectee2D.swap(that._projectee2D);
    _projectee3D.swap(that._projectee3D);
}

bool Projector::isEmpty2D() const
{
    return _projectee2D.isEmptyFT();
}

bool Projector::isEmpty3D() const
{
    return _projectee3D.isEmptyFT();
}

int Projector::mode() const
{
    return _mode;
}

void Projector::setMode(const int mode)
{
    _mode = mode;
}

int Projector::maxRadius() const
{
    return _maxRadius;
}

void Projector::setMaxRadius(const int maxRadius)
{
    _maxRadius = maxRadius;
}

int Projector::interp() const
{
    return _interp;
}

void Projector::setInterp(const int interp)
{
    _interp = interp;
}

int Projector::pf() const
{
    return _pf;
}

void Projector::setPf(const int pf)
{
    _pf = pf;
}

const Image& Projector::projectee2D() const
{
    return _projectee2D;
}

const Volume& Projector::projectee3D() const
{
    return _projectee3D;
}

void Projector::setProjectee(Image src, const unsigned int nThread)
{
    FFT fft;
    fft.bw(src, nThread);

    IMG_PAD_RL(_projectee2D, src, _pf, nThread);

    if (_projectee2D.isEmptyRL()) REPORT_ERROR("REAL SPACE EMPTY");

    _maxRadius = floor(MIN(_projectee2D.nColRL(),
                           _projectee2D.nRowRL()) / _pf / 2 - 1);

#ifdef VERBOSE_LEVEL_3
    CLOG(INFO, "LOGGER_SYS") << "Performing Grid Correction";
#endif

#ifdef PROJECTOR_CORRECT_CONVOLUTION_KERNEL

    gridCorrection(nThread);

#endif

    fft.fw(_projectee2D, nThread);
    _projectee2D.clearRL();
}

void Projector::setProjectee(Volume src, const unsigned int nThread)
{
    FFT fft;
    fft.bw(src, nThread);

    VOL_PAD_RL(_projectee3D, src, _pf, nThread);

    if (_projectee3D.isEmptyRL()) REPORT_ERROR("RL SPACE EMPTY");

    _maxRadius = floor(MIN_3(_projectee3D.nColRL(),
                             _projectee3D.nRowRL(),
                             _projectee3D.nSlcRL()) / _pf / 2 - 1);

#ifdef VERBOSE_LEVEL_3
    CLOG(INFO, "LOGGER_SYS") << "Performing Grid Correction";
#endif

#ifdef PROJECTOR_CORRECT_CONVOLUTION_KERNEL

    gridCorrection(nThread);

#endif

    fft.fw(_projectee3D, nThread);
    _projectee3D.clearRL();
}

/*void Projector::project(Image& dst,
                        const dmat22& mat) const
{
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
        {
            dvec2 newCor((double)(i * _pf), (double)(j * _pf));
            dvec2 oldCor = mat * newCor;

            dst.setFT(_projectee2D.getByInterpolationFT(oldCor(0),
                                                        oldCor(1),
                                                        _interp),
                      i,
                      j);
        }
}

void Projector::project(Image& dst,
                        const dmat33& mat) const
{
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
        {
            dvec3 newCor((double)(i * _pf), (double)(j * _pf), 0);
            dvec3 oldCor = mat * newCor;

            dst.setFT(_projectee3D.getByInterpolationFT(oldCor(0),
                                                        oldCor(1),
                                                        oldCor(2),
                                                        _interp),
                      i,
                      j);
        }
}

void Projector::project(Image& dst,
                        const dmat22& mat,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl) const
{
    for (int i = 0; i < nPxl; i++)
    {
        dvec2 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf));
        dvec2 oldCor = mat * newCor;

        dst[iPxl[i]] = _projectee2D.getByInterpolationFT(oldCor(0),
                                                         oldCor(1),
                                                         _interp);
    }
}

void Projector::project(Image& dst,
                        const dmat33& mat,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl) const
{
    for (int i = 0; i < nPxl; i++)
    {
        dvec3 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf), 0);
        dvec3 oldCor = mat * newCor;

        dst[iPxl[i]] = _projectee3D.getByInterpolationFT(oldCor(0),
                                                         oldCor(1),
                                                         oldCor(2),
                                                         _interp);
    }
}

void Projector::project(Complex* dst,
                        const dmat22& mat,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl) const
{
    for (int i = 0; i < nPxl; i++)
    {
        dvec2 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf));
        dvec2 oldCor = mat * newCor;

        dst[i] = _projectee2D.getByInterpolationFT(oldCor(0),
                                                   oldCor(1),
                                                   _interp);
    }
}

void Projector::project(Complex* dst,
                        const dmat33& mat,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl) const
{
    for (int i = 0; i < nPxl; i++)
    {
        dvec3 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf), 0);
        dvec3 oldCor = mat * newCor;

        dst[i] = _projectee3D.getByInterpolationFT(oldCor(0),
                                                   oldCor(1),
                                                   oldCor(2),
                                                   _interp);
    }
}*/

void Projector::project(Image& dst,
                        const dmat22& mat,
                        const unsigned int nThread) const
{
    #pragma omp parallel for schedule(dynamic) num_threads(nThread)
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
        {
            dvec2 newCor((double)(i * _pf), (double)(j * _pf));
            dvec2 oldCor = mat * newCor;

            dst.setFT(_projectee2D.getByInterpolationFT(oldCor(0),
                                                        oldCor(1),
                                                        _interp),
                      i,
                      j);
        }
}

void Projector::project(Image& dst,
                        const dmat33& mat,
                        const unsigned int nThread) const
{
    #pragma omp parallel for schedule(dynamic) num_threads(nThread)
    IMAGE_FOR_PIXEL_R_FT(_maxRadius)
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
        {
            dvec3 newCor((double)(i * _pf), (double)(j * _pf), 0);
            dvec3 oldCor = mat * newCor;

            dst.setFT(_projectee3D.getByInterpolationFT(oldCor(0),
                                                        oldCor(1),
                                                        oldCor(2),
                                                        _interp),
                      i,
                      j);
        }
}

void Projector::project(Image& dst,
                        const dmat22& mat,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl,
                        const unsigned int nThread) const
{
    #pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < nPxl; i++)
    {
        dvec2 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf));
        dvec2 oldCor = mat * newCor;

        dst[iPxl[i]] = _projectee2D.getByInterpolationFT(oldCor(0),
                                                         oldCor(1),
                                                         _interp);
    }
}

void Projector::project(Image& dst,
                        const dmat33& mat,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl,
                        const unsigned int nThread) const
{
    #pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < nPxl; i++)
    {
        dvec3 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf), 0);
        dvec3 oldCor = mat * newCor;

        dst[iPxl[i]] = _projectee3D.getByInterpolationFT(oldCor(0),
                                                         oldCor(1),
                                                         oldCor(2),
                                                         _interp);
    }
}

void Projector::project(Complex* dst,
                        const dmat22& mat,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl,
                        const unsigned int nThread) const
{
    #pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < nPxl; i++)
    {
        dvec2 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf));
        dvec2 oldCor = mat * newCor;

        dst[i] = _projectee2D.getByInterpolationFT(oldCor(0),
                                                   oldCor(1),
                                                   _interp);
    }
}

void Projector::project(Complex* dst,
                        const dmat33& mat,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl,
                        const unsigned int nThread) const
{
    #pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < nPxl; i++)
    {
        dvec3 newCor((double)(iCol[i] * _pf), (double)(iRow[i] * _pf), 0);
        dvec3 oldCor = mat * newCor;

        dst[i] = _projectee3D.getByInterpolationFT(oldCor(0),
                                                   oldCor(1),
                                                   oldCor(2),
                                                   _interp);
    }
}

void Projector::project(RFLOAT* dstR,
                        RFLOAT* dstI,
                        const dmat22& mat,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl,
                        const unsigned int nThread) const
{
    Complex* dst = new Complex[nPxl];

    project(dst, mat, iCol, iRow, nPxl, nThread);

    for (int i = 0; i < nPxl; i++)
    {
        dstR[i] = dst[i].dat[0];
        dstI[i] = dst[i].dat[1];
    }
}

void Projector::project(RFLOAT* dstR,
                        RFLOAT* dstI,
                        const dmat33& mat,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl,
                        const unsigned int nThread) const
{
    Complex* dst = new Complex[nPxl];

    project(dst, mat, iCol, iRow, nPxl, nThread);

    for (int i = 0; i < nPxl; i++)
    {
        dstR[i] = dst[i].dat[0];
        dstI[i] = dst[i].dat[1];
    }
}

//void Projector::project(Image& dst,
//                        const dmat22& rot,
//                        const dvec2& t) const
//{
//    project(dst, rot);
//    translate(dst, dst, _maxRadius, t(0), t(1));
//}
//
//void Projector::project(Image& dst,
//                        const dmat33& rot,
//                        const dvec2& t) const
//{
//    project(dst, rot);
//    translate(dst, dst, _maxRadius, t(0), t(1));
//}
//
//void Projector::project(Image& dst,
//                        const dmat22& rot,
//                        const dvec2& t,
//                        const int* iCol,
//                        const int* iRow,
//                        const int* iPxl,
//                        const int nPxl) const
//{
//    project(dst, rot, iCol, iRow, iPxl, nPxl);
//
//    translate(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl);
//}
//
//void Projector::project(Image& dst,
//                        const dmat33& rot,
//                        const dvec2& t,
//                        const int* iCol,
//                        const int* iRow,
//                        const int* iPxl,
//                        const int nPxl) const
//{
//    project(dst, rot, iCol, iRow, iPxl, nPxl);
//
//    translate(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl);
//}
//
//void Projector::project(Complex* dst,
//                        const dmat22& rot,
//                        const dvec2& t,
//                        const int nCol,
//                        const int nRow,
//                        const int* iCol,
//                        const int* iRow,
//                        const int nPxl) const
//{
//    project(dst, rot, iCol, iRow, nPxl);
//
//    translate(dst, dst, t(0), t(1), nCol, nRow, iCol, iRow, nPxl);
//}
//
//void Projector::project(Complex* dst,
//                        const dmat33& rot,
//                        const dvec2& t,
//                        const int nCol,
//                        const int nRow,
//                        const int* iCol,
//                        const int* iRow,
//                        const int nPxl) const
//{
//    project(dst, rot, iCol, iRow, nPxl);
//
//    translate(dst, dst, t(0), t(1), nCol, nRow, iCol, iRow, nPxl);
//}

void Projector::project(Image& dst,
                        const dmat22& rot,
                        const dvec2& t,
                        const unsigned int nThread) const
{
    project(dst, rot, nThread);

    translate(dst, dst, _maxRadius, t(0), t(1), nThread);
}

void Projector::project(Image& dst,
                        const dmat33& rot,
                        const dvec2& t,
                        const unsigned int nThread) const
{
    project(dst, rot, nThread);

    translate(dst, dst, _maxRadius, t(0), t(1), nThread);
}

void Projector::project(Image& dst,
                        const dmat22& rot,
                        const dvec2& t,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl,
                        const unsigned int nThread) const
{
    project(dst, rot, iCol, iRow, iPxl, nPxl, nThread);

    translate(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl, nThread);
}

void Projector::project(Image& dst,
                        const dmat33& rot,
                        const dvec2& t,
                        const int* iCol,
                        const int* iRow,
                        const int* iPxl,
                        const int nPxl,
                        const unsigned int nThread) const
{
    project(dst, rot, iCol, iRow, iPxl, nPxl, nThread);

    translate(dst, dst, t(0), t(1), iCol, iRow, iPxl, nPxl, nThread);
}

void Projector::project(Complex* dst,
                        const dmat22& rot,
                        const dvec2& t,
                        const int nCol,
                        const int nRow,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl,
                        const unsigned int nThread) const
{
    project(dst, rot, iCol, iRow, nPxl, nThread);

    translate(dst, dst, t(0), t(1), nCol, nRow, iCol, iRow, nPxl, nThread);
}

void Projector::project(Complex* dst,
                        const dmat33& rot,
                        const dvec2& t,
                        const int nCol,
                        const int nRow,
                        const int* iCol,
                        const int* iRow,
                        const int nPxl,
                        const unsigned int nThread) const
{
    project(dst, rot, iCol, iRow, nPxl, nThread);

    translate(dst, dst, t(0), t(1), nCol, nRow, iCol, iRow, nPxl, nThread);
}

void Projector::gridCorrection(const unsigned int nThread)
{
        if (_mode == MODE_2D)
        {
#ifdef VERBOSE_LEVEL_3
            CLOG(INFO, "LOGGER_SYS") << "Inverse Fourier Transform in Grid Correction";
#endif

#ifdef PROJECTOR_REMOVE_NEG
            #pragma omp parallel for num_threads(nThread)
            REMOVE_NEG(_projectee2D);
#endif

        if (_interp == LINEAR_INTERP)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nThread)
            IMAGE_FOR_EACH_PIXEL_RL(_projectee2D)
                _projectee2D.setRL(_projectee2D.getRL(i, j)
                                 / TIK_RL(NORM(i, j)
                                        / (_pf * _projectee2D.nColRL())),
                                   i,
                                   j);
        }
        else if (_interp == NEAREST_INTERP)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nThread)
            IMAGE_FOR_EACH_PIXEL_RL(_projectee2D)
                _projectee2D.setRL(_projectee2D.getRL(i, j)
                                 / NIK_RL(NORM(i, j)
                                        / (_pf * _projectee2D.nColRL())),
                                   i,
                                   j);
        }

#ifdef VERBOSE_LEVEL_3
            CLOG(INFO, "LOGGER_SYS") << "Fourier Transform in Grid Correction";
#endif
        }
        else if (_mode == MODE_3D)
        {
#ifdef VERBOSE_LEVEL_3
            CLOG(INFO, "LOGGER_SYS") << "Inverse Fourier Transform in Grid Correction";
#endif

#ifdef PROJECTOR_REMOVE_NEG
            #pragma omp parallel for num_threads(nThread)
            REMOVE_NEG(_projectee3D);
#endif

        if (_interp == LINEAR_INTERP)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nThread)
            VOLUME_FOR_EACH_PIXEL_RL(_projectee3D)
                _projectee3D.setRL(_projectee3D.getRL(i, j, k)
                                 / TIK_RL(NORM_3(i, j, k)
                                        / (_pf * _projectee3D.nColRL())),
                                   i,
                                   j,
                                   k);
        }
        else if (_interp == NEAREST_INTERP)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nThread)
            VOLUME_FOR_EACH_PIXEL_RL(_projectee3D)
                _projectee3D.setRL(_projectee3D.getRL(i, j, k)
                                 / NIK_RL(NORM_3(i, j, k)
                                        / (_pf * _projectee3D.nColRL())),
                                   i,
                                   j,
                                   k);
        }

#ifdef VERBOSE_LEVEL_3
            CLOG(INFO, "LOGGER_SYS") << "Fourier Transform in Grid Correction";
#endif
        }
        else
        {
            REPORT_ERROR("INEXISTENT_MODE");

            abort();
        }
}
