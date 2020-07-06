/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Reconstructor.h"

Reconstructor::Reconstructor()
{
    defaultInit();
}

Reconstructor::Reconstructor(const int mode,
                             const int size,
                             const int N,
                             const int pf,
                             const Symmetry* sym,
                             const RFLOAT a,
                             const RFLOAT alpha)
{
    defaultInit();

    init(mode, size, N, pf, sym, a, alpha);
}

Reconstructor::~Reconstructor()
{
#ifdef GPU_RECONSTRUCT
    if (_mode == MODE_3D)
    {
        _fft.fwDestroyPlan();
        _fft.bwDestroyPlan();
    }
#else
    _fft.fwDestroyPlan();
    _fft.bwDestroyPlan();
#endif
}

void Reconstructor::init(const int mode,
                         const int size,
                         const int N,
                         const int pf,
                         const Symmetry* sym,
                         const RFLOAT a,
                         const RFLOAT alpha)
{
    _mode = mode;
    _size = size;
    _N = N;
    _pf = pf;
    _sym = sym;

    _a = a;
    _alpha = alpha;

    // initialize the interpolation kernel
    
    ALOG(INFO, "LOGGER_RECO") << "Initialising Kernels";
    BLOG(INFO, "LOGGER_RECO") << "Initialising Kernels";

    _kernelFT.init(boost::bind(MKB_FT_R2,
                               boost::placeholders::_1,
#ifdef RECONSTRUCTOR_KERNEL_PADDING
                               _pf * _a,
#else
                               _a,
#endif
                               _alpha),
                   0,
                   TSGSL_pow_2(_pf * _a),
                   1e5);

    _kernelRL.init(boost::bind(MKB_RL_R2,
                               boost::placeholders::_1,
#ifdef RECONSTRUCTOR_KERNEL_PADDING
                               _pf * _a,
#else
                               _a,
#endif
                               _alpha),
                   0,
                   1,
                   1e5);

    setMaxRadius(_size/ 2 - CEIL(a));
    // _maxRadius = (_size / 2 - CEIL(a));
}

void Reconstructor::allocSpace(const unsigned int nThread)
{
    if (_mode == MODE_2D)
    {
        // Create Fourier Plans First, Then Allocate Space
        // For Save Memory Space

#ifndef GPU_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Creating Fourier Transform Plans";
        BLOG(INFO, "LOGGER_RECO") << "Creating Fourier Transform Plans";

        _fft.fwCreatePlan(PAD_SIZE, PAD_SIZE, nThread);
        _fft.bwCreatePlan(PAD_SIZE, PAD_SIZE, nThread);
#endif

        ALOG(INFO, "LOGGER_RECO") << "Allocating Spaces";
        BLOG(INFO, "LOGGER_RECO") << "Allocating Spaces";

        _F2D.alloc(PAD_SIZE, PAD_SIZE, FT_SPACE);
        _W2D.alloc(PAD_SIZE, PAD_SIZE, FT_SPACE);
        _C2D.alloc(PAD_SIZE, PAD_SIZE, FT_SPACE);
        _T2D.alloc(PAD_SIZE, PAD_SIZE, FT_SPACE);
    }
    else if (_mode == MODE_3D)
    {
        // Create Fourier Plans First, Then Allocate Space
        // For Save Memory Space

        ALOG(INFO, "LOGGER_RECO") << "Creating Fourier Transform Plans";
        BLOG(INFO, "LOGGER_RECO") << "Creating Fourier Transform Plans";

        _fft.fwCreatePlan(PAD_SIZE, PAD_SIZE, PAD_SIZE, nThread);
        _fft.bwCreatePlan(PAD_SIZE, PAD_SIZE, PAD_SIZE, nThread);

        ALOG(INFO, "LOGGER_RECO") << "Allocating Spaces";
        BLOG(INFO, "LOGGER_RECO") << "Allocating Spaces";

        _F3D.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
        _W3D.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
        _C3D.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
        _T3D.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);

    }
    else 
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    reset(nThread);
}

void Reconstructor::freeSpace()
{
    if (_mode == MODE_2D)
    {
        ALOG(INFO, "LOGGER_RECO") << "Freeing Spaces";
        BLOG(INFO, "LOGGER_RECO") << "Freeing Spaces";

#ifndef GPU_RECONSTRUCT
        _fft.fwDestroyPlan();
        _fft.bwDestroyPlan();
#endif

        _F2D.clear();
        _W2D.clear();
        _C2D.clear();
        _T2D.clear();
    }
    else if (_mode == MODE_3D)
    {
        ALOG(INFO, "LOGGER_RECO") << "Freeing Spaces";
        BLOG(INFO, "LOGGER_RECO") << "Freeing Spaces";

        _fft.fwDestroyPlan();
        _fft.bwDestroyPlan();
        
        _F3D.clear();
        _W3D.clear();
        _C3D.clear();
        _T3D.clear();

    }
    else 
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
}

void Reconstructor::resizeSpace(const int size)
{
#ifdef GPU_RECONSTRUCT
    if (_mode == MODE_3D)
    {
        _fft.fwDestroyPlan();
        _fft.bwDestroyPlan();
    }
#else
    _fft.fwDestroyPlan();
    _fft.bwDestroyPlan();
#endif

    _size = size;
}

void Reconstructor::reset(const unsigned int nThread)
{
    _iCol = NULL;
    _iRow = NULL;
    _iPxl = NULL;
    _iSig = NULL;

    _calMode = POST_CAL_MODE;

    _MAP = true;

    _gridCorr = true;

    _joinHalf = false;

    if (_mode == MODE_2D)
    {
        #pragma omp parallel for num_threads(nThread)
        SET_0_FT(_F2D);

        #pragma omp parallel for num_threads(nThread)
        SET_1_FT(_W2D);

        #pragma omp parallel for num_threads(nThread)
        SET_0_FT(_C2D);

        #pragma omp parallel for num_threads(nThread)
        SET_0_FT(_T2D);
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for num_threads(nThread)
        SET_0_FT(_F3D);

        #pragma omp parallel for num_threads(nThread)
        SET_1_FT(_W3D);

        #pragma omp parallel for num_threads(nThread)
        SET_0_FT(_C3D);

        #pragma omp parallel for num_threads(nThread)
        SET_0_FT(_T3D);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    _ox = 0;
    _oy = 0;
    _oz = 0;

    _counter = 0;
}

int Reconstructor::mode() const
{
    return _mode;
}

void Reconstructor::setMode(const int mode)
{
    _mode = mode;
}

bool Reconstructor::MAP() const
{
    return _MAP;
}

void Reconstructor::setMAP(const bool MAP)
{
    _MAP = MAP;
}

bool Reconstructor::gridCorr() const
{
    return _gridCorr;
}

void Reconstructor::setGridCorr(const bool gridCorr)
{
    _gridCorr = gridCorr;
}

bool Reconstructor::joinHalf() const
{
    return _joinHalf;
}

void Reconstructor::setJoinHalf(const bool joinHalf)
{
    _joinHalf = joinHalf;
}

void Reconstructor::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Reconstructor::setFSC(const vec& FSC)
{
    _FSC = FSC;
}

void Reconstructor::setTau(const vec& tau)
{
    _tau = tau;
}

void Reconstructor::setSig(const vec& sig)
{
    _sig = sig;
}

void Reconstructor::setOx(const double ox)
{
    _ox = ox;
}

void Reconstructor::setOy(const double oy)
{
    _oy = oy;
}

void Reconstructor::setOz(const double oz)
{
    _oz = oz;
}

void Reconstructor::setCounter(const int counter)
{
    _counter = counter;
}

double Reconstructor::ox() const
{
    return _ox;
}

double Reconstructor::oy() const
{
    return _oy;
}

double Reconstructor::oz() const
{
    return _oz;
}

int Reconstructor::counter() const
{
    return _counter;
}

int Reconstructor::maxRadius() const
{
    return _maxRadius;
}

void Reconstructor::setMaxRadius(const int maxRadius)
{
    _maxRadius = maxRadius;

    _tau = vec::Zero(_maxRadius + 2);
}

int Reconstructor::pf() const
{
    return _pf;
}

void Reconstructor::setPf(const int pf)
{
    _pf = pf;
}

int Reconstructor::N() const
{
    return _N;
}

void Reconstructor::setN(const int N)
{
    _N = N;
}

RFLOAT Reconstructor::getNF()
{
#ifdef RECONSTRUCTOR_KERNEL_PADDING
        return MKB_RL(0, _a * _pf, _alpha);
#else
        return MKB_RL(0, _a, _alpha);
#endif
}

void Reconstructor::preCal(int& nPxl,
                           const int* iCol,
                           const int* iRow,
                           const int* iPxl,
                           const int* iSig) const
{
    nPxl = _nPxl;

    iCol = _iCol;
    iRow = _iRow;
    iPxl = _iPxl;
    iSig = _iSig;
}

void Reconstructor::setPreCal(const int nPxl,
                              const int* iCol,
                              const int* iRow,
                              const int* iPxl,
                              const int* iSig)
{
    _calMode = PRE_CAL_MODE;

    _nPxl = nPxl;

    _iCol = iCol;
    _iRow = iRow;
    _iPxl = iPxl;
    _iSig = iSig;
}

void Reconstructor::insertDir(const dvec2& dir)
{
    insertDir(dir(0), dir(1), 0);
}

void Reconstructor::insertDir(const dvec3& dir)
{
    insertDir(dir(0), dir(1), dir(2));
}

void Reconstructor::insertDir(const double ox,
                              const double oy,
                              const double oz)
{
    #pragma omp atomic
    _ox += ox;

    #pragma omp atomic
    _oy += oy;

    #pragma omp atomic
    _oz += oz;

    #pragma omp atomic
    _counter +=1;
}

void Reconstructor::insert(const Image& src,
                           const Image& ctf,
                           const dmat22& rot,
                           const RFLOAT w)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_2D REPORT_ERROR("WRONG MODE");

    if (_calMode != POST_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");

    if ((src.nColRL() != _size) ||
        (src.nRowRL() != _size) ||
        (ctf.nColRL() != _size) ||
        (ctf.nRowRL() != _size))
        REPORT_ERROR("INCORRECT SIZE OF INSERTING IMAGE");
#endif

    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
        {
            #pragma omp atomic
            _tau(AROUND(NORM(i, j))) += w;

            dvec2 newCor((double)(i * _pf), (double)(j * _pf));
            dvec2 oldCor = rot * newCor;

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _F2D.addFT(src.getFTHalf(i, j)
                     * REAL(ctf.getFTHalf(i, j))
                     * w, 
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a, 
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _F2D.addFT(src.getFTHalf(i, j)
                     * REAL(ctf.getFTHalf(i, j))
                     * w, 
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _T2D.addFT(TSGSL_pow_2(REAL(ctf.getFTHalf(i, j)))
                     * w, 
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a,
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
             _T2D.addFT(TSGSL_pow_2(REAL(ctf.getFTHalf(i, j)))
                      * w, 
                        (RFLOAT)oldCor(0), 
                        (RFLOAT)oldCor(1));
#endif

#endif
        }
    }
}

void Reconstructor::insert(const Image& src,
                           const Image& ctf,
                           const dmat33& rot,
                           const RFLOAT w)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_3D REPORT_ERROR("WRONG MODE");

    if (_calMode != POST_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");

    if ((src.nColRL() != _size) ||
        (src.nRowRL() != _size) ||
        (ctf.nColRL() != _size) ||
        (ctf.nRowRL() != _size))
        REPORT_ERROR("INCORRECT SIZE OF INSERTING IMAGE");
#endif

        IMAGE_FOR_EACH_PIXEL_FT(src)
        {
            if (QUAD(i, j) < gsl_pow_2(_maxRadius))
            {
                #pragma omp atomic
                _tau(AROUND(NORM(i, j))) += w;

                const double* ptr = rot.data();
                double oldCor[3];
                oldCor[0] = (ptr[0] * i + ptr[3] * j) * _pf;
                oldCor[1] = (ptr[1] * i + ptr[4] * j) * _pf;
                oldCor[2] = (ptr[2] * i + ptr[5] * j) * _pf;

#ifdef RECONSTRUCTOR_MKB_KERNEL
                _F3D.addFT(src.getFTHalf(i, j)
                         * REAL(ctf.getFTHalf(i, j))
                         * w, 
                           (RFLOAT)oldCor[0], 
                           (RFLOAT)oldCor[1], 
                           (RFLOAT)oldCor[2], 
                           _pf * _a, 
                           _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                _F3D.addFT(src.getFTHalf(i, j)
                         * REAL(ctf.getFTHalf(i, j))
                         * w, 
                           (RFLOAT)oldCor[0], 
                           (RFLOAT)oldCor[1], 
                           (RFLOAT)oldCor[2]);
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
                _T3D.addFT(TSGSL_pow_2(REAL(ctf.getFTHalf(i, j)))
                         * w, 
                           (RFLOAT)oldCor[0], 
                           (RFLOAT)oldCor[1], 
                           (RFLOAT)oldCor[2],
                           _pf * _a,
                           _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                _T3D.addFT(TSGSL_pow_2(REAL(ctf.getFTHalf(i, j)))
                         * w, 
                           (RFLOAT)oldCor[0], 
                           (RFLOAT)oldCor[1], 
                           (RFLOAT)oldCor[2]);
#endif

#endif
            }
        }
}

void Reconstructor::insertP(const Image& src,
                            const Image& ctf,
                            const dmat22& rot,
                            const RFLOAT w,
                            const vec* sig)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_2D REPORT_ERROR("WRONG MODE");

    if (_calMode != PRE_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");
#endif

        for (int i = 0; i < _nPxl; i++)
        {
            #pragma omp atomic
            _tau(_iSig[i]) += (sig == NULL ? 1 : (*sig)(_iSig[i])) * w;

            dvec2 newCor((double)(_iCol[i]), (double)(_iRow[i]));
            dvec2 oldCor = rot * newCor;

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _F2D.addFT(src.iGetFT(_iPxl[i])
                     * REAL(ctf.iGetFT(_iPxl[i]))
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a, 
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _F2D.addFT(src.iGetFT(_iPxl[i])
                     * REAL(ctf.iGetFT(_iPxl[i]))
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _T2D.addFT(TSGSL_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a,
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _T2D.addFT(TSGSL_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#endif
        }
}

void Reconstructor::insertP(const Image& src,
                            const Image& ctf,
                            const dmat33& rot,
                            const RFLOAT w,
                            const vec* sig)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_3D REPORT_ERROR("WRONG MODE");

    if (_calMode != PRE_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");
#endif

    for (int i = 0; i < _nPxl; i++)
    {
        #pragma omp atomic
        _tau(_iSig[i]) += (sig == NULL ? 1 : (*sig)(_iSig[i])) * w;

        const double* ptr = rot.data();
        double oldCor[3];
        int iCol = _iCol[i];
        int iRow = _iRow[i];
        oldCor[0] = ptr[0] * iCol + ptr[3] * iRow;
        oldCor[1] = ptr[1] * iCol + ptr[4] * iRow;
        oldCor[2] = ptr[2] * iCol + ptr[5] * iRow;

#ifdef RECONSTRUCTOR_MKB_KERNEL
        _F3D.addFT(src.iGetFT(_iPxl[i])
                 * REAL(ctf.iGetFT(_iPxl[i]))
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0],
                   (RFLOAT)oldCor[1],
                   (RFLOAT)oldCor[2],
                   _pf * _a, 
                   _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        _F3D.addFT(src.iGetFT(_iPxl[i])
                 * REAL(ctf.iGetFT(_iPxl[i]))
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0],
                   (RFLOAT)oldCor[1],
                   (RFLOAT)oldCor[2]);
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
        _T3D.addFT(TSGSL_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0],
                   (RFLOAT)oldCor[1],
                   (RFLOAT)oldCor[2],
                   _pf * _a,
                   _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        _T3D.addFT(TSGSL_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0],
                   (RFLOAT)oldCor[1],
                   (RFLOAT)oldCor[2]);
#endif

#endif
    }
}

void Reconstructor::insertP(const RFLOAT* srcR,
                            const RFLOAT* srcI,
                            const RFLOAT* ctf,
                            const dmat22& rot,
                            const RFLOAT w,
                            const vec* sig)
{
    Complex* src = new Complex[_nPxl];

    for (int i = 0; i < _nPxl; i++)
    {
        src[i].dat[0] = srcR[i];
        src[i].dat[1] = srcI[i];
    }

    insertP(src, ctf, rot, w, sig);

    delete[] src;
}

void Reconstructor::insertP(const Complex* src,
                            const RFLOAT* ctf,
                            const dmat22& rot,
                            const RFLOAT w,
                            const vec* sig)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_2D REPORT_ERROR("WRONG MODE");

    if (_calMode != PRE_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");
#endif

#ifndef NAN_NO_CHECK
    
    SEGMENT_NAN_CHECK_COMPLEX(src, (size_t)_nPxl);
    SEGMENT_NAN_CHECK(ctf, (size_t)_nPxl);
    NAN_CHECK_DMAT22(rot);
    POINT_NAN_CHECK(w);

#endif

        for (int i = 0; i < _nPxl; i++)
        {
            #pragma omp atomic
            _tau(_iSig[i]) += (sig == NULL ? 1 : (*sig)(_iSig[i])) * w;

            dvec2 newCor((double)(_iCol[i]), (double)(_iRow[i]));
            dvec2 oldCor = rot * newCor;

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _F2D.addFT(src[i]
                     * ctf[i]
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a, 
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _F2D.addFT(src[i]
                     * ctf[i]
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _T2D.addFT(TSGSL_pow_2(ctf[i])
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a,
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _T2D.addFT(TSGSL_pow_2(ctf[i])
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#endif
        }
}

void Reconstructor::insertP(const RFLOAT* srcR,
                            const RFLOAT* srcI,
                            const RFLOAT* ctf,
                            const dmat33& rot,
                            const RFLOAT w,
                            const vec* sig)
{
    Complex* src = new Complex[_nPxl];

    for (int i = 0; i < _nPxl; i++)
    {
        src[i].dat[0] = srcR[i];
        src[i].dat[1] = srcI[i];
    }

    insertP(src, ctf, rot, w, sig);

    delete[] src;
}

void Reconstructor::insertP(const Complex* src,
                            const RFLOAT* ctf,
                            const dmat33& rot,
                            const RFLOAT w,
                            const vec* sig)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_3D REPORT_ERROR("WRONG MODE");

    if (_calMode != PRE_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");
#endif

#ifndef NAN_NO_CHECK
    
    SEGMENT_NAN_CHECK_COMPLEX(src, (size_t)_nPxl);
    SEGMENT_NAN_CHECK(ctf, (size_t)_nPxl);
    NAN_CHECK_DMAT33(rot);
    POINT_NAN_CHECK(w);

#endif

    for (int i = 0; i < _nPxl; i++)
    {
        #pragma omp atomic
        _tau(_iSig[i]) += (sig == NULL ? 1 : (*sig)(_iSig[i])) * w;

        const double* ptr = rot.data();
        double oldCor[3];
        int iCol = _iCol[i];
        int iRow = _iRow[i];
        oldCor[0] = ptr[0] * iCol + ptr[3] * iRow;
        oldCor[1] = ptr[1] * iCol + ptr[4] * iRow;
        oldCor[2] = ptr[2] * iCol + ptr[5] * iRow;

#ifdef RECONSTRUCTOR_MKB_KERNEL
        _F3D.addFT(src[i]
                 * ctf[i]
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0], 
                   (RFLOAT)oldCor[1], 
                   (RFLOAT)oldCor[2], 
                   _pf * _a, 
                   _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        _F3D.addFT(src[i]
                 * ctf[i]
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0], 
                   (RFLOAT)oldCor[1], 
                   (RFLOAT)oldCor[2]);
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
        _T3D.addFT(TSGSL_pow_2(ctf[i])
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0], 
                   (RFLOAT)oldCor[1], 
                   (RFLOAT)oldCor[2],
                   _pf * _a,
                   _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        _T3D.addFT(TSGSL_pow_2(ctf[i])
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0], 
                   (RFLOAT)oldCor[1], 
                   (RFLOAT)oldCor[2]);
#endif

#endif
    }
}

#ifdef GPU_INSERT
int Reconstructor::getModelDim(bool mode)
{
    if (mode)
        return _F3D.nSlcFT();
    else
        return _F2D.nRowFT();
}

int Reconstructor::getModelSize(bool mode)
{
    if (mode)
        return _F3D.sizeFT();
    else
        return _F2D.sizeFT();
}

TabFunction& Reconstructor::getTabFuncRL()
{
    return _kernelRL;
}

int Reconstructor::getTauSize()
{
     return _tau.size();
}

void Reconstructor::getF(Complex* modelF,
                         bool mode,
                         const unsigned int nThread)
{
    if (mode)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        for(size_t i = 0; i < _F3D.sizeFT(); i++)
            modelF[i] = _F3D[i];
    }
    else
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        for(size_t i = 0; i < _F2D.sizeFT(); i++)
            modelF[i] = _F2D[i];
    }
}

void Reconstructor::getT(RFLOAT* modelT,
                         bool mode,
                         const unsigned int nThread)
{
    if (mode)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        for(size_t i = 0; i < _T3D.sizeFT(); i++)
            modelT[i] = REAL(_T3D[i]);
    }
    else
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        for(size_t i = 0; i < _T2D.sizeFT(); i++)
            modelT[i] = REAL(_T2D[i]);
    }
}

void Reconstructor::getTau(RFLOAT* arrayTau,
                           const unsigned int nThread)
{
    #pragma omp parallel for schedule(dynamic) num_threads(nThread)
    for(size_t i = 0; i < _tau.size(); i++)
        arrayTau[i] = _tau(i);
}

void Reconstructor::resetF(Complex* modelF,
                           bool mode,
                           const unsigned int nThread)
{
    if (mode)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        for(size_t i = 0; i < _F3D.sizeFT(); i++)
            _F3D[i] = modelF[i];
    }
    else
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        for(size_t i = 0; i < _F2D.sizeFT(); i++)
            _F2D[i] = modelF[i];
    }
}

void Reconstructor::resetT(RFLOAT* modelT,
                           bool mode,
                           const unsigned int nThread)
{
    if (mode)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        for(size_t i = 0; i < _T3D.sizeFT(); i++)
            _T3D[i] = COMPLEX(modelT[i], 0);
    }
    else
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        for(size_t i = 0; i < _T2D.sizeFT(); i++)
            _T2D[i] = COMPLEX(modelT[i], 0);
    }
}

void Reconstructor::resetTau(RFLOAT* tau)
{
    for (size_t i = 0; i < _tau.size(); i++)
        _tau(i) = tau[i];
}
        
void Reconstructor::prepareTFG(std::vector<int>& iGPU,
                               std::vector<void*>& stream,
                               int nGPU)
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_RECO") << "Allreducing tau";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing tau";

    allReduceTau();

    ALOG(INFO, "LOGGER_RECO") << "Adding Tau of Wiener Factor";
    BLOG(INFO, "LOGGER_RECO") << "Adding Tau of Wiener Factor";

    if (_mode == MODE_2D)
    {
        for (size_t i = 1; i < _maxRadius + 2; i++)
        {
            _tau(i) /= M_PI * i;
        }

        IMAGE_FOR_EACH_PIXEL_FT(_T2D)
        {
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
            {
                _T2D.setFTHalf(COMPLEX(REAL(_T2D.getFTHalf(i, j)) + _tau[AROUND(NORM(i, j) / _pf)] * TAU_FACTOR, 0), i, j);
            }
        }
    }
    else if (_mode == MODE_3D)
    {
        for (size_t i = 1; i < _maxRadius + 2; i++)
        {
            _tau(i) /= 2 * M_PI * i * i;
        }
        
        VOLUME_FOR_EACH_PIXEL_FT(_T3D)
        {
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
            {
                _T3D.setFTHalf(COMPLEX(REAL(_T3D.getFTHalf(i, j, k)) + _tau[AROUND(NORM_3(i, j, k) / _pf)] * TAU_FACTOR, 0), i, j, k);
            }
        }
    }

    // only in 3D mode, symmetry should be considered
    IF_MODE_3D
    {
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
	    int nSymmetryElement = _sym->nSymmetryElement();
        double *symMat = new double[nSymmetryElement * 9];

        ALOG(INFO, "LOGGER_RECO") << "Prepare param for Symmetrizing TF";
        BLOG(INFO, "LOGGER_RECO") << "Prepare param for Symmetrizing TF";
        dmat33 L, R;   
        
	    for(int i = 0; i < nSymmetryElement; i++)
	    {
            _sym->get(L, R, i);
            Map<dmat33>(symMat + i * 9, 3, 3) = R;
	    }
        
        symetrizeTF(iGPU,
                    stream,
                    _F3D,
                    _T3D,
                    symMat,
                    nGPU,
                    nSymmetryElement,
                    _maxRadius,
                    _pf);
        
        delete[]symMat;
#endif
    }
}

#endif // GPU_INSERT

void Reconstructor::prepareTF(const unsigned int nThread)
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_RECO") << "Allreducing tau";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing tau";

    allReduceTau();

    ALOG(INFO, "LOGGER_RECO") << "Allreducing T";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing T";

    allReduceT(nThread);
    allReduceF();

    ALOG(INFO, "LOGGER_RECO") << "Adding Tau of Wiener Factor";
    BLOG(INFO, "LOGGER_RECO") << "Adding Tau of Wiener Factor";

    if (_mode == MODE_2D)
    {
        for (size_t i = 1; i < _maxRadius + 2; i++)
        {
            _tau(i) /= M_PI * i;
        }

        IMAGE_FOR_EACH_PIXEL_FT(_T2D)
        {
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
            {
                _T2D.setFTHalf(COMPLEX(REAL(_T2D.getFTHalf(i, j)) + _tau[AROUND(NORM(i, j) / _pf)] * TAU_FACTOR, 0), i, j);
            }
        }
    }
    else if (_mode == MODE_3D)
    {
        for (size_t i = 1; i < _maxRadius + 2; i++)
        {
            _tau(i) /= 2 * M_PI * i * i;
        }
        
        VOLUME_FOR_EACH_PIXEL_FT(_T3D)
        {
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
            {
                _T3D.setFTHalf(COMPLEX(REAL(_T3D.getFTHalf(i, j, k)) + _tau[AROUND(NORM_3(i, j, k) / _pf)] * TAU_FACTOR, 0), i, j, k);
            }
        }
    }

    // only in 3D mode, symmetry should be considered
    IF_MODE_3D
    {
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Symmetrizing T";
        BLOG(INFO, "LOGGER_RECO") << "Symmetrizing T";

        symmetrizeT(nThread);
#endif
    }

    ALOG(INFO, "LOGGER_RECO") << "Allreducing F";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing F";

    // only in 3D mode, symmetry should be considered
    IF_MODE_3D
    {
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Symmetrizing F";
        BLOG(INFO, "LOGGER_RECO") << "Symmetrizing F";

        symmetrizeF(nThread);
#endif
    }
}

void Reconstructor::normalise(const size_t nPar,
                              const unsigned int nThread)
{
    _tau.array() /= nPar;
}

void Reconstructor::reconstruct(Image& dst,
                                const unsigned int nThread)
{
    Volume tmp;

    reconstruct(tmp, nThread);

    dst.alloc(PAD_SIZE, PAD_SIZE, RL_SPACE);

    SLC_EXTRACT_RL(dst, tmp, 0);
}

void Reconstructor::prepareO()
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_RECO") << "Allreducing O";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing O";

    allReduceO();

    IF_MODE_3D
    {
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Symmetrizing O";
        BLOG(INFO, "LOGGER_RECO") << "Symmetrizing O";

        symmetrizeO();
#endif
    }

    _ox /= _counter;
    _oy /= _counter;
    _oz /= _counter;
}

void Reconstructor::reconstruct(Volume& dst,
                                const unsigned int nThread)
{
    IF_MASTER return;

#ifdef VERBOSE_LEVEL_2

    IF_MODE_2D
    {
        ALOG(INFO, "LOGGER_RECO") << "Reconstructing Under 2D Mode";
        BLOG(INFO, "LOGGER_RECO") << "Reconstructing Under 2D Mode";
    }

    IF_MODE_3D
    {
        ALOG(INFO, "LOGGER_RECO") << "Reconstructing Under 3D Mode";
        BLOG(INFO, "LOGGER_RECO") << "Reconstructing Under 3D Mode";
    }

#endif

    if (_MAP)
    {
        // Obviously, wiener_filter with FSC can be wrong when dealing with
        // preferrable orienation problem
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
        vec avg = vec::Zero(_maxRadius * _pf + 1);

        if (_mode == MODE_2D)
        {
            ringAverage(avg,
                        _T2D,
                        REAL,
                        _maxRadius * _pf - 1);
        }
        else if (_mode == MODE_3D)
        {
            shellAverage(avg,
                         _T3D,
                         REAL,
                         _maxRadius * _pf - 1,
                         nThread);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

        // the last two elements have low fidelity
        avg(_maxRadius * _pf - 1) = avg(_maxRadius * _pf - 2);
        avg(_maxRadius * _pf) = avg(_maxRadius * _pf - 2);

#ifdef VERBOSE_LEVEL_2
        ALOG(INFO, "LOGGER_SYS") << "End of Avg = "
                                 << avg(avg.size() - 5) << ", "
                                 << avg(avg.size() - 4) << ", "
                                 << avg(avg.size() - 3) << ", "
                                 << avg(avg.size() - 2) << ", "
                                 << avg(avg.size() - 1);
        BLOG(INFO, "LOGGER_SYS") << "End of Avg = "
                                 << avg(avg.size() - 5) << ", "
                                 << avg(avg.size() - 4) << ", "
                                 << avg(avg.size() - 3) << ", "
                                 << avg(avg.size() - 2) << ", "
                                 << avg(avg.size() - 1);
#endif

#endif

#ifdef VERBOSE_LEVEL_2
        ALOG(INFO, "LOGGER_SYS") << "End of FSC = " << _FSC(_FSC.size() - 1);
        BLOG(INFO, "LOGGER_SYS") << "End of FSC = " << _FSC(_FSC.size() - 1);
#endif

        if (_mode == MODE_2D)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nThread)
            IMAGE_FOR_EACH_PIXEL_FT(_T2D)
                if ((QUAD(i, j) >= TSGSL_pow_2(WIENER_FACTOR_MIN_R * _pf)) &&
                    (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf)))
                {
                    int u = AROUND(NORM(i, j));

                    RFLOAT FSC = (u / _pf >= _FSC.size())
                               ? 0
                               : _FSC(u / _pf);

                    FSC = TSGSL_MAX_RFLOAT(FSC_BASE_L, TSGSL_MIN_RFLOAT(FSC_BASE_H, FSC));

#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
                    FSC = sqrt(2 * FSC / (1 + FSC));
#else
                    if (_joinHalf) FSC = sqrt(2 * FSC / (1 + FSC));
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
                    _T2D.setFT(_T2D.getFT(i, j)
                             + COMPLEX((1 - FSC) / FSC * avg(u), 0),
                               i,
                               j);
#else
                    _T2D.setFT(_T2D.getFT(i, j) / FSC, i, j);
#endif
                }
        }
        else if (_mode == MODE_3D)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nThread)
            VOLUME_FOR_EACH_PIXEL_FT(_T3D)
                if ((QUAD_3(i, j, k) >= TSGSL_pow_2(WIENER_FACTOR_MIN_R * _pf)) &&
                    (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf)))
                {
                    int u = AROUND(NORM_3(i, j, k));

                    RFLOAT FSC = (u / _pf >= _FSC.size())
                               ? 0
                               : _FSC(u / _pf);

                    FSC = TSGSL_MAX_RFLOAT(FSC_BASE_L, TSGSL_MIN_RFLOAT(FSC_BASE_H, FSC));

#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
                    FSC = sqrt(2 * FSC / (1 + FSC));
#else
                    if (_joinHalf) FSC = sqrt(2 * FSC / (1 + FSC));
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
                    _T3D.setFT(_T3D.getFT(i, j, k)
                             + COMPLEX((1 - FSC) / FSC * avg(u), 0),
                               i,
                               j,
                               k);
#else
                    _T3D.setFT(_T3D.getFT(i, j, k) / FSC, i, j, k);
#endif
                }
        }
        else
        {
            REPORT_ERROR("INEXISTENT_MODE");

            abort();
        }
#endif
    }

#ifdef VERBOSE_LEVEL_2

    ALOG(INFO, "LOGGER_RECO") << "Initialising W";
    BLOG(INFO, "LOGGER_RECO") << "Initialising W";

#endif

    if (_mode == MODE_2D)
    {
        #pragma omp parallel for num_threads(nThread)
        IMAGE_FOR_EACH_PIXEL_FT(_W2D)
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
                _W2D.setFTHalf(COMPLEX(1, 0), i, j);
            else
                _W2D.setFTHalf(COMPLEX(0, 0), i, j);
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_FT(_W3D)
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
                _W3D.setFTHalf(COMPLEX(1, 0), i, j, k);
            else
                _W3D.setFTHalf(COMPLEX(0, 0), i, j, k);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    // make sure there is a minimum value for T
    if (_mode == MODE_2D)
    {
        #pragma omp parallel for num_threads(nThread)
        FOR_EACH_PIXEL_FT(_T2D)
            _T2D[i].dat[0] = TSGSL_MAX_RFLOAT(_T2D[i].dat[0], 1e-25);
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for num_threads(nThread)
        FOR_EACH_PIXEL_FT(_T3D)
            _T3D[i].dat[0] = TSGSL_MAX_RFLOAT(_T3D[i].dat[0], 1e-25);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

#ifndef NAN_NO_CHECK
    if (_mode == MODE_2D)
    {
        SEGMENT_NAN_CHECK_COMPLEX(_F2D.dataFT(), _F2D.sizeFT());
        SEGMENT_NAN_CHECK_COMPLEX(_W2D.dataFT(), _W2D.sizeFT());
        SEGMENT_NAN_CHECK_COMPLEX(_T2D.dataFT(), _T2D.sizeFT());
        SEGMENT_NAN_CHECK_COMPLEX(_C2D.dataFT(), _C2D.sizeFT());
    }
    else if (_mode == MODE_3D)
    {
        SEGMENT_NAN_CHECK_COMPLEX(_F3D.dataFT(), _F3D.sizeFT());
        SEGMENT_NAN_CHECK_COMPLEX(_W3D.dataFT(), _W3D.sizeFT());
        SEGMENT_NAN_CHECK_COMPLEX(_T3D.dataFT(), _T3D.sizeFT());
        SEGMENT_NAN_CHECK_COMPLEX(_C3D.dataFT(), _C3D.sizeFT());
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
#endif

    // TODO, debug
    // _gridCorr = false;

    if (_gridCorr)
    {
        RFLOAT diffC = TS_MAX_RFLOAT_VALUE;
        RFLOAT diffCPrev = TS_MAX_RFLOAT_VALUE;

        int m = 0;

        int nDiffCNoDecrease = 0;

        for (m = 0; m < MAX_N_ITER_BALANCE; m++)
        {
#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_RECO") << "Balancing Weights Round " << m;
            BLOG(INFO, "LOGGER_RECO") << "Balancing Weights Round " << m;

            ALOG(INFO, "LOGGER_RECO") << "Determining C";
            BLOG(INFO, "LOGGER_RECO") << "Determining C";

#endif
        
            if (_mode == MODE_2D)
            {
#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(_T2D.dataFT(), _T2D.sizeFT());
                SEGMENT_NAN_CHECK_COMPLEX(_W2D.dataFT(), _W2D.sizeFT());
#endif

                #pragma omp parallel for num_threads(nThread)
                FOR_EACH_PIXEL_FT(_C2D)
                    _C2D[i] = _T2D[i] * REAL(_W2D[i]);

#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(_C2D.dataFT(), _C2D.sizeFT());
#endif
            }
            else if (_mode == MODE_3D)
            {
#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(_T3D.dataFT(), _T3D.sizeFT());
                SEGMENT_NAN_CHECK_COMPLEX(_W3D.dataFT(), _W3D.sizeFT());
#endif

                #pragma omp parallel for num_threads(nThread)
                FOR_EACH_PIXEL_FT(_C3D)
                    _C3D[i] = _T3D[i] * REAL(_W3D[i]);

#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(_C3D.dataFT(), _C3D.sizeFT());
#endif
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }

#ifdef VERBOSE_LEVEL_2
            ALOG(INFO, "LOGGER_RECO") << "Convoluting C";
            BLOG(INFO, "LOGGER_RECO") << "Convoluting C";
#endif
            convoluteC(nThread);

#ifdef VERBOSE_LEVEL_2
            ALOG(INFO, "LOGGER_RECO") << "Re-Calculating W";
            BLOG(INFO, "LOGGER_RECO") << "Re-Calculating W";
#endif

            if (_mode == MODE_2D)
            {
#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(_W2D.dataFT(), _W2D.sizeFT());
                SEGMENT_NAN_CHECK_COMPLEX(_C2D.dataFT(), _C2D.sizeFT());
#endif

                #pragma omp parallel for schedule(dynamic) num_threads(nThread)
                IMAGE_FOR_EACH_PIXEL_FT(_W2D)
                    if (QUAD(i, j) < gsl_pow_2(_maxRadius * _pf))
                    {
                        /***
                        if (IS_NAN(ABS(_C2D.getFTHalf(i, j))))
                        {
                            CLOG(FATAL, "LOGGER_RECO") << "_C2D : "
                                                       << REAL(_C2D.getFTHalf(i, j))
                                                       << ", "
                                                       << IMAG(_C2D.getFTHalf(i, j));

                            abort();
                        }
                        ***/
                        
                        /***

                        _W2D.setFTHalf(COMPLEX(TSGSL_MIN_RFLOAT(REAL(_W2D.getFTHalf(i, j)) / TSGSL_MAX_RFLOAT(ABS(_C2D.getFTHalf(i, j)), 1e-6), 1.0 / NOISE_FACTOR), 0), i, j);
                        ***/

                        _W2D.setFTHalf(_W2D.getFTHalf(i, j)
                                     / TSGSL_MAX_RFLOAT(ABS(_C2D.getFTHalf(i, j)),
                                                   1e-6),
                                       i,
                                       j);

                        /***
                        if (IS_NAN(REAL(_W2D.getFTHalf(i, j)))
                         || IS_NAN(IMAG(_W2D.getFTHalf(i, j))))
                        {
                            CLOG(FATAL, "LOGGER_RECO") << "_W2D : "
                                                       << REAL(_W2D.getFTHalf(i, j))
                                                       << ", "
                                                       << IMAG(_W2D.getFTHalf(i, j));

                            CLOG(FATAL, "LOGGER_RECO") << "_C2D : "
                                                       << REAL(_C2D.getFTHalf(i, j))
                                                       << ", "
                                                       << IMAG(_C2D.getFTHalf(i, j))
                                                       << ", ABS = "
                                                       << ABS(_C2D.getFTHalf(i, j));

                            abort();
                        }
                        ***/
                    }

#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(_W2D.dataFT(), _W2D.sizeFT());
#endif
            }
            else if (_mode == MODE_3D)
            {
#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(_W3D.dataFT(), _W3D.sizeFT());
                SEGMENT_NAN_CHECK_COMPLEX(_C3D.dataFT(), _C3D.sizeFT());
#endif

                #pragma omp parallel for schedule(dynamic) num_threads(nThread)
                VOLUME_FOR_EACH_PIXEL_FT(_W3D)
                    if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
                    {
                        // _W3D.setFTHalf(COMPLEX(TSGSL_MIN_RFLOAT(REAL(_W3D.getFTHalf(i, j, k)) / TSGSL_MAX_RFLOAT(ABS(_C3D.getFTHalf(i, j, k)), 1e-6), 1.0 / NOISE_FACTOR), 0), i, j, k);
                        _W3D.setFTHalf(_W3D.getFTHalf(i, j, k)
                                     / TSGSL_MAX_RFLOAT(ABS(_C3D.getFTHalf(i, j, k)),
                                                   1e-6),
                                       i,
                                       j,
                                       k);
                    }

#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(_W3D.dataFT(), _W3D.sizeFT());
#endif
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }

#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_RECO") << "Calculating Distance to Total Balanced";
            BLOG(INFO, "LOGGER_RECO") << "Calculating Distance to Total Balanced";

#endif
            diffCPrev = diffC;
            diffC = checkC(nThread);
 
#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_SYS") << "After "
                                     << m
                                     << " Iterations, Distance to Total Balanced: "
                                     << diffC;
            BLOG(INFO, "LOGGER_SYS") << "After "
                                     << m
                                     << " Iterations, Distance to Total Balanced: "
                                     << diffC;

#endif

#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_RECO") << "Distance to Total Balanced: " << diffC;
            BLOG(INFO, "LOGGER_RECO") << "Distance to Total Balanced: " << diffC;

#endif

            if (diffC > diffCPrev * DIFF_C_DECREASE_THRES)
                nDiffCNoDecrease += 1;
            else
                nDiffCNoDecrease = 0;

            if ((diffC < DIFF_C_THRES) ||
                ((m >= MIN_N_ITER_BALANCE) &&
                (nDiffCNoDecrease == N_DIFF_C_NO_DECREASE))) 
                break;
        }
    }
    else
    {
        // no grid correction
        if (_mode == MODE_2D)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nThread)
            IMAGE_FOR_EACH_PIXEL_FT(_W2D)
                if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
                    _W2D.setFTHalf(COMPLEX(1.0
                                         / TSGSL_MAX_RFLOAT(ABS(_T2D.getFTHalf(i, j)),
                                                                1e-6),
                                           0),
                                   i,
                                   j);
        }
        else if (_mode == MODE_3D)
        {
            #pragma omp parallel for schedule(dynamic) num_threads(nThread)
            VOLUME_FOR_EACH_PIXEL_FT(_W3D)
                if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
                    _W3D.setFTHalf(COMPLEX(1.0
                                         / TSGSL_MAX_RFLOAT(ABS(_T3D.getFTHalf(i, j, k)),
                                                                1e-6),
                                           0),
                                   i,
                                   j,
                                   k);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }

    if (_mode == MODE_2D)
    {
#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(_F2D.dataFT(), _F2D.sizeFT());
        SEGMENT_NAN_CHECK_COMPLEX(_W2D.dataFT(), _W2D.sizeFT());
#endif

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Image";
        BLOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Image";

#endif

        Image padDst(_N * _pf, _N * _pf, FT_SPACE);

        #pragma omp parallel num_threads(nThread)
        SET_0_FT(padDst);

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";

#endif

        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        IMAGE_FOR_EACH_PIXEL_FT(_F2D)
        {
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
            {
                /***
                Complex result = _F2D.getFTHalf(i, j) * _W2D.getFTHalf(i, j);
                if ((TSGSL_isnan(REAL(result))) || (TSGSL_isnan(IMAG(result))))
                {
                    CLOG(FATAL, "LOGGER_RECO") << "_F2D : " << REAL(_F2D.getFTHalf(i, j)) << ", " << IMAG(_F2D.getFTHalf(i, j));
                    CLOG(FATAL, "LOGGER_RECO") << "_W2D : " << REAL(_W2D.getFTHalf(i, j)) << ", " << IMAG(_W2D.getFTHalf(i, j));
                    CLOG(FATAL, "LOGGER_RECO") << "_F2D * W2D : " << REAL(result) << ", " << IMAG(result);
                }
                ***/

                padDst.setFTHalf(_F2D.getFTHalf(i, j)
                               * _W2D.getFTHalf(i, j),
                                 i,
                                 j);
            }
        }

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming Padded Destination Image";
        BLOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming Padded Destination Image";

#endif

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(padDst.dataFT(), padDst.sizeFT());
#endif

        FFT fft;
        fft.bw(padDst, nThread);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK(padDst.dataRL(), padDst.sizeRL());
#endif

        Image imgDst;

        IMG_EXTRACT_RL(imgDst, padDst, 1.0 / _pf, nThread);

        dst.alloc(_N, _N, 1, RL_SPACE);

        #pragma omp parallel num_threads(nThread)
        IMAGE_FOR_EACH_PIXEL_RL(imgDst)
            dst.setRL(imgDst.getRL(i, j), i, j, 0);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK(dst.dataRL(), dst.sizeRL());
#endif
    }
    else if (_mode == MODE_3D)
    {
#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Volume";

#endif

        Volume padDst(_N * _pf, _N * _pf, _N * _pf, FT_SPACE);

        #pragma omp parallel num_threads(nThread)
        SET_0_FT(padDst);

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";

#endif

        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_FT(_F3D)
        {
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
            {
                padDst.setFTHalf(_F3D.getFTHalf(i, j, k)
                               * _W3D.getFTHalf(i, j ,k),
                                 i,
                                 j,
                                 k);
            }
        }

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming Padded Destination Volume";

#endif

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(padDst.dataFT(), padDst.sizeFT());
#endif

        FFT fft;
        fft.bw(padDst, nThread);
        
#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Extracting Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Extracting Destination Volume";

#endif

        VOL_EXTRACT_RL(dst, padDst, 1.0 / _pf, nThread);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifndef NAN_NO_CHECK
    SEGMENT_NAN_CHECK(dst.dataRL(), dst.sizeRL());
#endif

#ifdef VERBOSE_LEVEL_2

    ALOG(INFO, "LOGGER_RECO") << "Correcting Convolution Kernel";
    BLOG(INFO, "LOGGER_RECO") << "Correcting Convolution Kernel";

#endif

#ifdef RECONSTRUCTOR_MKB_KERNEL
    RFLOAT nf = MKB_RL(0, _a * _pf, _alpha);
#endif

    if (_mode == MODE_2D)
    {
        Image imgDst(_N, _N, RL_SPACE);

        SLC_EXTRACT_RL(imgDst, dst, 0);

        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        IMAGE_FOR_EACH_PIXEL_RL(imgDst)
        {
#ifdef RECONSTRUCTOR_MKB_KERNEL
            imgDst.setRL(imgDst.getRL(i, j)
                       / MKB_RL(NORM(i, j) / (_pf * _N),
                                _a * _pf,
                                _alpha)
                       * nf,
                         i,
                         j);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            imgDst.setRL(imgDst.getRL(i, j)
                       / TIK_RL(NORM(i, j) / (_pf * _N)),
                         i,
                         j);
#endif
        }

        SLC_REPLACE_RL(dst, imgDst, 0);
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_RL(dst)
        {
#ifdef RECONSTRUCTOR_MKB_KERNEL
            dst.setRL(dst.getRL(i, j, k)
                     / MKB_RL(NORM_3(i, j, k) / (_pf * _N),
                              _a * _pf,
                              _alpha)
                     * nf,
                       i,
                       j,
                       k);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            dst.setRL(dst.getRL(i, j, k)
                     / TIK_RL(NORM_3(i, j, k) / (_pf * _N)),
                       i,
                       j,
                       k);
#endif
        }
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

#ifdef VERBOSE_LEVEL_2

    ALOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";
    BLOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";

#endif

#endif // RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_REMOVE_NEG
    ALOG(INFO, "LOGGER_RECO") << "Removing Negative Values";
    BLOG(INFO, "LOGGER_RECO") << "Removing Negative Values";

    #pragma omp parallel for num_threads(nThread)
    REMOVE_NEG(dst);
#endif // RECONSTRUCT_REMOVE_NEG

#ifndef NAN_NO_CHECK
    SEGMENT_NAN_CHECK(dst.dataRL(), dst.sizeRL());
#endif
}

#ifdef GPU_RECONSTRUCT
int Reconstructor::getFSCSize()
{
    return _FSC.size();
}

vec Reconstructor::getFSC()
{
    return _FSC;
}

void Reconstructor::reconstructG(std::vector<int>& iGPU,
                                 std::vector<void*>& stream,
                                 Volume& dst,
                                 int nGPU,
                                 const unsigned int nThread)//LSQ:GPU 
{
    IF_MASTER return;

#ifdef VERBOSE_LEVEL_2
    IF_MODE_2D
    {
        ALOG(INFO, "LOGGER_RECO") << "Reconstructing Under 2D Mode";
        BLOG(INFO, "LOGGER_RECO") << "Reconstructing Under 2D Mode";
    }

    IF_MODE_3D
    {
        ALOG(INFO, "LOGGER_RECO") << "Reconstructing Under 3D Mode";
        BLOG(INFO, "LOGGER_RECO") << "Reconstructing Under 3D Mode";
    }
#endif
    
    RFLOAT* volumeT;
    RFLOAT* volumeW;
    RFLOAT* dev_W[nGPU];
    RFLOAT* dev_T[nGPU];

    if (_mode == MODE_2D)
    {
        size_t dimSize = _T2D.sizeFT();
        volumeT = (RFLOAT*)malloc(dimSize * sizeof(RFLOAT));
        volumeW = (RFLOAT*)malloc(dimSize * sizeof(RFLOAT));
	    
        #pragma omp parallel for num_threads(nThread)
        for(size_t i = 0; i < dimSize; i++)
	    {
            volumeT[i] = REAL(_T2D[i]);
            volumeW[i] = REAL(_W2D[i]);
	    }
        
        allocVolume(iGPU,
                    dev_T,
                    dev_W,
                    nGPU,
                    dimSize);
    }
    else if (_mode == MODE_3D)
    {
        size_t dimSize = _T3D.sizeFT();
        volumeT = (RFLOAT*)malloc(dimSize * sizeof(RFLOAT));
	    
        #pragma omp parallel for num_threads(nThread)
        for(size_t i = 0; i < dimSize; i++)
	    {
            volumeT[i] = REAL(_T3D[i]);
	    }

        allocVolume(iGPU,
                    dev_T,
                    dev_W,
                    nGPU,
                    dimSize);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
    // only in 3D mode, the MAP method is appropriate
    if (_MAP)
    {
        if (_mode == MODE_3D)
        {
            ExposePT(stream,
                     iGPU,
                     volumeT,
                     dev_T,
                     _FSC,
                     nGPU,
                     _maxRadius,
                     _pf,
                     _T3D.nSlcFT(),
                     _joinHalf,
                     WIENER_FACTOR_MIN_R);
            
            #pragma omp parallel for num_threads(nThread)
            for (size_t i = 0; i < _T3D.sizeFT(); i++)
	        {
                _T3D[i] = COMPLEX(volumeT[i], 0);
	        }
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }
#endif
 
    if (_gridCorr)
    {
#ifdef RECONSTRUCTOR_KERNEL_PADDING
        RFLOAT nf = MKB_RL(0, _a * _pf, _alpha);
#else
        RFLOAT nf = MKB_RL(0, _a, _alpha);
#endif
        if (_mode == MODE_3D)
        {
            gridCorrection(stream,
                           iGPU,
                           _C3D,
                           volumeT,
                           dev_W,
                           dev_T,
                           _kernelRL,
                           _fft,
                           nf,
                           nGPU,
                           _maxRadius,
                           _pf,
                           _N,
                           _MAP,
                           nThread);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }
    else
    {
        if (_mode == MODE_3D)
        {
            ExposeWT(stream,
                     iGPU,
                     volumeT,
                     dev_W,
                     dev_T,
                     nGPU,
                     _maxRadius,
                     _pf,
                     _MAP,
                     _T3D.nSlcFT());
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }

    free(volumeT);
    
    if (_mode == MODE_3D)
    {
#ifdef VERBOSE_LEVEL_2
        ALOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Volume";

        ALOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";

#endif

        Volume padDst(_N * _pf, 
                      _N * _pf, 
                      _N * _pf, 
                      FT_SPACE);
        //Volume padDstR(_N * _pf, _N * _pf, _N * _pf, RL_SPACE);

        ExposePFW(stream,
                  iGPU,
                  padDst,
                  _F3D,
                  dev_W,
                  nGPU,
                  _maxRadius,
                  _pf);
        
        FFT fft;

        fft.bw(padDst, nThread);//LSQ: the parameter maybe mistake for it's bw originally instead of bwMT
        
        //ExposePF(gpuIdx,
        //         padDst,
        //         padDstR,
        //         _F3D,
        //         volumeW,
        //         _maxRadius,
        //         _pf);
        
        //padDst.clearFT();

        //dst.alloc(AROUND((1.0 / _pf) * padDstR.nColRL()), 
        //          AROUND((1.0 / _pf) * padDstR.nRowRL()), 
        //          AROUND((1.0 / _pf) * padDstR.nSlcRL()), 
        //          FT_SPACE);

        //Volume dstN;
        //dstN.alloc(AROUND((1.0 / _pf) * padDstR.nColRL()), 
        //           AROUND((1.0 / _pf) * padDstR.nRowRL()), 
        //           AROUND((1.0 / _pf) * padDstR.nSlcRL()), 
        //           RL_SPACE);
        //
        //VOLUME_FOR_EACH_PIXEL_RL(dstN) 
        //    dstN.setRL(padDstR.getRL(i, j, k), i, j, k);

        //padDstR.clearRL();
        
        dst.alloc(AROUND((1.0 / _pf) * padDst.nColRL()), 
                  AROUND((1.0 / _pf) * padDst.nRowRL()), 
                  AROUND((1.0 / _pf) * padDst.nSlcRL()), 
                  RL_SPACE);

        #pragma omp parallel for num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_RL(dst) 
            dst.setRL(padDst.getRL(i, j, k), i, j, k);
        
        padDst.clearRL();

        RFLOAT nf = 0;
#ifdef RECONSTRUCTOR_MKB_KERNEL
        nf = MKB_RL(0, _a * _pf, _alpha);
#endif
        int padSize = _pf * _N;
        //int dim = dstN.nSlcRL();
        int dim = dst.nSlcRL();
        int slcSize = (dim / 2 + 1) * (dim / 2 + 1);
        RFLOAT *mkbRL = new RFLOAT[slcSize * (dim / 2 + 1)];
        
        #pragma omp parallel for num_threads(nThread)
        for (int k = 0; k <= dim / 2; k++) 
            for (int j = 0; j <= dim / 2; j++) 
                for (int i = 0; i <= dim / 2; i++) 
                {
                    size_t index = k * slcSize + j * (dim / 2 + 1) + i;
#ifdef RECONSTRUCTOR_MKB_KERNEL
                    mkbRL[index] = MKB_RL(NORM_3(i, j, k) / padSize,
                                      _a * _pf,
                                      _alpha);
#endif
#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                    mkbRL[index] = TIK_RL(NORM_3(i, j, k) / padSize);
#endif
                }
    
        ExposeCorrF(stream,
                    iGPU,
                    dst,
                    mkbRL,
                    nf,
                    nGPU);
#ifdef RECONSTRUCTOR_REMOVE_NEG
        REMOVE_NEG(dst);
#endif
        delete[] mkbRL;
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

#ifdef VERBOSE_LEVEL_2
    ALOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";
    BLOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";
#endif
}

#endif // GPU_RECONSTRUCT

void Reconstructor::allReduceF()
{

    ALOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere A";
    BLOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere B";

    MPI_Barrier(_hemi);

    if (_mode == MODE_2D)
    {

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(&_F2D[0], _F2D.sizeFT());
#endif

        MPI_Allreduce_Large(&_F2D[0],
                            2 * _F2D.sizeFT(),
                            TS_MPI_DOUBLE,
                            MPI_SUM,
                            _hemi);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(&_F2D[0], _F2D.sizeFT());
#endif

    }
    else if (_mode == MODE_3D)
    {

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(&_F3D[0], _F3D.sizeFT());
#endif

        MPI_Allreduce_Large(&_F3D[0],
                            2 * _F3D.sizeFT(),
                            TS_MPI_DOUBLE,
                            MPI_SUM,
                            _hemi);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(&_F3D[0], _F3D.sizeFT());
#endif
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    MPI_Barrier(_hemi);
}

void Reconstructor::allReduceT(const unsigned int nThread)
{
    ALOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere A";
    BLOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere B";

    MPI_Barrier(_hemi);

    if (_mode == MODE_2D)
    {

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(&_T2D[0], _T2D.sizeFT());
#endif

        MPI_Allreduce_Large(&_T2D[0],
                            2 * _T2D.sizeFT(),
                            TS_MPI_DOUBLE,
                            MPI_SUM,
                            _hemi);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(&_T2D[0], _T2D.sizeFT());
#endif

    }
    else if (_mode == MODE_3D)
    {

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(&_T3D[0], _T3D.sizeFT());
#endif

        MPI_Allreduce_Large(&_T3D[0],
                            2 * _T3D.sizeFT(),
                            TS_MPI_DOUBLE,
                            MPI_SUM,
                            _hemi);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(&_T3D[0], _T3D.sizeFT());
#endif
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    MPI_Barrier(_hemi);
}

void Reconstructor::allReduceO()
{
    ALOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere A";
    BLOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere B";

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_ox,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_oy,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_oz,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);
}

void Reconstructor::allReduceCounter()
{
    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_counter,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  _hemi);
}

void Reconstructor::allReduceTau()
{
    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _tau.data(),
                  _tau.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);
}

RFLOAT Reconstructor::checkC(const unsigned int nThread) const
{
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
    RFLOAT diff = 0;

    int counter = 0;
    
    // TODO use REDUCTION

    if (_mode == MODE_2D)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        IMAGE_FOR_EACH_PIXEL_FT(_C2D)
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
            {
                #pragma omp atomic
                diff += fabs(ABS(_C2D.getFT(i, j)) - 1);
                #pragma omp atomic
                counter += 1;
            }
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_FT(_C3D)
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
            {
                #pragma omp atomic
                diff += fabs(ABS(_C3D.getFT(i, j, k)) - 1);
                #pragma omp atomic
                counter += 1;
            }
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    return diff / counter;
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
    if (_mode == MODE_2D)
    {
        vector<RFLOAT> diff(_C2D.sizeFT(), 0);
        
        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        IMAGE_FOR_EACH_PIXEL_FT(_C2D)
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
                diff[_C2D.iFTHalf(i, j)] = fabs(ABS(_C2D.getFTHalf(i, j)) - 1);

        return *std::max_element(diff.begin(), diff.end());
    }
    else if (_mode == MODE_3D)
    {
        vector<RFLOAT> diff(_C3D.sizeFT(), 0);

        #pragma omp parallel for schedule(dynamic) num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_FT(_C3D)
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
                diff[_C3D.iFTHalf(i, j, k)] = fabs(ABS(_C3D.getFTHalf(i, j, k)) - 1);

        return *std::max_element(diff.begin(), diff.end());
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
#endif
}

void Reconstructor::convoluteC(const unsigned int nThread)
{
#ifdef RECONSTRUCTOR_KERNEL_PADDING
    RFLOAT nf = MKB_RL(0, _a * _pf, _alpha);
#else
    RFLOAT nf = MKB_RL(0, _a, _alpha);
#endif

    if (_mode == MODE_2D)
    {
#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(_C2D.dataFT(), _C2D.sizeFT());
#endif

        _fft.bwExecutePlan(_C2D,nThread);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK(_C2D.dataRL(), _C2D.sizeRL());
#endif

        #pragma omp parallel for num_threads(nThread)
        IMAGE_FOR_EACH_PIXEL_RL(_C2D)
            _C2D.setRL(_C2D.getRL(i, j)
                     * _kernelRL(QUAD(i, j) / TSGSL_pow_2(_N * _pf))
                     / nf,
                       i,
                       j);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK(_C2D.dataRL(), _C2D.sizeRL());
#endif

        _fft.fwExecutePlan(_C2D);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(_C2D.dataFT(), _C2D.sizeFT());
#endif

        _C2D.clearRL();
    }
    else if (_mode == MODE_3D)
    {
#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(_C3D.dataFT(), _C3D.sizeFT());
#endif

        _fft.bwExecutePlan(_C3D, nThread);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK(_C3D.dataRL(), _C3D.sizeRL());
#endif

        #pragma omp parallel for num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_RL(_C3D)
            _C3D.setRL(_C3D.getRL(i, j, k)
                     * _kernelRL(QUAD_3(i, j, k) / TSGSL_pow_2(_N * _pf))
                     / nf,
                       i,
                       j,
                       k);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK(_C3D.dataRL(), _C3D.sizeRL());
#endif

        _fft.fwExecutePlan(_C3D);

#ifndef NAN_NO_CHECK
        SEGMENT_NAN_CHECK_COMPLEX(_C3D.dataFT(), _C3D.sizeFT());
#endif

        _C3D.clearRL();
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
}

void Reconstructor::symmetrizeF(const unsigned int nThread)
{
    if (_sym != NULL)
        SYMMETRIZE_FT(_F3D, _F3D, *_sym, _maxRadius * _pf + 1, LINEAR_INTERP, nThread);
    else
        CLOG(WARNING, "LOGGER_SYS") << "Symmetry Information Not Assigned in Reconstructor";
}

void Reconstructor::symmetrizeT(const unsigned int nThread)
{
    if (_sym != NULL)
        SYMMETRIZE_FT(_T3D, _T3D, *_sym, _maxRadius * _pf + 1, LINEAR_INTERP, nThread);
    else
        CLOG(WARNING, "LOGGER_SYS") << "Symmetry Information Not Assigned in Reconstructor";
}

void Reconstructor::symmetrizeO()
{
    if (_sym != NULL)
    {
        dmat33 L, R;

        dvec3 result = dvec3(_ox, _oy, _oz);

        for (int i = 0; i < _sym->nSymmetryElement(); i++)
        {
            _sym->get(L, R, i);

            result += R * dvec3(_ox, _oy, _oz);
        }

        _counter *= (1 + _sym->nSymmetryElement());

        _ox = result(0);
        _oy = result(1);
        _oz = result(2);
    }
    else
        CLOG(WARNING, "LOGGER_SYS") << "Symmetry Information Not Assigned in Reconstructor";
}
/*
void Reconstructor::getF3D(Volume& dst)
{
   dst.swap(_F3D);
}

 void Reconstructor::getT3D(Volume& dst)
{
   dst.swap(_T3D);
}

void Reconstructor::setF3D(Volume& src)
{
   _F3D.swap(src);
}

 void Reconstructor::setT3D(Volume& src)
{
   _T3D.swap(src);
} 
*/
Volume& Reconstructor::getF3D()
{
   return _F3D;
}

Volume& Reconstructor::getT3D()
{
   return _T3D;
}
