/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang, Bing Li, Heng Guo
 * Dependency:
 * Test:
 * Execution: * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Optimiser.h"
#include "core/serializeImage.h"
#include "core/memoryDistribution.h"

#ifdef GPU_VERSION

void Optimiser::setGPUEnv()
{
    _nGPU = 0;
    _iGPU.clear();
    
    NT_MASTER
    {
        int rank, size;
        MPI_Comm_size(_hemi, &size);
        MPI_Comm_rank(_hemi, &rank);
        
        _gpusPerProcess.resize(size);
        
        int *recvbuf = (int*)malloc(size * sizeof(int));
        memset(recvbuf, 0, size * sizeof(int));

        readGPUPARA(_para.gpus,
                    _iGPU,
                    _nGPU);
        
        gpuCheck(_stream,
                 _iGPU,
                 _nGPU);

        recvbuf[rank] = _nGPU;

        MPI_Allgather(&_nGPU, 1, MPI_INT, recvbuf, 1, MPI_INT, _hemi);
        
        for (int i = 0; i < size; i++)
            _gpusPerProcess[i] = recvbuf[i];
    
        free(recvbuf);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void Optimiser::destoryGPUEnv()
{
    NT_MASTER
    {
        gpuEnvDestory(_stream,
                      _iGPU,
                      _nGPU);

        _nGPU = 0;
        _gpusPerProcess.clear();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}
#endif

Optimiser::~Optimiser()
{
    clear();

    _fftImg.fwDestroyPlan();
    _fftImg.bwDestroyPlan();
}

OptimiserPara& Optimiser::para()
{
    return _para;
}

void Optimiser::setPara(const OptimiserPara& para)
{
    _para = para;
}

void Optimiser::init()
{

#ifdef GPU_VERSION
    MLOG(INFO, "LOGGER_GPU") << "Setting Up GPU Devices for Each Process";
    setGPUEnv();
#endif

    if (_para.mode == MODE_2D)
    {
        MLOG(INFO, "LOGGER_INIT") << "The Program is Running under 2D Mode";
    }
    else if (_para.mode == MODE_3D)
    {
        MLOG(INFO, "LOGGER_INIT") << "The Program is Running under 3D Mode";
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    MLOG(INFO, "LOGGER_INIT") << "Setting MPI Environment of _model";
    _model.setMPIEnv(_commSize, _commRank, _hemi, _slav);

    MLOG(INFO, "LOGGER_INIT") << "Setting up Symmetry";
    _sym.init(_para.sym);
    MLOG(INFO, "LOGGER_INIT") << "Symmetry Group : " << _sym.pgGroup();
    MLOG(INFO, "LOGGER_INIT") << "Symmetry Order : " << _sym.pgOrder();
    MLOG(INFO, "LOGGER_INIT") << "Number of Symmetry Element : " << 1 + _sym.nSymmetryElement();

    MLOG(INFO, "LOGGER_INIT") << "Number of Class(es): " << _para.k;

    MLOG(INFO, "LOGGER_INIT") << "Initialising FFTW Plan";

    _fftImg.fwCreatePlan(_para.size, _para.size, _para.nThreadsPerProcess);
    _fftImg.bwCreatePlan(_para.size, _para.size, _para.nThreadsPerProcess);

    MLOG(INFO, "LOGGER_INIT") << "Initialising Class Distribution";
    _cDistr.resize(_para.k);

    if (_para.mode == MODE_3D)
    {
        MLOG(INFO, "LOGGER_INIT") << "Modifying the Number of Sampling Points Used in Global Search Scanning Phase";

        _para.mS = GSL_MAX_INT(_para.mS, MIN_M_S * (1 + _sym.nSymmetryElement()));
    }

    /***
    Symmetry sym;
    sym.init("I");
    ***/

    MLOG(INFO, "LOGGER_INIT") << "Passing Parameters to _model";
    _model.init(_para.mode,
                _para.gSearch,
                _para.lSearch,
                _para.cSearch,
                _para.coreFSC,
                AROUND(_para.maskRadius / _para.pixelSize),
                _para.maskFSC,
                &_mask,
                _para.goldenStandard,
                _para.k,
                _para.size,
                0,
                _para.pf,
                _para.pixelSize,
                _para.a,
                _para.alpha,
                // &sym);
                &_sym);

    MLOG(INFO, "LOGGER_INIT") << "Determining Search Type";

    if (_para.gSearch)
    {
        _searchType = SEARCH_TYPE_GLOBAL;

        MLOG(INFO, "LOGGER_INIT") << "Search Type : Global";
    }
    else if (_para.lSearch)
    {
        _searchType = SEARCH_TYPE_LOCAL;

        MLOG(INFO, "LOGGER_INIT") << "Search Type : Local";
    }
    else if (_para.cSearch)
    {
        _searchType = SEARCH_TYPE_CTF;

        MLOG(INFO, "LOGGER_INIT") << "Search Type : CTF";
    }
    else
    {
        _searchType = SEARCH_TYPE_STOP;

        MLOG(INFO, "LOGGER_INIT") << "Search Type : Stop";
    }

    _model.setSearchType(_searchType);

    /***
    MLOG(INFO, "LOGGER_INIT") << "Initialising Upper Boundary of Reconstruction";

    _model.updateRU();
    ***/

    /***
    MLOG(INFO, "LOGGER_INIT") << "Information Under "
                              << _para.ignoreRes
                              << " Angstrom will be Ignored during Comparison";

                              ***/
    _rL = FLOOR(resA2P(1.0 / _para.ignoreRes, _para.size, _para.pixelSize));
    //_rL = 0;
    //_rL = 1.5;
    //_rL = 3.5;
    //_rL = 6;
    //_rL = resA2P(1.0 / (2 * _para.maskRadius), _para.size, _para.pixelSize);
    //_rL = resA2P(1.0 / _para.maskRadius, _para.size, _para.pixelSize);

    MLOG(INFO, "LOGGER_INIT") << "Information Under "
                              << _rL
                              << " Pixels in Fourier Space will be Ignored during Comparison";

    MLOG(INFO, "LOGGER_INIT") << "Checking Radius of Mask";

    /***
    CLOG(INFO, "LOGGER_SYS") << "_para.size / 2 = " << _para.size / 2;
    CLOG(INFO, "LOGGER_SYS") << "CEIL(_para.maskRadius / _para.pxielSize) = "
                             << CEIL(_para.maskRadius / _para.pixelSize);
    ***/

    if (_para.size / 2 - CEIL(_para.maskRadius / _para.pixelSize) < 1)
    {
        MLOG(WARNING, "LOGGER_SYS") << "Inproper radius of mask, modified it to half of image size.";
        _para.maskRadius = FLOOR(_para.size * _para.pixelSize / 2);
    }

    //_rS = AROUND(resA2P(1.0 / _para.sclCorRes, _para.size, _para.pixelSize)) + 1;

    if (_para.gSearch)
    {
        MLOG(INFO, "LOGGER_INIT") << "Information Under "
                                  << _para.sclCorRes
                                  << " Angstrom will be Used for Performing Intensity Scale Correction";

        _rS = AROUND(resA2P(1.0 / _para.sclCorRes, _para.size, _para.pixelSize)) + 1;

        MLOG(INFO, "LOGGER_INIT") << "Information Under "
                                  << _rS
                                  << " (Pixel) will be Used for Performing Intensity Scale Correction";

    }
    else
    {
         MLOG(INFO, "LOGGER_INIT") << "Information Under "
                                   << _para.initRes
                                   << " Angstrom will be Used for Performing Intensity Scale Correction";

         _rS = AROUND(resA2P(1.0 / _para.initRes, _para.size, _para.pixelSize)) + 1;

         MLOG(INFO, "LOGGER_INIT") << "Information Under "
                                   << _rS
                                   << " (Pixel) will be Used for Performing Intensity Scale Correction";
    }

    MLOG(INFO, "LOGGER_INIT") << "Setting Frequency Upper Boundary during Global Search";

    RFLOAT globalSearchRes = GSL_MIN_DBL(_para.globalSearchRes,
                                             R_GLOBAL_FACTOR * _para.maskRadius / pow(1 + _sym.nSymmetryElement(), 1.0 / 3));

    _model.setRGlobal(AROUND(resA2P(1.0 / globalSearchRes,
                                        _para.size,
                                        _para.pixelSize)) + 1);

    MLOG(INFO, "LOGGER_INIT") << "Global Search Resolution Limit : "
                                  << globalSearchRes
                                  << " (Angstrom), "
                                  << _model.rGlobal()
                                  << " (Pixel)";

    MLOG(INFO, "LOGGER_INIT") << "Setting Parameters: _r, _iter";

    _iter = 0;

    _r = AROUND(resA2P(1.0 / _para.initRes, _para.size, _para.pixelSize)) + 1;
    _model.setR(_r);
    _model.setRInit(_r);
    //_model.setRPrev(_r);
    //_model.setRT(_r);

    MLOG(INFO, "LOGGER_INIT") << "Setting MPI Environment of _exp";
    _db.setMPIEnv(_commSize, _commRank, _hemi, _slav);

    MLOG(INFO, "LOGGER_INIT") << "Opening Database File";
    //_db.openDatabase(newDatabaseName);
    _db.openDatabase(_para.db, _para.outputDirFullPath,  _commRank);

    MLOG(INFO, "LOGGER_INIT") << "Shuffling Particles";
    _db.shuffle();

    MLOG(INFO, "LOGGER_INIT") << "Assigning Particles to Each Process";
    _db.assign();

    MLOG(INFO, "LOGGER_INIT") << "Indexing the Offset in Database";
    _db.index();

    MLOG(INFO, "LOGGER_INIT") << "Appending Initial References into _model";
    initRef();

    MLOG(INFO, "LOGGER_INIT") << "Broadcasting Total Number of 2D Images";
    bCastNPar();

    MLOG(INFO, "LOGGER_INIT") << "Total Number of Images: " << _nPar;

    /***
    int nClass = FLOOR(_nPar * CLASS_BALANCE_FACTOR / MIN_N_IMAGES_PER_CLASS);

    if (nClass < _para.k)
    {
        MLOG(FATAL, "LOGGER_INIT") << "According to Total Number of Images, "
                                   << "Maximum "
                                   << nClass
                                   << " Classes is Recommended for Classification";
        abort();
    }
    ***/

    if ((_para.maskFSC) ||
        (_para.performMask && !_para.autoMask))
    {
        MLOG(INFO, "LOGGER_INIT") << "Reading Mask";

        initMask();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_INIT") << "Mask Read";
#endif
    }

    NT_MASTER
    {
        ALOG(INFO, "LOGGER_INIT") << "Initialising IDs of 2D Images";
        BLOG(INFO, "LOGGER_INIT") << "Initialising IDs of 2D Images";

        initID();
        
        _nChange.clear();
        _nChange.resize(_ID.size());
        
        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            _nChange[l] = 0;
        }

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "IDs of 2D Images Initialised";
        BLOG(INFO, "LOGGER_INIT") << "IDs of 2D Images Initialised";
#endif

        ALOG(INFO, "LOGGER_INIT") << "Assigning Memory Distribution for MemoryBazaar Objects";
        BLOG(INFO, "LOGGER_INIT") << "Assigning Memory Distribution for MemoryBazaar Objects";

       assignMemoryDistribution(_md,
                                 TSGSL_MAX_RFLOAT(_para.maximumMemoryUsagePerProcessGB * GIGABYTE - (RFLOAT)referenceMemorySize(_para.size, _para.pf, _para.mode, _para.k), GIGABYTE),
                                 serializeSize(Image(_para.size, _para.size, RL_SPACE)),
                                 1,
                                 _ID.size(),
                                 3 * omp_get_max_threads());

        std::cout << "md.nStallImg = " << _md.nStallImg << std::endl;
        std::cout << "md.nStallImgOri = " << _md.nStallImgOri << std::endl;
        std::cout << "md.nStallDatPR = " << _md.nStallDatPR << std::endl;
        std::cout << "md.nStallDatPI = " << _md.nStallDatPI << std::endl;
        std::cout << "md.nStallCtfP = " << _md.nStallCtfP << std::endl;
        std::cout << "md.nStallSigRcpP = " << _md.nStallSigRcpP << std::endl;

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "Memory Distribution for MemoryBazaar Objects Assigned";
        BLOG(INFO, "LOGGER_INIT") << "Memory Distribution for MemoryBazaar Objects Assigned";
#endif

        ALOG(INFO, "LOGGER_INIT") << "Setting Parameter _N";
        BLOG(INFO, "LOGGER_INIT") << "Setting Parameter _N";

        allReduceN();

        ALOG(INFO, "LOGGER_INIT") << "Number of Images in Hemisphere A: " << _N;
        BLOG(INFO, "LOGGER_INIT") << "Number of Images in Hemisphere B: " << _N;

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "Parameter _N Set";
        BLOG(INFO, "LOGGER_INIT") << "Parameter _N Set";
#endif

#ifdef OPTIMISER_LOG_MEM_USAGE
        CHECK_MEMORY_USAGE("Before Initialsing 2D Images");
#endif

        ALOG(INFO, "LOGGER_INIT") << "Initialising 2D Images";
        BLOG(INFO, "LOGGER_INIT") << "Initialising 2D Images";

        initImg();

#ifdef OPTIMISER_LOG_MEM_USAGE
        CHECK_MEMORY_USAGE("After Initialising 2D Images");
#endif

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "2D Images Initialised";
        BLOG(INFO, "LOGGER_INIT") << "2D Images Initialised";
#endif

#ifdef OPTIMISER_LOG_MEM_USAGE
        CHECK_MEMORY_USAGE("Before Initialsing CTFs");
#endif

        ALOG(INFO, "LOGGER_INIT") << "Generating CTFs";
        BLOG(INFO, "LOGGER_INIT") << "Generating CTFs";

        initCTF();

#ifdef OPTIMISER_LOG_MEM_USAGE
        CHECK_MEMORY_USAGE("After Initialsing CTFs");
#endif

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "CTFs Generated";
        BLOG(INFO, "LOGGER_INIT") << "CTFs Generated";
#endif

        ALOG(INFO, "LOGGER_INIT") << "Initialising Particle Filters";
        BLOG(INFO, "LOGGER_INIT") << "Initialising Particle Filters";

        initParticles();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_INIT") << "Particle Filters Initialised";
        BLOG(INFO, "LOGGER_INIT") << "Particle Filters Initialised";
#endif

        if (!_para.gSearch)
        {
            ALOG(INFO, "LOGGER_INIT") << "Loading Particle Filters";
            BLOG(INFO, "LOGGER_INIT") << "Loading Particle Filters";

            loadParticles();

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(_hemi);

            ALOG(INFO, "LOGGER_INIT") << "Particle Filters Loaded";
            BLOG(INFO, "LOGGER_INIT") << "Particle Filters Loaded";
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION

            ALOG(INFO, "LOGGER_INIT") << "Re-Centring Images";
            BLOG(INFO, "LOGGER_INIT") << "Re-Centring Images";

            reCentreImg();

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(_hemi);

            ALOG(INFO, "LOGGER_INIT") << "Images Re-Centred";
            BLOG(INFO, "LOGGER_INIT") << "Images Re-Centred";
#endif
#endif

#ifdef OPTIMISER_MASK_IMG

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Masking Images";
#ifdef GPU_VERSION
            reMaskImgG();
#else
            reMaskImg();
#endif

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(_hemi);

            ALOG(INFO, "LOGGER_INIT") << "Images Re-Masked";
            BLOG(INFO, "LOGGER_INIT") << "Images Re-Masked";
#endif
#endif
        }
    }

    MLOG(INFO, "LOGGER_INIT") << "Broadcasting Information of Groups";

    bcastGroupInfo();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_INIT") << "Information of Groups Broadcasted";
#endif

    NT_MASTER
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Solvent Flattening";

        if ((_para.globalMask) || (_searchType != SEARCH_TYPE_GLOBAL))
            solventFlatten(_para.performMask);
        else
            solventFlatten(false);

        ALOG(INFO, "LOGGER_INIT") << "Setting Up Projectors and Reconstructors of _model";
        BLOG(INFO, "LOGGER_INIT") << "Setting Up Projectors and Reconstructors of _model";

        _model.initProjReco(_para.nThreadsPerProcess);
    }

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_INIT") << "Projectors and Reconstructors Set Up";
#endif

    if (strcmp(_para.initModel, "") != 0)
    {
        MLOG(INFO, "LOGGER_INIT") << "Re-balancing Intensity Scale";

        if (_para.gSearch)
        {
            MLOG(INFO, "LOGGER_INIT") << "Re-balancing Intensity Scale Using Random Projections";

            correctScale(true, false, false);
        }
        else
        {
            MLOG(INFO, "LOGGER_INIT") << "Re-balancing Intensity Scale Using Given Projections";

            correctScale(true, true, false);
        }

        NT_MASTER
        {
            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Refreshing Projectors After Intensity Scale Correction";
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Refreshing Projectors After Intensity Scale Correction";

            _model.refreshProj(_para.nThreadsPerProcess);
        }

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_INIT") << "Intensity Scale Re-balanced";
#endif
    }

    NT_MASTER
    {
        if (_para.gSearch)
        {
            ALOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma";
            BLOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma";

            initSigma();
        }
        else
        {
            ALOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma Using Given Projections";
            BLOG(INFO, "LOGGER_INIT") << "Estimating Initial Sigma Using Given Projections";

            allReduceSigma(false);
        }
    }

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_INIT") << "Sigma Initialised";
#endif
}

struct Sp
{
    RFLOAT _w;
    size_t _k;
    size_t _iR;
    size_t _iT;

    Sp() : _w(-TS_MAX_RFLOAT_VALUE), _k(0), _iR(0), _iT(0) {};

    Sp(const RFLOAT w,
       const size_t k,
       const size_t iR,
       const size_t iT)
    {
        _w = w;
        _k = k;
        _iR = iR;
        _iT = iT;
    };
};

struct SpWeightComparator
{
    bool operator()(const Sp& a, const Sp& b) const
    {
        return a._w > b._w;
    }
};

void Optimiser::expectation()
{
    IF_MASTER return;

    int nPer = 0;

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space for Pre-calcuation in Expectation";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space for Pre-calcuation in Expectation";

    allocPreCalIdx(_r, _rL);

    if (_searchType == SEARCH_TYPE_GLOBAL)
    {
        if (_searchType != SEARCH_TYPE_CTF)
        {
#ifndef PIXEL_MAJOR
            allocPreCal(true, false, false);
#else
            allocPreCal(true, true, false);
#endif
        }
        else
        {
#ifndef PIXEL_MAJOR
            allocPreCal(true, false, true);
#else
            allocPreCal(true, true, true);
#endif
        }

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space for Pre-calcuation in Expectation Allocated";
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space for Pre-calcuation in Expectation Allocated";

        // initialse a particle filter

        int nR;
        if (_para.mode == MODE_2D)
        {
            nR = _para.mS;
        }
        else if (_para.mode == MODE_3D)
        {
            nR = _para.mS / (1 + _sym.nSymmetryElement());
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

        int nT = GSL_MAX_INT(30,
                             AROUND(M_PI
                                  * TSGSL_pow_2(_para.transS
                                            * TSGSL_cdf_chisq_Qinv(0.5, 2))
                                  * _para.transSearchFactor));

        RFLOAT scanMinStdR;
        if (_para.mode == MODE_2D)
        {
            scanMinStdR = 1.0 / _para.mS;
        }
        else if (_para.mode == MODE_3D)
        {
            scanMinStdR =  pow(_para.mS, -1.0 / 3);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

        RFLOAT scanMinStdT = 1.0
                           / TSGSL_cdf_chisq_Qinv(INIT_OUTSIDE_CONFIDENCE_AREA, 2)
                           / sqrt(_para.transSearchFactor * M_PI);

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Minimum Standard Deviation of Rotation in Scanning Phase: "
                                   << scanMinStdR;
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Minimum Standard Deviation of Translation in Scanning Phase: "
                                   << scanMinStdT;

        Particle par = _parAll[0].copy();

        par.reset(nR, nT, 1);

        for (size_t t = 0; t < (size_t)_para.k; t++)
        {
            FOR_EACH_2D_IMAGE
            {
                par.copy(_parAll[t * _ID.size() + l]);
            }
        }

        dmat22 rot2D;
        dmat33 rot3D;
        dvec2 t;

        // generate "translations"

        Complex* traP = (Complex*)TSFFTW_malloc(nT * _nPxl * sizeof(Complex));

        #pragma omp parallel for schedule(dynamic) private(t)
        for (size_t m = 0; m < (size_t)nT; m++)
        {
            par.t(t, m);

            translate(traP + m * _nPxl,
                      t(0),
                      t(1),
                      _para.size,
                      _para.size,
                      _iCol,
                      _iRow,
                      _nPxl,
                      1);
        }

        mat wC = mat::Zero(_ID.size(), _para.k);

        vector<mat> wR(_para.k, mat::Zero(_ID.size(), nR));
        vector<mat> wT(_para.k, mat::Zero(_ID.size(), nT));

        _nR = 0;

        // the same arrangement with _par
        RFLOAT* baseLine = new RFLOAT[_ID.size() * _para.k];

        for (size_t t = 0; t < (size_t)_para.k; t++)
        {
            #pragma omp parallel for
            FOR_EACH_2D_IMAGE
            {
                baseLine[t * _ID.size() + l] = GSL_NAN;
            }
        }

        // determining batch size

        size_t batchSize = GSL_MAX_INT(GSL_MIN_INT(AROUND(SCANNING_PHASE_BATCH_MEMORY_USAGE / _nPxl / sizeof(Complex)), _para.k * nR), 1);

        CLOG(INFO, "LOGGER_SYS") << "batchSize = " << batchSize;


        // t -> class
        // m -> rotation
        // n -> translation

        RFLOAT* poolPriRotPR = (RFLOAT*)TSFFTW_malloc(batchSize * _nPxl * sizeof(RFLOAT));
        RFLOAT* poolPriRotPI = (RFLOAT*)TSFFTW_malloc(batchSize * _nPxl * sizeof(RFLOAT));
        RFLOAT* poolPriAllPR = (RFLOAT*)TSFFTW_malloc(batchSize * _nPxl * sizeof(RFLOAT));
        RFLOAT* poolPriAllPI = (RFLOAT*)TSFFTW_malloc(batchSize * _nPxl * sizeof(RFLOAT));

        // will be converted to dvp, for storing logDataVSPrior result for each image against a certain prior (projection from reference in a certain rotation and translation)
        RFLOAT *poolSIMDResult = (RFLOAT*)TSFFTW_malloc(batchSize * _ID.size() * sizeof(RFLOAT));

        /***
        for (size_t t = 0; t < (size_t)_para.k; t++)
        {
            for (size_t m = 0; m < (size_t)(nR - 1) / batchSize + 1; m++)
        ***/
        for (size_t tm = 0; tm < (size_t)(_para.k * nR - 1) / batchSize + 1; tm++)
        {
                size_t chunk = ((tm + 1) * batchSize <= nR * _para.k) ? batchSize : (nR * _para.k - tm * batchSize);

                // CLOG(INFO, "LOGGER_SYS") << "tm = " << tm;
                // CLOG(INFO, "LOGGER_SYS") << "chunk = " << chunk;

                RFLOAT* priRotPR = poolPriRotPR;
                RFLOAT* priRotPI = poolPriRotPI;
                RFLOAT* priAllPR = poolPriAllPR;
                RFLOAT* priAllPI = poolPriAllPI;

                RFLOAT* SIMDResult = poolSIMDResult;

                // perform projection

                #pragma omp parallel for schedule(dynamic) private(rot2D, rot3D)
                for (size_t tnp = 0; tnp < chunk; tnp++)
                {
                    size_t t = (tm * batchSize + tnp) / nR; // class index
                    size_t m = (tm * batchSize + tnp) - t * nR; // rotation index

                    if (_para.mode == MODE_2D)
                    {
                        // par.rot(rot2D, m * batchSize + np);
                        par.rot(rot2D, m);

                        _model.proj(t).project(priRotPR + tnp * _nPxl, priRotPI + tnp * _nPxl, rot2D, _iCol, _iRow, _nPxl, 1);
                    }
                    else if (_para.mode == MODE_3D)
                    {
                        //par.rot(rot3D, m * batchSize + np);
                        par.rot(rot3D, m);

                        _model.proj(t).project(priRotPR + tnp * _nPxl, priRotPI + tnp * _nPxl, rot3D, _iCol, _iRow, _nPxl, 1);
                    }
                    else
                   {
                        REPORT_ERROR("INEXISTENT MODE");

                        abort();
                    }
                }

                for (size_t n = 0; n < (size_t)nT; n++)
                {
                    #pragma omp parallel for schedule(dynamic)
                    for (size_t tnp = 0; tnp < chunk; tnp++)
                    {
                        for (int i = 0; i < _nPxl; i++)
                        {
                            priAllPR[i + tnp * _nPxl] = traP[_nPxl * n + i].dat[0] * priRotPR[i + tnp * _nPxl] - traP[_nPxl * n + i].dat[1] * priRotPI[i + tnp * _nPxl];

                            priAllPI[i + tnp * _nPxl] = traP[_nPxl * n + i].dat[1] * priRotPR[i + tnp * _nPxl] + traP[_nPxl * n + i].dat[0] * priRotPI[i + tnp * _nPxl];
                        }
                    }

                    // higher logDataVSPrior, higher probability

                    memset(SIMDResult, '\0', batchSize * _ID.size() * sizeof(RFLOAT));

                    RFLOAT* dvp = logDataVSPrior(_datPR,
                                                 _datPI,
                                                 priAllPR,
                                                 priAllPI,
                                                 _ctfP,
                                                 _sigRcpP,
                                                 _ID.size(),
                                                 chunk,
                                                 _nPxl,
                                                 SIMDResult,
                                                 omp_get_max_threads());

#ifndef NAN_NO_CHECK

                    SEGMENT_NAN_CHECK(dvp, _ID.size());

#endif

                    #pragma omp parallel for schedule(dynamic)
                    FOR_EACH_2D_IMAGE
                    {
                        for (size_t tnp = 0; tnp < chunk; tnp++)
                        {
                            size_t t = (tm * batchSize + tnp) / nR; // class index
                            size_t m = (tm * batchSize + tnp) - t * nR; // rotation index

                            size_t baseLineIndex = t * _ID.size() + l;

                            if (TSGSL_isnan(baseLine[baseLineIndex]))
                            {
                                baseLine[baseLineIndex] = dvp[tnp * _ID.size() + l];
                            }
                            else
                            {
                                if (dvp[tnp * _ID.size() + l] > baseLine[baseLineIndex])
                                {
                                    RFLOAT offset = dvp[tnp * _ID.size() + l] - baseLine[baseLineIndex];

                                    RFLOAT nf = exp(-offset);

                                    wC(l, t) *= nf;

                                    // wC.row(l) *= nf;

                                    /***
                                    for (int td = 0; td < _para.k; td++)
                                    {
                                        wR[td].row(l) *= nf;
                                        wT[td].row(l) *= nf;
                                    }
                                    ***/

                                    wR[t].row(l) *= nf;
                                    wT[t].row(l) *= nf;

                                    baseLine[baseLineIndex] += offset;
                                }
                            }

                            RFLOAT w = exp(dvp[tnp * _ID.size() + l] - baseLine[baseLineIndex]);

                            wC(l, t) += w * (_parAll[t * _ID.size() + l].wR(m) * _parAll[t * _ID.size() + l].wT(n));

                            wR[t](l, m) += w * _parAll[t * _ID.size() + l].wT(n);

                            wT[t](l, n) += w * _parAll[t * _ID.size() + l].wR(m);
                        }
                    }
                }

                // #pragma omp atomic
                _nR += chunk;

                // #pragma omp critical  (line833)
                if (_nR > (int)(nR * _para.k / 10))
                {
                    _nR = 0;

                    nPer += 1;

                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << nPer * 10
                                               << "\% Initial Phase of Global Search Performed";
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << nPer * 10
                                               << "\% Initial Phase of Global Search Performed";
                }
        }

        /***
        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            vec baseLineT = vec::Zero(_para.k);

            vec ratio = vec::Zero(_para.k);

            for (size_t t = 0; t < _para.k; t++)
            {
                baseLineT(t) = baseLine[t * _ID.size() + l];
            }

            if ((l == 0) && (_commRank == HEMI_A_LEAD))
            {
                for (size_t t = 0; t < _para.k; t++)
                {
                    std::cout << "t = " << t << ", baseLine(t) = " << baseLineT(t) << std::endl;
                }
            }

            RFLOAT maxBaseLine = baseLineT(value_max_index(baseLineT));

            for (size_t t = 0; t < _para.k; t++)
            {
                ratio(t) = exp(baseLineT(t) - maxBaseLine);
            }

            RFLOAT hh = ratio(value_max_index(ratio)) * PEAK_FACTOR_C;

            for (size_t t = 0; t < _para.k; t++)
            {
                if (ratio(t) < hh)
                {
                    ratio(t) = 0;
                }
                else
                {
                    ratio(t) -= hh;
                }
            }

            if ((l == 0) && (_commRank == HEMI_A_LEAD))
            {
                for (size_t t = 0; t < _para.k; t++)
                {
                    std::cout << "t = " << t << ", ratio(t) = " << ratio(t) << std::endl;
                }
            }

            _iRef[l] = drawWithWeightIndex(ratio);

            if ((_iRef[l] < 0) ||
                (_iRef[l] >= _para.k))
            {
                CLOG(FATAL, "LOGGER_SYS") << "_iRef[l] = " << _iRef[l];

                abort();
            }

            _par[l] = &(_parAll[_iRef[l] * _ID.size() + l]);
        }
        ***/

        TSFFTW_free(poolSIMDResult);
        TSFFTW_free(poolPriRotPR);
        TSFFTW_free(poolPriRotPI);
        TSFFTW_free(poolPriAllPR);
        TSFFTW_free(poolPriAllPI);

        delete[] baseLine;

        // reset weights of particle filter

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            /***
#ifdef OPTIMISER_SAVE_PARTICLES
            if (_ID[l] < N_SAVE_IMG)
            {
                _par[l]->sort();

                char filename[FILE_NAME_LENGTH];

                snprintf(filename,
                         sizeof(filename),
                         "C_Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_C, true);
            }
#endif
            ***/

            // FOR_DEBUG

            /***
            for (int iR = 0; iR < nR; iR++)
                _par[l]->setUR(wR[cls](l, iR), iR);
            for (int iT = 0; iT < nT; iT++)
                _par[l]->setUT(wT[cls](l, iT), iT);
            ***/

            for (size_t t = 0; t < _para.k; t++)
            {
                Particle* par = &(_parAll[t * _ID.size() + l]);

                for (int iR = 0; iR < nR; iR++)
                {
                    // _par[l]->setUR(wR[_iRef[l]](l, iR), iR);

                    par->setUR(wR[t](l, iR), iR);
                }

                for (int iT = 0; iT < nT; iT++)
                {
                    // _par[l]->setUT(wT[_iRef[l]](l, iT), iT);

                    par->setUT(wT[t](l, iT), iT);
                }

#ifdef OPTIMISER_PEAK_FACTOR_R
                // _par[l]->setPeakFactor(PAR_R);
                // _par[l]->keepHalfHeightPeak(PAR_R);

                par->setPeakFactor(PAR_R);
                par->keepHalfHeightPeak(PAR_R);
#endif

#ifdef OPTIMISER_PEAK_FACTOR_T
                // _par[l]->setPeakFactor(PAR_T);
                // _par[l]->keepHalfHeightPeak(PAR_T);
                par->setPeakFactor(PAR_T);
                par->keepHalfHeightPeak(PAR_T);
#endif


            /***
#ifdef OPTIMISER_SAVE_PARTICLES
            if (_ID[l] < N_SAVE_IMG)
            {
                _par[l]->sort();

                char filename[FILE_NAME_LENGTH];

                snprintf(filename,
                         sizeof(filename),
                         "R_Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_R, true);
                snprintf(filename,
                         sizeof(filename),
                         "T_Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_T, true);
                snprintf(filename,
                         sizeof(filename),
                         "D_Particle_%04d_Round_%03d_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_D, true);
            }
#endif
            ***/

                // _par[l]->resample(_para.mLR, PAR_R);
                // _par[l]->resample(_para.mLT, PAR_T);

                par->resample(_para.mLR, PAR_R);
                par->resample(_para.mLT, PAR_T);

                // _par[l]->calVari(PAR_R);
                // _par[l]->calVari(PAR_T);

                par->calVari(PAR_R);
                par->calVari(PAR_T);

#ifdef PARTICLE_RHO
                // _par[l].setRho(0);

                par->setRho(0);
            // if there is only two resampled points in translation, it is possible making pho be 1
            // then it will crash down
            // make rho to be 0
#endif

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_SCAN_SET_MIN_STD_WITH_PERTURB
                    par->setK1(TSGSL_MAX_RFLOAT((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                 ? _para.perturbFactorSGlobal
                                                 : _para.perturbFactorSLocal))
                                                 * MIN_STD_FACTOR * scanMinStdR,
                                                 par->k1()));
#else
                    par->setK1(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * scanMinStdR,
                                                par->k1()));
#endif
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_SCAN_SET_MIN_STD_WITH_PERTURB
                par->setK1(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                          ? _para.perturbFactorSGlobal
                                                          : _para.perturbFactorSLocal))
                                                  * MIN_STD_FACTOR * scanMinStdR),
                                          par->k1()));
                par->setK2(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                          ? _para.perturbFactorSGlobal
                                                          : _para.perturbFactorSLocal))
                                                  * MIN_STD_FACTOR * scanMinStdR),
                                          par->k2()));

                par->setK3(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                          ? _para.perturbFactorSGlobal
                                                          : _para.perturbFactorSLocal))
                                                   * MIN_STD_FACTOR * scanMinStdR),
                                          par->k3()));
#else
                par->setK1(TSGSL_MAX_RFLOAT(TSGSL_pow_2(MIN_STD_FACTOR * scanMinStdR),
                                               par->k1()));
                par->setK2(TSGSL_MAX_RFLOAT(TSGSL_pow_2(MIN_STD_FACTOR * scanMinStdR),
                                               par->k2()));
                par->setK3(TSGSL_MAX_RFLOAT(TSGSL_pow_2(MIN_STD_FACTOR * scanMinStdR),
                                               par->k3()));
#endif
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }

#ifdef OPTIMISER_SCAN_SET_MIN_STD_WITH_PERTURB
                par->setS0(TSGSL_MAX_RFLOAT(1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                           ? _para.perturbFactorSGlobal
                                           : _para.perturbFactorSLocal)
                                    * MIN_STD_FACTOR * scanMinStdT,
                                      par->s0()));

                par->setS1(TSGSL_MAX_RFLOAT(1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                           ? _para.perturbFactorSGlobal
                                           : _para.perturbFactorSLocal)
                                    * MIN_STD_FACTOR * scanMinStdT,
                                      par->s1()));
#else
                par->setS0(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * scanMinStdT,
                                           par->s0()));
                par->setS1(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * scanMinStdT,
                                           par->s1()));
#endif

                /***
#ifdef OPTIMISER_SAVE_PARTICLES
                if (_ID[l] < N_SAVE_IMG)
                {
                _par[l]->sort();

                char filename[FILE_NAME_LENGTH];
                snprintf(filename,
                         sizeof(filename),
                         "C_Particle_%04d_Round_%03d_Resampled_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_C);
                snprintf(filename,
                         sizeof(filename),
                         "R_Particle_%04d_Round_%03d_Resampled_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_R);
                snprintf(filename,
                         sizeof(filename),
                         "T_Particle_%04d_Round_%03d_Resampled_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_T);
                snprintf(filename,
                         sizeof(filename),
                         "D_Particle_%04d_Round_%03d_Resampled_Initial.par",
                         _ID[l],
                         _iter);
                save(filename, _par[l], PAR_D);
                }
#endif
                ***/
            }
        }

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initial Phase of Global Search Performed.";
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initial Phase of Global Search Performed.";

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initial Phase of Global Search in Hemisphere A Performed";
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initial Phase of Global Search in Hemisphere B Performed";
#endif

        TSFFTW_free(traP);

        if (_searchType != SEARCH_TYPE_CTF)
            freePreCal(false);
        else
            freePreCal(true);
    }

#ifdef OPTIMISER_PARTICLE_FILTER

    if (_searchType != SEARCH_TYPE_CTF)
        allocPreCal(true, false, false);
    else
        allocPreCal(true, false, true);

    _nP.resize(_ID.size(), 0);

    _nF = 0;
    _nI = 0;

    nPer = 0;

    // TODO, remove
    Complex* poolPriRotP = (Complex*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(Complex));
    // poolPriRotP = (Complex*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(Complex));

    // TODO, remove
    Complex* poolPriAllP = (Complex*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(Complex));

    RFLOAT* poolPriAllPR = (RFLOAT*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(RFLOAT));
    RFLOAT* poolPriAllPI = (RFLOAT*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(RFLOAT));

    Complex* poolTraP = (Complex*)TSFFTW_malloc(_para.mLT * _nPxl * omp_get_max_threads() * sizeof(Complex));

    RFLOAT* poolCtfP;

    if (_searchType == SEARCH_TYPE_CTF)
        poolCtfP = (RFLOAT*)TSFFTW_malloc(_para.mLD * _nPxl * omp_get_max_threads() * sizeof(RFLOAT));

    // new, for non-alignment classification
    mat wC = mat::Zero(_ID.size(), _para.k);

    vector<mat> wR(_para.k, mat::Zero(_ID.size(), _para.mLR));
    vector<mat> wT(_para.k, mat::Zero(_ID.size(), _para.mLT));
    vector<mat> wD(_para.k, mat::Zero(_ID.size(), _para.mLD));

    // the same arrangement with _par
    RFLOAT* baseLine = new RFLOAT[_ID.size() * _para.k];

    for (size_t t = 0; t < (size_t)_para.k; t++)
    {
        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            baseLine[t * _ID.size() + l] = GSL_NAN;
        }
    }

    MemoryBazaarDustman<RFLOAT, BaseType, 4> datPRDustman(&_datPR);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> datPIDustman(&_datPI);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> ctfPDustman(&_ctfP);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> sigRcpPDustman(&_sigRcpP);
    #pragma omp parallel for schedule(dynamic) firstprivate(datPRDustman, datPIDustman, ctfPDustman, sigRcpPDustman)
    FOR_EACH_2D_IMAGE
    {
        _datPR.endLastVisit(l * _nPxl);
        _datPI.endLastVisit(l * _nPxl);

        if (_searchType != SEARCH_TYPE_CTF)
        {
            _ctfP.endLastVisit(l * _nPxl);
        }

        _sigRcpP.endLastVisit(l * _nPxl);

        // TODO, remove
        Complex* priRotP = poolPriRotP + _nPxl * omp_get_thread_num();

        // TODO, remove
        Complex* priAllP = poolPriAllP + _nPxl * omp_get_thread_num();

        RFLOAT* priAllPR = poolPriAllPR + _nPxl * omp_get_thread_num();
        RFLOAT* priAllPI = poolPriAllPI + _nPxl * omp_get_thread_num();

        /***
        // debug
        if (_commRank == HEMI_A_LEAD)
        {
            # pragma omp critical
            std::cout << "_para.alignR = " << _para.alignR << std::endl;
            # pragma omp critical
            std::cout << "_para.alignT = " << _para.alignT << std::endl;
            # pragma omp critical
            std::cout << "_para.alignD = " << _para.alignD << std::endl;
            # pragma omp critical
            std::cout << "phase rounds = " << ((_para.alignR || _para.alignT || _para.alignD) ? MAX_N_PHASE_PER_ITER : 2) << std::endl;
        }
        // end debug
        ***/

        for (size_t t = 0; t < _para.k; t++)
        {
            Particle* par = &(_parAll[t * _ID.size() + l]);

            /**
            if (par == _par[l])
            {
            ***/

            int nPhaseWithNoVariDecrease = 0;

#ifdef OPTIMISER_COMPRESS_CRITERIA
            double variR = DBL_MAX;
            double variT = DBL_MAX;
            double variD = DBL_MAX;
#else
            double k1 = 1;
            double k2 = 1;
            double k3 = 1;
            double tVariS0 = 5 * _para.transS;
            double tVariS1 = 5 * _para.transS;
            double dVari = 5 * _para.ctfRefineS;
#endif

            for (int phase = ((_searchType == SEARCH_TYPE_GLOBAL) ? 1 : 0); phase < ((_para.alignR || _para.alignT || _para.alignD) ? MAX_N_PHASE_PER_ITER : 1); phase++)
            {
                // wR, wT, wD and baseLine should be reset
                // in this section, the class ID, i.e., k, and the image ID, i.e., l, are given
                wR[t].row(l) = vec::Zero(_para.mLR);
                wT[t].row(l) = vec::Zero(_para.mLT);
                wD[t].row(l) = vec::Zero(_para.mLD);

                baseLine[t * _ID.size() + l] = GSL_NAN;

#ifdef OPTIMISER_GLOBAL_PERTURB_LARGE
                if (phase == (_searchType == SEARCH_TYPE_GLOBAL) ? 1 : 0)
#else
                if (phase == 0)
#endif
                {
                    // _par[l]->perturb(_para.perturbFactorL, PAR_R);
                    // _par[l]->perturb(_para.perturbFactorL, PAR_T);

                    if (_para.alignR)
                    {
                        par->perturb(_para.perturbFactorL, PAR_R);
                    }
                    
                    if (_para.alignT)
                    {
                        par->perturb(_para.perturbFactorL, PAR_T);
                    }

                    if (_para.alignD && _searchType == SEARCH_TYPE_CTF)
                    {
                        par->initD(_para.mLD, _para.ctfRefineS);
                    }
                }
                else
                {
                    if (_para.alignR)
                    {
                        par->perturb((_searchType == SEARCH_TYPE_GLOBAL)
                                   ? _para.perturbFactorSGlobal
                                   : _para.perturbFactorSLocal,
                                     PAR_R);
                    }

                    if (_para.alignT)
                    {
                        par->perturb((_searchType == SEARCH_TYPE_GLOBAL)
                                   ? _para.perturbFactorSGlobal
                                   : _para.perturbFactorSLocal,
                                     PAR_T);
                    }

                    if (_para.alignD && _searchType == SEARCH_TYPE_CTF)
                    {
                        par->perturb(_para.perturbFactorSCTF, PAR_D);
                    }
                }

                // RFLOAT baseLine = GSL_NAN;

                // vec wC = vec::Zero(1);
                // vec wR = vec::Zero(_para.mLR);
                // vec wT = vec::Zero(_para.mLT);
                // vec wD = vec::Zero(_para.mLD);

                // size_t c;
                dmat22 rot2D;
                dmat33 rot3D;
                double d;
                dvec2 tran;

                Complex* traP = poolTraP + par->nT() * _nPxl * omp_get_thread_num();

                FOR_EACH_T(*par)
                {
                    par->t(tran, iT);

                    /***
                    // debug
                    if ((l == 0) && (_commRank == HEMI_A_LEAD))
                    {
                        # pragma omp critical
                        std::cout << "t = " << t << ", tran = " << tran << std::endl;
                    }
                    // end debug
                    ***/

                    translate(traP + iT * _nPxl,
                              tran(0),
                              tran(1),
                              _para.size,
                              _para.size,
                              _iCol,
                              _iRow,
                              _nPxl,
                              1);
                }

                RFLOAT* ctfP;

                if (_searchType == SEARCH_TYPE_CTF)
                {
                    ctfP = poolCtfP + par->nD() * _nPxl * omp_get_thread_num();

                    FOR_EACH_D(*par)
                    {
                        par->d(d, iD);

                        for (int i = 0; i < _nPxl; i++)
                        {
                            RFLOAT ki = _K1[l]
                                      * _defocusP[l * _nPxl + i]
                                      * d
                                      * TSGSL_pow_2(_frequency[i])
                                      + _K2[l]
                                      * TSGSL_pow_4(_frequency[i])
                                      - _ctfAttr[l].phaseShift;

                            ctfP[_nPxl * iD + i] = -TS_SQRT(1 - TSGSL_pow_2(_ctfAttr[l].amplitudeContrast))
                                                 * TS_SIN(ki)
                                                 + _ctfAttr[l].amplitudeContrast
                                                 * TS_COS(ki);
                        }
                    }
                }

                FOR_EACH_R(*par)
                {
                    if (_para.mode == MODE_2D)
                    {
                        par->rot(rot2D, iR);

                        /*
                        // debug
                        if ((l == 0) && (_commRank == HEMI_A_LEAD))
                        {
                            # pragma omp critical
                            std::cout << "t = " << t << ", rot2D = " << rot2D << std::endl;
                        }
                        // end debug
                        */
                    }
                    else if (_para.mode == MODE_3D)
                    {
                        par->rot(rot3D, iR);

                        /*
                        // debug
                        if ((l == 0) && (_commRank == HEMI_A_LEAD))
                        {
                            # pragma omp critical
                            std::cout << "t = " << t << ", rot3D = " << rot3D << std::endl;
                        }
                        // end debug
                        */
                    }
                    else
                    {
                        REPORT_ERROR("INEXISTENT MODE");

                        abort();
                    }

                    if (_para.mode == MODE_2D)
                    {
                        _model.proj(t).project(priRotP,
                                               rot2D,
                                               _iCol,
                                               _iRow,
                                               _nPxl,
                                               1);
                    }
                    else if (_para.mode == MODE_3D)
                    {
                        _model.proj(t).project(priRotP,
                                               rot3D,
                                               _iCol,
                                               _iRow,
                                               _nPxl,
                                               1);
                    }
                    else
                    {
                        REPORT_ERROR("INEXISTENT MODE");

                        abort();
                    }

                    FOR_EACH_T(*par)
                    {
                        for (int i = 0; i < _nPxl; i++)
                        {
                            // TODO, remove
                            priAllP[i] = traP[_nPxl * iT + i] * priRotP[i];

                            // Complex multiplication
                            // RFLOAT tR = traP[_nPxl * iT + i].dat[0];
                            // RFLOAT tI = traP[_nPxl * iT + i].dat[1];

                            // priAllPR[i] = tR * priRotPR[i] - tI * priRotPI[i];
                            // priAllPI[i] = tR * priRotPI[i] + tI * priRotPR[i];
                            priAllPR[i] = priAllP[i].dat[0];
                            priAllPI[i] = priAllP[i].dat[1];
                        }

                        FOR_EACH_D(*par)
                        {
                            par->d(d, iD);

                            RFLOAT w;

                            RFLOAT* datPR = &_datPR[l * _nPxl];
                            RFLOAT* datPI = &_datPI[l * _nPxl];
                            RFLOAT* sigRcpP = &_sigRcpP[l * _nPxl];

                            if (_searchType != SEARCH_TYPE_CTF)
                            {
                                RFLOAT* ctfP = &_ctfP[l * _nPxl];

                                w = logDataVSPrior(datPR,
                                                   datPI,
                                                   // _datPI + l * _nPxl,
                                                   priAllPR,
                                                   priAllPI,
                                                   ctfP,
                                                   // _ctfP + l * _nPxl,
                                                   //_sigRcpP + l * _nPxl,
                                                   sigRcpP,
                                                   _nPxl);
                            }
                            else
                            {
                                w = logDataVSPrior(datPR,
                                                   datPI,
                                                   // _datPI + l * _nPxl,
                                                   priAllPR,
                                                   priAllPI,
                                                   ctfP + iD * _nPxl,
                                                   // _sigRcpP + l * _nPxl,
                                                   sigRcpP,
                                                   _nPxl);
                            }

                            baseLine[t * _ID.size() + l] = TSGSL_isnan(baseLine[t * _ID.size() + l]) ? w : baseLine[t * _ID.size() + l];

                            if (w > baseLine[t * _ID.size() + l])
                            {
                                RFLOAT nf = exp(baseLine[t * _ID.size() + l] - w);

                                wC(l, t) *= nf;

                                // wR[t] *= nf;
                                // wT[t] *= nf;
                                // wD[t] *= nf;

                                wR[t].row(l) *= nf;
                                wT[t].row(l) *= nf;
                                wD[t].row(l) *= nf;

                                baseLine[t * _ID.size() + l] = w;
                            }

                            /* 
                            // for debug
                            if ((l == 0) && (_commRank == HEMI_A_LEAD))
                            {
                                std::cout << "t = " << t << ", baseLine(t) = " << baseLine[t * _ID.size() + l] << std::endl;
                            }
                            // end debug
                            */

                            RFLOAT s = exp(w - baseLine[t * _ID.size() + l]);

                            wC(l, t) += s * (par->wR(iR) * par->wT(iT) * par->wD(iD));
                            wR[t](l, iR) += s * (par->wT(iT) * par->wD(iD));
                            wT[t](l, iT) += s * (par->wR(iR) * par->wD(iD));
                            wD[t](l, iD) += s * (par->wR(iR) * par->wT(iT));
                        }
                    }
                }

                // par->setUC(wC(0), 0);

                for (int iR = 0; iR < _para.mLR; iR++)
                {
                    par->setUR(wR[t](l, iR), iR);
                }

#ifdef OPTIMISER_PEAK_FACTOR_R
                par->keepHalfHeightPeak(PAR_R);
#endif

                for (int iT = 0; iT < _para.mLT; iT++)
                {
                    par->setUT(wT[t](l, iT), iT);
                }

#ifdef OPTIMISER_PEAK_FACTOR_T
                par->keepHalfHeightPeak(PAR_T);
#endif

                if (_searchType == SEARCH_TYPE_CTF)
                {
                    for (int iD = 0; iD < _para.mLD; iD++)
                    {
                        par->setUD(wD[t](l, iD), iD);
                    }

#ifdef OPTIMISER_PEAK_FACTOR_D
                    if (phase == 0)
                    {
                        par->setPeakFactor(PAR_D);
                    }

                    par->keepHalfHeightPeak(PAR_D);
#endif
                }

#ifdef OPTIMISER_SAVE_PARTICLES
                if (_ID[l] < N_SAVE_IMG)
                {
                    par->sort();

                    char filename[FILE_NAME_LENGTH];

                    /***
                    snprintf(filename,
                             sizeof(filename),
                             "C_Particle_%04d_Round_%03d_%03d.par",
                             _ID[l],
                             _iter,
                             phase);
                    save(filename, *par, PAR_C, true);
                    ***/
                    snprintf(filename,
                             sizeof(filename),
                             "R_Particle_%04d_Round_%03d_%03d.par",
                             _ID[l],
                             _iter,
                             phase);
                    save(filename, *par, PAR_R, true);
                    snprintf(filename,
                             sizeof(filename),
                             "T_Particle_%04d_Round_%03d_%03d.par",
                             _ID[l],
                             _iter,
                             phase);
                    save(filename, *par, PAR_T, true);
                    snprintf(filename,
                             sizeof(filename),
                             "D_Particle_%04d_Round_%03d_%03d.par",
                             _ID[l],
                             _iter,
                             phase);
                    save(filename, *par, PAR_D, true);
                }
#endif

                if (_para.alignR)
                {
                    par->calRank1st(PAR_R);
                    par->calVari(PAR_R);
                    par->resample(_para.mLR, PAR_R);
                }

                if (_para.alignT)
                {
                    par->calRank1st(PAR_T);
                    par->calVari(PAR_T);
                    par->resample(_para.mLT, PAR_T);
                }

                if (_para.alignD && (_searchType == SEARCH_TYPE_CTF))
                {
                    par->calRank1st(PAR_D);
                    par->calVari(PAR_D);
                    par->resample(_para.mLD, PAR_D);
                }

            /***
            RFLOAT k1 = _par[l]->k1();
            RFLOAT s0 = _par[l]->s0();
            RFLOAT s1 = _par[l]->s1();

            _par[l]->resample(_para.mLR, PAR_R);
            _par[l]->resample(_para.mLT, PAR_T);

            _par[l]->calVari(PAR_R);
            _par[l]->calVari(PAR_T);

            _par[l]->setK1(TSGSL_MAX_RFLOAT(k1 * gsl_pow_2(MIN_STD_FACTOR
                                                   * pow(_par[l]->nR(), -1.0 / 3)),
                                      _par[l]->k1()));

            _par[l]->setS0(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * s0 / sqrt(_par[l]->nT()), _par[l]->s0()));

            _par[l]->setS1(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * s1 / sqrt(_par[l]->nT()), _par[l]->s1()));
            ***/

                if (phase >= ((_searchType == SEARCH_TYPE_GLOBAL)
                              ? MIN_N_PHASE_PER_ITER_GLOBAL
                              : MIN_N_PHASE_PER_ITER_LOCAL))
                {
#ifdef OPTIMISER_COMPRESS_CRITERIA
                    double variRCur;
                    double variTCur;
                    double variDCur;
#else
                    double k1Cur;
                    double k2Cur;
                    double k3Cur;
                    double tVariS0Cur;
                    double tVariS1Cur;
                    double dVariCur;
#endif

#ifdef OPTIMISER_COMPRESS_CRITERIA
                    variRCur = par->variR();
                    variTCur = par->variT();
                    variDCur = par->variD();
#else
                    par->vari(k1Cur, k2Cur, k3Cur, tVariS0Cur, tVariS1Cur, dVariCur);
#endif

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_COMPRESS_CRITERIA
                    if ((variRCur < variR * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (variTCur < variT * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (variDCur < variD * PARTICLE_FILTER_DECREASE_FACTOR))
#else
                    if ((k1Cur < k1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (tVariS0Cur < tVariS0 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (tVariS1Cur < tVariS1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (dVariCur < dVari * PARTICLE_FILTER_DECREASE_FACTOR))
#endif
                    {
                        // there is still room for searching
                        nPhaseWithNoVariDecrease = 0;
                    }
                    else
                        nPhaseWithNoVariDecrease += 1;
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_COMPRESS_CRITERIA
                    if ((variRCur < variR * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (variTCur < variT * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (variDCur < variD * PARTICLE_FILTER_DECREASE_FACTOR))
#else
                    if ((k1Cur < k1 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                        (k2Cur < k2 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                        (k3Cur < k3 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                        (tVariS0Cur < tVariS0 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (tVariS1Cur < tVariS1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                        (dVariCur < dVari * PARTICLE_FILTER_DECREASE_FACTOR))
#endif
                    {
                        // there is still room for searching
                        nPhaseWithNoVariDecrease = 0;
                    }
                    else
                        nPhaseWithNoVariDecrease += 1;
                }
                else
                {
                    REPORT_ERROR("EXISTENT MODE");

                    abort();
                }

#ifdef OPTIMISER_COMPRESS_CRITERIA

#ifndef NAN_NO_CHECK
                POINT_NAN_CHECK(par->compressR());
                POINT_NAN_CHECK(par->compressT());
#endif

                if (variRCur < variR) variR = variRCur;
                if (variTCur < variT) variT = variTCur;
                if (variDCur < variD) variD = variDCur;
#else
                // make tVariS0, tVariS1, rVari the smallest variance ever got
                if (k1Cur < k1) k1 = k1Cur;
                if (k2Cur < k2) k2 = k2Cur;
                if (k3Cur < k3) k3 = k3Cur;
                if (tVariS0Cur < tVariS0) tVariS0 = tVariS0Cur;
                if (tVariS1Cur < tVariS1) tVariS1 = tVariS1Cur;
                if (dVariCur < dVari) dVari = dVariCur;
#endif

                // break if in a few continuous searching, there is no improvement
                if (nPhaseWithNoVariDecrease == N_PHASE_WITH_NO_VARI_DECREASE)
                {
                    _nP[l] = phase;

                    #pragma omp atomic
                    _nF += phase;

                    #pragma omp atomic
                    _nI += 1;

                    break;
                }
            }
        } // phase end
        // } // if statement end
        } // class end

        #pragma omp critical  (line1495)
        if (_nI > (int)(_ID.size() * _para.k / 10))
        {
            _nI = 0;

            nPer += 1;

            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << nPer * 10 << "\% Expectation Performed";
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << nPer * 10 << "\% Expectation Performed";
        }

#ifdef OPTIMISER_SAVE_PARTICLES
        if (_ID[l] < N_SAVE_IMG)
        {
            char filename[FILE_NAME_LENGTH];

            /***
            snprintf(filename,
                     sizeof(filename),
                     "C_Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, *(_par[l]), PAR_C);
            ***/
            snprintf(filename,
                     sizeof(filename),
                     "R_Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, *(_par[l]), PAR_R);
            snprintf(filename,
                     sizeof(filename),
                     "T_Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, *(_par[l]), PAR_T);
            snprintf(filename,
                     sizeof(filename),
                     "D_Particle_%04d_Round_%03d_Final.par",
                     _ID[l],
                     _iter);
            save(filename, *(_par[l]), PAR_D);
        }
#endif
    } // image end

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            vec baseLineT = vec::Zero(_para.k);

            vec ratio = vec::Zero(_para.k);

            for (size_t t = 0; t < _para.k; t++)
            {
                baseLineT(t) = baseLine[t * _ID.size() + l];
            }

            /*
            if ((l == 0) && (_commRank == HEMI_A_LEAD))
            {
                for (size_t t = 0; t < _para.k; t++)
                {
                    std::cout << "t = " << t << ", baseLine(t) = " << baseLineT(t) << std::endl;
                }
            }
            */

            RFLOAT maxBaseLine = baseLineT(value_max_index(baseLineT));

            for (size_t t = 0; t < _para.k; t++)
            {
                ratio(t) = exp(baseLineT(t) - maxBaseLine);
            }

            /*
            // add [0, 1%] uniform distribution noise here
            
            gsl_rng* engine = get_random_engine();

            RFLOAT range = 0.01 * ratio(value_min_index(ratio));

            for (size_t t = 0; t < _para.k; t++)
            {
                ratio(t) += range * TSGSL_rng_uniform(engine);
            }
            */

            RFLOAT hh = ratio(value_max_index(ratio)) * PEAK_FACTOR_C;

            for (size_t t = 0; t < _para.k; t++)
            {
                if (ratio(t) < hh)
                {
                    ratio(t) = 0;
                }
                else
                {
                    ratio(t) -= hh;
                }
            }

            /* debug
            if ((l == 0) && (_commRank == HEMI_A_LEAD))
            {
                for (size_t t = 0; t < _para.k; t++)
                {
                    std::cout << "t = " << t << ", ratio(t) = " << ratio(t) << std::endl;
                }
            }
            */

            _iRefPrev[l] = _iRef[l];

            _iRef[l] = drawWithWeightIndex(ratio);

            /*
            if ((_iRef[l] < 0) ||
                (_iRef[l] >= _para.k))
            {
                CLOG(FATAL, "LOGGER_SYS") << "_iRef[l] = " << _iRef[l];

                abort();
            }
            */

            _par[l] = &(_parAll[_iRef[l] * _ID.size() + l]);
        }

    TSFFTW_free(poolPriRotP);
    TSFFTW_free(poolPriAllP);
    TSFFTW_free(poolPriAllPR);
    TSFFTW_free(poolPriAllPI);

    TSFFTW_free(poolTraP);

    if (_searchType == SEARCH_TYPE_CTF)
        TSFFTW_free(poolCtfP);

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space for Pre-calculation in Expectation";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space for Pre-calculation in Expectation";

    if (_searchType != SEARCH_TYPE_CTF)
        freePreCal(false);
    else
        freePreCal(true);

#endif // OPTIMISER_PARTICLE_FILTER

    freePreCalIdx();
}

#ifdef GPU_VERSION
void Optimiser::expectationG()
{
    IF_MASTER return;

    int nPer = 0;

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space for Pre-calculation in Expectation";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space for Pre-calculation in Expectation";

    long memUsage = memoryCheckRM();
    printf("expect global memory check work begin:%dG!\n", memUsage / MEGABYTE);
    
    allocPreCalIdx(_r, _rL);

    printf("Round:%d, Before expectation GPU memory check!\n", _iter);
    gpuMemoryCheck(_iGPU,
                   _commRank,
                   _nGPU);

    int *deviCol[_nGPU];
    int *deviRow[_nGPU];

    #pragma omp parallel for num_threads(_nGPU)
    for (int i = 0; i < _nGPU; i++)
    {
        ExpectPreidx(_iGPU[i],
                     &deviCol[i],
                     &deviRow[i],
                     _iCol,
                     _iRow,
                     _nPxl);
    }

    if (_searchType == SEARCH_TYPE_GLOBAL)
    {
        if (_searchType != SEARCH_TYPE_CTF)
            allocPreCal(true, false, false);
        else
            allocPreCal(true, false, true);

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space for Pre-calculation in Expectation Allocated";
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space for Pre-calculation in Expectation Allocated";

        // initialse a particle filter

        int nR;
        if (_para.mode == MODE_2D)
        {
            nR = _para.mS;
        }
        else if (_para.mode == MODE_3D)
        {
            nR = _para.mS / (1 + _sym.nSymmetryElement());
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

        int nT = GSL_MAX_INT(30,
                             AROUND(M_PI
                                  * TSGSL_pow_2(_para.transS
                                            * TSGSL_cdf_chisq_Qinv(0.5, 2))
                                  * _para.transSearchFactor));

        RFLOAT scanMinStdR;
        if (_para.mode == MODE_2D)
        {
            scanMinStdR = 1.0 / _para.mS;
        }
        else if (_para.mode == MODE_3D)
        {
            scanMinStdR =  pow(_para.mS, -1.0 / 3);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

        RFLOAT scanMinStdT = 1.0
                           / TSGSL_cdf_chisq_Qinv(INIT_OUTSIDE_CONFIDENCE_AREA, 2)
                           / sqrt(_para.transSearchFactor * M_PI);

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Minimum Standard Deviation of Rotation in Scanning Phase: "
                                   << scanMinStdR;
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Minimum Standard Deviation of Translation in Scanning Phase: "
                                   << scanMinStdT;

        Particle par = _parAll[0].copy();

        par.reset(nR, nT, 1);

        for (size_t t = 0; t < (size_t)_para.k; t++)
        {
            FOR_EACH_2D_IMAGE
            {
                par.copy(_parAll[t * _ID.size() + l]);
            }
        }
        //float time_use = 0;
        //struct timeval start;
        //struct timeval end;

        //gettimeofday(&start, NULL);

        int nImg = _ID.size();
        RFLOAT* weightC = (RFLOAT*)malloc(_ID.size() * _para.k * sizeof(RFLOAT));
        RFLOAT* weightR = (RFLOAT*)malloc(_ID.size() * _para.k * nR * sizeof(RFLOAT));
        RFLOAT* weightT = (RFLOAT*)malloc(_ID.size() * _para.k * nT * sizeof(RFLOAT));

        double* pr = new double[nR];
        double* pt = new double[nT];

        for (int i = 0; i < nR; i++)
            pr[i] = par.wR(i);
        for (int i = 0; i < nT; i++)
            pt[i] = par.wT(i);

        if (_para.mode == MODE_3D)
        {
            double* trans = new double[nT * 2];
            double* rot = new double[nR * 4];
            RFLOAT* baseL = new RFLOAT[_ID.size()];

            for (int k = 0; k < nT; k++)
                Map<dvec2>(trans + k * 2, 2, 1) = par.t().row(k).transpose();

            for (int k = 0; k < nR; k++)
                Map<dvec4>(rot + k * 4, 4, 1) = par.r().row(k).transpose();

            Complex* devrotP[_stream.size()];
            Complex* devtraP[_nGPU];
            double* devRotMat[_nGPU];
            double* devpR[_nGPU];
            double* devpT[_nGPU];
            
            ExpectRotran(_iGPU,
                         _stream,
                         devrotP,
                         devtraP,
                         devRotMat,
                         devpR,
                         devpT,
                         trans,
                         rot,
                         pr,
                         pt,
                         deviCol,
                         deviRow,
                         nR,
                         nT,
                         _para.size,
                         _nPxl,
                         _nGPU);

            delete[] trans;
            delete[] rot;

            Complex* rotP = (Complex*)TSFFTW_malloc((long long)nR * _nPxl * sizeof(Complex));
            Complex* vol;

            RFLOAT *pglk_datPR = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_datPI = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_ctfP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_sigRcpP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));

            hostRegister(pglk_datPR, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_datPI, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_ctfP, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_sigRcpP, IMAGE_BATCH * _nPxl);
                
            for (size_t t = 0; t < (size_t)_para.k; t++)
            {
                vol = &((const_cast<Volume&>(_model.proj(t).projectee3D()))[0]);

                ExpectProject(_iGPU, 
                              _stream,
                              vol,
                              rotP,
                              devrotP,
                              devRotMat,
                              deviCol,
                              deviRow,
                              nR,
                              _model.proj(t).pf(),
                              _model.proj(t).interp(),
                              _model.proj(t).projectee3D().nSlcFT(),
                              _nPxl,
                              _nGPU);

                for (int l = 0; l < nImg;)
                {
                    if (l >= nImg)
                        break;

                    int batch = (l + IMAGE_BATCH < nImg)
                              ? IMAGE_BATCH : (nImg - l);

                    RFLOAT *temp_datPR;
                    RFLOAT *temp_datPI;
                    RFLOAT *temp_ctfP;
                    RFLOAT *temp_sigP;
                    for (int i = 0; i < batch; i++)
                    {
                        temp_datPR = &_datPR[(l + i) * _nPxl];
                        temp_datPI = &_datPI[(l + i) * _nPxl];
                        temp_ctfP = &_ctfP[(l + i) * _nPxl];
                        temp_sigP = &_sigRcpP[(l + i) * _nPxl];
                        
                        for (int p = 0; p < _nPxl; p++)
                        {
                            pglk_datPR[p + i * _nPxl] = temp_datPR[p];
                            pglk_datPI[p + i * _nPxl] = temp_datPI[p];
                            pglk_ctfP[p + i * _nPxl] = temp_ctfP[p];
                            pglk_sigRcpP[p + i * _nPxl] = temp_sigP[p];
                        }
                    }

                    ExpectGlobal3D(_iGPU,
                                   _stream,
                                   devrotP,
                                   devtraP,
                                   rotP,
                                   devpR,
                                   devpT,
                                   pglk_datPR,
                                   pglk_datPI,
                                   pglk_ctfP,
                                   pglk_sigRcpP,
                                   weightC + t * nImg + l,
                                   weightR + (long long)t * nImg * nR + l * nR,
                                   weightT + (long long)t * nImg * nT + l * nT,
                                   baseL + l,
                                   nR,
                                   nT,
                                   _nPxl,
                                   batch,
                                   _nGPU);
                    
                    l += batch;
                }
            }

            hostFree(pglk_datPR);
            hostFree(pglk_datPI);
            hostFree(pglk_ctfP);
            hostFree(pglk_sigRcpP);

            free(pglk_datPR);
            free(pglk_datPI);
            free(pglk_ctfP);
            free(pglk_sigRcpP);

            freeRotran(_iGPU,
                       devrotP,
                       devtraP,
                       devRotMat,
                       devpR,
                       devpT,
                       _nGPU);

            delete[] baseL;
            TSFFTW_free(rotP);
        }
        else
        {
            int sizeModel = _model.proj(0).projectee2D().sizeFT();
            Complex* vol = new Complex[sizeModel * _para.k];
            Complex* temp;

            for (int k = 0; k < _para.k; k++)
            {
                temp = &((const_cast<Image&>(_model.proj(k).projectee2D()))[0]);
                for (int z = 0; z < sizeModel; z++)
                    vol[k * sizeModel + z] = temp[z];
            }

            double* trans = new double[nT * 2];
            double* rot = new double[nR * 2];

            for (int k = 0; k < nT; k++)
                Map<dvec2>(trans + k * 2, 2, 1) = par.t().row(k).transpose();

            for (int k = 0; k < nR; k++)
            {
                rot[k * 2] = (par.r())(k, 0);
                rot[k * 2 + 1] = (par.r())(k, 1);

            }

            Complex* devtraP[_nGPU];
            double* devnR[_nGPU];
            double* devpR[_nGPU];
            double* devpT[_nGPU];
            std::vector<void*> symArray;
            std::vector<void*> texObject;
            
            ExpectRotran2D(_iGPU,
                           _stream,
                           symArray,
                           texObject,
                           vol,
                           devtraP,
                           devnR,
                           devpR,
                           devpT,
                           trans,
                           rot,
                           pr,
                           pt,
                           deviCol,
                           deviRow,
                           _para.k,
                           nR,
                           nT,
                           _para.size,
                           _model.proj(0).projectee2D().nRowFT(),
                           _nPxl,
                           _nGPU);

            delete[] vol;
            delete[] trans;
            delete[] rot;
            
            RFLOAT *pglk_datPR = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_datPI = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_ctfP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_sigRcpP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));

            hostRegister(pglk_datPR, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_datPI, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_ctfP, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_sigRcpP, IMAGE_BATCH * _nPxl);
                
            for (int l = 0; l < nImg;)
            {
                if (l >= nImg)
                    break;

                int batch = (l + IMAGE_BATCH < nImg)
                          ? IMAGE_BATCH : (nImg - l);

                RFLOAT *temp_datPR;
                RFLOAT *temp_datPI;
                RFLOAT *temp_ctfP;
                RFLOAT *temp_sigP;
                for (int i = 0; i < batch; i++)
                {
                    temp_datPR = &_datPR[(l + i) * _nPxl];
                    temp_datPI = &_datPI[(l + i) * _nPxl];
                    temp_ctfP = &_ctfP[(l + i) * _nPxl];
                    temp_sigP = &_sigRcpP[(l + i) * _nPxl];
                    
                    for (int p = 0; p < _nPxl; p++)
                    {
                        pglk_datPR[p + i * _nPxl] = temp_datPR[p];
                        pglk_datPI[p + i * _nPxl] = temp_datPI[p];
                        pglk_ctfP[p + i * _nPxl] = temp_ctfP[p];
                        pglk_sigRcpP[p + i * _nPxl] = temp_sigP[p];
                    }
                }

                ExpectGlobal2D(_iGPU,
                               _stream,
                               texObject,
                               devtraP,
                               devnR,
                               devpR,
                               devpT,
                               pglk_datPR,
                               pglk_datPI,
                               pglk_ctfP,
                               pglk_sigRcpP,
                               weightC + l * _para.k,
                               weightR + (long long)l * _para.k * nR,
                               weightT + (long long)l * _para.k * nT,
                               deviCol,
                               deviRow,
                               _para.k,
                               nR,
                               nT,
                               _model.proj(0).pf(),
                               _model.proj(0).interp(),
                               _para.size,
                               _model.proj(0).projectee2D().nRowFT(),
                               _nPxl,
                               batch,
                               _nGPU);

                l += batch;
            }

            hostFree(pglk_datPR);
            hostFree(pglk_datPI);
            hostFree(pglk_ctfP);
            hostFree(pglk_sigRcpP);

            free(pglk_datPR);
            free(pglk_datPI);
            free(pglk_ctfP);
            free(pglk_sigRcpP);

            freeRotran2D(_iGPU,
                         symArray,
                         texObject,
                         devtraP,
                         devnR,
                         devpR,
                         devpT,
                         _para.k,
                         _nGPU);

        }

        delete[] pr;
        delete[] pt;

        // reset weights of particle filter

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
            for (size_t t = 0; t < _para.k; t++)
            {
                Particle* par = &(_parAll[t * _ID.size() + l]);

                long long rShift;
                if (_para.mode == MODE_3D)
                {
                    rShift = (long long)t * nImg * nR + l * nR;
                }
                else
                {
                    rShift = (long long)l * _para.k * nR + t * nR;
                }
                
                for (int iR = 0; iR < nR; iR++)
                {
                    par->setUR(weightR[rShift + iR], iR);
                }

                long long tShift;
                if (_para.mode == MODE_3D)
                {
                    tShift = (long long)t * nImg * nT + l * nT;
                }
                else
                {
                    tShift = (long long)l * _para.k * nT + t * nT;
                }
                
                for (int iT = 0; iT < nT; iT++)
                {
                    par->setUT(weightT[tShift + iT], iT);
                }

#ifdef OPTIMISER_PEAK_FACTOR_R
                par->setPeakFactor(PAR_R);
                par->keepHalfHeightPeak(PAR_R);
#endif

#ifdef OPTIMISER_PEAK_FACTOR_T
                par->setPeakFactor(PAR_T);
                par->keepHalfHeightPeak(PAR_T);
#endif

                par->resample(_para.mLR, PAR_R);
                par->resample(_para.mLT, PAR_T);

                par->calVari(PAR_R);
                par->calVari(PAR_T);

#ifdef PARTICLE_RHO
                par->setRho(0);
                // if there is only two resampled points in translation, it is possible making pho be 1
                // then it will crash down
                // make rho to be 0
#endif

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_SCAN_SET_MIN_STD_WITH_PERTURB
                    par->setK1(TSGSL_MAX_RFLOAT((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                 ? _para.perturbFactorSGlobal
                                                 : _para.perturbFactorSLocal))
                                                 * MIN_STD_FACTOR * scanMinStdR,
                                                 par->k1()));
#else
                    par->setK1(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * scanMinStdR,
                                                par->k1()));
#endif
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_SCAN_SET_MIN_STD_WITH_PERTURB
                    par->setK1(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                              ? _para.perturbFactorSGlobal
                                                              : _para.perturbFactorSLocal))
                                                      * MIN_STD_FACTOR * scanMinStdR),
                                              par->k1()));
                    par->setK2(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                              ? _para.perturbFactorSGlobal
                                                              : _para.perturbFactorSLocal))
                                                      * MIN_STD_FACTOR * scanMinStdR),
                                              par->k2()));

                    par->setK3(TSGSL_MAX_RFLOAT(TSGSL_pow_2((1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                                              ? _para.perturbFactorSGlobal
                                                              : _para.perturbFactorSLocal))
                                                       * MIN_STD_FACTOR * scanMinStdR),
                                              par->k3()));
#else
                    par->setK1(TSGSL_MAX_RFLOAT(TSGSL_pow_2(MIN_STD_FACTOR * scanMinStdR),
                                                   par->k1()));
                    par->setK2(TSGSL_MAX_RFLOAT(TSGSL_pow_2(MIN_STD_FACTOR * scanMinStdR),
                                                   par->k2()));
                    par->setK3(TSGSL_MAX_RFLOAT(TSGSL_pow_2(MIN_STD_FACTOR * scanMinStdR),
                                               par->k3()));
#endif
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }

#ifdef OPTIMISER_SCAN_SET_MIN_STD_WITH_PERTURB
                par->setS0(TSGSL_MAX_RFLOAT(1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                           ? _para.perturbFactorSGlobal
                                           : _para.perturbFactorSLocal)
                                    * MIN_STD_FACTOR * scanMinStdT,
                                      par->s0()));

                par->setS1(TSGSL_MAX_RFLOAT(1.0 / ((_searchType == SEARCH_TYPE_GLOBAL)
                                           ? _para.perturbFactorSGlobal
                                           : _para.perturbFactorSLocal)
                                    * MIN_STD_FACTOR * scanMinStdT,
                                      par->s1()));
#else
                par->setS0(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * scanMinStdT,
                                           par->s0()));
                par->setS1(TSGSL_MAX_RFLOAT(MIN_STD_FACTOR * scanMinStdT,
                                           par->s1()));
#endif
            }
        }

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initial Phase of Global Search Performed.";
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initial Phase of Global Search Performed.";

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(_hemi);

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initial Phase of Global Search in Hemisphere A Performed";
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initial Phase of Global Search in Hemisphere B Performed";
#endif

        delete[] weightC;
        delete[] weightR;
        delete[] weightT;

        if (_searchType != SEARCH_TYPE_CTF)
            freePreCal(false);
        else
            freePreCal(true);

        //gettimeofday(&end, NULL);
        //time_use=(end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000;
        //if (_commRank == HEMI_A_LEAD)
        //    printf("Expectation globalA time_use:%lf\n", time_use);
        //else
        //    printf("Expectation globalB time_use:%lf\n", time_use);
    }

    long memUsageG = memoryCheckRM();
    printf("expect global memory check work done:%dG!\n", memUsageG / MEGABYTE);
    
#ifdef OPTIMISER_PARTICLE_FILTER

    //float time_use = 0;
    //struct timeval start;
    //struct timeval end;

    //gettimeofday(&start, NULL);

    if (_searchType != SEARCH_TYPE_CTF)
        allocPreCal(true, false, false);
    else
        allocPreCal(true, false, true);

    RFLOAT* devfreQ[_nGPU];

    if(_searchType == SEARCH_TYPE_CTF)
    {
        #pragma omp parallel for
        for (int i = 0; i < _nGPU; i++)
        {
            ExpectPrefre(_iGPU[i],
                         &devfreQ[i],
                         _frequency,
                         _nPxl);
        }
    }

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Local Search PreImg & frequency done.";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Local Search PreImg & frequency done.";

    _nP.resize(_ID.size(), 0);

    _nF = 0;
    _nI = 0;

    nPer = 0;

    int streamNum = _stream.size() / _nGPU;
    int buffNum = _nGPU * streamNum;

    ManagedArrayTexture *mgr2D[_nGPU * _para.k];
    ManagedArrayTexture *mgr3D[_nGPU];

    int interp = _model.proj(0).interp();
    int vdim;
    if (_para.mode == MODE_2D)
        vdim = _model.proj(0).projectee2D().nRowFT();
    else
        vdim = _model.proj(0).projectee3D().nSlcFT();

    if (_para.mode == MODE_2D)
    {
        #pragma omp parallel for
        for(int i = 0; i < _nGPU; i++)
        {
            for (int j = 0; j < _para.k; j++)
            {
                mgr2D[i * _para.k + j] = new ManagedArrayTexture();
                mgr2D[i * _para.k + j]->Init(_para.mode, vdim, _iGPU[i]);
                Complex* temp = &((const_cast<Image&>(_model.proj(j).projectee2D()))[0]);
                int sizeModel = _model.proj(j).projectee2D().sizeFT();
                ExpectLocalV2D(_iGPU[i],
                               mgr2D[i * _para.k + j],
                               temp,
                               sizeModel);
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for(int i = 0; i < _nGPU; i++)
        {
            mgr3D[i] = new ManagedArrayTexture();
            mgr3D[i]->Init(_para.mode, vdim, _iGPU[i]);
        }
    }

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Local Search texture object done.";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Local Search texture object done.";

    int cpyNum = omp_get_max_threads() / _nGPU;
    int cpyNumL = (omp_get_max_threads() % _nGPU == 0) ? cpyNum : cpyNum + 1;

    RFLOAT* devdatPR[_nGPU];
    RFLOAT* devdatPI[_nGPU];
    RFLOAT* devctfP[_nGPU];
    RFLOAT* devsigP[_nGPU];
    RFLOAT* devdefO[_nGPU];

    #pragma omp parallel for
    for (int i = 0; i < _nGPU; i++)
    {
        ExpectLocalIn(_iGPU[i],
                      &devdatPR[i],
                      &devdatPI[i],
                      &devctfP[i],
                      &devdefO[i],
                      &devsigP[i],
                      _nPxl,
                      cpyNumL,
                      _searchType);
    }

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Local Search GPU Image alloc done.";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Local Search GPU Image alloc done.";

    ManagedCalPoint *mcp[buffNum];

    #pragma omp parallel for
    for (int i = 0; i < _nGPU; i++)
    {
        for (int j = 0; j < streamNum; j++)
        {
            mcp[i * streamNum + j] = new ManagedCalPoint();
            mcp[i * streamNum + j]->Init(_para.mode,
                                         _searchType,
                                         _iGPU[i],
                                         _para.mLR,
                                         _para.mLT,
                                         _para.mLD,
                                         _nPxl);
        }
    }

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Local Search GPU Calculate buffer alloc done.";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Local Search GPU Calculate buffer alloc done.";

    RFLOAT* wC[omp_get_max_threads()];
    RFLOAT* wR[omp_get_max_threads()];
    RFLOAT* wT[omp_get_max_threads()];
    RFLOAT* wD[omp_get_max_threads()];
    double* oldR[omp_get_max_threads()];
    double* oldT[omp_get_max_threads()];
    double* oldD[omp_get_max_threads()];
    double* trans[omp_get_max_threads()];
    double* dpara[omp_get_max_threads()];
    double* rot[omp_get_max_threads()];

    RFLOAT* baseLine = new RFLOAT[_ID.size() * _para.k];
    omp_lock_t* mtx = new omp_lock_t[_nGPU];

    #pragma omp parallel for
    for(int i = 0; i < _nGPU; i++)
    {
        omp_init_lock(&mtx[i]);
    }

    for (int i = 0; i < omp_get_max_threads(); i++)
    {
        int gpuIdx;
        if (i / cpyNum > _nGPU)
            gpuIdx = i - _nGPU * cpyNum;
        else if (i / cpyNum == _nGPU)
            gpuIdx = i % cpyNum;
        else
            gpuIdx = i / cpyNum;

        ExpectLocalHostA(_iGPU[gpuIdx],
                         &wC[i],
                         &wR[i],
                         &wT[i],
                         &wD[i],
                         &oldR[i],
                         &oldT[i],
                         &oldD[i],
                         &trans[i],
                         &rot[i],
                         &dpara[i],
                         _para.mLR,
                         _para.mLT,
                         _para.mLD,
                         _searchType);
    }

    MemoryBazaarDustman<RFLOAT, BaseType, 4> datPRDustman(&_datPR);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> datPIDustman(&_datPI);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> ctfPDustman(&_ctfP);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> sigRcpPDustman(&_sigRcpP);

    for (int cls = 0; cls < _para.k; cls++)
    {
        if (_para.mode == MODE_3D)
        {
            Complex* temp = &((const_cast<Volume&>(_model.proj(cls).projectee3D()))[0]);
            #pragma omp parallel for
            for (int i = 0; i < _nGPU; i++)
            {
                ExpectLocalV3D(_iGPU[i],
                               mgr3D[i],
                               temp,
                               vdim);
            }
        }
        
        nPer = 0;
        #pragma omp parallel for schedule(dynamic) firstprivate(datPRDustman, datPIDustman, ctfPDustman, sigRcpPDustman)
        FOR_EACH_2D_IMAGE
        {
            Particle* par = &(_parAll[cls * _ID.size() + l]);
        
            _datPR.endLastVisit(l * _nPxl);
            _datPI.endLastVisit(l * _nPxl);

            if (_searchType != SEARCH_TYPE_CTF)
            {
                _ctfP.endLastVisit(l * _nPxl);
            }

            _sigRcpP.endLastVisit(l * _nPxl);

            RFLOAT* datPR = &_datPR[l * _nPxl];
            RFLOAT* datPI = &_datPI[l * _nPxl];
            RFLOAT* sigRcpP = &_sigRcpP[l * _nPxl];
            RFLOAT* ctfP;
            if (_searchType != SEARCH_TYPE_CTF)
            {
                ctfP = &_ctfP[l * _nPxl];
            }

            int threadId = omp_get_thread_num();
            int gpuIdx;
            if (threadId / cpyNum > _nGPU)
                gpuIdx = threadId - _nGPU * cpyNum;
            else if (threadId / cpyNum == _nGPU)
                gpuIdx = threadId % cpyNum;
            else
                gpuIdx = threadId / cpyNum;

            omp_set_lock(&mtx[gpuIdx]);

            if (threadId < _nGPU * cpyNum)
            {
                ExpectLocalP(_iGPU[gpuIdx],
                             devdatPR[gpuIdx],
                             devdatPI[gpuIdx],
                             devctfP[gpuIdx],
                             devsigP[gpuIdx],
                             devdefO[gpuIdx],
                             datPR,
                             datPI,
                             ctfP,
                             sigRcpP,
                             _defocusP + l * _nPxl,
                             threadId % cpyNum,
                             _nPxl,
                             _searchType);
            }
            else
            {
                ExpectLocalP(_iGPU[gpuIdx],
                             devdatPR[gpuIdx],
                             devdatPI[gpuIdx],
                             devctfP[gpuIdx],
                             devsigP[gpuIdx],
                             devdefO[gpuIdx],
                             datPR,
                             datPI,
                             ctfP,
                             sigRcpP,
                             _defocusP + l * _nPxl,
                             cpyNum,
                             _nPxl,
                             _searchType);
            }

            omp_unset_lock(&mtx[gpuIdx]);

            int nPhaseWithNoVariDecrease = 0;

#ifdef OPTIMISER_COMPRESS_CRITERIA
            double variR = DBL_MAX;
            double variT = DBL_MAX;
            double variD = DBL_MAX;
#else
            double k1 = 1;
            double k2 = 1;
            double k3 = 1;
            double tVariS0 = 5 * _para.transS;
            double tVariS1 = 5 * _para.transS;
            double dVari = 5 * _para.ctfRefineS;
#endif
            for (int phase = ((_searchType == SEARCH_TYPE_GLOBAL) ? 1 : 0); phase < ((_para.alignR || _para.alignT || _para.alignD) ? MAX_N_PHASE_PER_ITER : 1); phase++)
            {
#ifdef OPTIMISER_GLOBAL_PERTURB_LARGE
                if (phase == (_searchType == SEARCH_TYPE_GLOBAL) ? 1 : 0)
#else
                if (phase == 0)
#endif
                {
                    if (_para.alignR)
                    {
                        par->perturb(_para.perturbFactorL, PAR_R);
                    }
                    
                    if (_para.alignT)
                    {
                        par->perturb(_para.perturbFactorL, PAR_T);
                    }

                    if (_para.alignD && _searchType == SEARCH_TYPE_CTF)
                    {
                        par->initD(_para.mLD, _para.ctfRefineS);
                    }
                }
                else
                {
                    if (_para.alignR)
                    {
                        par->perturb((_searchType == SEARCH_TYPE_GLOBAL)
                                   ? _para.perturbFactorSGlobal
                                   : _para.perturbFactorSLocal,
                                     PAR_R);
                    }

                    if (_para.alignT)
                    {
                        par->perturb((_searchType == SEARCH_TYPE_GLOBAL)
                                   ? _para.perturbFactorSGlobal
                                   : _para.perturbFactorSLocal,
                                     PAR_T);
                    }

                    if (_para.alignD && _searchType == SEARCH_TYPE_CTF)
                    {
                        par->perturb(_para.perturbFactorSCTF, PAR_D);
                    }
                }

                for (int itr = 0; itr < _para.mLR; itr++)
                    oldR[threadId][itr] = par->wR(itr);

                for (int itr = 0; itr < _para.mLT; itr++)
                    oldT[threadId][itr] = par->wT(itr);

                for (int itr = 0; itr < par->nD(); itr++)
                    oldD[threadId][itr] = par->wD(itr);

                dvec2 t;
                for (int k = 0; k < _para.mLT; k++)
                {
                    par->t(t, k);
                    trans[threadId][k * 2] = t(0);
                    trans[threadId][k * 2 + 1] = t(1);
                }

                if (_para.mode == MODE_2D)
                {
                    dvec4 r;
                    for (int k = 0; k < _para.mLR; k++)
                    {
                        par->quaternion(r, k);
                        rot[threadId][k * 2] = r(0);
                        rot[threadId][k * 2 + 1] = r(1);
                    }
                }
                else
                {
                    dvec4 r;
                    for (int k = 0; k < _para.mLR; k++)
                    {
                        par->quaternion(r, k);
                        rot[threadId][k * 4] = r(0);
                        rot[threadId][k * 4 + 1] = r(1);
                        rot[threadId][k * 4 + 2] = r(2);
                        rot[threadId][k * 4 + 3] = r(3);
                    }
                }

                if (_searchType == SEARCH_TYPE_CTF)
                {
                    for (int k = 0; k < par->nD(); k++)
                        dpara[threadId][k] = (par->d())(k);
                }

                int streamId;
                int datId;
                int datShift;
                if (threadId < _nGPU * cpyNum)
                {
                    datShift = threadId % cpyNum;
                    streamId = (threadId % cpyNum) % streamNum;
                    datId = gpuIdx * streamNum + streamId;
                }
                else
                {
                    datShift = cpyNum;
                    streamId = cpyNum % streamNum;
                    datId = gpuIdx * streamNum + streamId;
                }

                omp_set_lock(&mtx[gpuIdx]);

                ExpectLocalRTD(_iGPU[gpuIdx],
                               mcp[datId],
                               oldR[threadId],
                               oldT[threadId],
                               oldD[threadId],
                               trans[threadId],
                               rot[threadId],
                               dpara[threadId]);

                if(_para.mode == MODE_2D)
                {
                    if (_searchType == SEARCH_TYPE_CTF)
                    {
                        ExpectLocalPreI2D(_iGPU[gpuIdx],
                                          datShift,
                                          mgr2D[gpuIdx * _para.k + cls],
                                          mcp[datId],
                                          devdefO[gpuIdx],
                                          devfreQ[gpuIdx],
                                          deviCol[gpuIdx],
                                          deviRow[gpuIdx],
                                          _ctfAttr[l].phaseShift,
                                          _ctfAttr[l].amplitudeContrast,
                                          _K1[l],
                                          _K2[l],
                                          _para.pf,
                                          _para.size,
                                          vdim,
                                          _nPxl,
                                          interp);
                    }
                    else
                    {
                        ExpectLocalPreI2D(_iGPU[gpuIdx],
                                          datShift,
                                          mgr2D[gpuIdx * _para.k + cls],
                                          mcp[datId],
                                          devdefO[gpuIdx],
                                          devfreQ[gpuIdx],
                                          deviCol[gpuIdx],
                                          deviRow[gpuIdx],
                                          _ctfAttr[l].phaseShift,
                                          _ctfAttr[l].amplitudeContrast,
                                          0,
                                          0,
                                          _para.pf,
                                          _para.size,
                                          vdim,
                                          _nPxl,
                                          interp);
                    }
                }
                else
                {
                    if (_searchType == SEARCH_TYPE_CTF)
                    {
                        ExpectLocalPreI3D(_iGPU[gpuIdx],
                                          datShift,
                                          mgr3D[gpuIdx],
                                          mcp[datId],
                                          devdefO[gpuIdx],
                                          devfreQ[gpuIdx],
                                          deviCol[gpuIdx],
                                          deviRow[gpuIdx],
                                          _ctfAttr[l].phaseShift,
                                          _ctfAttr[l].amplitudeContrast,
                                          _K1[l],
                                          _K2[l],
                                          _para.pf,
                                          _para.size,
                                          vdim,
                                          _nPxl,
                                          interp);
                    }
                    else
                    {
                        ExpectLocalPreI3D(_iGPU[gpuIdx],
                                          datShift,
                                          mgr3D[gpuIdx],
                                          mcp[datId],
                                          devdefO[gpuIdx],
                                          devfreQ[gpuIdx],
                                          deviCol[gpuIdx],
                                          deviRow[gpuIdx],
                                          _ctfAttr[l].phaseShift,
                                          _ctfAttr[l].amplitudeContrast,
                                          0,
                                          0,
                                          _para.pf,
                                          _para.size,
                                          vdim,
                                          _nPxl,
                                          interp);
                    }
                }

                ExpectLocalM(_iGPU[gpuIdx],
                             datShift,
                             //l,
                             mcp[datId],
                             devdatPR[gpuIdx],
                             devdatPI[gpuIdx],
                             devctfP[gpuIdx],
                             devsigP[gpuIdx],
                             wC[threadId],
                             wR[threadId],
                             wT[threadId],
                             wD[threadId],
                             baseLine + cls * _ID.size() + l, 
                             _nPxl);

                omp_unset_lock(&mtx[gpuIdx]);

                for (int iR = 0; iR < _para.mLR; iR++)
                    par->setUR(wR[threadId][iR], iR);

#ifdef OPTIMISER_PEAK_FACTOR_R
                par->keepHalfHeightPeak(PAR_R);
#endif

                for (int iT = 0; iT < _para.mLT; iT++)
                    par->setUT(wT[threadId][iT], iT);

#ifdef OPTIMISER_PEAK_FACTOR_T
                par->keepHalfHeightPeak(PAR_T);
#endif

                if (_searchType == SEARCH_TYPE_CTF)
                {
                    for (int iD = 0; iD < _para.mLD; iD++)
                        par->setUD(wD[threadId][iD], iD);

#ifdef OPTIMISER_PEAK_FACTOR_D
                    if (phase == 0) par->setPeakFactor(PAR_D);

                    par->keepHalfHeightPeak(PAR_D);
#endif
                }

#ifdef OPTIMISER_SAVE_PARTICLES
                if (_ID[l] < N_SAVE_IMG)
                {
                    par->sort();

                    char filename[FILE_NAME_LENGTH];

                    snprintf(filename,
                             sizeof(filename),
                             "R_Particle_%04d_Round_%03d_%03d.par",
                             _ID[l],
                             _iter,
                             phase);
                    save(filename, *par, PAR_R, true);
                    snprintf(filename,
                             sizeof(filename),
                             "T_Particle_%04d_Round_%03d_%03d.par",
                             _ID[l],
                             _iter,
                             phase);
                    save(filename, *par, PAR_T, true);
                    snprintf(filename,
                             sizeof(filename),
                             "D_Particle_%04d_Round_%03d_%03d.par",
                             _ID[l],
                             _iter,
                             phase);
                    save(filename, *par, PAR_D, true);
                }
#endif

                if (_para.alignR)
                {
                    par->calRank1st(PAR_R);
                    par->calVari(PAR_R);
                    par->resample(_para.mLR, PAR_R);
                }

                if (_para.alignT)
                {
                    par->calRank1st(PAR_T);
                    par->calVari(PAR_T);
                    par->resample(_para.mLT, PAR_T);
                }

                if (_para.alignD && (_searchType == SEARCH_TYPE_CTF))
                {
                    par->calRank1st(PAR_D);
                    par->calVari(PAR_D);
                    par->resample(_para.mLD, PAR_D);
                }

                if (phase >= ((_searchType == SEARCH_TYPE_GLOBAL)
                              ? MIN_N_PHASE_PER_ITER_GLOBAL
                              : MIN_N_PHASE_PER_ITER_LOCAL))
                {
#ifdef OPTIMISER_COMPRESS_CRITERIA
                    double variRCur;
                    double variTCur;
                    double variDCur;
#else
                    double k1Cur;
                    double k2Cur;
                    double k3Cur;
                    double tVariS0Cur;
                    double tVariS1Cur;
                    double dVariCur;
#endif

#ifdef OPTIMISER_COMPRESS_CRITERIA
                    variRCur = par->variR();
                    variTCur = par->variT();
                    variDCur = par->variD();
#else
                    par->vari(k1Cur, k2Cur, k3Cur, tVariS0Cur, tVariS1Cur, dVariCur);
#endif

                    if (_para.mode == MODE_2D)
                    {
#ifdef OPTIMISER_COMPRESS_CRITERIA
                        if ((variRCur < variR * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (variTCur < variT * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (variDCur < variD * PARTICLE_FILTER_DECREASE_FACTOR))
#else
                        if ((k1Cur < k1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (tVariS0Cur < tVariS0 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (tVariS1Cur < tVariS1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (dVariCur < dVari * PARTICLE_FILTER_DECREASE_FACTOR))
#endif
                        {
                            // there is still room for searching
                            nPhaseWithNoVariDecrease = 0;
                        }
                        else
                            nPhaseWithNoVariDecrease += 1;
                    }
                    else if (_para.mode == MODE_3D)
                    {
#ifdef OPTIMISER_COMPRESS_CRITERIA
                        if ((variRCur < variR * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (variTCur < variT * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (variDCur < variD * PARTICLE_FILTER_DECREASE_FACTOR))
#else
                        if ((k1Cur < k1 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                            (k2Cur < k2 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                            (k3Cur < k3 * gsl_pow_2(PARTICLE_FILTER_DECREASE_FACTOR)) ||
                            (tVariS0Cur < tVariS0 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (tVariS1Cur < tVariS1 * PARTICLE_FILTER_DECREASE_FACTOR) ||
                            (dVariCur < dVari * PARTICLE_FILTER_DECREASE_FACTOR))
#endif
                        {
                            // there is still room for searching
                            nPhaseWithNoVariDecrease = 0;
                        }
                        else
                            nPhaseWithNoVariDecrease += 1;
                    }
                    else
                    {
                        REPORT_ERROR("EXISTENT MODE");

                        abort();
                    }

#ifdef OPTIMISER_COMPRESS_CRITERIA

#ifndef NAN_NO_CHECK
                    POINT_NAN_CHECK(par->compressR());
                    POINT_NAN_CHECK(par->compressT());
#endif

                    if (variRCur < variR) variR = variRCur;
                    if (variTCur < variT) variT = variTCur;
                    if (variDCur < variD) variD = variDCur;
#else
                    // make tVariS0, tVariS1, rVari the smallest variance ever got
                    if (k1Cur < k1) k1 = k1Cur;
                    if (k2Cur < k2) k2 = k2Cur;
                    if (k3Cur < k3) k3 = k3Cur;
                    if (tVariS0Cur < tVariS0) tVariS0 = tVariS0Cur;
                    if (tVariS1Cur < tVariS1) tVariS1 = tVariS1Cur;
                    if (dVariCur < dVari) dVari = dVariCur;
#endif

                    // break if in a few continuous searching, there is no improvement
                    if (nPhaseWithNoVariDecrease == N_PHASE_WITH_NO_VARI_DECREASE)
                    {
                        _nP[l] = phase;

                        #pragma omp atomic
                        _nF += phase;

                        #pragma omp atomic
                        _nI += 1;

                        break;
                    }
                }
            }//phase end

            #pragma omp critical  (line1495)
            if (_nI > (int)(_ID.size() * _para.k / 10))
            {
                _nI = 0;

                nPer += 1;

                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", class " << cls << ", " << nPer * 10 << "\% Expectation Performed";
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", class " << cls << ", " << nPer * 10 << "\% Expectation Performed";
            }

#ifdef OPTIMISER_SAVE_PARTICLES
            if (_ID[l] < N_SAVE_IMG)
            {
                char filename[FILE_NAME_LENGTH];

                snprintf(filename,
                         sizeof(filename),
                         "R_Particle_%04d_Round_%03d_Final.par",
                         _ID[l],
                         _iter);
                save(filename, *(_par[l]), PAR_R);
                snprintf(filename,
                         sizeof(filename),
                         "T_Particle_%04d_Round_%03d_Final.par",
                         _ID[l],
                         _iter);
                save(filename, *(_par[l]), PAR_T);
                snprintf(filename,
                         sizeof(filename),
                         "D_Particle_%04d_Round_%03d_Final.par",
                         _ID[l],
                         _iter);
                save(filename, *(_par[l]), PAR_D);
            }
#endif
        }//image end

    }//class end

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        vec baseLineT = vec::Zero(_para.k);

        vec ratio = vec::Zero(_para.k);

        for (size_t t = 0; t < _para.k; t++)
        {
            baseLineT(t) = baseLine[t * _ID.size() + l];
        }

        RFLOAT maxBaseLine = baseLineT(value_max_index(baseLineT));

        for (size_t t = 0; t < _para.k; t++)
        {
            ratio(t) = exp(baseLineT(t) - maxBaseLine);
        }

        RFLOAT hh = ratio(value_max_index(ratio)) * PEAK_FACTOR_C;

        for (size_t t = 0; t < _para.k; t++)
        {
            if (ratio(t) < hh)
            {
                ratio(t) = 0;
            }
            else
            {
                ratio(t) -= hh;
            }
        }

        _iRefPrev[l] = _iRef[l];

        _iRef[l] = drawWithWeightIndex(ratio);

        _par[l] = &(_parAll[_iRef[l] * _ID.size() + l]);
    }

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space for Pre-calculation in Expectation";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space for Pre-calculation in Expectation";

    for (int i = 0; i < omp_get_max_threads(); i++)
    {
        int gpuIdx;
        if (i / cpyNum > _nGPU)
            gpuIdx = i - _nGPU * cpyNum;
        else if (i / cpyNum == _nGPU)
            gpuIdx = i % cpyNum;
        else
            gpuIdx = i / cpyNum;

        ExpectLocalHostF(_iGPU[gpuIdx],
                         &wC[i],
                         &wR[i],
                         &wT[i],
                         &wD[i],
                         &oldR[i],
                         &oldT[i],
                         &oldD[i],
                         &trans[i],
                         &rot[i],
                         &dpara[i],
                         _searchType);
    }

    #pragma omp parallel for
    for (int i = 0; i < _nGPU; i++)
    {
        ExpectLocalFin(_iGPU[i],
                       &devdatPR[i],
                       &devdatPI[i],
                       &devctfP[i],
                       &devdefO[i],
                       &devfreQ[i],
                       &devsigP[i],
                       _searchType);
    }

    delete[] mtx;
    for (int i = 0; i < buffNum; i++)
        delete mcp[i];

    if (_para.mode == MODE_2D)
    {
        for (int i = 0; i < _nGPU * _para.k; i++)
            delete mgr2D[i];
    }
    else
    {
        for (int i = 0; i < _nGPU; i++)
            delete mgr3D[i];
    }

    if (_searchType != SEARCH_TYPE_CTF)
        freePreCal(false);
    else
        freePreCal(true);

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space in Expectation GPU";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space in Expectation GPU";

    //gettimeofday(&end, NULL);
    //time_use=(end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000;
    //if (_commRank == HEMI_A_LEAD)
    //    printf("Expectation LocalA time_use:%lf\n", time_use);
    //else
    //    printf("Expectation LocalB time_use:%lf\n", time_use);
#endif // OPTIMISER_PARTICLE_FILTER

    #pragma omp parallel for num_threads(_nGPU)
    for (int i = 0; i < _nGPU; i++)
    {
        ExpectFreeIdx(_iGPU[i],
                      &deviCol[i],
                      &deviRow[i]);
    }

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space GPU iCol & iRow";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space GPU iCol & iRow";

    freePreCalIdx();
    
    long memUsageL = memoryCheckRM();
    printf("expect local memory check work done:%dG!\n", memUsageL / MEGABYTE);
    
    printf("Round:%d, after expectation GPU memory check!\n", _iter);
    gpuMemoryCheck(_iGPU,
                   _commRank,
                   _nGPU);
}
#endif

void Optimiser::maximization()
{
#ifdef OPTIMISER_NORM_CORRECTION
    if ((_iter != 0) && (_searchType != SEARCH_TYPE_GLOBAL))
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Normalisation Noise";

        normCorrection();
    }
#endif

#ifdef OPTIMISER_REFRESH_SIGMA
    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Generating Sigma for the Next Iteration";

    allReduceSigma(_para.groupSig);

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Sigma Generated for the Next Iteration";
#endif

#endif

#ifdef OPTIMISER_CORRECT_SCALE
    if ((_searchType == SEARCH_TYPE_GLOBAL) &&
        (_para.groupScl) &&
        (_iter != 0))
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-balancing Intensity Scale for Each Group";

        correctScale(false, true);

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Intensity Scale Re-balanced for Each Group";
#endif
    }
#endif

    if (!_para.skipR)
    {

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_RECONSTRUCT_FREE_IMG_STACK_TO_SAVE_MEM

        if (_searchType != SEARCH_TYPE_GLOBAL)
        {
#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("Before Freeing Image Stacks in Reconstruction");

#endif
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Image Stacks";

            FOR_EACH_2D_IMAGE
            {
                _img[l].clear();
            }

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Image Stacks Freed";
#endif

#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("After Freeing Image Stacks in Reconstruction");

#endif
        }

#endif
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space in Reconstructor(s)";

        NT_MASTER
        {
            for (int t = 0; t < _para.k; t++)
                _model.reco(t).allocSpace(_para.nThreadsPerProcess);
        }

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space Allocated in Reconstructor(s)";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructing Reference(s)";

        reconstructRef(true, true, true, false, false);

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference(s) Reconstructed";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space in Reconstructor(s)";

        NT_MASTER
        {
            for (int t = 0; t < _para.k; t++)
                _model.reco(t).freeSpace();
        }

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space Freed in Reconstructor(s)";
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_RECONSTRUCT_FREE_IMG_STACK_TO_SAVE_MEM

        if (_searchType != SEARCH_TYPE_GLOBAL)
        {
#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("Before Allocating Image Stacks in Reconstruction");

#endif
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Image Stacks";

            FOR_EACH_2D_IMAGE
            {
                _img[l].alloc(_para.size, _para.size, FT_SPACE);

                SET_0_FT(_img[l]);
            }

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Image Stacks Allocated";
#endif

#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("After Allocating Image Stacks in Reconstruction");

#endif
        }

#endif
#endif
    }
    else
    {
        _model.setFSC(mat::Constant(_model.rU(), _para.k, 1));
    }
}

void Optimiser::run()
{
    //MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Initialising Optimiser";
    MLOG(INFO, "LOGGER_ROUND") << "Initialising Optimiser";

    init();

#ifdef OPIMISER_LOG_MEM_USAGE

    CHECK_MEMORY_USAGE("After Initialising Optimiser");

#endif

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Some Data";

#ifdef OPTIMISER_SAVE_IMAGES

    saveImages();

#endif

    /***
    saveCTFs();
    saveBinImages();
    saveLowPassImages();
    ***/

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef OPTIMISER_SAVE_SIGMA
    saveSig();
#endif

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Entering Iteration";
    for (_iter = 0; _iter < _para.iterMax; _iter++)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Round " << _iter;

        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Search Type ( Round "
                                       << _iter
                                       << " ) : Global Search";
        }
        else if (_searchType == SEARCH_TYPE_LOCAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Search Type ( Round "
                                       << _iter
                                       << " ) : Local Search";
        }
        else if (_searchType == SEARCH_TYPE_CTF)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Search Type ( Round "
                                       << _iter
                                       << " ) : CTF Search";
        }
        else
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Search Type ( Round "
                                       << _iter
                                       << " ) : Stop Search";

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Exitting Searching";

            break;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if ((_iter == 0) || (!_para.skipE))
        {
#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("Before Performing Expectation");

#endif

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Performing Expectation";

#ifdef GPU_VERSION
            //float time_use = 0;
            //struct timeval start;
            //struct timeval end;

            //gettimeofday(&start, NULL);
            expectationG();
            //gettimeofday(&end, NULL);
            //time_use=(end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000;
            //if (_commRank == HEMI_A_LEAD)
            //    printf("itr:%d, ExpectationA time_use:%lf\n", _iter, time_use);
            //else if (_commRank == HEMI_B_LEAD)
            //    printf("itr:%d, ExpectationB time_use:%lf\n", _iter, time_use);
#else
            //float time_use = 0;
            //struct timeval start;
            //struct timeval end;

            //gettimeofday(&start, NULL);
            expectation();
            //gettimeofday(&end, NULL);
            //time_use=(end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec) / 1000000;
            //if (_commRank == HEMI_A_LEAD)
            //    printf("itr:%d, ExpectationA time_use:%lf\n", _iter, time_use);
            //else if (_commRank == HEMI_B_LEAD)
            //    printf("itr:%d, ExpectationB time_use:%lf\n", _iter, time_use);
#endif

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Waiting for All Processes Finishing Expectation";

#ifdef VERBOSE_LEVEL_1
            ILOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Expectation Accomplished, with Filtering "
                                       << _nF
                                       << " Times over "
                                       << _ID.size()
                                       << " Images";
#endif

            MPI_Barrier(MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "All Processes Finishing Expectation";

#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("After Performing Expectation");

#endif
        }

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Determining Percentage of Images Belonging to Each Class";

        refreshClassDistr();

        for (int t = 0; t < _para.k; t++)
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << _cDistr(t) * 100
                                       << "\% Percentage of Images Belonging to Class "
                                       << t;

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Percentage of Images Belonging to Each Class Determined";
#endif

#ifdef OPTIMISER_SAVE_BEST_PROJECTIONS

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Best Projections";
        saveBestProjections();

#endif

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Best Projections Saved";
#endif

        if (_para.saveTHUEachIter)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Database";

            saveDatabase();

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Database Saved";
#endif
        }

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Calculating Variance of Rotation and Translation";

        refreshVariance();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Variance of Rotation and Translation Calculated";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Rotation Variance ( Round "
                                   << _iter
                                   << " ) : "
                                   << _model.rVari();

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Translation Variance ( Round "
                                   << _iter
                                   << " ) : "
                                   << _model.tVariS0()
                                   << ", "
                                   << _model.tVariS1();

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Standard Deviation of Rotation Variance : "
                                   << _model.stdRVari();

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Standard Deviation of Translation Variance : "
                                   << _model.stdTVariS0()
                                   << ", "
                                   << _model.stdTVariS1();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Variance of Rotation and Translation Calculated";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Calculating Changes of Rotation Between Iterations";
        refreshRotationChange();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Changes of Rotation Between Iterations Calculated";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Average Rotation Change : " << _model.rChange();
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Standard Deviation of Rotation Change : "
                                   << _model.stdRChange();

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Calculating Changes of Classification Between Iterations";
        refreshClassChange();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Changes of Classification Between Iterations Calculated";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Classification Change : " << _model.cChange();

        if (!_para.skipM)
        {
#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("Before Performing Maximization");

#endif

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Performing Maximization";

            maximization();

#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("After Performing Maximization");

#endif
        }
        else
        {
            _model.setFSC(mat::Constant(_model.rU(), _para.k, 1));
        }

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION

        if (_searchType != SEARCH_TYPE_GLOBAL)
        {
#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("Before Re-Centring Images");

#endif
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Centring Images";

            reCentreImg();

#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("After Re-Centring Images");

#endif

#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("Before Re-Masking Images");

#endif

#ifdef OPTIMISER_MASK_IMG
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Masking Images";
#ifdef GPU_VERSION
            reMaskImgG();
#else
            reMaskImg();
#endif
#endif

#ifdef OPTIMISER_LOG_MEM_USAGE

            CHECK_MEMORY_USAGE("After Re-Masking Images");

#endif
        }

#endif

        /***
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION

        if (_searchType != SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Centring Images";

            reCentreImg();
        }
        else
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Loading Images from Original Images";

            _img.clear();
            FOR_EACH_2D_IMAGE
                _img.push_back(_imgOri[l].copyImage());
        }

#else

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Loading Images from Original Images";

        _img.clear();
        FOR_EACH_2D_IMAGE
            _img.push_back(_imgOri[l].copyImage());

#endif

#ifdef OPTIMISER_MASK_IMG
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Masking Images";
        reMaskImg();
#endif
        ***/

#ifdef OPTIMISER_SAVE_SIGMA
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Sigma";
        saveSig();
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Maximization Performed";

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Calculating SNR(s)";
        _model.refreshSNR();

#ifdef OPTIMISER_SAVE_FSC
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving FSC(s)";
        saveFSC();
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Class Information";
        saveClassInfo();

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Current Cutoff Frequency: "
                                   << _r - 1
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_r - 1,
                                                   _para.size,
                                                   _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "FSC Area Below Cutoff Frequency: "
                                   << _model.fscArea();

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Recording Current Resolution";

        _resReport = _model.resolutionP(_para.thresReportFSC, false);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Current Resolution for Report, ( Round "
                                   << _iter
                                   << " ) : "
                                   << _resReport
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_resReport, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        _model.setRes(_resReport);

        _resCutoff = _model.resolutionP(_para.thresCutoffFSC, false);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Current Resolution for Cutoff, ( Round "
                                   << _iter
                                   << " ) : "
                                   << _resCutoff
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_resCutoff, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Calculating FSC Area";

        _model.setFSCArea(_model.fsc().topRows(_resCutoff).sum());

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Updating Cutoff Frequency in Model";

        _model.updateR(_para.thresCutoffFSC);

        if (_para.k == 1)
        {
#ifdef MODEL_DETERMINE_INCREASE_R_R_CHANGE
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Increasing Cutoff Frequency or Not: "
                                       << _model.increaseR()
                                       << ", as the Rotation Change is "
                                       << _model.rChange()
                                       << " and the Previous Rotation Change is "
                                       << _model.rChangePrev();
#endif

#ifdef MODEL_DETERMINE_INCREASE_R_T_VARI
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Increasing Cutoff Frequency or Not: "
                                       << _model.increaseR()
                                       << ", as the Translation Variance is "
                                       << _model.tVariS0()
                                       << ", "
                                       << _model.tVariS1()
                                       << ", and the Previous Translation Variance is "
                                       << _model.tVariS0Prev()
                                       << ", "
                                       << _model.tVariS1Prev();
#endif

#ifdef MODEL_DETERMINE_INCREASE_FSC
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Increasing Cutoff Frequency or Not: "
                                       << _model.increaseR()
                                       << ", as the FSC Area is "
                                       << _model.fscArea()
                                       << ", and the Previous FSC Area is "
                                       << _model.fscAreaPrev();
#endif
        }
        else
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Increasing Cutoff Frequency or Not: "
                                       << _model.increaseR()
                                       << ", as the Classification Change is "
                                       << _model.cChange()
                                       << ", and the Previous Classification Change is "
                                       << _model.cChangePrev();
        }

        if (_model.r() > _model.rT())
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Resetting Parameters Determining Increase Frequency";

            //_model.resetTVari();
            //_model.resetFSCArea();

            //_model.resetCChange();
            _model.resetRChange();

            _model.setNRChangeNoDecrease(0);

            _model.setNTopResNoImprove(0);
            _model.setIncreaseR(false);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Recording Current Highest Frequency";

            _model.setRT(_model.r());
        }

#ifdef OPTIMISER_SOLVENT_FLATTEN

        /***
        if ((_para.globalMask) || (_searchType != SEARCH_TYPE_GLOBAL))
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Solvent Flattening";

            solventFlatten(_para.performMask);
        }
        ***/

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Solvent Flattening";

        if ((_para.globalMask) || (_searchType != SEARCH_TYPE_GLOBAL))
            solventFlatten(_para.performMask);
        else
            solventFlatten(false);

#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Determining the Search Type of the Next Iteration";
        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            _searchType = _model.searchType();

            if (_para.performMask &&
                _para.autoMask &&
                (_searchType == SEARCH_TYPE_LOCAL))
            {
                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "A Mask Should be Generated";

                _genMask = true;
            }
        }
        else
            _searchType = _model.searchType();

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Recording Top Resolution";
        if (_resReport > _model.resT())
            _model.setResT(_resReport);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Updating Cutoff Frequency";
        _r = _model.r();

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "New Cutoff Frequency: "
                                   << _r - 1
                                   << " (Spatial), "
                                   << 1.0 / resP2A(_r - 1, _para.size, _para.pixelSize)
                                   << " (Angstrom)";

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Updating Frequency Boundary of Reconstructor";
        _model.updateRU();

        NT_MASTER
        {
            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Refreshing Projectors";
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Refreshing Projectors";

            _model.refreshProj(_para.nThreadsPerProcess);

            /***
            if (_searchType == SEARCH_TYPE_CTF)
            {
                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Resetting to Nyquist Limit in CTF Refine";
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Resetting to Nyquist Limit in CTF Refine";

                _model.setRU(maxR());
            }
            ***/

            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Resetting Reconstructors";
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Resetting Reconstructors";

            _model.resetReco(_para.thresReportFSC);
        }
    }

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Preparing to Reconstruct Reference(s) at Nyquist";

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Resetting to Nyquist Limit";
    _model.setMaxRU();

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Refreshing Reconstructors";
    NT_MASTER
    {
        _model.resetReco(_para.thresReportFSC);
    }

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructing References(s) at Nyquist";

#ifdef OPTIMISER_RECONSTRUCT_FREE_IMG_STACK_TO_SAVE_MEM

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Image Stacks";

    FOR_EACH_2D_IMAGE
    {
        _img[l].clear();
    }

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Image Stacks Freed";
#endif

#endif

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space in Reconstructor(s)";

    NT_MASTER
    {
        for (int t = 0; t < _para.k; t++)
            _model.reco(t).allocSpace(_para.nThreadsPerProcess);
    }

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space Allocated in Reconstructor(s)";
#endif

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructing Final Reference(s)";

    if (_para.subtract)
    {
        reconstructRef(true, false, false, false, true);
    }
    else
    {
        reconstructRef(true, false, true, false, true);
    }

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space in Reconstructor(s)";

    NT_MASTER
    {
        for (int t = 0; t < _para.k; t++)
            _model.reco(t).freeSpace();
    }

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space Freed in Reconstructor(s)";
#endif

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Final Reference(s) Reconstructed";
#endif

    if (!_para.subtract)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Final Class Information";

        saveClassInfo(true);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Final FSC(s)";

        saveFSC(true);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Final .thu File";

        saveDatabase(true);
    }

    if (_para.subtract)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Loading Images from Original Images";

        _img.cleanUp();

        _img.setUp(omp_get_max_threads(), _ID.size(), _md.nStallImg, serializeSize(Image(_para.size, _para.size, RL_SPACE)), 1, _para.cacheDirectory);

        FOR_EACH_2D_IMAGE
        {
            _img[l] = _imgOri[l].copyImage();
            // _img.push_back(_imgOri[l].copyImage());
        }

#ifdef OPTIMISER_MASK_IMG
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Re-Masking Images";
#ifdef GPU_VERSION
        reMaskImgG();
#else
        reMaskImg();
#endif
#endif

        if (strcmp(_para.regionCentre, "") != 0)
        {
            ImageFile imf(_para.regionCentre, "rb");
            imf.readMetaData();

            Volume cr;

            imf.readVolume(cr);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Recording Region Centre";
            _regionCentre = centroid(cr, _para.nThreadsPerProcess);

            /***
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Inversing Mask for Subtraction";

            Volume tmp(_para.size, _para.size, _para.size, RL_SPACE);

            #pragma omp parallel for
            SET_1_RL(tmp);

            #pragma omp parallel for
            SUB_RL(tmp, _mask);

            softMask(tmp, tmp, _para.maskRadius / _para.pixelSize, EDGE_WIDTH_RL, 0);

            _mask.swap(tmp);
            ***/
        }
        else
        {
            _regionCentre = vec3::Zero();
        }


        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Subtracting Masked Region Reference From Images";

        _r = maxR();

        _model.setR(_r);

        for (int pass = 0; pass < 2; pass++)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Entering Pass " << pass << " of Subtraction";

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Averaging Reference(s) From Two Hemispheres";
            _model.avgHemi();

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference(s) From Two Hemispheres Averaged";
#endif

            NT_MASTER
            {
                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Masking Reference(s)";
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Masking Reference(s)";

                if (pass == 0)
                    solventFlatten(false);
                else
                    solventFlatten(true);

#ifdef VERBOSE_LEVEL_1
                MPI_Barrier(_hemi);

                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference(s) Masked";
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference(s) Masked";
#endif

                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Refreshing Projectors";
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Refreshing Projectors";

                _model.refreshProj(_para.nThreadsPerProcess);

#ifdef VERBOSE_LEVEL_1
                MPI_Barrier(_hemi);

                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Projectors Refreshed";
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Projectors Refreshed";
#endif
            }

            if (pass == 0)
            {
#ifdef OPTIMISER_NORM_CORRECTION
                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Normalising Noise";

                normCorrection();

#ifdef VERBOSE_LEVEL_1
                MPI_Barrier(MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Noise Normalised";
#endif

#endif

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Refreshing Reconstructors";

                NT_MASTER
                {
                    _model.resetReco(_para.thresReportFSC);
                }

#ifdef VERBOSE_LEVEL_1
                MPI_Barrier(MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructors Refreshed";
#endif

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space in Reconstructor(s)";

                NT_MASTER
                {
                    for (int t = 0; t < _para.k; t++)
                    _model.reco(t).allocSpace(_para.nThreadsPerProcess);
                }

#ifdef VERBOSE_LEVEL_1
                MPI_Barrier(MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Space Allocated in Reconstructor(s)";
#endif

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructing References(s) at Nyquist After Normalising Noise";

                reconstructRef(true, false, false, false, true);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space in Reconstructor(s)";

                NT_MASTER
                {
                    for (int t = 0; t < _para.k; t++)
                    _model.reco(t).freeSpace();
                }


#ifdef VERBOSE_LEVEL_1
                MPI_Barrier(MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "References(s) at Nyquist After Normalising Noise Reconstructed";
#endif
            }

#ifdef VERBOSE_LEVEL_1
            MPI_Barrier(MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Finishing Pass " << pass << " of Subtraction";
#endif
        }

#ifdef OPTIMISER_SAVE_BEST_PROJECTIONS

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Best Projections";
        saveBestProjections();

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Best Projections Saved";
#endif

#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Masked Region Reference Subtracted Images";

        if (_para.symmetrySubtract)
        {
            saveSubtract(true, _para.reboxSize);
        }
        else
        {
            saveSubtract(false, _para.reboxSize);
        }


#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Masked Region Reference Subtracted Images Saved";
#endif

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Database of Masked Region Reference Subtracted Images";
        saveDatabase(true, true, _para.symmetrySubtract);

#ifdef VERBOSE_LEVEL_1
        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Database of Masked Region Reference Subtracted Images Saved";
#endif
    }

#ifdef GPU_VERSION
    MLOG(INFO, "LOGGER_GPU") << "Destory GPU variable for Each Process";
    destoryGPUEnv();
#endif
}

void Optimiser::clear()
{
    _img.cleanUp();
    _par.clear();
    _ctf.clear();
}

void Optimiser::bCastNPar()
{
    _nPar = _db.nParticle();
}

void Optimiser::allReduceN()
{
    IF_MASTER return;

    _N = _db.nParticleRank();

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_N, 1, MPI_INT, MPI_SUM, _hemi);

    MPI_Barrier(_hemi);
}

int Optimiser::size() const
{
    return _para.size;
}

int Optimiser::maxR() const
{
    return size() / 2 - 1;
}

void Optimiser::bcastGroupInfo()
{
    ALOG(INFO, "LOGGER_INIT") << "Storing GroupID";

    _groupID.clear();

    NT_MASTER
        FOR_EACH_2D_IMAGE
            _groupID.push_back(_db.groupID(_ID[l]));

    MLOG(INFO, "LOGGER_INIT") << "Getting Number of Groups from Database";

    _nGroup = _db.nGroup();

    MLOG(INFO, "LOGGER_INIT") << "Number of Group: " << _nGroup;

    MLOG(INFO, "LOGGER_INIT") << "Setting Up Space for Storing Sigma";
    NT_MASTER
    {
        _sig.resize(_nGroup, maxR() + 1);
        _sigRcp.resize(_nGroup, maxR());

        _svd.resize(_nGroup, maxR() + 1);
    }

    MLOG(INFO, "LOGGER_INIT") << "Setting Up Space for Storing Intensity Scale";
    _scale.resize(_nGroup);
}

void Optimiser::initRef()
{
    FFT fft;

    if (strcmp(_para.initModel, "") != 0)
    {
        MLOG(INFO, "LOGGER_INIT") << "Read Initial Model from Hard-disk";

        Volume ref;

        ImageFile imf(_para.initModel, "rb");
        imf.readMetaData();
        imf.readVolume(ref);

        if (_para.mode == MODE_2D)
        {
            if ((ref.nColRL() != _para.size) ||
                (ref.nRowRL() != _para.size) ||
                (ref.nSlcRL() != 1))
            {
                CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of Appending Reference"
                                          << ": size = " << _para.size
                                          << ", nCol = " << ref.nColRL()
                                          << ", nRow = " << ref.nRowRL()
                                          << ", nSlc = " << ref.nSlcRL();

                abort();
            }
        }
        else if (_para.mode == MODE_3D)
        {
            if ((ref.nColRL() != _para.size) ||
                (ref.nRowRL() != _para.size) ||
                (ref.nSlcRL() != _para.size))
            {
                CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of Appending Reference"
                                          << ": size = " << _para.size
                                          << ", nCol = " << ref.nColRL()
                                          << ", nRow = " << ref.nRowRL()
                                          << ", nSlc = " << ref.nSlcRL();

                abort();
            }
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

#ifdef OPTIMISER_INIT_REF_REMOVE_NEG
        #pragma omp parallel for
        FOR_EACH_PIXEL_RL(ref)
            if (ref(i) < 0) ref(i) = 0;
#endif

        _model.clearRef();

        for (int t = 0; t < _para.k; t++)
        {
            if (_para.mode == MODE_2D)
            {
                _model.appendRef(ref.copyVolume());
            }
            else if (_para.mode == MODE_3D)
            {
                _model.appendRef(ref.copyVolume());
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }

            fft.fw(_model.ref(t), _para.nThreadsPerProcess);
            // _model.ref(t).clearRL();
        }
    }
    else
    {
        MLOG(INFO, "LOGGER_INIT") << "Initial Model is not Provided";

        if (_para.mode == MODE_2D)
        {
            Image ref(_para.size,
                      _para.size,
                      RL_SPACE);

            /***
            IMAGE_FOR_EACH_PIXEL_RL(ref)
            {
                if (NORM(i, j) < _para.maskRadius / _para.pixelSize)
                    ref.setRL(1, i, j);
                else
                    ref.setRL(0, i, j);
            }
            ***/

            // softMask(ref, _para.maskRadius / _para.pixelSize, EDGE_WIDTH_RL);

            SET_0_RL(ref);

            Volume volRef(_para.size,
                          _para.size,
                          1,
                          RL_SPACE);

            COPY_RL(volRef, ref);

            _model.clearRef();

            for (int t = 0; t < _para.k; t++)
            {
                _model.appendRef(volRef.copyVolume());

                fft.fw(_model.ref(t), _para.nThreadsPerProcess);
                // _model.ref(t).clearRL();
            }
        }
        else if (_para.mode == MODE_3D)
        {
            Volume ref(_para.size,
                       _para.size,
                       _para.size,
                       RL_SPACE);

            /***
            VOLUME_FOR_EACH_PIXEL_RL(ref)
            {
                if (NORM_3(i, j, k) < _para.maskRadius / _para.pixelSize)
                    ref.setRL(1, i, j, k);
                else
                    ref.setRL(0, i, j, k);
            }
            ***/

            // softMask(ref, 0.7 * _para.maskRadius / _para.pixelSize, EDGE_WIDTH_RL);

            SET_0_RL(ref);

            _model.clearRef();

            for (int t = 0; t < _para.k; t++)
            {
                _model.appendRef(ref.copyVolume());

                fft.fw(_model.ref(t), _para.nThreadsPerProcess);
                // _model.ref(t).clearRL();
            }

        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }
}

void Optimiser::initMask()
{
    ImageFile imf(_para.mask, "rb");
    imf.readMetaData();

    imf.readVolume(_mask);
}

void Optimiser::initID()
{
    _ID.clear();

    for (int i = _db.start(); i <= _db.end(); i++)
        _ID.push_back(i);
}

void Optimiser::initImg()
{
    ALOG(INFO, "LOGGER_INIT") << "Reading Images from Disk";
    BLOG(INFO, "LOGGER_INIT") << "Reading Images from Disk";

    _img.cleanUp();

    // _img.resize(_ID.size());
    _img.setUp(omp_get_max_threads(), _ID.size(), _md.nStallImg, serializeSize(Image(_para.size, _para.size, RL_SPACE)), 1, _para.cacheDirectory);

    int nPer = 0;
    int nImg = 0;

#ifdef OPTIMISER_LOG_MEM_USAGE

    CHECK_MEMORY_USAGE("Before Reading 2D Images");

#endif

    string imgName;

    MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
    #pragma omp parallel for private(imgName) firstprivate(imgDustman)
    FOR_EACH_2D_IMAGE
    {
        nImg += 1;

        #pragma omp critical
        if (nImg >= (int)_ID.size() / 10)
        {
            nPer += 1;

            ALOG(INFO, "LOGGER_SYS") << nPer * 10 << "\% Percentage of Images Read";
            BLOG(INFO, "LOGGER_SYS") << nPer * 10 << "\% Percentage of Images Read";

            nImg = 0;
        }

        #pragma omp critical
        imgName = _db.path(_ID[l]);

        if (imgName.find('@') == string::npos)
        {
            ImageFile imf((string(_para.parPrefix) + imgName).c_str(), "rb");
            imf.readMetaData();
            imf.readImage(_img[l]);
        }
        else
        {
            if (imgName.find('$') == string::npos)
            {
                int nSlc = atoi(imgName.substr(0, imgName.find('@')).c_str()) - 1;
                string filename = string(_para.parPrefix) + imgName.substr(imgName.find('@') + 1);

                ImageFile imf(filename.c_str(), "rb");
                imf.readMetaData();
                imf.readImage(_img[l], nSlc);
            }
            else
            {
                int nSlc = atoi(imgName.substr(imgName.find('$') + 1, imgName.find('@')).c_str()) - 1;
                string filename = string(_para.parPrefix) + imgName.substr(imgName.find('@') + 1);

                ImageFile imf(filename.c_str(), "rb");
                imf.readMetaData();
                imf.readImage(_img[l], nSlc);
            }

            /***
            int nSlc = atoi(imgName.substr(0, imgName.find('@')).c_str()) - 1;
            string filename = string(_para.parPrefix) + imgName.substr(imgName.find('@') + 1);
            
            ImageFile imf(filename.c_str(), "rb");
            imf.readMetaData();
            imf.readImage(_img[l], nSlc);
            ***/
        }

        if ((_img[l].nColRL() != _para.size) ||
            (_img[l].nRowRL() != _para.size))
        {
            CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of 2D Images, "
                                      << "Should be "
                                      << _para.size
                                      << " x "
                                      << _para.size
                                      << ", but "
                                      << _img[l].nColRL()
                                      << " x "
                                      << _img[l].nRowRL()
                                      << " Input.";

            abort();
        }
    }

#ifdef OPTIMISER_LOG_MEM_USAGE
    CHECK_MEMORY_USAGE("After Reading 2D Images");
#endif

#ifdef VERBOSE_LEVEL_1
    ILOG(INFO, "LOGGER_INIT") << "Images Read from Disk";
#endif

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Images Read from Disk";
    BLOG(INFO, "LOGGER_INIT") << "Images Read from Disk";
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    ALOG(INFO, "LOGGER_INIT") << "Setting 0 to Offset between Images and Original Images";
    BLOG(INFO, "LOGGER_INIT") << "Setting 0 to Offset between Images and Original Images";

    _offset = vector<dvec2>(_ID.size(), dvec2(0, 0));

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Offset between Images and Original Images are Set to 0";
    BLOG(INFO, "LOGGER_INIT") << "Offset between Images and Original Images are Set to 0";
#endif
#endif

    ALOG(INFO, "LOGGER_INIT") << "Subtracting Mean of Noise, Making the Noise Have Zero Mean";
    BLOG(INFO, "LOGGER_INIT") << "Subtracting Mean of Noise, Making the Noise Have Zero Mean";

    substractBgImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Mean of Noise Subtracted";
    BLOG(INFO, "LOGGER_INIT") << "Mean of Noise Subtracted";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Performing Statistics of 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Performing Statistics of 2D Images";

    statImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Statistics Performed of 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Statistics Performed of 2D Images";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images Before Normalising";
    BLOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images Before Normalising";

    displayStatImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Statistics of 2D Images Before Normalising Displayed";
    BLOG(INFO, "LOGGER_INIT") << "Statistics of 2D Images Before Normalising Displayed";
#endif

#ifdef OPTIMISER_LOG_MEM_USAGE
    CHECK_MEMORY_USAGE("Before Masking on 2D Images");
#endif

    ALOG(INFO, "LOGGER_INIT") << "Masking on 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Masking on 2D Images";

    maskImg();

#ifdef OPTIMISER_LOG_MEM_USAGE
    CHECK_MEMORY_USAGE("After Masking on 2D Images");
#endif

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "2D Images Masked";
    BLOG(INFO, "LOGGER_INIT") << "2D Images Masked";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Normalising 2D Images, Making the Noise Have Standard Deviation of 1";
    BLOG(INFO, "LOGGER_INIT") << "Normalising 2D Images, Making the Noise Have Standard Deviation of 1";

    normaliseImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "2D Images Normalised";
    BLOG(INFO, "LOGGER_INIT") << "2D Images Normalised";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images After Normalising";
    BLOG(INFO, "LOGGER_INIT") << "Displaying Statistics of 2D Images After Normalising";

    displayStatImg();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Statistics of 2D Images After Normalising Displayed";
    BLOG(INFO, "LOGGER_INIT") << "Statistics of 2D Images After Normalising Displayed";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Performing Fourier Transform on 2D Images";
    BLOG(INFO, "LOGGER_INIT") << "Performing Fourier Transform on 2D Images";

#ifdef OPTIMISER_LOG_MEM_USAGE
    CHECK_MEMORY_USAGE("Before Performing Fourier Transform on 2D Images");
#endif

    fwImg();

#ifdef OPTIMISER_LOG_MEM_USAGE
    CHECK_MEMORY_USAGE("After Performing Fourier Transform on 2D Images");
#endif

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Fourier Transform on 2D Images Performed";
    BLOG(INFO, "LOGGER_INIT") << "Fourier Transform on 2D Images Performed";
#endif
}

void Optimiser::statImg()
{
    int nPer = 0;
    int nImg = 0;

    RFLOAT mean = 0;
    RFLOAT stdN = 0;
    RFLOAT stdD = 0;
    RFLOAT stdS = 0;
    RFLOAT stdStdN = 0;

    MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
    #pragma omp parallel for reduction(+:mean, stdN, stdD, stdS, stdStdN) firstprivate(imgDustman)
    FOR_EACH_2D_IMAGE
    {
        #pragma omp atomic
        nImg += 1;

        #pragma omp critical (line2477)
        if (nImg >= (int)_ID.size() / 10)
        {
            nPer += 1;

            ALOG(INFO, "LOGGER_SYS") << nPer * 10 << "\% Percentage of Images Performed Statistics";
            BLOG(INFO, "LOGGER_SYS") << nPer * 10 << "\% Percentage of Images Performed Statistics";

            nImg = 0;
        }

#ifdef OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION
        mean += regionMean(_img[l],
                           _para.maskRadius / _para.pixelSize,
                           0,
                           1);
#else
        mean += regionMean(_img[l],
                           _para.size / 2,
                           0,
                           1);
#endif

#ifdef OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION
        stdN += bgStddev(0,
                         _img[l],
                         _para.maskRadius / _para.pixelSize);
#else
        stdN += bgStddev(0,
                         _img[l],
                         _para.size / 2);
#endif

        stdD += stddev(0, _img[l]);

#ifdef OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION
        stdStdN += gsl_pow_2(bgStddev(0,
                                      _img[l],
                                      _para.maskRadius / _para.pixelSize));
#else
        stdStdN += gsl_pow_2(bgStddev(0,
                                      _img[l],
                                      _para.size / 2));
#endif
    }

    _mean = mean;
    _stdN = stdN;
    _stdD = stdD;
    _stdS = stdS;
    _stdStdN = stdStdN;

#ifdef VERBOSE_LEVEL_1
    ILOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Performing Statistics on Images Accomplished";
#endif

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_mean, 1, TS_MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_stdN, 1, TS_MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_stdD, 1, TS_MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Allreduce(MPI_IN_PLACE, &_stdStdN, 1, TS_MPI_DOUBLE, MPI_SUM, _hemi);

    MPI_Barrier(_hemi);

    _mean /= _N;

    _stdN /= _N;
    _stdD /= _N;

    _stdStdN /= _N;

    _stdS = _stdD - _stdN;

    _stdStdN = sqrt(_stdStdN - TSGSL_pow_2(_stdN));
}

void Optimiser::displayStatImg()
{
    ALOG(INFO, "LOGGER_INIT") << "Mean of Centre : " << _mean;

    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Noise  : " << _stdN;
    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Data   : " << _stdD;
    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Signal : " << _stdS;

    ALOG(INFO, "LOGGER_INIT") << "Standard Deviation of Standard Deviation of Noise : "
                              << _stdStdN;

    BLOG(INFO, "LOGGER_INIT") << "Mean of Centre : " << _mean;

    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Noise  : " << _stdN;
    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Data   : " << _stdD;
    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Signal : " << _stdS;

    BLOG(INFO, "LOGGER_INIT") << "Standard Deviation of Standard Deviation of Noise : "
                              << _stdStdN;
}

void Optimiser::substractBgImg()
{
    MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
    #pragma omp parallel for firstprivate(imgDustman)
    FOR_EACH_2D_IMAGE
    {
        RFLOAT bgMean, bgStddev;

#ifdef OPTIMISER_INIT_IMG_NORMALISE_OUT_MASK_REGION
        bgMeanStddev(bgMean,
                     bgStddev,
                     _img[l],
                     _para.maskRadius / _para.pixelSize);
#else
        bgMeanStddev(bgMean,
                     bgStddev,
                     _img[l],
                     _para.size / 2);
#endif

        FOR_EACH_PIXEL_RL(_img[l])
        {
            _img[l](i) -= bgMean;
            _img[l](i) /= bgStddev;
        }

        /***
        RFLOAT bg = background(_img[l],
                               _para.maskRadius / _para.pixelSize,
                               EDGE_WIDTH_RL);

        FOR_EACH_PIXEL_RL(_img[l])
            _img[l](i) -= bg;
        ***/
    }
}

void Optimiser::maskImg()
{
    // _imgOri.clear();
    _imgOri.cleanUp();

    _imgOri.setUp(omp_get_max_threads(), _ID.size(), _md.nStallImgOri, serializeSize(Image(_para.size, _para.size, RL_SPACE)), 1, _para.cacheDirectory);

    // _imgOri.resize(_ID.size());

    FOR_EACH_2D_IMAGE
    {
        // _imgOri.push_back(_img[l].copyImage());
        _imgOri[l] = _img[l].copyImage();
    }

#ifdef OPTIMISER_MASK_IMG
    if (_para.zeroMask)
    {
        MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
        #pragma omp parallel for firstprivate(imgDustman)
        FOR_EACH_2D_IMAGE
            softMask(_img[l],
                     _img[l],
                     _para.maskRadius / _para.pixelSize,
                     EDGE_WIDTH_RL,
                     0,
                     1);
    }
    else
    {
        MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
        #pragma omp parallel for firstprivate(imgDustman)
        FOR_EACH_2D_IMAGE
            softMask(_img[l],
                     _img[l],
                     _para.maskRadius / _para.pixelSize,
                     EDGE_WIDTH_RL,
                     0,
                     _stdN,
                     1);
    }
#endif
}

void Optimiser::normaliseImg()
{
    RFLOAT scale = 1.0 / _stdN;

    MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
    MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
    #pragma omp parallel for firstprivate(imgDustman, imgOriDustman)
    FOR_EACH_2D_IMAGE
    {
        _img.endLastVisit(l);
        _imgOri.endLastVisit(l);

        SCALE_RL(_img[l], scale);
        SCALE_RL(_imgOri[l], scale);
    }

    _stdN *= scale;
    _stdD *= scale;
    _stdS *= scale;
}

void Optimiser::fwImg()
{
    FOR_EACH_2D_IMAGE
    {
        _fftImg.fwExecutePlan(_img[l]);
        // _img[l].clearRL();

        _fftImg.fwExecutePlan(_imgOri[l]);
        // _imgOri[l].clearRL();
    }
}

void Optimiser::bwImg()
{
    FOR_EACH_2D_IMAGE
    {
        _fftImg.bwExecutePlan(_img[l], _para.nThreadsPerProcess);
        _img[l].clearFT();

        _fftImg.bwExecutePlan(_imgOri[l], _para.nThreadsPerProcess);
        _imgOri[l].clearFT();
    }
}

void Optimiser::initCTF()
{
    IF_MASTER return;

    _ctfAttr.clear();
    _ctf.clear();

    CTFAttr ctfAttr;
#ifdef GPU_VERSION
    RFLOAT *dpara = (RFLOAT*)malloc(sizeof(RFLOAT) * _ID.size());

    FOR_EACH_2D_IMAGE
    {
        _db.ctf(ctfAttr, _ID[l]);

        dpara[l] = _db.d(_ID[l]);

        _ctfAttr.push_back(ctfAttr);

#ifndef OPTIMISER_CTF_ON_THE_FLY
        _ctf.push_back(Image(size(), size(), FT_SPACE));
#endif
    }

#ifndef OPTIMISER_CTF_ON_THE_FLY
    int dimSizeFT = _para.size * (_para.size / 2 + 1);
    int nImg = _ID.size();
    int batch = IMAGE_BATCH;
    Complex *ctfData;
    ctfData = (Complex*)malloc(sizeof(Complex) * IMAGE_BATCH * dimSizeFT);

    hostRegister(ctfData,
                 IMAGE_BATCH * dimSizeFT);

    hostRegister(dpara,
                 _ID.size());

    for (int l = 0; l < nImg;)
    {
        if (l >= nImg)
            break;

        batch = (l + IMAGE_BATCH < nImg)
              ? IMAGE_BATCH : (nImg - l);

        for (int i = 0; i < batch; i++)
        {
            for (int n = 0; n < dimSizeFT; n++)
                ctfData[i * dimSizeFT + n] = _ctf[l + i][n];
        }

        GCTFinit(_stream,
                 _iGPU,
                 ctfData,
                 _ctfAttr,
                 dpara + l,
                 _para.pixelSize,
                 _para.size,
                 l,
                 batch,
                 _nGPU);

        for (int i = 0; i < batch; i++)
        {
            for (int n = 0; n < dimSizeFT; n++)
                _ctf[l + i][n] = ctfData[i * dimSizeFT + n];
        }

        l += batch;
    }

    hostFree(dpara);
    hostFree(ctfData);
    free(ctfData);
#endif
    free(dpara);
#else

    FOR_EACH_2D_IMAGE
    {
        _db.ctf(ctfAttr, _ID[l]);

        _ctfAttr.push_back(ctfAttr);

#ifndef OPTIMISER_CTF_ON_THE_FLY
        _ctf.push_back(Image(size(), size(), FT_SPACE));
#endif
    }

#ifndef OPTIMISER_CTF_ON_THE_FLY
    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
#ifdef VERBOSE_LEVEL_3
        ALOG(INFO, "LOGGER_SYS") << "Initialising CTF for Image " << _ID[l];
        BLOG(INFO, "LOGGER_SYS") << "Initialising CTF for Image " << _ID[l];
#endif

        CTF(_ctf[l],
            _para.pixelSize,
            _ctfAttr[l].voltage,
            _ctfAttr[l].defocusU * _db.d(_ID[l]),
            _ctfAttr[l].defocusV * _db.d(_ID[l]),
            _ctfAttr[l].defocusTheta,
            _ctfAttr[l].Cs,
            _ctfAttr[l].amplitudeContrast,
            _ctfAttr[l].phaseShift,
            1);
    }
#endif

#endif
}

void Optimiser::correctScale(const bool init,
                             const bool coord,
                             const bool group)
{
    ALOG(INFO, "LOGGER_SYS") << "Refreshing Scale";
    BLOG(INFO, "LOGGER_SYS") << "Refreshing Scale";

    refreshScale(coord, group);

    IF_MASTER return;

    ALOG(INFO, "LOGGER_SYS") << "Correcting Scale";
    BLOG(INFO, "LOGGER_SYS") << "Correcting Scale";

    if (init)
    {
        for (int l = 0; l < _para.k; l++)
        {
            #pragma omp parallel for
            SCALE_FT(_model.ref(l), _scale(_groupID[0] - 1));
        }
    }
    else
    {
        MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
        MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
        #pragma omp parallel for firstprivate(imgDustman, imgOriDustman)
        FOR_EACH_2D_IMAGE
        {
            _img.endLastVisit(l);
            _imgOri.endLastVisit(l);

            FOR_EACH_PIXEL_FT(_img[l])
            {
                _img[l][i] /= _scale(_groupID[l] - 1);
                _imgOri[l][i] /= _scale(_groupID[l] - 1);
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < _nGroup; i++)
        {
            _sig.row(i) /= TSGSL_pow_2(_scale(i));
        }
    }
}

void Optimiser::initSigma()
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_INIT") << "Calculating Average Image";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Average Image";

#ifdef OPTIMISER_SIGMA_MASK
    Image avg = _img[0].copyImage();
#else
    Image avg = _imgOri[0].copyImage();
#endif

    for (size_t l = 1; l < _ID.size(); l++)
    {
        MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
        MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
        #pragma omp parallel for firstprivate(imgDustman, imgOriDustman)
#ifdef OPTIMISER_SIGMA_MASK
        ADD_FT(avg, _img[l]);
#else
        ADD_FT(avg, _imgOri[l]);
#endif
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &avg[0],
                  2 * avg.sizeFT(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    #pragma omp parallel for
    SCALE_FT(avg, 1.0 / _N);

    ALOG(INFO, "LOGGER_INIT") << "Calculating Average Power Spectrum";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Average Power Spectrum";

    vec avgPs = vec::Zero(maxR());

    MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
    MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
    #pragma omp parallel for firstprivate(imgDustman, imgOriDustman)
    FOR_EACH_2D_IMAGE
    {
        _img.endLastVisit(l);
        _imgOri.endLastVisit(l);

        vec ps(maxR());

        // powerSpectrum(ps, _imgOri[l], maxR());

#ifdef OPTIMISER_SIGMA_MASK
        powerSpectrum(ps, _img[l], maxR(), 1);
#else
        powerSpectrum(ps, _imgOri[l], maxR(), 1);
#endif

        #pragma omp critical  (line2742)
        avgPs += ps;
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  avgPs.data(),
                  maxR(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);


    avgPs /= _N;

    ALOG(INFO, "LOGGER_INIT") << "Calculating Expectation for Initializing Sigma";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Expectation for Initializing Sigma";

    vec psAvg(maxR());
    for (int i = 0; i < maxR(); i++)
    {
        psAvg(i) = ringAverage(i,
                               avg,
                               function<RFLOAT(const Complex)>(&gsl_real_imag_sum));
        psAvg(i) = TSGSL_pow_2(psAvg(i));
    }

    // avgPs -> average power spectrum
    // psAvg -> expectation of pixels
    ALOG(INFO, "LOGGER_INIT") << "Substract avgPs and psAvg for _sig";
    BLOG(INFO, "LOGGER_INIT") << "Substract avgPs and psAvg for _sig";

    _sig.leftCols(_sig.cols() - 1).rowwise() = (avgPs - psAvg).transpose() / 2;

    ALOG(INFO, "LOGGER_INIT") << "Calculating Reciprocal of Sigma";
    BLOG(INFO, "LOGGER_INIT") << "Calculating Reciprocal of Sigma";

    for (int i = 0; i < _nGroup; i++)
        for (int j = 0; j < maxR(); j++)
            _sigRcp(i, j) = -0.5 / _sig(i, j);
}

void Optimiser::initParticles()
{
    IF_MASTER return;

    _parAll.clear();

    // _par.resize(_ID.size());
    _parAll.resize(_ID.size() * _para.k);

    for (size_t t = 0; t < _para.k; t++)
    {
        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {
#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Initialising Particle Filter for Image " << _ID[l];
            BLOG(INFO, "LOGGER_SYS") << "Initialising Particle Filter for Image " << _ID[l];
#endif
            _parAll[t * _ID.size() + l].init(_para.mode,
                                             _para.transS,
                                             TRANS_Q,
                                             &_sym);
        }
    }

    _iRefPrev.resize(_ID.size());

    _iRef.resize(_ID.size());

    _par.resize(_ID.size());

    gsl_rng* engine = get_random_engine();

    #pragma omp parallel for
    FOR_EACH_2D_IMAGE
    {
        _iRefPrev[l] = gsl_rng_uniform_int(engine, _para.k);

        _iRef[l] = gsl_rng_uniform_int(engine, _para.k);

        _par[l] = &_parAll[_iRef[l] * _ID.size() + l]; // in initialisation, assign it to the first class
    }
}

void Optimiser::avgStdR(RFLOAT& stdR)
{
    IF_MASTER return;

    /***
    stdR = 0;

    FOR_EACH_2D_IMAGE
        stdR += _db.stdR(_ID[l]);

    MPI_Allreduce(MPI_IN_PLACE,
                 &stdR,
                 1,
                 TS_MPI_DOUBLE,
                 MPI_SUM,
                 _hemi);

    stdR /= _N;
    ***/
}

void Optimiser::avgStdT(RFLOAT& stdT)
{
    IF_MASTER return;

    /***
    stdT = 0;

    FOR_EACH_2D_IMAGE
    {
        stdT += _db.stdTX(_ID[l]);
        stdT += _db.stdTY(_ID[l]);
    }

    MPI_Allreduce(MPI_IN_PLACE,
                  &stdT,
                  1,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    stdT /= _N;
    stdT /= 2;
    ***/
}

void Optimiser::loadParticles()
{
    IF_MASTER return;

    // size_t cls;
    dvec4 quat;
    dvec2 tran;
    double d;

    double k1, k2, k3, stdTX, stdTY, stdD, score;

    #pragma omp parallel for private(quat, tran, d, k1, k2, k3, stdTX, stdTY, stdD, score)
    FOR_EACH_2D_IMAGE
    {
        #pragma omp critical (line2883)
        {
            quat = _db.quat(_ID[l]);
            tran = _db.tran(_ID[l]);
            d = _db.d(_ID[l]);

            k1 = _db.k1(_ID[l]);
            k2 = _db.k2(_ID[l]);
            k3 = _db.k3(_ID[l]);

            stdTX = _db.stdTX(_ID[l]);
            stdTY = _db.stdTY(_ID[l]);
            stdD = _db.stdD(_ID[l]);

            score = _db.score(_ID[l]);
        }

        for (size_t t = 0; t < _para.k; t++)
        {
            _parAll[t * _ID.size() + l].load(_para.mLR,
                                             _para.mLT,
                                             1,
                                             quat,
                                             k1,
                                             k2,
                                             k3,
                                             tran,
                                             stdTX,
                                             stdTY,
                                             d,
                                             stdD,
                                             score);
        }
    }
}

void Optimiser::refreshRotationChange()
{
    vec rc = vec::Zero(_nPar);

    NT_MASTER
    {
        FOR_EACH_2D_IMAGE
        {
            RFLOAT diff = _par[l]->diffTopR();

            rc(_ID[l]) = diff;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE,
                  rc.data(),
                  rc.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    RFLOAT mean, std;
    TSGSL_sort(rc.data(), 1, _nPar);

    stat_MAS(mean, std, rc, _nPar);

    _model.setRChange(mean);
    _model.setStdRChange(std);
}

void Optimiser::refreshClassChange()
{
    int cc = 0;

    NT_MASTER
    {
        FOR_EACH_2D_IMAGE
        {
            if (_iRefPrev[l] != _iRef[l])
            {
                cc += 1;
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE,
                  &cc,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    _model.setCChange((RFLOAT)cc / _nPar);
}

void Optimiser::refreshClassDistr()
{
    _cDistr = vec::Zero(_para.k);

    NT_MASTER
    {
        size_t cls;

        #pragma omp parallel for private(cls)
        FOR_EACH_2D_IMAGE
        {
            for (int k = 0; k < _para.k; k++)
            {
                // _par[l]->rand(cls);

                cls = _iRef[l];

                #pragma omp atomic
                _cDistr(cls) += 1;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  _cDistr.data(),
                  _cDistr.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    //_cDistr.array() /= (_nPar * _para.k);
    _cDistr.array() /= _cDistr.sum();
}

void Optimiser::determineBalanceClass(umat2& dst,
                                      const RFLOAT thres)
{
    int num = 0;

    for (int t = 0; t < _para.k; t++)
        if (_cDistr(t) < thres / _para.k)
            num++;

#ifdef VERBOSE_LEVEL_1
    MLOG(INFO, "LOGGER_SYS") << num << " Classes Empty and Needing Resigning";
#endif

    dst = umat2::Zero(num, 2);

    IF_MASTER
    {
        dvec cum = dvec::Zero(_para.k);

        for (int t = 0; t < _para.k; t++)
        {
            if (_cDistr(t) < thres / _para.k)
                cum(t) = 0;
            else
                cum(t) = _cDistr(t) - (thres / _para.k);
        }

        cum.array() /= cum.sum();

        cum = d_cumsum(cum);

#ifdef VERBOSE_LEVEL_1
        MLOG(INFO, "LOGGER_SYS") << "Summation of Percentage Calculated";
#endif

        gsl_rng* engine = get_random_engine();

        int i = 0;

        for (int t = 0; t < _para.k; t++)
        {
            if (_cDistr(t) < thres / _para.k)
            {
                RFLOAT indice = TSGSL_ran_flat(engine, 0, 1);

                int j = 0;
                while (cum(j) < indice) j++;

                MLOG(INFO, "LOGGER_SYS") << "Class " << t << " is Empty ( Round "
                                         << _iter
                                         << " ), Resigned it with Class "
                                         << j;

                dst(i, 0) = t;
                dst(i, 1) = j;

                i++;
            }
        }
    }

    MPI_Bcast(dst.data(),
              dst.size(),
              MPI_LONG,
              MASTER_ID,
              MPI_COMM_WORLD);
}

void Optimiser::balanceClass(const umat2& bm)
{
    for (int i = 0; i < bm.rows(); i++)
    {
        NT_MASTER _model.ref(bm(i, 0)) = _model.ref(bm(i, 1)).copyVolume();
    }
}

/***
void Optimiser::balanceClass(const RFLOAT thres,
                             const bool refreshDistr)
{
    int cls;
    RFLOAT num = _cDistr.maxCoeff(&cls);

    for (int t = 0; t < _para.k; t++)
        if (_cDistr(t) < thres / _para.k)
        {
            MLOG(INFO, "LOGGER_SYS") << "Class " << t << " is Empty ( Round "
                                     << _iter
                                     << " ), Resigned it with Class "
                                     << cls;

            NT_MASTER _model.ref(t) = _model.ref(cls).copyVolume();

            if (refreshDistr) _cDistr(t) = num;
        }

    if (refreshDistr) _cDistr.array() /= _cDistr.sum();
}
***/

void Optimiser::refreshVariance()
{
    vec rv = vec::Zero(_nPar);
    vec t0v = vec::Zero(_nPar);
    vec t1v = vec::Zero(_nPar);

#ifdef OPTIMISER_REFRESH_VARIANCE_BEST_CLASS

    int bestClass = _model.bestClass(_para.thresCutoffFSC, false);

#ifdef VERBOSE_LEVEL_1
    MLOG(INFO, "LOGGER_SYS") << "Best Class is " << bestClass;
#endif

#endif

    NT_MASTER
    {
        double rVari, tVariS0, tVariS1, dVari;

        #pragma omp parallel for private(rVari, tVariS0, tVariS1, dVari)
        FOR_EACH_2D_IMAGE
        {
#ifdef OPTIMISER_REFRESH_VARIANCE_BEST_CLASS
            size_t cls;
            _par[l]->rand(cls);

            // ADDED
            cls = _par[l].topC();

            if (cls == (size_t)bestClass)
            {
                _par[l]->vari(rVari,
                             tVariS0,
                             tVariS1,
                             dVari);
            }
            else
            {
                rVari = GSL_NAN;
                tVariS0 = GSL_NAN;
                tVariS1 = GSL_NAN;
                dVari = GSL_NAN;
            }
#else
            _par[l]->vari(rVari,
                         tVariS0,
                         tVariS1,
                         dVari);
#endif

            rv(_ID[l]) = rVari;
            t0v(_ID[l]) = tVariS0;
            t1v(_ID[l]) = tVariS1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  rv.data(),
                  rv.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  t0v.data(),
                  t0v.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  t1v.data(),
                  t1v.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef OPTIMISER_REFRESH_VARIANCE_BEST_CLASS
    int num = 0;
    for (int i = 0; i < _nPar; i++)
        if (!TSGSL_isnan(rv(i))) num++;

#ifdef VERBOSE_LEVEL_1
    MLOG(INFO, "LOGGER_SYS") << num << " Particles Belonging to Best Class";
#endif

    vec rvt = vec::Zero(num);
    vec t0vt = vec::Zero(num);
    vec t1vt = vec::Zero(num);

    int j = 0;
    for (int i = 0; i < _nPar; i++)
    {
        if (!TSGSL_isnan(rv(i)))
        {
            rvt(j) = rv(i);
            t0vt(j) = t0v(i);
            t1vt(j) = t1v(i);

            j++;
        }
    }

    rv = rvt;
    t0v = t0vt;
    t1v = t1vt;
#endif

    ALOG(INFO, "LOGGER_SYS") << "Maximum Rotation Variance: " << rv.maxCoeff();
    BLOG(INFO, "LOGGER_SYS") << "Maximum Rotation Variance: " << rv.maxCoeff();

    RFLOAT mean, std;

    stat_MAS(mean, std, rv, rv.size());

    _model.setRVari(mean);
    _model.setStdRVari(std);

    stat_MAS(mean, std, t0v, t0v.size());

    _model.setTVariS0(mean);
    _model.setStdTVariS0(std);

    stat_MAS(mean, std, t1v, t1v.size());

    _model.setTVariS1(mean);
    _model.setStdTVariS1(std);
}

void Optimiser::refreshScale(const bool coord,
                             const bool group)
{
    if (_iter != 0)
        _rS = _model.resolutionP(_para.thresSclCorFSC, false);

    if (_rS > _r)
    {
        MLOG(WARNING, "LOGGER_SYS") << "_rS is Larger than _r, Set _rS to _r";
        _rS = _r;
    }

#ifdef OPTIMISER_REFRESH_SCALE_ZERO_FREQ_NO_COORD
    if (!coord) _rS = 1;
#endif

    MLOG(INFO, "LOGGER_SYS") << "Upper Boundary Frequency for Scale Correction: "
                             << _rS;

    mat mXA = mat::Zero(_nGroup, _rS);
    mat mAA = mat::Zero(_nGroup, _rS);

    vec sXA = vec::Zero(_rS);
    vec sAA = vec::Zero(_rS);

    NT_MASTER
    {
        Image img(size(), size(), FT_SPACE);

        dmat22 rot2D;
        dmat33 rot3D;
        dvec2 tran;
        double d;

        FOR_EACH_2D_IMAGE
        {
#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Projecting from the Initial Reference from a Random Rotation for Image " << _ID[l];
            BLOG(INFO, "LOGGER_SYS") << "Projecting from the Initial Reference from a Random Rotation for Image " << _ID[l];
#endif

            if (!coord)
            {
                if (_para.mode == MODE_2D)
                {
                    randRotate2D(rot2D);
#ifdef VERBOSE_LEVEL_3
                ALOG(INFO, "LOGGER_SYS") << "The Random Rotation Matrix is " << rot2D;
                BLOG(INFO, "LOGGER_SYS") << "The Random Rotation Matrix is " << rot2D;
#endif
                }
                else if (_para.mode == MODE_3D)
                {
                    randRotate3D(rot3D);
#ifdef VERBOSE_LEVEL_3
                ALOG(INFO, "LOGGER_SYS") << "The Random Rotation Matrix is " << rot3D;
                BLOG(INFO, "LOGGER_SYS") << "The Random Rotation Matrix is " << rot3D;
#endif
                }
                else
                    REPORT_ERROR("INEXISTENT MODE");

                if (_para.mode == MODE_2D)
                {
                    _model.proj(0).project(img, rot2D, _para.nThreadsPerProcess);
                }
                else if (_para.mode == MODE_3D)
                {
                    _model.proj(0).project(img, rot3D, _para.nThreadsPerProcess);
                }
                else
                    REPORT_ERROR("INEXISTENT MODE");
                d = _db.d(_ID[l]);
            }
            else
            {
                if (_para.mode == MODE_2D)
                {
                    _par[l]->rank1st(rot2D, tran, d);
                }
                else if (_para.mode == MODE_3D)
                {
                    _par[l]->rank1st(rot3D, tran, d);
                }
                else
                    REPORT_ERROR("INEXISTENT MODE");

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_SCALE_MASK
                    _model.proj(_iRef[l]).project(img, rot2D, tran, _para.nThreadsPerProcess);
#else
                    _model.proj(_iRef[l]).project(img, rot2D, tran - _offset[l], _para.nThreadsPerProcess);
#endif
#else
                    _model.proj(_iRef[l]).project(img, rot2D, tran, _para.nThreadsPerProcess);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_SCALE_MASK
                    _model.proj(_iRef[l]).project(img, rot3D, tran, _para.nThreadsPerProcess);
#else
                    _model.proj(_iRef[l]).project(img, rot3D, tran - _offset[l], _para.nThreadsPerProcess);
#endif
#else
                    _model.proj(_iRef[l]).project(img, rot3D, tran, _para.nThreadsPerProcess);
#endif
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }
            }

#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Calculating Intensity Scale for Image " << l;
            BLOG(INFO, "LOGGER_SYS") << "Calculating Intensity Scale for Image " << l;
#endif

#ifdef OPTIMISER_REFRESH_SCALE_ZERO_FREQ_NO_COORD
            RFLOAT rL = coord ? _rL : 0;
#else
            RFLOAT rL = _rL;
#endif

#ifdef OPTIMISER_CTF_ON_THE_FLY
            Image ctf(_para.size, _para.size, FT_SPACE);
            CTF(ctf,
                _para.pixelSize,
                _ctfAttr[l].voltage,
                _ctfAttr[l].defocusU * d,
                _ctfAttr[l].defocusV * d,
                _ctfAttr[l].defocusTheta,
                _ctfAttr[l].Cs,
                _ctfAttr[l].amplitudeContrast,
                _ctfAttr[l].phaseShift,
                CEIL(_rS) + 1,
                _para.nThreadsPerProcess);
#ifdef OPTIMISER_SCALE_MASK
            scaleDataVSPrior(sXA,
                             sAA,
                             _img[l],
                             img,
                             ctf,
                             _rS,
                             rL);
#else
            scaleDataVSPrior(sXA,
                             sAA,
                             _imgOri[l],
                             img,
                             ctf,
                             _rS,
                             rL);
#endif
#else
#ifdef OPTIMISER_SCALE_MASK
            scaleDataVSPrior(sXA,
                             sAA,
                             _img[l],
                             img,
                             _ctf[l],
                             _rS,
                             rL);
#else
            scaleDataVSPrior(sXA,
                             sAA,
                             _imgOri[l],
                             img,
                             _ctf[l],
                             _rS,
                             rL);
#endif
#endif

#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Accumulating Intensity Scale Information from Image " << l;
            BLOG(INFO, "LOGGER_SYS") << "Accumulating Intensity Scale Information from Image " << l;
#endif

            if (group)
            {
                mXA.row(_groupID[l] - 1) += sXA.transpose();
                mAA.row(_groupID[l] - 1) += sAA.transpose();
            }
            else
            {
                mXA.row(0) += sXA.transpose();
                mAA.row(0) += sAA.transpose();
            }
        }
    }

#ifdef VERBOSE_LEVEL_1
    ILOG(INFO, "LOGGER_SYS") << "Intensity Scale Information Calculated";
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Accumulating Intensity Scale Information from All Processes";

    MPI_Allreduce(MPI_IN_PLACE,
                  mXA.data(),
                  mXA.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  mAA.data(),
                  mAA.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    if (group)
    {
        for (int i = 0; i < _nGroup; i++)
        {
#ifdef OPTIMISER_REFRESH_SCALE_SPECTRUM
            RFLOAT sum = 0;
            int count = 0;

            for (int r = (int)rL; r < _rS; r++)
            {
                sum += mXA(i, r) / mAA(i, r);
                count += 1;
            }

            _scale(i) = sum / count;
#else
            _scale(i) = mXA.row(i).sum() / mAA.row(i).sum();
#endif
        }
    }
    else
    {
#ifdef OPTIMISER_REFRESH_SCALE_SPECTRUM
        RFLOAT sum = 0;
        int count = 0;

        for (int r = (int)rL; r < _rS; r++)
        {
            sum += mXA(0, r) / mAA(0, r);
            count += 1;
        }

        for (int i = 0; i < _nGroup; i++)
            _scale(i) = sum / count;
#else
        for (int i = 0; i < _nGroup; i++)
            _scale(i) = mXA.row(0).sum() / mAA.row(0).sum();
#endif
    }

#ifdef OPTIMISER_REFRESH_SCALE_ABSOLUTE
    _scale = _scale.array().abs();
#endif

    RFLOAT medianScale = median(_scale, _scale.size());

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Median Intensity Scale: " << medianScale;

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Removing Extreme Values from Intensity Scale";

    for (int i = 0; i < _nGroup; i++)
    {
        if (fabs(_scale(i)) > fabs(medianScale * 5))
            _scale(i) = medianScale * 5;
        else if (fabs(_scale(i)) < fabs(medianScale / 5))
            _scale(i) = medianScale / 5;
    }

    RFLOAT meanScale = _scale.mean();

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Average Intensity Scale: " << meanScale;

    if (meanScale < 0)
    {
        REPORT_ERROR("AVERAGE INTENSITY SCALE SHOULD NOT BE SMALLER THAN ZERO");
        abort();
    }

    /***
    if (medianScale * meanScale < 0)
        CLOG(FATAL, "LOGGER_SYS") << "Median and Mean of Intensity Scale Should Have the Same Sign";
    ***/

    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Standard Deviation of Intensity Scale: "
                               << TSGSL_stats_sd(_scale.data(), 1, _scale.size());

    /***
    if (!init)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Making Average Intensity Scale be 1";

        for (int i = 0; i < _nGroup; i++)
            _scale(i) /= fabs(meanScale);
    }
    ***/

    IF_MASTER
    {
#ifdef VERBOSE_LEVEL_2
        for (int i = 0; i < _nGroup; i++)
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Scale of Group " << i << " is " << _scale(i);
#endif
    }
}

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
void Optimiser::reCentreImg()
{
    IF_MASTER return;

    dvec2 tran;

    MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
    MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
    #pragma omp parallel for private(tran) firstprivate(imgDustman, imgOriDustman)
    FOR_EACH_2D_IMAGE
    {
        _img.endLastVisit(l);
        _imgOri.endLastVisit(l);

        _par[l]->rank1st(tran);

        _offset[l](0) -= tran(0);
        _offset[l](1) -= tran(1);

        translate(_img[l],
                  _imgOri[l],
                  _offset[l](0),
                  _offset[l](1),
                  1);

        for (int t = 0; t < _para.k; t++)
        {
            _parAll[t * _ID.size() + l].setT(_parAll[t * _ID.size() + l].t().rowwise() - tran.transpose());

            _parAll[t * _ID.size() + l].setTopT(_parAll[t * _ID.size() + l].topT() - tran);
            _parAll[t * _ID.size() + l].setTopTPrev(_parAll[t * _ID.size() + l].topTPrev() - tran);
        }

        /***
        _par[l]->setT(_par[l]->t().rowwise() - tran.transpose());

        _par[l]->setTopT(_par[l]->topT() - tran);
        _par[l]->setTopTPrev(_par[l]->topTPrev() - tran);
        ***/
    }
}
#endif

void Optimiser::reMaskImg()
{
    IF_MASTER return;

#ifdef OPTIMISER_MASK_IMG
    if (_para.zeroMask)
    {
        Image mask(_para.size, _para.size, RL_SPACE);

        softMask(mask,
                 _para.maskRadius / _para.pixelSize,
                 EDGE_WIDTH_RL,
                 _para.nThreadsPerProcess);

        FOR_EACH_2D_IMAGE
        {
            _fftImg.bwExecutePlan(_img[l], _para.nThreadsPerProcess);

            MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
            #pragma omp parallel for firstprivate(imgDustman)
            MUL_RL(_img[l], mask);

            _fftImg.fwExecutePlan(_img[l]);

            // _img[l].clearRL();
        }
    }
    else
    {
        //TODO Make the background a noise with PowerSpectrum of sigma2
    }
#endif
}

#ifdef GPU_VERSION

void Optimiser::reMaskImgG()
{
    IF_MASTER return;

#ifdef OPTIMISER_MASK_IMG
    if (_para.zeroMask)
    {
        int dimSizeFT = _para.size * (_para.size / 2 + 1);
        int nImg = _ID.size();
        int batch = IMAGE_BATCH;
        Complex *imgData;
        imgData = (Complex*)malloc(sizeof(Complex) * IMAGE_BATCH * dimSizeFT);

        hostRegister(imgData,
                     IMAGE_BATCH * dimSizeFT);

        for (int l = 0; l < nImg;)
        {
            if (l >= nImg)
                break;

            batch = (l + IMAGE_BATCH < nImg)
                  ? IMAGE_BATCH : (nImg - l);

            for (int i = 0; i < batch; i++)
            {
                for (int n = 0; n < dimSizeFT; n++)
                    imgData[i * dimSizeFT + n] = _img[l + i][n];
            }

            reMask(_stream,
                   _iGPU,
                   imgData,
                   _para.maskRadius,
                   _para.pixelSize,
                   EDGE_WIDTH_RL,
                   _para.size,
                   batch,
                   _nGPU);
            
            for (int i = 0; i < batch; i++)
            {
                for (int n = 0; n < dimSizeFT; n++)
                    _img[l + i][n] = imgData[i * dimSizeFT + n];
            }

            l += batch;
        }

        hostFree(imgData);
        free(imgData);
    }
    else
    {
        //TODO Make the background a noise with PowerSpectrum of sigma2
    }
#endif
}

#endif // GPU_VERSION

void Optimiser::normCorrection()
{
    RFLOAT rNorm = TSGSL_MIN_RFLOAT(_r, _model.resolutionP(0.75, false));

    vec norm = vec::Zero(_nPar);

    dmat22 rot2D;
    dmat33 rot3D;

    dvec2 tran;

    double d;

    NT_MASTER
    {
        MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
        MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
        #pragma omp parallel for private(rot2D, rot3D, tran, d) firstprivate(imgDustman, imgOriDustman)
        FOR_EACH_2D_IMAGE
        {
            _img.endLastVisit(l);
            _imgOri.endLastVisit(l);

#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Calculating Power Spectrum of Remains of Image " << _ID[l];

            BLOG(INFO, "LOGGER_SYS") << "Calculating Power Spectrum of Remains of Image " << _ID[l];
#endif

            Image img(size(), size(), FT_SPACE);

            SET_0_FT(img);

#ifdef VERBOSE_LEVEL_3
            ALOG(INFO, "LOGGER_SYS") << "Projecting Reference for Image " << _ID[l];
            BLOG(INFO, "LOGGER_SYS") << "Projecting Reference for Image " << _ID[l];
#endif

                if (_para.mode == MODE_2D)
                {
                    _par[l]->rank1st(rot2D, tran, d);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_NORM_MASK
                    _model.proj(_iRef[l]).project(img, rot2D, tran, 1);
#else
                    _model.proj(_iRef[l]).project(img, rot2D, tran - _offset[l], 1);
#endif
#else
                    _model.proj(_iRef[l]).project(img, rot2D, tran, 1);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
                    _par[l]->rank1st(rot3D, tran, d);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
#ifdef OPTIMISER_NORM_MASK
                    _model.proj(_iRef[l]).project(img, rot3D, tran, 1);
#else
                    _model.proj(_iRef[l]).project(img, rot3D, tran - _offset[l], 1);
#endif
#else
                    _model.proj(_iRef[l]).project(img, rot3D, tran, 1);
#endif
                }

#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(img.dataFT(), img.sizeFT());
#endif

#ifdef VERBOSE_LEVEL_3
                ALOG(INFO, "LOGGER_SYS") << "Applying CTF on Projection of Reference for Image " << _ID[l];
                BLOG(INFO, "LOGGER_SYS") << "Applying CTF on Projection of Reference for Image " << _ID[l];
#endif

                if (_searchType != SEARCH_TYPE_CTF)
                {
#ifdef OPTIMISER_CTF_ON_THE_FLY
                    Image ctf(_para.size, _para.size, FT_SPACE);

                    SET_0_FT(ctf);

                    CTF(ctf,
                        _para.pixelSize,
                        _ctfAttr[l].voltage,
                        _ctfAttr[l].defocusU * _db.d(_ID[l]),
                        _ctfAttr[l].defocusV * _db.d(_ID[l]),
                        _ctfAttr[l].defocusTheta,
                        _ctfAttr[l].Cs,
                        _ctfAttr[l].amplitudeContrast,
                        _ctfAttr[l].phaseShift,
                        CEIL(rNorm) + 1,
                        1);

                    FOR_EACH_PIXEL_FT(img)
                        img[i] *= REAL(ctf[i]);
#else
                    FOR_EACH_PIXEL_FT(img)
                        img[i] *= REAL(_ctf[l][i]);
#endif
                }
                else
                {
                    Image ctf(_para.size, _para.size, FT_SPACE);

                    SET_0_FT(ctf);

                    CTF(ctf,
                        _para.pixelSize,
                        _ctfAttr[l].voltage,
                        _ctfAttr[l].defocusU * d,
                        _ctfAttr[l].defocusV * d,
                        _ctfAttr[l].defocusTheta,
                        _ctfAttr[l].Cs,
                        _ctfAttr[l].amplitudeContrast,
                        _ctfAttr[l].phaseShift,
                        1);

                    FOR_EACH_PIXEL_FT(img)
                        img[i] *= REAL(ctf[i]);
                }

#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(img.dataFT(), img.sizeFT());
#endif

#ifdef OPTIMISER_ADJUST_2D_IMAGE_NOISE_ZERO_MEAN
                _img[l][0] = img[0];
                _imgOri[l][0] = img[0];
#endif

#ifdef VERBOSE_LEVEL_3
                ALOG(INFO, "LOGGER_SYS") << "Determining Remain of Image " << _ID[l];
                BLOG(INFO, "LOGGER_SYS") << "Determining Remain of Image " << _ID[l];
#endif

                NEG_FT(img);

#ifdef OPTIMISER_NORM_MASK
                ADD_FT(img, _img[l]);
#else
                ADD_FT(img, _imgOri[l]);
#endif

#ifndef NAN_NO_CHECK
                SEGMENT_NAN_CHECK_COMPLEX(img.dataFT(), img.sizeFT());
#endif

                IMAGE_FOR_EACH_PIXEL_FT(img)
                {
                    if ((QUAD(i, j) >= TSGSL_pow_2(_rL)) &&
                        (QUAD(i, j) < TSGSL_pow_2(rNorm)))
                        norm(_ID[l]) += ABS2(img.getFTHalf(i, j));
                }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  norm.data(),
                  norm.size(),
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_SYS") << "Max of Norm of Noise : "
                             << TSGSL_stats_max(norm.data(), 1, norm.size());

    MLOG(INFO, "LOGGER_SYS") << "Min of Norm of Noise : "
                             << TSGSL_stats_min(norm.data(), 1, norm.size());

    RFLOAT m = median(norm, norm.size());

    MLOG(INFO, "LOGGER_SYS") << "Mean of Norm of Noise : "
                             << m;

    NT_MASTER
    {
        MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
        MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
        #pragma omp parallel for firstprivate(imgDustman, imgOriDustman)
        FOR_EACH_2D_IMAGE
        {
            _img.endLastVisit(l);
            _imgOri.endLastVisit(l);

            FOR_EACH_PIXEL_FT(_img[l])
            {
                _img[l][i] *= sqrt(m / norm(_ID[l]));
                _imgOri[l][i] *= sqrt(m / norm(_ID[l]));
            }
        }
    }
}

void Optimiser::allReduceSigma(const bool group)
/***
void Optimiser::allReduceSigma(const bool mask,
                               const bool group)
***/
{
    IF_MASTER return;

#ifdef OPTIMISER_SIGMA_WHOLE_FREQUENCY
    int rSig = maxR();
#else
    int rSig = _r;
#endif

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Clearing Up Sigma";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Clearing Up Sigma";

    // set re-calculating part to zero
    _sig.leftCols(rSig).setZero();
    _sig.rightCols(1).setZero();

    // mat sigM = _sig; // masked sigma
    // mat sigN = _sig; // no-masked sigma

    mat sigM = mat::Zero(_sig.rows(), _sig.cols()); // masked sigma
    mat sigN = mat::Zero(_sig.rows(), _sig.cols()); // no-masked sigma

    _svd.leftCols(rSig).setZero();
    _svd.rightCols(1).setZero();

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Recalculating Sigma";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Recalculating Sigma";

    dmat22 rot2D;
    dmat33 rot3D;

    dvec2 tran;

    double d;

    omp_lock_t* mtx = new omp_lock_t[_nGroup];

    #pragma omp parallel for
    for (int l = 0; l < _nGroup; l++)
        omp_init_lock(&mtx[l]);

    MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
    MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
    #pragma omp parallel for private(rot2D, rot3D, tran, d) schedule(dynamic) firstprivate(imgDustman, imgOriDustman)
    FOR_EACH_2D_IMAGE
    {
        _img.endLastVisit(l);
        _imgOri.endLastVisit(l);

#ifdef OPTIMISER_SIGMA_RANK1ST
        for (int m = 0; m < 1; m++)
#else
        for (int m = 0; m < _para.mReco; m++)
#endif
        {
#ifdef OPTIMIDSER_SIGMA_GRADING
            RFLOAT w;

            if (_para.parGra)
                w = _par[l]->compressR();
            else
                w = 1;
#else
            RFLOAT w = 1;
#endif

            Image imgM(size(), size(), FT_SPACE);
            Image imgN(size(), size(), FT_SPACE);

            SET_0_FT(imgM);
            SET_0_FT(imgN);

            vec vSigM(rSig);
            vec vSigN(rSig);

            vec sSVD(rSig);
            vec dSVD(rSig);

            if (_para.mode == MODE_2D)
            {
#ifdef OPTIMISER_SIGMA_RANK1ST
                _par[l]->rank1st(rot2D, tran, d);
#else
                _par[l]->rand(rot2D, tran, d);
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                 _model.proj(_iRef[l]).project(imgM, rot2D, tran, 1);
                 _model.proj(_iRef[l]).project(imgN, rot2D, tran - _offset[l], 1);
#else
                 _model.proj(_iRef[l]).project(imgM, rot2D, tran, 1);
                 _model.proj(_iRef[l]).project(imgN, rot2D, tran, 1);
#endif
            }
            else if (_para.mode == MODE_3D)
            {
#ifdef OPTIMISER_SIGMA_RANK1ST
                _par[l]->rank1st(rot3D, tran, d);
#else
                _par[l]->rand(rot3D, tran, d);
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                _model.proj(_iRef[l]).project(imgM, rot3D, tran, 1);
                _model.proj(_iRef[l]).project(imgN, rot3D, tran - _offset[l], 1);
#else
                _model.proj(_iRef[l]).project(imgM, rot3D, tran, 1);
                _model.proj(_iRef[l]).project(imgN, rot3D, tran, 1);
#endif
            }

            if (_searchType != SEARCH_TYPE_CTF)
            {
#ifdef OPTIMISER_CTF_ON_THE_FLY
                Image ctf(_para.size, _para.size, FT_SPACE);
                CTF(ctf,
                    _para.pixelSize,
                    _ctfAttr[l].voltage,
                    _ctfAttr[l].defocusU * _db.d(_ID[l]),
                    _ctfAttr[l].defocusV * _db.d(_ID[l]),
                    _ctfAttr[l].defocusTheta,
                    _ctfAttr[l].Cs,
                    _ctfAttr[l].amplitudeContrast,
                    _ctfAttr[l].phaseShift,
                    CEIL(rSig) + 1,
                    1);

                FOR_EACH_PIXEL_FT(imgM)
                    imgM[i] *= REAL(ctf[i]);
                FOR_EACH_PIXEL_FT(imgN)
                    imgN[i] *= REAL(ctf[i]);
#else
                FOR_EACH_PIXEL_FT(imgM)
                    imgM[i] *= REAL(_ctf[l][i]);
                FOR_EACH_PIXEL_FT(imgN)
                    imgN[i] *= REAL(_ctf[l][i]);
#endif
            }
            else
            {
                Image ctf(_para.size, _para.size, FT_SPACE);
                CTF(ctf,
                    _para.pixelSize,
                    _ctfAttr[l].voltage,
                    _ctfAttr[l].defocusU * d,
                    _ctfAttr[l].defocusV * d,
                    _ctfAttr[l].defocusTheta,
                    _ctfAttr[l].Cs,
                    _ctfAttr[l].amplitudeContrast,
                    _ctfAttr[l].phaseShift,
                    1);

                FOR_EACH_PIXEL_FT(imgM)
                    imgM[i] *= REAL(ctf[i]);
                FOR_EACH_PIXEL_FT(imgN)
                    imgN[i] *= REAL(ctf[i]);
            }

            powerSpectrum(sSVD, imgM, rSig, 1);
            powerSpectrum(dSVD, _img[l], rSig, 1);

            NEG_FT(imgM);
            NEG_FT(imgN);

            ADD_FT(imgM, _img[l]);
            ADD_FT(imgN, _imgOri[l]);

            powerSpectrum(vSigM, imgM, rSig, 1);
            powerSpectrum(vSigN, imgN, rSig, 1);

            if (group)
            {
                omp_set_lock(&mtx[_groupID[l] - 1]);

                sigM.row(_groupID[l] - 1).head(rSig) += w * vSigM.transpose() / 2;
                sigM(_groupID[l] - 1, sigM.cols() - 1) += w;

                sigN.row(_groupID[l] - 1).head(rSig) += w * vSigN.transpose() / 2;
                sigN(_groupID[l] - 1, sigN.cols() - 1) += w;

                for (int i = 0; i < rSig; i++)
                    _svd(_groupID[l] - 1, i) += w * sqrt(sSVD(i) / dSVD(i));
                _svd(_groupID[l] - 1, _svd.cols() - 1) += w;

                omp_unset_lock(&mtx[_groupID[l] - 1]);
            }
            else
            {
                omp_set_lock(&mtx[0]);

                sigM.row(0).head(rSig) += w * vSigM.transpose() / 2;
                sigM(0, sigM.cols() - 1) += w;

                sigN.row(0).head(rSig) += w * vSigN.transpose() / 2;
                sigN(0, sigN.cols() - 1) += w;

                for (int i = 0; i < rSig; i++)
                    _svd(0, i) += w * sqrt(sSVD(i) / dSVD(i));
                _svd(0, _svd.cols() - 1) += w;

                omp_unset_lock(&mtx[0]);
            }
        }
    }

    delete[] mtx;

    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Averaging Sigma of Images Belonging to the Same Group";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Averaging Sigma of Images Belonging to the Same Group";

    MPI_Allreduce(MPI_IN_PLACE,
                  sigM.data(),
                  rSig * _nGroup,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  sigM.col(sigM.cols() - 1).data(),
                  _nGroup,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  sigN.data(),
                  rSig * _nGroup,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  sigN.col(sigN.cols() - 1).data(),
                  _nGroup,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _svd.data(),
                  rSig * _nGroup,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  _svd.col(_svd.cols() - 1).data(),
                  _nGroup,
                  TS_MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    if (group)
    {
        #pragma omp parallel for
        for (int i = 0; i < _sig.rows(); i++)
        {
            sigM.row(i).head(rSig) /= sigM(i, sigM.cols() - 1);
            sigN.row(i).head(rSig) /= sigN(i, sigN.cols() - 1);
            _svd.row(i).head(rSig) /= _svd(i, _svd.cols() - 1);
        }
    }
    else
    {
        sigM.row(0).head(rSig) /= sigM(0, sigM.cols() - 1);
        sigN.row(0).head(rSig) /= sigN(0, sigN.cols() - 1);
        _svd.row(0).head(rSig) /= _svd(0, _svd.cols() - 1);

        #pragma omp parallel for
        for (int i = 1; i < _sig.rows(); i++)
        {
            sigM.row(i).head(rSig) = sigM.row(0).head(rSig);
            sigN.row(i).head(rSig) = sigN.row(0).head(rSig);
            _svd.row(i).head(rSig) = _svd.row(0).head(rSig);
        }
    }

    #pragma omp parallel for
    for (int i = rSig; i < _sig.cols() - 1; i++)
    {
        sigM.col(i) = sigM.col(rSig - 1);
        sigN.col(i) = sigN.col(rSig - 1);
        _svd.col(i) = _svd.col(rSig - 1);
    }

    RFLOAT alpha = sqrt(M_PI * gsl_pow_2(_para.maskRadius / (_para.size * _para.pixelSize)));

    // ALOG(INFO, "LOGGER_SYS") << "alpha = " << alpha;

    #pragma omp parallel for
    for (int i = 0; i < _nGroup; i++)
        for (int j = 0; j < _sig.cols() - 1; j++)
        {
            // _sig(i, j) = gsl_pow_2(alpha) * sigN(i, j);

            RFLOAT ratio = GSL_MIN_DBL(1, _svd(i, j));

#ifdef OPTIMISER_SIGMA_MASK
            _sig(i, j) = sigM(i, j);
#else
            _sig(i, j) = ratio * sigM(i, j) + (1 - ratio) * alpha * sigN(i, j);
#endif
        }

    #pragma omp parallel for
    for (int i = 0; i < _nGroup; i++)
        for (int j = 0; j < rSig; j++)
            _sigRcp(i, j) = -0.5 / _sig(i, j);
}

void Optimiser::reconstructRef(const bool fscFlag,
                               const bool avgFlag,
                               const bool fscSave,
                               const bool avgSave,
                               const bool finished)
{
    FFT fft;

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space for Pre-calculation in Reconstruction";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Allocating Space for Pre-calculation in Reconstruction";

    allocPreCalIdx(_model.rU(), 0);

    if (_searchType != SEARCH_TYPE_CTF)
        allocPreCal(false, false, false);
    else
        allocPreCal(false, false, true);

    CHECK_MEMORY_USAGE("Reconstruct Insert begin!!!");
    long memUsageRM = memoryCheckRM();
    printf("insert memory check work begin:%dG!\n", memUsageRM / MEGABYTE);
#ifdef GPU_VERSION
    Complex *modelF;
    RFLOAT *modelT;
#endif

    NT_MASTER
    {
#ifdef GPU_VERSION
        printf("Round:%d, before insert image GPU memory check!\n", _iter);
        gpuMemoryCheck(_iGPU,
                       _commRank,
                       _nGPU);
#endif

        if ((_para.parGra) && (_para.k != 1))
        {
            ALOG(WARNING, "LOGGER_ROUND") << "Round " << _iter << ", " << "PATTICLE GRADING IS ONLY RECOMMENDED IN REFINEMENT, NOT CLASSIFICATION";
            BLOG(WARNING, "LOGGER_ROUND") << "Round " << _iter << ", " << "PATTICLE GRADING IS ONLY RECOMMENDED IN REFINEMENT, NOT CLASSIFICATION";
        }

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Inserting High Probability 2D Images into Reconstructor";
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Inserting High Probability 2D Images into Reconstructor";

        for (int t = 0; t < _para.k; t++)
            _model.reco(t).setPreCal(_nPxl, _iColPad, _iRowPad, _iPxl, _iSig);

        bool cSearch = ((_searchType == SEARCH_TYPE_CTF) ||
                        ((_para.cSearch) &&
                        (_searchType == SEARCH_TYPE_STOP)));

#ifdef GPU_VERSION
        Complex *dev_F[_nGPU];
        RFLOAT *dev_T[_nGPU];
        RFLOAT *devTau[_nGPU];
        RFLOAT *arrayTau;
        double *dev_O[_nGPU];
        double *arrayO;
        int *dev_C[_nGPU];
        int *arrayC;
        int *deviCol[_nGPU];
        int *deviRow[_nGPU];
        int *deviSig[_nGPU];
        int vdim = _model.reco(0).getModelDim(_para.mode);
        int modelSize = _model.reco(0).getModelSize(_para.mode);
        int tauSize = _model.reco(0).getTauSize();
        int nImg = _ID.size();

        if (_para.mode == MODE_2D)
        {
            modelF = (Complex*)malloc(_para.k * modelSize * sizeof(Complex));
            modelT = (RFLOAT*)malloc(_para.k * modelSize * sizeof(RFLOAT));
            arrayTau = (RFLOAT*)malloc(_para.k * tauSize * sizeof(RFLOAT));
            arrayO = new double[_para.k * 2];
            arrayC = new int[_para.k];
            
            for (int t = 0; t < _para.k; t++)
            {
                _model.reco(t).getF(modelF + t * modelSize,
                                    _para.mode,
                                    _para.nThreadsPerProcess);
                _model.reco(t).getT(modelT + t * modelSize,
                                    _para.mode,
                                    _para.nThreadsPerProcess);
                _model.reco(t).getTau(arrayTau + t * tauSize,
                                      _para.nThreadsPerProcess);
                
                arrayO[t * 2] = _model.reco(t).ox();
                arrayO[t * 2 + 1] = _model.reco(t).oy();
                arrayC[t] = _model.reco(t).counter();
            }
        }
        else
        {
            modelF = (Complex*)malloc(modelSize * sizeof(Complex));
            modelT = (RFLOAT*)malloc(modelSize * sizeof(RFLOAT));
            arrayTau = (RFLOAT*)malloc(_para.k * tauSize * sizeof(RFLOAT));
            arrayO = new double[_para.k * 3];
            arrayC = new int[_para.k];

            for (int t = 0; t < _para.k; t++)
            {
                _model.reco(t).getTau(arrayTau + t * tauSize,
                                      _para.nThreadsPerProcess);
                arrayO[t * 3] = _model.reco(t).ox();
                arrayO[t * 3 + 1] = _model.reco(t).oy();
                arrayO[t * 3 + 2] = _model.reco(t).oz();
                arrayC[t] = _model.reco(t).counter();
            }
        }
        
        allocFTO(_iGPU,
                 _stream,
                 modelF,
                 dev_F,
                 modelT,
                 dev_T,
                 arrayTau,
                 devTau,
                 arrayO,
                 dev_O,
                 arrayC,
                 dev_C,
                 _iColPad,
                 deviCol,
                 _iRowPad,
                 deviRow,
                 _iSig,
                 deviSig,
                 _para.mode,
                 _para.k,
                 tauSize,
                 vdim,
                 _nPxl,
                 _nGPU);

        if (_para.mode == MODE_2D)
        {
            RFLOAT *w = (RFLOAT*)malloc(_ID.size() * sizeof(RFLOAT));
            double *offS = (double*)malloc(_ID.size() * 2 * sizeof(double));
            double *nr = (double*)malloc(_para.mReco * _ID.size() * 2 * sizeof(double));
            double *nt = (double*)malloc(_para.mReco * _ID.size() * 2 * sizeof(double));
            double *nd = (double*)malloc(_para.mReco * _ID.size() * sizeof(double));
            CTFAttr* ctfaData = (CTFAttr*)malloc(_ID.size() * sizeof(CTFAttr));
            int *nc = (int*)malloc(_para.mReco * _ID.size() * sizeof(int));

            #pragma omp parallel for
            FOR_EACH_2D_IMAGE
            {
                if (_searchType != SEARCH_TYPE_STOP)
                {
                    _par[l]->calScore();
                }
                
                if (_para.parGra && _para.k == 1)
                    w[l] = _par[l]->compressR();
                else
                    w[l] = 1;

                w[l] /= _para.mReco;

                if (cSearch)
                {
                    ctfaData[l].voltage           = _ctfAttr[l].voltage;
                    ctfaData[l].defocusU          = _ctfAttr[l].defocusU;
                    ctfaData[l].defocusV          = _ctfAttr[l].defocusV;
                    ctfaData[l].defocusTheta      = _ctfAttr[l].defocusTheta;
                    ctfaData[l].Cs                = _ctfAttr[l].Cs;
                    ctfaData[l].amplitudeContrast = _ctfAttr[l].amplitudeContrast;
                    ctfaData[l].phaseShift        = _ctfAttr[l].phaseShift;
                }

                offS[l * 2] = _offset[l](0);
                offS[l * 2 + 1] = _offset[l](1);

                int shift = l * _para.mReco;
                for (int m = 0; m < _para.mReco; m++)
                {
                    dvec4 quat;
                    dvec2 tran;
                    double d;

                    _par[l]->rank1st(quat, tran, d);
                    
                    nc[shift + m] = _iRef[l];
                    nt[(shift + m) * 2] = tran(0);
                    nt[(shift + m) * 2 + 1] = tran(1);
                    nr[(shift + m) * 2] = quat(0);
                    nr[(shift + m) * 2 + 1] = quat(1);
                    if (cSearch)
                    {
                        nd[shift + m] = d;
                    }
                }
            }

            volumeCopy2D(_iGPU,
                         modelF,
                         dev_F,
                         modelT,
                         dev_T,
                         _para.k,
                         vdim,
                         _nGPU);

            RFLOAT *pglk_datPR = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_datPI = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_ctfP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
            RFLOAT *pglk_sigRcpP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));

            hostRegister(pglk_sigRcpP, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_datPR, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_datPI, IMAGE_BATCH * _nPxl);
            hostRegister(pglk_ctfP, IMAGE_BATCH * _nPxl);
            
            for (int l = 0; l < nImg;)
            {
                if (l >= nImg)
                    break;

                int batch = (l + IMAGE_BATCH < nImg)
                          ? IMAGE_BATCH : (nImg - l);

                RFLOAT *temp_datPR;
                RFLOAT *temp_datPI;
                RFLOAT *temp_ctfP;
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                RFLOAT *temp_sigP;
#endif
                for (int i = 0; i < batch; i++)
                {
                    temp_datPR = &_datPR[(l + i) * _nPxl];
                    temp_datPI = &_datPI[(l + i) * _nPxl];
                    memcpy((void*)(pglk_datPR + i * _nPxl),
                           (void*)temp_datPR,
                           _nPxl * sizeof(RFLOAT));
                    memcpy((void*)(pglk_datPI + i * _nPxl),
                           (void*)temp_datPI,
                           _nPxl * sizeof(RFLOAT));
                    
                    if (!cSearch)
                    {
                        temp_ctfP = &_ctfP[(l + i) * _nPxl];
                        memcpy((void*)(pglk_ctfP + i * _nPxl),
                               (void*)temp_ctfP,
                               _nPxl * sizeof(RFLOAT));
                    }
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                    temp_sigP = &_sigP[(l + i) * _nPxl];
                    memcpy((void*)(pglk_sigRcpP + i * _nPxl),
                           (void*)temp_sigP,
                           _nPxl * sizeof(RFLOAT));
#endif
                }

                InsertI2D(_iGPU, _stream, dev_F, dev_T, devTau, dev_O, dev_C, pglk_datPR, pglk_datPI, pglk_ctfP, pglk_sigRcpP, 
                          w + l, offS + l * 2, nr + l * _para.mReco * 2, nt + l * _para.mReco * 2, nd + l * _para.mReco, 
                          nc + l * _para.mReco, ctfaData + l, deviCol, deviRow, deviSig, _para.pixelSize, cSearch, _para.k, 
                          _para.pf, _nPxl,  _para.mReco, tauSize, _para.size, vdim, batch, _nGPU);
                
                l += batch;
            }

            hostFree(pglk_datPR);
            hostFree(pglk_datPI);
            hostFree(pglk_ctfP);
            hostFree(pglk_sigRcpP);
            
            free(pglk_datPR);
            free(pglk_datPI);
            free(pglk_ctfP);
            free(pglk_sigRcpP);
            
            allReduceFTO(_iGPU,
                         _gpusPerProcess,
                         _stream,
                         modelF,
                         dev_F,
                         modelT,
                         dev_T,
                         arrayTau,
                         devTau,
                         arrayO,
                         dev_O,
                         arrayC,
                         dev_C,
                         _hemi,
                         _para.mode,
                         0,
                         _para.k,
                         tauSize,
                         vdim,
                         _nGPU);

            for (int t = 0; t < _para.k; t++)
            {
                _model.reco(t).resetF(modelF + t * modelSize,
                                      _para.mode,
                                      _para.nThreadsPerProcess);
                _model.reco(t).resetT(modelT + t * modelSize,
                                      _para.mode,
                                      _para.nThreadsPerProcess);
                _model.reco(t).resetTau(arrayTau + t * tauSize);
                _model.reco(t).setOx(arrayO[t * 2]);
                _model.reco(t).setOy(arrayO[t * 2 + 1]);
                _model.reco(t).setCounter(arrayC[t]);
            }
            
            delete[]w;
            delete[]offS;
            delete[]nc;
            delete[]nr;
            delete[]nt;
            delete[]nd;
            delete[]ctfaData;

        }
        else if (_para.mode == MODE_3D)
        {
            if (_para.k != 1)
            {
                RFLOAT *w = (RFLOAT*)malloc(_ID.size() * sizeof(RFLOAT));
                double *offS = (double*)malloc(_ID.size() * 2 * sizeof(double));
                CTFAttr* ctfaData = (CTFAttr*)malloc(_ID.size() * sizeof(CTFAttr));
                int *nc = (int*)malloc(_para.k * _ID.size() * sizeof(int));

                #pragma omp parallel for
                for(size_t i = 0; i < _para.k * _ID.size(); i++)
                    nc[i] = 0;

                #pragma omp parallel for
                FOR_EACH_2D_IMAGE
                {
                    if (_para.parGra && _para.k == 1)
                        w[l] = _par[l]->compressR();
                    else
                        w[l] = 1;

                    w[l] /= _para.mReco;

                    if (cSearch)
                    {
                        ctfaData[l].voltage           = _ctfAttr[l].voltage;
                        ctfaData[l].defocusU          = _ctfAttr[l].defocusU;
                        ctfaData[l].defocusV          = _ctfAttr[l].defocusV;
                        ctfaData[l].defocusTheta      = _ctfAttr[l].defocusTheta;
                        ctfaData[l].Cs                = _ctfAttr[l].Cs;
                        ctfaData[l].amplitudeContrast = _ctfAttr[l].amplitudeContrast;
                        ctfaData[l].phaseShift        = _ctfAttr[l].phaseShift;
                    }

                    offS[l * 2] = _offset[l](0);
                    offS[l * 2 + 1] = _offset[l](1);

                    for (int m = 0; m < _para.mReco; m++)
                    {
                        nc[_iRef[l] * _ID.size() + l]++;
                    }
                }

                double *nr;
                double *nt;
                double *nd;

                RFLOAT *pglk_datPR = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
                RFLOAT *pglk_datPI = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
                RFLOAT *pglk_ctfP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
                RFLOAT *pglk_sigRcpP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));

                hostRegister(pglk_sigRcpP, IMAGE_BATCH * _nPxl);
                hostRegister(pglk_datPR, IMAGE_BATCH * _nPxl);
                hostRegister(pglk_datPI, IMAGE_BATCH * _nPxl);
                hostRegister(pglk_ctfP, IMAGE_BATCH * _nPxl);
                
                int temp = 0;
                for (int t = 0; t < _para.k; t++)
                {
                    temp = 0;
                    int shiftc = t * _ID.size();
                    for (size_t l = 0; l < _ID.size(); l++)
                    {
                        if (nc[shiftc + l] > temp)
                            temp = nc[shiftc + l];
                    }

                    if (temp != 0)
                    {
                        nr = (double*)malloc(temp * _ID.size() * 4 * sizeof(double));
                        nt = (double*)malloc(temp * _ID.size() * 2 * sizeof(double));
                        nd = (double*)malloc(temp * _ID.size() * sizeof(double));

                        #pragma omp parallel for
                        FOR_EACH_2D_IMAGE
                        {
                            int shift = l * temp;
                            for (int m = 0; m < temp; m++)
                            {
                                dvec4 quat;
                                dvec2 tran;
                                double d;
                                
                                _par[l]->rank1st(quat, tran, d);

                                nt[(shift + m) * 2] = tran(0);
                                nt[(shift + m) * 2 + 1] = tran(1);
                                nr[(shift + m) * 4] = quat(0);
                                nr[(shift + m) * 4 + 1] = quat(1);
                                nr[(shift + m) * 4 + 2] = quat(2);
                                nr[(shift + m) * 4 + 3] = quat(3);
                                if (cSearch)
                                {
                                    nd[shift + m] = d;
                                }
                            }
                        }

                        _model.reco(t).getF(modelF,
                                            _para.mode,
                                            _para.nThreadsPerProcess);
                        _model.reco(t).getT(modelT,
                                            _para.mode,
                                            _para.nThreadsPerProcess);
            
                        volumeCopy3D(_iGPU,
                                     modelF,
                                     dev_F,
                                     modelT,
                                     dev_T,
                                     vdim,
                                     _nGPU);

                        for (int l = 0; l < nImg;)
                        {
                            if (l >= nImg)
                                break;

                            int batch = (l + IMAGE_BATCH < nImg)
                                      ? IMAGE_BATCH : (nImg - l);

                            RFLOAT *temp_datPR;
                            RFLOAT *temp_datPI;
                            RFLOAT *temp_ctfP;
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                            RFLOAT *temp_sigP;
#endif
                            for (int i = 0; i < batch; i++)
                            {
                                temp_datPR = &_datPR[(l + i) * _nPxl];
                                temp_datPI = &_datPI[(l + i) * _nPxl];
                                memcpy((void*)(pglk_datPR + i * _nPxl),
                                       (void*)temp_datPR,
                                       _nPxl * sizeof(RFLOAT));
                                memcpy((void*)(pglk_datPI + i * _nPxl),
                                       (void*)temp_datPI,
                                       _nPxl * sizeof(RFLOAT));
                                
                                if (!cSearch)
                                {
                                    temp_ctfP = &_ctfP[(l + i) * _nPxl];
                                    memcpy((void*)(pglk_ctfP + i * _nPxl),
                                           (void*)temp_ctfP,
                                           _nPxl * sizeof(RFLOAT));
                                }
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                                temp_sigP = &_sigP[(l + i) * _nPxl];
                                memcpy((void*)(pglk_sigRcpP + i * _nPxl),
                                       (void*)temp_sigP,
                                       _nPxl * sizeof(RFLOAT));
#endif
                            }

                            InsertFT(_iGPU, _stream,dev_F, dev_T, devTau, dev_O, dev_C, pglk_datPR, 
                                     pglk_datPI, pglk_ctfP, pglk_sigRcpP, w + l, offS + l * 2, 
                                     nr + l * temp * 4, nt + l * temp * 2, nd + l * temp, nc + shiftc + l, 
                                     ctfaData + l, deviCol, deviRow, deviSig, _para.pixelSize, cSearch, t, 
                                     _para.pf, _nPxl, temp, tauSize, batch, _para.size, vdim, _nGPU);
                            
                            l += batch;
                        }

                        allReduceFTO(_iGPU,
                                     _gpusPerProcess,
                                     _stream,
                                     modelF,
                                     dev_F,
                                     modelT,
                                     dev_T,
                                     arrayTau,
                                     devTau,
                                     arrayO,
                                     dev_O,
                                     arrayC,
                                     dev_C,
                                     _hemi,
                                     _para.mode,
                                     t,
                                     _para.k,
                                     tauSize,
                                     vdim,
                                     _nGPU);

                        _model.reco(t).resetF(modelF,
                                              _para.mode,
                                              _para.nThreadsPerProcess);
                        _model.reco(t).resetT(modelT,
                                              _para.mode,
                                              _para.nThreadsPerProcess);
                        _model.reco(t).resetTau(arrayTau + t * tauSize);
                        
                        _model.reco(t).setOx(arrayO[t * 3]);
                        _model.reco(t).setOy(arrayO[t * 3 + 1]);
                        _model.reco(t).setOz(arrayO[t * 3 + 2]);
                        _model.reco(t).setCounter(arrayC[t]);
                        
                        delete[]nr;
                        delete[]nt;
                        delete[]nd;
                    }
                }

                hostFree(pglk_datPR);
                hostFree(pglk_datPI);
                hostFree(pglk_ctfP);
                hostFree(pglk_sigRcpP);

                free(pglk_datPR);
                free(pglk_datPI);
                free(pglk_ctfP);
                free(pglk_sigRcpP);
                
                delete[]w;
                delete[]offS;
                delete[]nc;
                delete[]ctfaData;
            }
            else
            {
                RFLOAT* w = (RFLOAT*)malloc(_ID.size() * sizeof(RFLOAT));
                double* offS = (double*)malloc(_ID.size() * 2 * sizeof(double));
                double* nr = (double*)malloc(_para.mReco * _ID.size() * 4 * sizeof(double));
                double* nt = (double*)malloc(_para.mReco * _ID.size() * 2 * sizeof(double));
                double* nd = (double*)malloc(_para.mReco * _ID.size() * sizeof(double));
                CTFAttr* ctfaData = (CTFAttr*)malloc(_ID.size() * sizeof(CTFAttr));

                #pragma omp parallel for
                FOR_EACH_2D_IMAGE
                {
                    if (_para.parGra && _para.k == 1)
                        w[l] = _par[l]->compressR();
                    else
                        w[l] = 1;

                    w[l] /= _para.mReco;

                    if (cSearch)
                    {
                        ctfaData[l].voltage           = _ctfAttr[l].voltage;
                        ctfaData[l].defocusU          = _ctfAttr[l].defocusU;
                        ctfaData[l].defocusV          = _ctfAttr[l].defocusV;
                        ctfaData[l].defocusTheta      = _ctfAttr[l].defocusTheta;
                        ctfaData[l].Cs                = _ctfAttr[l].Cs;
                        ctfaData[l].amplitudeContrast = _ctfAttr[l].amplitudeContrast;
                        ctfaData[l].phaseShift        = _ctfAttr[l].phaseShift;
                    }

                    offS[l * 2] = _offset[l](0);
                    offS[l * 2 + 1] = _offset[l](1);

                    int shift = l * _para.mReco;
                    for (int m = 0; m < _para.mReco; m++)
                    {
                        dvec4 quat;
                        dvec2 tran;
                        double d;

                        _par[l]->rank1st(quat, tran, d);

                        nt[(shift + m) * 2] = tran(0);
                        nt[(shift + m) * 2 + 1] = tran(1);
                        nr[(shift + m) * 4] = quat(0);
                        nr[(shift + m) * 4 + 1] = quat(1);
                        nr[(shift + m) * 4 + 2] = quat(2);
                        nr[(shift + m) * 4 + 3] = quat(3);
                        if (cSearch)
                        {
                            nd[shift + m] = d;
                        }
                    }
                }

                _model.reco(0).getF(modelF,
                                    _para.mode,
                                    _para.nThreadsPerProcess);
                _model.reco(0).getT(modelT,
                                    _para.mode,
                                    _para.nThreadsPerProcess);
            
                volumeCopy3D(_iGPU,
                             modelF,
                             dev_F,
                             modelT,
                             dev_T,
                             vdim,
                             _nGPU);

                RFLOAT *pglk_datPR = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
                RFLOAT *pglk_datPI = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
                RFLOAT *pglk_ctfP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));
                RFLOAT *pglk_sigRcpP = (RFLOAT*)malloc(IMAGE_BATCH * _nPxl * sizeof(RFLOAT));

                hostRegister(pglk_sigRcpP, IMAGE_BATCH * _nPxl);
                hostRegister(pglk_datPR, IMAGE_BATCH * _nPxl);
                hostRegister(pglk_datPI, IMAGE_BATCH * _nPxl);
                hostRegister(pglk_ctfP, IMAGE_BATCH * _nPxl);
                
                for (int l = 0; l < nImg;)
                {
                    if (l >= nImg)
                        break;

                    int batch = (l + IMAGE_BATCH < nImg)
                              ? IMAGE_BATCH : (nImg - l);

                    RFLOAT *temp_datPR;
                    RFLOAT *temp_datPI;
                    RFLOAT *temp_ctfP;
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                    RFLOAT *temp_sigP;
#endif
                    for (int i = 0; i < batch; i++)
                    {
                        temp_datPR = &_datPR[(l + i) * _nPxl];
                        temp_datPI = &_datPI[(l + i) * _nPxl];
                        memcpy((void*)(pglk_datPR + i * _nPxl),
                               (void*)temp_datPR,
                               _nPxl * sizeof(RFLOAT));
                        memcpy((void*)(pglk_datPI + i * _nPxl),
                               (void*)temp_datPI,
                               _nPxl * sizeof(RFLOAT));
                        
                        if (!cSearch)
                        {
                            temp_ctfP = &_ctfP[(l + i) * _nPxl];
                            memcpy((void*)(pglk_ctfP + i * _nPxl),
                                   (void*)temp_ctfP,
                                   _nPxl * sizeof(RFLOAT));
                        }
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                        temp_sigP = &_sigP[(l + i) * _nPxl];
                        memcpy((void*)(pglk_sigRcpP + i * _nPxl),
                               (void*)temp_sigP,
                               _nPxl * sizeof(RFLOAT));
#endif
                    }

                    InsertFT(_iGPU, _stream, dev_F, dev_T, devTau, dev_O, dev_C, pglk_datPR, pglk_datPI, 
                             pglk_ctfP, pglk_sigRcpP, w + l, offS + l * 2, nr + l * _para.mReco * 4, 
                             nt + l * _para.mReco * 2, nd + l * _para.mReco, ctfaData + l, deviCol, deviRow, 
                             deviSig, _para.pixelSize, cSearch, _para.pf, _nPxl, _para.mReco, tauSize, batch, 
                             _para.size, vdim, _nGPU);
                    
                    l += batch;
                }

                hostFree(pglk_datPR);
                hostFree(pglk_datPI);
                hostFree(pglk_ctfP);
                hostFree(pglk_sigRcpP);

                free(pglk_datPR);
                free(pglk_datPI);
                free(pglk_ctfP);
                free(pglk_sigRcpP);
                
                allReduceFTO(_iGPU,
                             _gpusPerProcess,
                             _stream,
                             modelF,
                             dev_F,
                             modelT,
                             dev_T,
                             arrayTau,
                             devTau,
                             arrayO,
                             dev_O,
                             arrayC,
                             dev_C,
                             _hemi,
                             _para.mode,
                             0,
                             _para.k,
                             tauSize,
                             vdim,
                             _nGPU);

                _model.reco(0).resetF(modelF,
                                      _para.mode,
                                      _para.nThreadsPerProcess);
                _model.reco(0).resetT(modelT,
                                      _para.mode,
                                      _para.nThreadsPerProcess);
                _model.reco(0).resetTau(arrayTau);
                
                _model.reco(0).setOx(arrayO[0]);
                _model.reco(0).setOy(arrayO[1]);
                _model.reco(0).setOz(arrayO[2]);
                _model.reco(0).setCounter(arrayC[0]);
                        
                free(w);
                free(offS);
                free(nr);
                free(nt);
                free(nd);
                free(ctfaData);
            }
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");
            abort();
        }

        freeFTO(_iGPU,
                modelF,
                dev_F,
                modelT,
                dev_T,
                arrayTau,
                devTau,        
                arrayO        ,
                dev_O,
                arrayC,
                dev_C,
                deviCol,
                deviRow,
                deviSig,
                _nGPU);

        delete[] arrayO;
        delete[] arrayC;
        delete[] arrayTau;
#else
        // Complex* poolTransImgP = (Complex*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(Complex));

        RFLOAT* poolTransImgPR = (RFLOAT*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(RFLOAT));
        RFLOAT* poolTransImgPI = (RFLOAT*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(RFLOAT));

        MemoryBazaarDustman<RFLOAT, BaseType, 4> datPRDustman(&_datPR);
        MemoryBazaarDustman<RFLOAT, BaseType, 4> datPIDustman(&_datPI);
        MemoryBazaarDustman<RFLOAT, BaseType, 4> ctfPDustman(&_ctfP);
        #pragma omp parallel for firstprivate(datPRDustman, datPIDustman, ctfPDustman)
        FOR_EACH_2D_IMAGE
        {
            RFLOAT* ctf;

            RFLOAT w;

            if (_searchType != SEARCH_TYPE_STOP)
            {
                // allow user change score when only performing a reconstruction without expectation
                _par[l]->calScore();
            }

            if ((_para.parGra) && (_para.k == 1))
            {
                w = _par[l]->compressR();
            }
            else
            {
                w = 1;
            }

            w /= _para.mReco;

            // Complex* transImgP = poolTransImgP + _nPxl * omp_get_thread_num();
            RFLOAT* transImgPR = poolTransImgPR + _nPxl * omp_get_thread_num();
            RFLOAT* transImgPI = poolTransImgPI + _nPxl * omp_get_thread_num();

            // Complex* orignImgP = _datP + _nPxl * l;
            RFLOAT* originImgPR = &_datPR[_nPxl * l];
            RFLOAT* originImgPI = &_datPI[_nPxl * l];

            for (int m = 0; m < _para.mReco; m++)
            {
                dvec4 quat;
                dvec2 tran;
                double d;

                if (_para.mode == MODE_2D)
                {
                    _par[l]->rank1st(quat, tran, d);

                    dmat22 rot2D;

                    rotate2D(rot2D, dvec2(quat(0), quat(1)));

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                    translate(transImgPR,
                              transImgPI,
                              originImgPR,
                              originImgPI,
                              -(tran - _offset[l])(0),
                              -(tran - _offset[l])(1),
                              _para.size,
                              _para.size,
                              _iCol,
                              _iRow,
                              _nPxl,
                              1);
#else
                    translate(transImgPR,
                              transImgPI,
                              originImgPR,
                              originImgPI,
                              -(tran)(0),
                              -(tran)(1),
                              _para.size,
                              _para.size,
                              _iCol,
                              _iRow,
                              _nPxl,
                              1);
#endif

                    if (cSearch)
                    {
                        ctf = (RFLOAT*)TSFFTW_malloc(_nPxl * sizeof(RFLOAT));

                        CTF(ctf,
                            _para.pixelSize,
                            _ctfAttr[l].voltage,
                            _ctfAttr[l].defocusU * d,
                            _ctfAttr[l].defocusV * d,
                            _ctfAttr[l].defocusTheta,
                            _ctfAttr[l].Cs,
                            _ctfAttr[l].amplitudeContrast,
                            _ctfAttr[l].phaseShift,
                            _para.size,
                            _para.size,
                            _iCol,
                            _iRow,
                            _nPxl,
                            1);
                    }
                    else
                    {
                        // ctf = _ctfP + _nPxl * l;
                        ctf = &(_ctfP[_nPxl * l]);
                    }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                    vec sig = _sig.row(_groupID[l] - 1).transpose();

                    _model.reco(_iRef[l]).insertP(transImgPR,
                                             transImgPI,
                                             ctf,
                                             rot2D,
                                             w,
                                             &sig);
#else
                    _model.reco(_iRef[l]).insertP(transImgPR,
                                             transImgPI,
                                             ctf,
                                             rot2D,
                                             w);
#endif

                    if (cSearch) TSFFTW_free(ctf);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                    dvec2 dir = -rot2D * (tran - _offset[l]);
#else
                    dvec2 dir = -rot2D * tran;
#endif
                    _model.reco(_iRef[l]).insertDir(dir);
                }

                else if (_para.mode == MODE_3D)
                {
                    _par[l]->rank1st(quat, tran, d);

                    dmat33 rot3D;

                    rotate3D(rot3D, quat);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                    translate(transImgPR,
                              transImgPI,
                              originImgPR,
                              originImgPI,
                              -(tran - _offset[l])(0),
                              -(tran - _offset[l])(1),
                              _para.size,
                              _para.size,
                              _iCol,
                              _iRow,
                              _nPxl,
                              1);
#else
                    translate(transImgPR,
                              transImgPI,
                              originImgPR,
                              originImgPI,
                              -(tran)(0),
                              -(tran)(1),
                              _para.size,
                              _para.size,
                              _iCol,
                              _iRow,
                              _nPxl,
                              1);
#endif

                    if (cSearch)
                    {
                        ctf = (RFLOAT*)TSFFTW_malloc(_nPxl * sizeof(RFLOAT));

                        CTF(ctf,
                            _para.pixelSize,
                            _ctfAttr[l].voltage,
                            _ctfAttr[l].defocusU * d,
                            _ctfAttr[l].defocusV * d,
                            _ctfAttr[l].defocusTheta,
                            _ctfAttr[l].Cs,
                            _ctfAttr[l].amplitudeContrast,
                            _ctfAttr[l].phaseShift,
                            _para.size,
                            _para.size,
                            _iCol,
                            _iRow,
                            _nPxl,
                            1);
                    }
                    else
                    {
                        ctf = &(_ctfP[_nPxl * l]);
                    }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                    vec sig = _sig.row(_groupID[l] - 1).transpose();

                    _model.reco(_iRef[l]).insertP(transImgPR,
                                             transImgPI,
                                             ctf,
                                             rot3D,
                                             w,
                                             &sig);
#else
                    _model.reco(_iRef[l]).insertP(transImgPR,
                                             transImgPI,
                                             ctf,
                                             rot3D,
                                             w);
#endif

                    if (cSearch) TSFFTW_free(ctf);

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                    dvec3 dir = -rot3D * dvec3((tran - _offset[l])[0],
                                           (tran - _offset[l])[1],
                                           0);
#else
                    dvec3 dir = -rot3D * dvec3(tran[0], tran[1], 0);
#endif
                    _model.reco(_iRef[l]).insertDir(dir);
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }
            }
        }
#endif

#ifdef VERBOSE_LEVEL_2
        ILOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Inserting Images Into Reconstructor(s) Accomplished";
#endif

        MPI_Barrier(_hemi);

#ifdef GPU_VERSION
        printf("Round:%d, after insert image GPU memory check!\n", _iter);
        gpuMemoryCheck(_iGPU,
                       _commRank,
                       _nGPU);
#endif

        for (int t = 0; t < _para.k; t++)
        {
            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Preparing Content in Reconstructor of Reference "
                                       << t;
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Preparing Content in Reconstructor of Reference "
                                       << t;
            _model.reco(t).allReduceCounter();
#ifdef GPU_VERSION
            _model.reco(t).prepareTFG(_iGPU,
                                      _stream,
                                      _nGPU);
#else
            _model.reco(t).prepareTF(_para.nThreadsPerProcess);
#endif
        }


        if (_para.refAutoRecentre)
        {
            for (int t = 0; t < _para.k; t++)
            {
                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Preparing Content in Reconstructor of Reference "
                                       << t;
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Preparing Content in Reconstructor of Reference "
                                       << t;

                _model.reco(t).prepareO();

                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Estimated X-Offset, Y-Offset and Z-Offset of Reference "
                                           << t
                                           << ": "
                                           << _model.reco(t).ox()
                                           << ", "
                                           << _model.reco(t).oy()
                                           << ", "
                                           << _model.reco(t).oz()
                                           << " (Pixel)";
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Estimated X-Offset, Y-Offset and Z-Offset of Reference "
                                           << t
                                           << ": "
                                           << _model.reco(t).ox()
                                           << ", "
                                           << _model.reco(t).oy()
                                           << ", "
                                           << _model.reco(t).oz()
                                           << " (Pixel)";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    CHECK_MEMORY_USAGE("Reconstruct Insert done!!!");
    long memUsageD = memoryCheckRM();
    printf("insert memory check work done:%dG!\n", memUsageD / MEGABYTE);

#ifdef OPTIMISER_BALANCE_CLASS
    umat2 bm;
#endif

    if (fscFlag)
    {
        NT_MASTER
        {
#ifdef GPU_RECONSTRUCT
            int vdim = _model.reco(0).getModelDim(_para.mode);
            int modelSize = _model.reco(0).getModelSize(_para.mode);
#endif
            
            #pragma omp parallel for num_threads(_para.nThreadsPerProcess)
            for (int t = 0; t < _para.k; t++)
            {
                _model.reco(t).setMAP(false);
#ifdef OPTIMISER_RECONSTRUCT_JOIN_HALF
                _model.reco(t).setJoinHalf(true);
#else
                _model.reco(t).setJoinHalf(false);
#endif

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_2D_GRID_CORR
                    _model.reco(t).setGridCorr(true);
#else
                    _model.reco(t).setGridCorr(false);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_3D_GRID_CORR
                    _model.reco(t).setGridCorr(true);
#else
                    _model.reco(t).setGridCorr(false);
#endif
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }
            }

            if (_para.mode == MODE_2D)
            {
                int kbatch = CLASS_BATCH;
#ifdef GPU_RECONSTRUCT
                RFLOAT nf = _model.reco(0).getNF();
                bool map = _model.reco(0).MAP();
                bool gridCorr = _model.reco(0).gridCorr();
                bool joinHalf = _model.reco(0).joinHalf();
                int fscMatSize = _model.reco(0).getFSCSize();
                int maxRadius = _model.reco(0).maxRadius();
                int pf = _model.reco(0).pf();
                int _N = _model.reco(0).N();
                
                Complex *refArray;
                RFLOAT *fscMat;
                RFLOAT *ox;
                RFLOAT *oy;
                refArray = (Complex*)malloc(kbatch * _N * (_N / 2 + 1) * sizeof(Complex)); 
                fscMat = (RFLOAT*)malloc(kbatch * fscMatSize * sizeof(RFLOAT));
                ox = (RFLOAT*)malloc(kbatch * sizeof(RFLOAT));
                oy = (RFLOAT*)malloc(kbatch * sizeof(RFLOAT));
#endif
                for (int t = 0; t < _para.k;)
                {
                    if (t >= _para.k)
                        break;
                    
                    kbatch = (t + CLASS_BATCH > _para.k) 
                           ? (_para.k - t) : CLASS_BATCH;
#ifdef GPU_RECONSTRUCT
                    #pragma omp parallel for num_threads(_para.nThreadsPerProcess)
                    for (int b = 0; b < kbatch; b++)
                    {
                        vec FSC = _model.reco(t + b).getFSC();
                        Map<vec>(fscMat + b * fscMatSize, 
                                 FSC.rows(), 
                                 FSC.cols()) = FSC;
                        ox[b] = (RFLOAT)(-_model.reco(t + b).ox());
                        oy[b] = (RFLOAT)(-_model.reco(t + b).oy());
                    }
                    
                    reconstructG2D(_iGPU,
                                   _stream,
                                   _model.reco(0).getTabFuncRL(),
                                   refArray,
                                   modelF + t * modelSize,
                                   modelT + t * modelSize,
                                   fscMat,
                                   nf,
                                   map,
                                   gridCorr,
                                   joinHalf,
                                   fscMatSize,
                                   maxRadius,
                                   pf,
                                   _N,
                                   vdim,
                                   kbatch,
                                   _para.nThreadsPerProcess,
                                   _nGPU);
                    
                    if (_mask.isEmptyRL() && _para.refAutoRecentre)
                    {
                        TranslateI2D(_stream,
                                     _iGPU,
                                     refArray,
                                     ox,
                                     oy,
                                     kbatch, 
                                     _model.rU(),
                                     _N,
                                     _nGPU);
                    }

                    for (int b = 0; b < kbatch; b++)
                    {
                        #pragma omp parallel for
                        SET_0_FT(_model.ref(t + b));

                        COPY_FT(_model.ref(t + b), (refArray + b * _N * (_N / 2 + 1)));
                    }
#else
                    for (int b = 0; b < kbatch; b++)
                    {
                        Volume ref;
                        _model.reco(t + b).reconstruct(ref, 
                                                       _para.nThreadsPerProcess);
                        fft.fw(ref, 
                               _para.nThreadsPerProcess);
                        
                        if (_mask.isEmptyRL() && _para.refAutoRecentre)
                        {
                            Image img(_para.size, _para.size, FT_SPACE);

                            SLC_EXTRACT_FT(img, ref, 0);
                            
                            translate(img, 
                                      img, 
                                      _model.rU(), 
                                      -_model.reco(t).ox(), 
                                      -_model.reco(t).oy(), 
                                      _para.nThreadsPerProcess);

                            SLC_REPLACE_FT(ref, img, 0);
                        } 
#ifndef NAN_NO_CHECK
                        SEGMENT_NAN_CHECK_COMPLEX(ref.dataFT(), ref.sizeFT());
#endif
                        #pragma omp parallel for
                        SET_0_FT(_model.ref(t + b));

                        COPY_FT(_model.ref(t + b), ref);
                    }
#endif
                    t += kbatch;
                }
#ifdef GPU_RECONSTRUCT
                free(refArray); 
                free(fscMat); 
                free(ox); 
                free(oy); 
#endif
            }
            else if (_para.mode == MODE_3D)
            {
                for (int t = 0; t < _para.k; t++)
                {
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructing Reference "
                                               << t
                                               << " for Determining FSC";
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructing Reference "
                                               << t
                                               << " for Determining FSC";
                    Volume ref;
#ifdef GPU_RECONSTRUCT
                    _model.reco(t).reconstructG(_iGPU,
                                                _stream,
                                                ref,
                                                _nGPU, 
                                                _para.nThreadsPerProcess);
                    
#else
                    _model.reco(t).reconstruct(ref, _para.nThreadsPerProcess);

#ifndef NAN_NO_CHECK
                    SEGMENT_NAN_CHECK(ref.dataRL(), ref.sizeRL());
#endif

#ifdef VERBOSE_LEVEL_2
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Fourier Transforming Reference " << t;
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Fourier Transforming Reference " << t;
#endif
#endif
                    fft.fw(ref, _para.nThreadsPerProcess);

#ifndef NAN_NO_CHECK
                    SEGMENT_NAN_CHECK_COMPLEX(ref.dataFT(), ref.sizeFT());
#endif

                    if (_mask.isEmptyRL() && _para.refAutoRecentre)
                    {
                        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Centring Reference " << t;
                        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Centring Reference " << t;

                        if (_sym.pgGroup() == PG_CN)
                        {
#ifdef GPU_RECONSTRUCT
                            TranslateI(_iGPU,
                                       _stream,
                                       ref,
                                       -_model.reco(t).ox(),
                                       -_model.reco(t).oy(),
                                       -_model.reco(t).oz(),
                                       _nGPU,
                                       _model.rU());
#else
                            translate(ref, 
                                      ref, 
                                      _model.rU(), 
                                      -_model.reco(t).ox(), 
                                      -_model.reco(t).oy(), 
                                      -_model.reco(t).oz(), 
                                      _para.nThreadsPerProcess);
#endif
                        }
                    }

#ifdef VERBOSE_LEVEL_2
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference " << t << "Fourier Transformed";
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference " << t << "Fourier Transformed";
#endif
                    #pragma omp parallel for
                    SET_0_FT(_model.ref(t));

                    COPY_FT(_model.ref(t), ref);
                }
            }
        }

#ifndef NAN_NO_CHECK
        NT_MASTER
        {
            for (int t = 0; t < _para.k; t++)
            {
                SEGMENT_NAN_CHECK_COMPLEX(_model.ref(t).dataFT(), _model.ref(t).sizeFT());
            }
        }
#endif

        if (fscSave && (_para.saveRefEachIter || finished))
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Reference(s)";

            if (_para.mode == MODE_2D)
            {
#ifdef OPTIMISER_2D_SAVE_JOIN_MAP
                saveMapJoin(finished);
#else
                saveMapHalf(finished);
#endif
            }
            else if (_para.mode == MODE_3D)
            {
                if (_para.k == 1)
                {
                    saveMapHalf(finished);
                }
                else
                {
#ifdef OPTIMISER_3D_SAVE_JOIN_MAP
                    saveMapJoin(finished);
#else
                    saveMapHalf(finished);
#endif
                }
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }
        }

#ifndef NAN_NO_CHECK
        NT_MASTER
        {
            for (int t = 0; t < _para.k; t++)
            {
                SEGMENT_NAN_CHECK_COMPLEX(_model.ref(t).dataFT(), _model.ref(t).sizeFT());
            }
        }
#endif

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef OPTIMISER_BALANCE_CLASS

        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Determining How to Balancing Class(es)";

            determineBalanceClass(bm, CLASS_BALANCE_FACTOR);

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Balancing Class(es)";

            balanceClass(bm);
            /***
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
            balanceClass(CLASS_BALANCE_FACTOR, false);
#else
            balanceClass(CLASS_BALANCE_FACTOR, true);
#endif
            ***/

#ifdef VERBOSE_LEVEL_1

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Percentage of Images Belonging to Each Class After Balancing";

            for (int t = 0; t < _para.k; t++)
                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << _cDistr(t) * 100
                                           << "\% Percentage of Images Belonging to Class "
                                           << t;
#endif
        }

#endif

#ifndef NAN_NO_CHECK
        NT_MASTER
        {
            for (int t = 0; t < _para.k; t++)
            {
                SEGMENT_NAN_CHECK_COMPLEX(_model.ref(t).dataFT(), _model.ref(t).sizeFT());
            }
        }
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
        _model.compareTwoHemispheres(true, false, AVERAGE_TWO_HEMISPHERE_THRES, _para.nThreadsPerProcess);
#else
        _model.compareTwoHemispheres(true, true, AVERAGE_TWO_HEMISPHERE_THRES, _para.nThreadsPerProcess);
#endif
    }

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
    if (avgFlag)
    {
        NT_MASTER
        {
#ifdef GPU_RECONSTRUCT
            int vdim = _model.reco(0).getModelDim(_para.mode);
            int modelSize = _model.reco(0).getModelSize(_para.mode);
#endif
            #pragma omp parallel for num_threads(_para.nThreadsPerProcess)
            for (int t = 0; t < _para.k; t++)
            {
                _model.reco(t).setMAP(true);
#ifdef OPTIMISER_RECONSTRUCT_JOIN_HALF
                _model.reco(t).setJoinHalf(true);
#else
                _model.reco(t).setJoinHalf(false);
#endif

                if (_para.mode == MODE_2D)
                {
#ifdef OPTIMISER_2D_GRID_CORR
                    _model.reco(t).setGridCorr(true);
#else
                    _model.reco(t).setGridCorr(false);
#endif
                }
                else if (_para.mode == MODE_3D)
                {
#ifdef OPTIMISER_3D_GRID_CORR
                    _model.reco(t).setGridCorr(true);
#else
                    _model.reco(t).setGridCorr(false);
#endif
                }
                else
                {
                    REPORT_ERROR("INEXISTENT MODE");

                    abort();
                }
            }
            
            if (_para.mode == MODE_2D)
            {
                int kbatch = CLASS_BATCH;
#ifdef GPU_RECONSTRUCT
                RFLOAT nf = _model.reco(0).getNF();
                bool map = _model.reco(0).MAP();
                bool gridCorr = _model.reco(0).gridCorr();
                bool joinHalf = _model.reco(0).joinHalf();
                int fscMatSize = _model.reco(0).getFSCSize();
                int maxRadius = _model.reco(0).maxRadius();
                int pf = _model.reco(0).pf();
                int _N = _model.reco(0).N();
                Complex *refArray;
                refArray = (Complex*)malloc(kbatch * _N * (_N / 2 + 1) * sizeof(Complex)); 
                RFLOAT *fscMat;
                fscMat = (RFLOAT*)malloc(kbatch * fscMatSize * sizeof(RFLOAT));
                RFLOAT *ox;
                ox = (RFLOAT*)malloc(kbatch * sizeof(RFLOAT));
                RFLOAT *oy;
                oy = (RFLOAT*)malloc(kbatch * sizeof(RFLOAT));
#endif
                for (int t = 0; t < _para.k;)
                {
                    if (t >= _para.k)
                        break;
                    
                    kbatch = (t + CLASS_BATCH > _para.k) 
                           ? (_para.k - t) : CLASS_BATCH;
#ifdef GPU_RECONSTRUCT
                    #pragma omp parallel for num_threads(_para.nThreadsPerProcess)
                    for (int b = 0; b < kbatch; b++)
                    {
                        vec FSC = _model.reco(t + b).getFSC();
                        Map<vec>(fscMat + b * fscMatSize, 
                                 FSC.rows(), 
                                 FSC.cols()) = FSC;
                        ox[b] = (RFLOAT)(-_model.reco(t + b).ox());
                        oy[b] = (RFLOAT)(-_model.reco(t + b).oy());
                    }
                    
                    reconstructG2D(_iGPU,
                                   _stream,
                                   _model.reco(0).getTabFuncRL(),
                                   refArray,
                                   modelF + t * modelSize,
                                   modelT + t * modelSize,
                                   fscMat,
                                   nf,
                                   map,
                                   gridCorr,
                                   joinHalf,
                                   fscMatSize,
                                   maxRadius,
                                   pf,
                                   _N,
                                   vdim,
                                   kbatch,
                                   _para.nThreadsPerProcess,
                                   _nGPU);
                    
                    if (_mask.isEmptyRL() && _para.refAutoRecentre)
                    {
                        TranslateI2D(_stream,
                                     _iGPU,
                                     refArray,
                                     ox,
                                     oy,
                                     kbatch, 
                                     _model.rU(),
                                     _N,
                                     _nGPU);
                    }

                    for (int b = 0; b < kbatch; b++)
                    {
                        #pragma omp parallel for
                        SET_0_FT(_model.ref(t + b));

                        COPY_FT(_model.ref(t + b), (refArray + b * _N * (_N / 2 + 1)));
                    }
#else
                    for (int b = 0; b < kbatch; b++)
                    {
                        Volume ref;
                        _model.reco(t + b).reconstruct(ref, 
                                                       _para.nThreadsPerProcess);
                        fft.fw(ref, 
                               _para.nThreadsPerProcess);
                        
                        if (_mask.isEmptyRL() && _para.refAutoRecentre)
                        {
                            Image img(_para.size, _para.size, FT_SPACE);

                            SLC_EXTRACT_FT(img, ref, 0);
                            
                            translate(img, 
                                      img, 
                                      _model.rU(), 
                                      -_model.reco(t).ox(), 
                                      -_model.reco(t).oy(), 
                                      _para.nThreadsPerProcess);

                            SLC_REPLACE_FT(ref, img, 0);
                        } 
#ifndef NAN_NO_CHECK
                        SEGMENT_NAN_CHECK_COMPLEX(ref.dataFT(), ref.sizeFT());
#endif
                        #pragma omp parallel for
                        SET_0_FT(_model.ref(t + b));

                        COPY_FT(_model.ref(t + b), ref);
                    }
#endif
                    t += kbatch;
                }
#ifdef GPU_RECONSTRUCT
                free(refArray); 
                free(fscMat); 
                free(ox); 
                free(oy); 
#endif
            }
            else if (_para.mode == MODE_3D)
            {
                for (int t = 0; t < _para.k; t++)
                {
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructing Reference "
                                               << t
                                               << " for Next Iteration";
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reconstructing Reference "
                                               << t
                                               << " for Next Iteration";

                    Volume ref;
#ifdef GPU_RECONSTRUCT
                    _model.reco(t).reconstructG(_iGPU,
                                                _stream,
                                                ref,
                                                _nGPU, 
                                                _para.nThreadsPerProcess);
                    
#else
                    _model.reco(t).reconstruct(ref, _para.nThreadsPerProcess);

#ifndef NAN_NO_CHECK
                    SEGMENT_NAN_CHECK(ref.dataRL(), ref.sizeRL());
#endif

#ifdef VERBOSE_LEVEL_2
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Fourier Transforming Reference " << t;
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Fourier Transforming Reference " << t;
#endif
#endif
                    fft.fw(ref, _para.nThreadsPerProcess);

#ifndef NAN_NO_CHECK
                    SEGMENT_NAN_CHECK_COMPLEX(ref.dataFT(), ref.sizeFT());
#endif

                    if (_mask.isEmptyRL() && _para.refAutoRecentre)
                    {
                        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Centring Reference " << t;
                        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Centring Reference " << t;

                        if (_sym.pgGroup() == PG_CN)
                        {
#ifdef GPU_RECONSTRUCT
                            TranslateI(_iGPU,
                                       _stream,
                                       ref,
                                       -_model.reco(t).ox(),
                                       -_model.reco(t).oy(),
                                       -_model.reco(t).oz(),
                                       _nGPU,
                                       _model.rU());
#else
                            translate(ref, 
                                      ref, 
                                      _model.rU(), 
                                      -_model.reco(t).ox(), 
                                      -_model.reco(t).oy(), 
                                      -_model.reco(t).oz(), 
                                      _para.nThreadsPerProcess);
#endif
                        }
                    }

#ifdef VERBOSE_LEVEL_2
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference " << t << "Fourier Transformed";
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference " << t << "Fourier Transformed";
#endif
                    #pragma omp parallel for
                    SET_0_FT(_model.ref(t));

                    COPY_FT(_model.ref(t), ref);
                }
            }
        }


        if (avgSave && (_para.saveRefEachIter || finished))
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Reference(s)";

            if (_para.mode == MODE_2D)
            {
#ifdef OPTIMISER_2D_SAVE_JOIN_MAP
                saveMapJoin(finished);
#else
                saveMapHalf(finished);
#endif
            }
            else if (_para.mode == MODE_3D)
            {
                if (_para.k == 1)
                {
                    saveMapHalf(finished);
                }
                else
                {
#ifdef OPTIMISER_3D_SAVE_JOIN_MAP
                    saveMapJoin(finished);
#else
                    saveMapHalf(finished);
#endif
                }
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef OPTIMISER_BALANCE_CLASS

        if (_searchType == SEARCH_TYPE_GLOBAL)
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Balancing Class(es)";

            //balanceClass(CLASS_BALANCE_FACTOR, true);
            balanceClass(bm);

#ifdef VERBOSE_LEVEL_1

            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Percentage of Images Belonging to Each Class After Balancing";

            for (int t = 0; t < _para.k; t++)
                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << _cDistr(t) * 100
                                           << "\% Percentage of Images Belonging to Class "
                                           << t;
#endif
        }

#endif

        _model.compareTwoHemispheres(false, true, AVERAGE_TWO_HEMISPHERE_THRES, _para.nThreadsPerProcess);
    }
#endif

    long memUsage = memoryCheckRM();
    printf("reconstruct memory check work done:%dG!\n", memUsage / MEGABYTE);

#ifdef GPU_VERSION
    NT_MASTER
    {
        free(modelF);
        free(modelT);
    }
#endif

    if (_searchType != SEARCH_TYPE_CTF)
        freePreCal(false);
    else
        freePreCal(true);

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space for Pre-calcuation in Reconstruction";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Freeing Space for Pre-calcuation in Reconstruction";

    freePreCalIdx();

    MPI_Barrier(MPI_COMM_WORLD);

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference(s) Reconstructed";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Reference(s) Reconstructed";
}

void Optimiser::solventFlatten(const bool mask)
{
    if ((_searchType == SEARCH_TYPE_GLOBAL) && mask)
    {
        MLOG(WARNING, "LOGGER_ROUND") << "Round " << _iter << ", " << "PERFORM REFERENCE MASKING DURING GLOBAL SEARCH. NOT RECOMMMENDED.";
    }

    IF_MASTER return;

    for (int t = 0; t < _para.k; t++)
    {
#ifdef OPTIMISER_SOLVENT_FLATTEN_LOW_PASS
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Low Pass Filter on Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Low Pass Filter on Reference " << t;

        lowPassFilter(_model.ref(t),
                      _model.ref(t),
                      (RFLOAT)_r  / _para.size,
                      (RFLOAT)EDGE_WIDTH_FT / _para.size,
                      _para.nThreadsPerProcess);
#endif

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Inverse Fourier Transforming Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Inverse Fourier Transforming Reference " << t;

        FFT fft;
        fft.bw(_model.ref(t), _para.nThreadsPerProcess);

#ifdef OPTIMISER_SOLVENT_FLATTEN_STAT_REMOVE_BG

        RFLOAT bgMean, bgStddev;

        bgMeanStddev(bgMean,
                     bgStddev,
                     _model.ref(t),
                     _para.size / 2,
                     _para.maskRadius / _para.pixelSize);

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Mean of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bgMean;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Mean of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bgMean;
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Standard Deviation of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bgStddev;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Standard Deviation of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bgStddev;

        //RFLOAT bgThres = bgMean + bgStddev * TSGSL_cdf_gaussian_Qinv(0.01, 1);
        RFLOAT bgThres = bgMean + bgStddev * TSGSL_cdf_gaussian_Qinv(1e-3, 1);

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Threshold for Removing Background of Reference "
                                   << t
                                   << ": "
                                   << bgThres;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Threshold for Removing Background of Reference "
                                   << t
                                   << ": "
                                   << bgThres;

        #pragma omp parallel for
        FOR_EACH_PIXEL_RL(_model.ref(t))
            if (_model.ref(t)(i) < bgThres)
                _model.ref(t)(i) = bgMean;

        #pragma omp parallel for
        FOR_EACH_PIXEL_RL(_model.ref(t))
                _model.ref(t)(i) -= bgMean;
#endif

#ifdef OPTIMISER_SOLVENT_FLATTEN_SUBTRACT_BG
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Subtracting Background from Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Subtracting Background from Reference " << t;

        RFLOAT bg = regionMean(_model.ref(t),
                               _para.maskRadius / _para.pixelSize + EDGE_WIDTH_RL,
                               _para.nThreadsPerProcess);

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Mean of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bg;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Mean of Background Noise of Reference "
                                   << t
                                   << ": "
                                   << bg;

        #pragma omp parallel for
        FOR_EACH_PIXEL_RL(_model.ref(t))
            (_model.ref(t))(i) -= bg;
#endif

#ifdef OPTIMISER_SOLVENT_FLATTEN_REMOVE_NEG
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Removing Negative Values from Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Removing Negative Values from Reference " << t;

        #pragma omp parallel for
        REMOVE_NEG(_model.ref(t));
#endif

        if (mask && !_mask.isEmptyRL())
        {
            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Performing Reference Masking";
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Performing Reference Masking";

            if (_para.mode == MODE_2D)
            {
                REPORT_ERROR("2D MODE DO NOT SUPPORTS PROVIDED MASK");

                abort();
            }
            else if (_para.mode == MODE_3D)
            {
#ifdef OPTIMISER_SOLVENT_FLATTEN_LOW_PASS_MASK

                fft.fw(_mask, _para.nThreadsPerProcess);
                // _mask.clearRL();

                Volume lowPassMask(_para.size, _para.size, _para.size, FT_SPACE);

                lowPassFilter(lowPassMask,
                              _mask,
                              (RFLOAT)_r / _para.size,
                              (RFLOAT)EDGE_WIDTH_FT / _para.size,
                              _para.nThreadsPerProcess);

                fft.bw(lowPassMask, _para.nThreadsPerProcess);

                fft.bw(_mask, _para.nThreadsPerProcess);

#ifdef OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
                softMask(_model.ref(t), _model.ref(t), lowPassMask, 0, _para.nThreadsPerProcess);
#else
                softMask(_model.ref(t), _model.ref(t), lowPassMask, _para.nThreadsPerProcess);
#endif

#else

#ifdef OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
                softMask(_model.ref(t), _model.ref(t), _mask, 0, _para.nThreadsPerProcess);
#else
                softMask(_model.ref(t), _model.ref(t), _mask, _para.nThreadsPerProcess);
#endif

#endif
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }
        }
        else
        {
            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Performing Solvent Flatten of Reference " << t;
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Performing Solvent Flatten of Reference " << t;

            if (_para.mode == MODE_2D)
            {
                Image ref(_para.size,
                          _para.size,
                          RL_SPACE);

                SLC_EXTRACT_RL(ref, _model.ref(t), 0);

#ifdef OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
                softMask(ref,
                         ref,
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL,
                         0,
                         _para.nThreadsPerProcess);
#else
                softMask(ref,
                         ref,
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL,
                         _para.nThreadsPerProcess);
#endif

                COPY_RL(_model.ref(t), ref);
            }
            else if (_para.mode == MODE_3D)
            {
#ifdef OPTIMISER_SOLVENT_FLATTEN_MASK_ZERO
                softMask(_model.ref(t),
                         _model.ref(t),
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL,
                         0,
                         _para.nThreadsPerProcess);
#else
                softMask(_model.ref(t),
                         _model.ref(t),
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL,
                         _para.nThreadsPerProcess);
#endif
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }
        }

        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Fourier Transforming Reference " << t;
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Fourier Transforming Reference " << t;

        fft.fw(_model.ref(t), _para.nThreadsPerProcess);
        // _model.ref(t).clearRL();
    }
}

void Optimiser::allocPreCalIdx(const RFLOAT rU,
                               const RFLOAT rL)
{
    IF_MASTER return;

    _iPxl = new int[_imgOri[0].sizeFT()];

    _iCol = new int[_imgOri[0].sizeFT()];

    _iRow = new int[_imgOri[0].sizeFT()];

    _iSig = new int[_imgOri[0].sizeFT()];

    _iColPad = new int[_imgOri[0].sizeFT()];

    _iRowPad = new int[_imgOri[0].sizeFT()];

    RFLOAT rU2 = TSGSL_pow_2(rU);
    RFLOAT rL2 = TSGSL_pow_2(rL);

    _nPxl = 0;

    IMAGE_FOR_PIXEL_R_FT(rU + 1)
    {
        if ((i == 0) && (j < 0)) continue;

        RFLOAT u = QUAD(i, j);

        if ((u < rU2) && (u >= rL2))
        {
            int v = AROUND(NORM(i, j));

            if ((v < rU) && (v >= rL))
            {
                _iPxl[_nPxl] = _imgOri[0].iFTHalf(i, j);

                _iCol[_nPxl] = i;

                _iRow[_nPxl] = j;

                _iSig[_nPxl] = v;

                _iColPad[_nPxl] = i * _para.pf;

                _iRowPad[_nPxl] = j * _para.pf;

                _nPxl++;
            }
        }
    }
}

void Optimiser::allocPreCal(const bool mask,
                            const bool pixelMajor,
                            const bool ctf)
{
    IF_MASTER return;

    // RFLOAT ratio = (M_PI / 4) / (2 * (RFLOAT)_nPxl / (_para.size * _para.size));

    RFLOAT ratio = (M_PI / 8 * _para.size * _para.size) / ((RFLOAT)_nPxl);

    // RFLOAT ratio = 1;

    std::cout << "Round " << _iter << ", ratio = " << ratio << std::endl;
    std::cout << "Round " << _nPxl << ", _nPxl = " << _nPxl << std::endl;
    std::cout << "Round " << _ID.size() << ", _ID.size() = " << _ID.size() << std::endl;

    // divide 4, as there are 4 containers in each stall

    _datPR.setUp(omp_get_max_threads(), _ID.size() * _nPxl, GSL_MIN(_ID.size() / 4, AROUND(ratio * _md.nStallDatPR)), sizeof(RFLOAT), _nPxl, _para.cacheDirectory);

    std::cout << "Round " << _iter << ", nStall of _datPR = " << _datPR.nStall() << std::endl;

    _datPI.setUp(omp_get_max_threads(), _ID.size() * _nPxl, GSL_MIN(_ID.size() / 4, AROUND(ratio * _md.nStallDatPI)), sizeof(RFLOAT), _nPxl, _para.cacheDirectory);

    std::cout << "Round " << _iter << ", nStall of _datPI = " << _datPI.nStall() << std::endl;

    _sigRcpP.setUp(omp_get_max_threads(), _ID.size() * _nPxl, GSL_MIN(_ID.size() / 4, AROUND(ratio * _md.nStallSigRcpP)), sizeof(RFLOAT), _nPxl, _para.cacheDirectory);

    std::cout << "Round " << _iter << ", nStall of _sigRcpP = " << _sigRcpP.nStall() << std::endl;

    // uvec si = shuffledIndex(_ID.size());
    uvec si = uvec::Zero(_ID.size());
    for (size_t l = 0; l < _ID.size(); l++)
        si(l) = l;

    MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
    MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> datPRDustman(&_datPR);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> datPIDustman(&_datPI);
    MemoryBazaarDustman<RFLOAT, BaseType, 4> sigRcpPDustman(&_sigRcpP);
    #pragma omp parallel for firstprivate(imgDustman, imgOriDustman, datPRDustman, datPIDustman, sigRcpPDustman)
    FOR_EACH_2D_IMAGE
    {
        _img.endLastVisit();
        _imgOri.endLastVisit();
        _datPR.endLastVisit();
        _datPI.endLastVisit();
        _sigRcpP.endLastVisit();

        size_t rl = si(l);

        for (size_t i = 0; i < _nPxl; i++)
        {
            Complex data = mask ? _img[rl].iGetFT(_iPxl[i]) : _imgOri[rl].iGetFT(_iPxl[i]);

            _datPR[pixelMajor
                 ? (i * _ID.size() + rl)
                 : (_nPxl * rl + i)] = data.dat[0];

            _datPI[pixelMajor
                 ? (i * _ID.size() + rl)
                 : (_nPxl * rl + i)] = data.dat[1];

            _sigRcpP[pixelMajor
                   ? (i * _ID.size() + rl)
                   : (_nPxl * rl + i)] = _sigRcp(_groupID[rl] - 1, _iSig[i]);
        }
    }

    if (!ctf)
    {
        _ctfP.setUp(omp_get_max_threads(), _ID.size() * _nPxl, GSL_MIN(_ID.size() / 4, AROUND(ratio * _md.nStallCtfP)), sizeof(RFLOAT), _nPxl, _para.cacheDirectory);

        std::cout << "Round " << _iter << ", nStall of _ctfP = " << _ctfP.nStall() << std::endl;

#ifdef OPTIMISER_CTF_ON_THE_FLY
        RFLOAT* poolCTF = (RFLOAT*)TSFFTW_malloc(_nPxl * omp_get_max_threads() * sizeof(RFLOAT));
#endif

        MemoryBazaarDustman<RFLOAT, BaseType, 4> ctfPDustman(&_ctfP);
        #pragma omp parallel for firstprivate(ctfPDustman)
        FOR_EACH_2D_IMAGE
        {
            size_t rl = si(l);

#ifdef OPTIMISER_CTF_ON_THE_FLY
            RFLOAT* ctf = poolCTF + _nPxl * omp_get_thread_num();

            CTF(ctf,
                _para.pixelSize,
                _ctfAttr[rl].voltage,
                _ctfAttr[rl].defocusU,
                _ctfAttr[rl].defocusV,
                _ctfAttr[rl].defocusTheta,
                _ctfAttr[rl].Cs,
                _ctfAttr[rl].amplitudeContrast,
                _ctfAttr[rl].phaseShift,
                _para.size,
                _para.size,
                _iCol,
                _iRow,
                _nPxl,
                1);

            for (size_t i = 0; i < _nPxl; i++)
            {
                _ctfP[pixelMajor
                    ? (i * _ID.size() + rl)
                    : (_nPxl * rl + i)] = ctf[i];
            }
#else
            for (size_t i = 0; i < _nPxl; i++)
            {
                _ctfP[pixelMajor
                    ? (i * _ID.size() + rl)
                    : (_nPxl * rl + i)] = REAL(_ctf[rl].iGetFT(_iPxl[i]));
            }
#endif
        }

#ifdef OPTIMISER_CTF_ON_THE_FLY
        TSFFTW_free(poolCTF);
#endif
    }
    else
    {
        _frequency = (RFLOAT*)TSFFTW_malloc(_nPxl * sizeof(RFLOAT));
        //_frequency = new RFLOAT[_nPxl];

        _defocusP = (RFLOAT*)TSFFTW_malloc(_ID.size() * _nPxl * sizeof(RFLOAT));
        //_defocusP = new RFLOAT[_ID.size() * _nPxl];

        _K1 = (RFLOAT*)TSFFTW_malloc(_ID.size() * sizeof(RFLOAT));
        //_K1 = new RFLOAT[_ID.size()];

        _K2 = (RFLOAT*)TSFFTW_malloc(_ID.size() * sizeof(RFLOAT));
        //_K2 = new RFLOAT[_ID.size()];

        for (size_t i = 0; i < _nPxl; i++)
            _frequency[i] = NORM(_iCol[i],
                                 _iRow[i])
                          / _para.size
                          / _para.pixelSize;

        #pragma omp parallel for
        FOR_EACH_2D_IMAGE
        {

            size_t rl = si(l);

            for (size_t i = 0; i < _nPxl; i++)
            {
                RFLOAT angle = atan2(_iRow[i],
                                     _iCol[i])
                             - _ctfAttr[rl].defocusTheta;

                RFLOAT defocus = -(_ctfAttr[rl].defocusU
                                 + _ctfAttr[rl].defocusV
                                 + (_ctfAttr[rl].defocusU - _ctfAttr[rl].defocusV)
                                 * cos(2 * angle))
                                 / 2;

                _defocusP[pixelMajor
                        ? (i * _ID.size() + rl)
                        : (_nPxl * rl + i)] = defocus;
            }

            RFLOAT lambda = 12.2643274 / sqrt(_ctfAttr[rl].voltage
                                            * (1 + _ctfAttr[rl].voltage * 0.978466e-6));

            _K1[rl] = M_PI * lambda;
            _K2[rl] = M_PI_2 * _ctfAttr[rl].Cs * TSGSL_pow_3(lambda);
        }
    }
}

void Optimiser::freePreCalIdx()
{
    IF_MASTER return;

    delete[] _iPxl;
    delete[] _iCol;
    delete[] _iRow;
    delete[] _iSig;

    delete[] _iColPad;
    delete[] _iRowPad;
}

void Optimiser::freePreCal(const bool ctf)
{
    IF_MASTER return;

    _datPR.cleanUp();

    //TSFFTW_free(_datPI);

    _datPI.cleanUp();

    //TSFFTW_free(_sigP);
    // _sigP.cleanUp();

    //TSFFTW_free(_sigRcpP);

    _sigRcpP.cleanUp();

    /***
    delete[] _datP;
    delete[] _ctfP;
    delete[] _sigRcpP;
    ***/

    if (!ctf)
    {
        // TSFFTW_free(_ctfP);
        _ctfP.cleanUp();
    }
    else
    {
        TSFFTW_free(_frequency);
        //delete[] _frequency;
        TSFFTW_free(_defocusP);
        //delete[] _defocusP;
        TSFFTW_free(_K1);
        TSFFTW_free(_K2);
        //delete[] _K1;
        //delete[] _K2;
    }
}

void Optimiser::writeDescInfo(FILE *file) const
{

    fprintf(file, "#0:VOLTAGE\tFLOAT\t18.9f\n");
    fprintf(file, "#1:DEFOCUS_U\tFLOAT\t18.9f\n");
    fprintf(file, "#2:DEFOCUS_V\tFLOAT\t18.9f\n");
    fprintf(file, "#3:DEFOCUS_THETA\tFLOAT\t18.9f\n");
    fprintf(file, "#4:CS\tFLOAT\t18.9f\n");
    fprintf(file, "#5:AMPLITUTDE_CONTRAST\tFLOAT\t18.9f\n");
    fprintf(file, "#6:PHASE_SHIFT\tFLOAT\t18.9f\n");
    fprintf(file, "#7:PARTICLE_PATH\tSTRING\n");
    fprintf(file, "#8:MICROGRAPH_PATH\tSTRING\n");
    fprintf(file, "#9:COORDINATE_X\tFLOAT\t18.9f\n");
    fprintf(file, "#10:COORDINATE_Y\tFLOAT\t18.9f\n");
    fprintf(file, "#11:GROUP_ID\tINT\t6d\n");
    fprintf(file, "#12:CLASS_ID\tINT\t6d\n");
    fprintf(file, "#13QUATERNION_0\tFLOAT\t18.9f\n");
    fprintf(file, "#14:QUATERNION_1\tFLOAT\t18.9f\n");
    fprintf(file, "#15:QUATERNION_2\tFLOAT\t18.9f\n");
    fprintf(file, "#16:QUATERNION_3\tFLOAT\t18.9f\n");
    fprintf(file, "#17:K1\tFLOAT\t18.9f\n");
    fprintf(file, "#18:K2\tFLOAT\t18.9f\n");
    fprintf(file, "#19:K3\tFLOAT\t18.9f\n");
    fprintf(file, "#20:TRANSLATION_X\tFLOAT\t18.9f\n");
    fprintf(file, "#21:TRANSLATION_Y\tFLOAT\t18.9f\n");
    fprintf(file, "#22:STD_TRANSLATION_X\tFLOAT\t18.9f\n");
    fprintf(file, "#23:STD_TRANSLATION_Y\tFLOAT\t18.9f\n");
    fprintf(file, "#24:DEFOCUS_FACTOR\tFLOAT\t18.9f\n");
    fprintf(file, "#25:STD_DEFOCUS_FACTOR\tFLOAT\t18.9f\n");
    fprintf(file, "#26:SCORE\tFLOAT\t18.9f\n");
    fprintf(file, "#27:CHANGENUM\tINT\t6d\n\n");

}

void Optimiser::saveDatabase(const bool finished,
                             const bool subtract,
                             const bool symmetrySubtract) const
{
    IF_MASTER return;

    char filename[FILE_NAME_LENGTH];

    //if (subtract)
    //    sprintf(filename, "%sMeta_Subtract.thu", _para.dstPrefix);

    if (finished)
        sprintf(filename, "%sMeta_Final.thu", _para.dstPrefix);
    else
        sprintf(filename, "%sMeta_Round_%03d.thu", _para.dstPrefix, _iter);

    bool flag;
    MPI_Status status;

    if (_commRank != 1)
        MPI_Recv(&flag, 1, MPI_C_BOOL, _commRank - 1, 0, MPI_COMM_WORLD, &status);

    FILE* file = (_commRank == 1)
               ? fopen(filename, "w")
               : fopen(filename, "a");

    dvec4 quat;
    dvec2 tran;
    double df;

    double k1, k2, k3, s0, s1, s;

    char subtractPath[FILE_WORD_LENGTH];

    dmat33 rotB; // rot for base left closet
    dmat33 rotC; // rot for every left closet

    writeDescInfo(file);

    FOR_EACH_2D_IMAGE
    {
        _par[l]->rank1st(quat, tran, df);

        _par[l]->vari(k1, k2, k3, s0, s1, s);

        rotate3D(rotB, quat);

        if (subtract)
        {
            for (int i = -1; i < (symmetrySubtract ? _sym.nSymmetryElement() : 0); i++)
            {
                if (i == -1)
                {
                    rotC = rotB;
                }
                else
                {
                    dmat33 L, R;

                    _sym.get(L, R, i);
                    rotC = R.transpose() * rotB;
                }

                quaternion(quat, rotC);

                snprintf(subtractPath,
                         sizeof(subtractPath),
                         "%012ld@Subtract_Rank_%06d.mrcs",
                         l + _ID.size() * (i + 1) + 1,
                         _commRank);

                fprintf(file,
                        "%18.9lf %18.9lf %18.9lf %18.9lf %18.9lf %18.9lf %18.9lf \
                         %s %s %18.9lf %18.9lf \
                         %6d %6lu \
                         %18.9lf %18.9lf %18.9lf %18.9lf \
                         %18.9lf %18.9lf %18.9lf \
                         %18.9lf %18.9lf %18.9lf %18.9lf \
                         %18.9lf %18.9lf \
                         %18.9lf %6d\n",
                         _ctfAttr[l].voltage,
                         _ctfAttr[l].defocusU,
                         _ctfAttr[l].defocusV,
                         _ctfAttr[l].defocusTheta,
                         _ctfAttr[l].Cs,
                         _ctfAttr[l].amplitudeContrast,
                         _ctfAttr[l].phaseShift,
                         subtractPath,
                         _db.micrographPath(_ID[l]).c_str(),
                         _db.coordX(_ID[l]),
                         _db.coordY(_ID[l]),
                         _groupID[l],
                         _iRef[l],
                         quat(0),
                         quat(1),
                         quat(2),
                         quat(3),
                         k1,
                         k2,
                         k3,
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                         tran(0) - _offset[l](0),
                         tran(1) - _offset[l](1),
#else
                         tran(0),
                         tran(1),
#endif
                         s0,
                         s1,
                         df,
                         s,
                         _par[l]->compressR(),
                         _nChange[l]);
            }

        }
        else
        {
            fprintf(file,
                    "%18.9lf %18.9lf %18.9lf %18.9lf %18.9lf %18.9lf %18.9lf \
                     %s %s %18.9lf %18.9lf \
                     %6d %6lu \
                     %18.9lf %18.9lf %18.9lf %18.9lf \
                     %18.9lf %18.9lf %18.9lf \
                     %18.9lf %18.9lf %18.9lf %18.9lf \
                     %18.9lf %18.9lf \
                     %18.9lf %6d\n",
                     _ctfAttr[l].voltage,
                     _ctfAttr[l].defocusU,
                     _ctfAttr[l].defocusV,
                     _ctfAttr[l].defocusTheta,
                     _ctfAttr[l].Cs,
                     _ctfAttr[l].amplitudeContrast,
                     _ctfAttr[l].phaseShift,
                     _db.path(_ID[l]).c_str(),
                     _db.micrographPath(_ID[l]).c_str(),
                     _db.coordX(_ID[l]),
                     _db.coordY(_ID[l]),
                     _groupID[l],
                     _iRef[l],
                     quat(0),
                     quat(1),
                     quat(2),
                     quat(3),
                     k1,
                     k2,
                     k3,
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                     tran(0) - _offset[l](0),
                     tran(1) - _offset[l](1),
#else
                     tran(0),
                     tran(1),
#endif
                     s0,
                     s1,
                     df,
                     s,
                     _par[l]->compressR(),
                     _nChange[l]);
        }

    }

    fclose(file);

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving .thu File To Path: " << filename;
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving .thu File To Path: " << filename;

    if (_commRank != _commSize - 1)
        MPI_Send(&flag, 1, MPI_C_BOOL, _commRank + 1, 0, MPI_COMM_WORLD);
}

void Optimiser::saveSubtract(const bool symmetrySubtract,
                             const unsigned int reboxSize)
{
    IF_MASTER return;

    if (reboxSize > _para.size)
    {
        ALOG(FATAL, "LOGGER_SYS") << "Round " << _iter << ", " << "RE-BOXING SIZE CAN NOT BE LARGER THAN THE ORIGINAL SIZE";
        BLOG(FATAL, "LOGGER_SYS") << "Round " << _iter << ", " << "RE-BOXING SIZE CAN NOT BE LARGER THAN THE ORIGINAL SIZE";

        abort();
    }

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Masked Region Reference Subtracted Images";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Masked Region Reference Subtracted Images";

    char filename[FILE_NAME_LENGTH];

    sprintf(filename, "%sSubtract_Rank_%06d.mrcs", _para.dstPrefix, _commRank);

    ImageFile imf;

    imf.openStack(filename, reboxSize, _ID.size() * (symmetrySubtract ? (1 + _sym.nSymmetryElement()) : 1), _para.pixelSize);

    Image result(_para.size, _para.size, FT_SPACE);
    Image diff(_para.size, _para.size, FT_SPACE);

    Image box(reboxSize, reboxSize, RL_SPACE);

    dmat33 rotB; // rot for base left closet
    dmat33 rotC; // rot for every left closet
    dvec2 tran;
    double d;

    FOR_EACH_2D_IMAGE
    {
        if (_para.mode == MODE_2D)
        {
            ALOG(FATAL, "LOGGER_ROUND") << "Round " << _iter << ", " << "SAVE SUBTRACT DOES NOT SUPPORT 2D MODE";
            BLOG(FATAL, "LOGGER_ROUND") << "Round " << _iter << ", " << "SAVE SUBTRACT DOES NOT SUPPORT 2D MODE";

            abort();
        }

        _par[l]->rank1st(rotB, tran, d);

#ifdef OPTIMISER_CTF_ON_THE_FLY

        Image ctf(_para.size, _para.size, FT_SPACE);
        CTF(ctf,
            _para.pixelSize,
            _ctfAttr[l].voltage,
            _ctfAttr[l].defocusU * d,
            _ctfAttr[l].defocusV * d,
            _ctfAttr[l].defocusTheta,
            _ctfAttr[l].Cs,
            _ctfAttr[l].amplitudeContrast,
            _ctfAttr[l].phaseShift,
            _para.nThreadsPerProcess);

#endif

        for (int i = -1; i < (symmetrySubtract ? _sym.nSymmetryElement() : 0); i++)
        {
            #pragma omp parallel for
            SET_0_FT(result);

            #pragma omp parallel for
            SET_0_FT(diff);

            if (i == -1)
            {
                rotC = rotB;
            }
            else
            {
                dmat33 L, R;

                _sym.get(L, R, i);
                rotC = R.transpose() * rotB;
            }

            _model.proj(_iRef[l]).project(result, rotC, tran - _offset[l], _para.nThreadsPerProcess);

            MemoryBazaarDustman<Image, DerivedType, 4> imgOriDustman(&_imgOri);
            #pragma omp parallel for firstprivate(imgOriDustman)
            FOR_EACH_PIXEL_FT(diff)
            {
#ifdef OPTIMISER_CTF_ON_THE_FLY
                diff[i] = _imgOri[l][i] - result[i] * REAL(ctf[i]);
#else
                diff[i] = _imgOri[l][i] - result[i] * REAL(_ctf[l][i]);
#endif
            }

            dvec3 regionTrans = rotC.transpose() * dvec3(_regionCentre(0),
                                                         _regionCentre(1),
                                                         _regionCentre(2));

            translate(diff,
                      diff,
                      -tran(0) + _offset[l](0) - regionTrans(0),
                      -tran(1) + _offset[l](1) - regionTrans(1),
                      _para.nThreadsPerProcess);

            if (i == -1)
            {
                _par[l]->setT(_par[l]->t().rowwise() - (tran - _offset[l]).transpose());
                _par[l]->setTopT(_par[l]->topT() - tran + _offset[l]);
                _par[l]->setTopTPrev(_par[l]->topTPrev() - tran + _offset[l]);
            }

            _fftImg.bwExecutePlan(diff, _para.nThreadsPerProcess);

            IMG_BOX_RL(box, diff, _para.nThreadsPerProcess);

            imf.writeStack(box, l + _ID.size() * (i + 1));

            _fftImg.fwExecutePlan(diff);
        }
    }

    imf.closeStack();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Masked Region Reference Subtracted Images Saved";
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Masked Region Reference Subtracted Images Saved";
#endif

    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Masked Region Reference Subtracted Image File To Path: " << filename;
    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Masked Region Reference Subtracted Image File To Path: " << filename;
}

void Optimiser::saveBestProjections()
{
    IF_MASTER return;

    FFT fft;

    Image result(_para.size, _para.size, FT_SPACE);
    Image diff(_para.size, _para.size, FT_SPACE);
    char filename[FILE_NAME_LENGTH];

    dmat22 rot2D;
    dmat33 rot3D;
    dvec2 tran;
    double d;

    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            #pragma omp parallel for
            SET_0_FT(result);

            #pragma omp parallel for
            SET_0_FT(diff);

            if (_para.mode == MODE_2D)
            {
                _par[l]->rank1st(rot2D, tran, d);

                _model.proj(_iRef[l]).project(result, rot2D, tran, _para.nThreadsPerProcess);
            }
            else if (_para.mode == MODE_3D)
            {
                _par[l]->rank1st(rot3D, tran, d);

                _model.proj(_iRef[l]).project(result, rot3D, tran, _para.nThreadsPerProcess);
            }
            else
                REPORT_ERROR("INEXISTENT MODE");

            sprintf(filename, "%sResult_%04d_Round_%03d.bmp", _para.dstPrefix, _ID[l], _iter);

            fft.bw(result, _para.nThreadsPerProcess);
            result.saveRLToBMP(filename);
            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Result Round BMP File To Path: " << filename;
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Result Round BMP File To Path: " << filename;
            fft.fw(result, _para.nThreadsPerProcess);

#ifdef OPTIMISER_CTF_ON_THE_FLY
            // TODO
#else
            MemoryBazaarDustman<Image, DerivedType, 4> imgDustman(&_img);
            #pragma omp parallel for firstprivate(imgDustman)
            FOR_EACH_PIXEL_FT(diff)
                diff[i] = _img[l][i] - result[i] * REAL(_ctf[l][i]);
#endif

            sprintf(filename, "%sDiff_%04d_Round_%03d.bmp", _para.dstPrefix, _ID[l], _iter);
            fft.bw(diff, _para.nThreadsPerProcess);
            diff.saveRLToBMP(filename);
            ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Diff Round BMP File To Path: " << filename;
            BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Diff Round BMP File To Path: " << filename;
            fft.fw(diff, _para.nThreadsPerProcess);
        }
    }

}

void Optimiser::saveImages()
{
    IF_MASTER return;

    char filename[FILE_NAME_LENGTH];
    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            sprintf(filename, "Fourier_Image_%04d.bmp", _ID[l]);

            _imgOri[l].saveFTToBMP(filename, 0.01);

            sprintf(filename, "Image_%04d.bmp", _ID[l]);

            _fftImg.bwExecutePlan(_imgOri[l], _para.nThreadsPerProcess);
            _imgOri[l].saveRLToBMP(filename);
            _fftImg.fwExecutePlan(_imgOri[l]);
        }
    }
}

void Optimiser::saveCTFs()
{
    IF_MASTER return;

    char filename[FILE_NAME_LENGTH];
    FOR_EACH_2D_IMAGE
    {
        if (_ID[l] < N_SAVE_IMG)
        {
            sprintf(filename, "CTF_%04d.bmp", _ID[l]);

#ifdef OPTIMISER_CTF_ON_THE_FLY
            // TODO
#else
            _ctf[l].saveFTToBMP(filename, 0.01);
#endif
        }
    }
}

void Optimiser::saveMapHalf(const bool finished)
{
    if ((_commRank != HEMI_A_LEAD) &&
        (_commRank != HEMI_B_LEAD))
        return;

    FFT fft;

    ImageFile imf;
    char filename[FILE_NAME_LENGTH];

    for (int t = 0; t < _para.k; t++)
    {
        if (_para.mode == MODE_2D)
        {
            if (_commRank == HEMI_A_LEAD)
            {
                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Reference " << t;

                Image ref(_para.size,
                          _para.size,
                          FT_SPACE);

                SLC_EXTRACT_FT(ref, _model.ref(t), 0);

                /***
                if (finished)
                    sprintf(filename, "%sFT_Reference_%03d_A_Final.bmp", _para.dstPrefix, t);
                else
                    sprintf(filename, "%sFT_Reference_%03d_A_Round_%03d.bmp", _para.dstPrefix, t, _iter);

                ref.saveFTToBMP(filename, 0.001);
                ***/

                fft.bw(ref, _para.nThreadsPerProcess);

                softMask(ref,
                         ref,
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL,
                         0,
                         _para.nThreadsPerProcess);

                if (finished)
                    sprintf(filename, "%sReference_%03d_A_Final.bmp", _para.dstPrefix, t);
                else
                    sprintf(filename, "%sReference_%03d_A_Round_%03d.bmp", _para.dstPrefix, t, _iter);

                ref.saveRLToBMP(filename);
                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Reference File To Path:  " << filename;
            }
            else if (_commRank == HEMI_B_LEAD)
            {
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Reference " << t;

                Image ref(_para.size,
                          _para.size,
                          FT_SPACE);

                SLC_EXTRACT_FT(ref, _model.ref(t), 0);

                /***
                if (finished)
                    sprintf(filename, "%sFT_Reference_%03d_B_Final.bmp", _para.dstPrefix, t);
                else
                    sprintf(filename, "%sFT_Reference_%03d_B_Round_%03d.bmp", _para.dstPrefix, t, _iter);

                ref.saveFTToBMP(filename, 0.001);
                ***/

                fft.bw(ref, _para.nThreadsPerProcess);

                softMask(ref,
                         ref,
                         _para.maskRadius / _para.pixelSize,
                         EDGE_WIDTH_RL,
                         0,
                         _para.nThreadsPerProcess);

                if (finished)
                    sprintf(filename, "%sReference_%03d_B_Final.bmp", _para.dstPrefix, t);
                else
                    sprintf(filename, "%sReference_%03d_B_Round_%03d.bmp", _para.dstPrefix, t, _iter);

                ref.saveRLToBMP(filename);
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Reference File To Path:  " << filename;
            }
        }
        else if (_para.mode == MODE_3D)
        {
            Volume lowPass(_para.size,
                           _para.size,
                           _para.size,
                           FT_SPACE);

            if (finished)
            {
                lowPass = _model.ref(t).copyVolume();

                fft.bw(lowPass, _para.nThreadsPerProcess);
            }
            else
            {
#ifdef OPTIMISER_SAVE_LOW_PASS_REFERENCE
                lowPassFilter(lowPass,
                              _model.ref(t),
                              (RFLOAT)_resReport / _para.size,
                              (RFLOAT)EDGE_WIDTH_FT / _para.size,
                              _para.nThreadsPerProcess);
#else
                lowPass = _model.ref(t).copyVolume();
#endif

                fft.bw(lowPass, _para.nThreadsPerProcess);
            }

            if (_commRank == HEMI_A_LEAD)
            {
                ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Reference " << t;

                if (finished)
                {
                    sprintf(filename, "%sReference_%03d_A_Final.mrc", _para.dstPrefix, t);
                }
                else
                {
                    sprintf(filename, "%sReference_%03d_A_Round_%03d.mrc", _para.dstPrefix, t, _iter);
                }

                imf.readMetaData(lowPass);
                imf.writeVolume(filename, lowPass, _para.pixelSize);

                if(finished)
                {
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << " Saving Final Reference File To Path:" << filename;
                }
                else
                {
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << " Saving Round Reference File To Path:" << filename;
                }
            }
            else if (_commRank == HEMI_B_LEAD)
            {
                BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Reference " << t;

                if (finished)
                {
                    sprintf(filename, "%sReference_%03d_B_Final.mrc", _para.dstPrefix, t);
                }
                else
                {
                    sprintf(filename, "%sReference_%03d_B_Round_%03d.mrc", _para.dstPrefix, t, _iter);
                }

                imf.readMetaData(lowPass);
                imf.writeVolume(filename, lowPass, _para.pixelSize);
                if(finished)
                {
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << " Saving Final Reference File To Path:" << filename;
                }
                else
                {
                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << " Saving Round Reference File To Path:" << filename;
                }
}
        }
    }
}

void Optimiser::saveMapJoin(const bool finished)
{
    FFT fft;

    ImageFile imf;
    char filename[FILE_NAME_LENGTH];

    if (_para.mode == MODE_2D)
    {
        IF_MASTER
        {
            MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Stack of Reference(s)";

            if (finished)
                sprintf(filename, "%sReference_Final.mrcs", _para.dstPrefix);
            else
                sprintf(filename, "%sReference_Round_%03d.mrcs", _para.dstPrefix, _iter);

            imf.openStack(filename, _para.size, _para.k, _para.pixelSize);
        }

        for (int l = 0; l < _para.k; l++)
        {
            IF_MASTER
            {
                Image ref(_para.size, _para.size, FT_SPACE);

                Image A(_para.size, _para.size, FT_SPACE);
                Image B(_para.size, _para.size, FT_SPACE);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Receiving Reference " << l << " from Hemisphere A";

                MPI_Recv_Large(&A[0],
                               2 * A.sizeFT(),
                               TS_MPI_DOUBLE,
                               HEMI_A_LEAD,
                               l,
                               MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Receiving Reference " << l << " from Hemisphere B";

                MPI_Recv_Large(&B[0],
                               2 * B.sizeFT(),
                               TS_MPI_DOUBLE,
                               HEMI_B_LEAD,
                               l,
                               MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Averaging Two Hemispheres";
                FOR_EACH_PIXEL_FT(ref)
                    ref[i] = (A[i] + B[i]) / 2;

                fft.bw(ref, _para.nThreadsPerProcess);

                imf.writeStack(ref, l);
                if(finished)
                {
                    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Stack of Final Reference File To Path: " << filename;
                }
                else
                {
                    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Stack of Round Reference File To Path: " << filename;
                }
            }
            else
            {
                if ((_commRank == HEMI_A_LEAD) ||
                    (_commRank == HEMI_B_LEAD))
                {
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Sending Reference "
                                                 << l
                                                 << " from Hemisphere A";

                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Sending Reference "
                                                 << l
                                                 << " from Hemisphere B";

                    MPI_Ssend_Large(&_model.ref(l)[0],
                                    2 * _model.ref(l).sizeFT(),
                                    TS_MPI_DOUBLE,
                                    MASTER_ID,
                                    l,
                                    MPI_COMM_WORLD);
                }
            }
        }

        IF_MASTER imf.closeStack();
    }
    else if (_para.mode == MODE_3D)
    {
        for (int l = 0; l < _para.k; l++)
        {
            IF_MASTER
            {
                Volume ref(_para.size, _para.size, _para.size, FT_SPACE);

                Volume A(_para.size, _para.size, _para.size, FT_SPACE);
                Volume B(_para.size, _para.size, _para.size, FT_SPACE);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Receiving Reference " << l << " from Hemisphere A";

                MPI_Recv_Large(&A[0],
                               2 * A.sizeFT(),
                               TS_MPI_DOUBLE,
                               HEMI_A_LEAD,
                               l,
                               MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Receiving Reference " << l << " from Hemisphere B";

                MPI_Recv_Large(&B[0],
                               2 * B.sizeFT(),
                               TS_MPI_DOUBLE,
                               HEMI_B_LEAD,
                               l,
                               MPI_COMM_WORLD);

                MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Averaging Two Hemispheres";
                FOR_EACH_PIXEL_FT(ref)
                    ref[i] = (A[i] + B[i]) / 2;

                fft.bw(ref, _para.nThreadsPerProcess);

                if (finished)
                    sprintf(filename, "%sReference_%03d_Final.mrc", _para.dstPrefix, l);
                else
                    sprintf(filename, "%sReference_%03d_Round_%03d.mrc", _para.dstPrefix, l, _iter);

                imf.readMetaData(ref);
                imf.writeVolume(filename, ref, _para.pixelSize);
                if(finished)
                {
                    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Final Reference File To Path :" << filename;
                }
                else
                {
                    MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Round Reference File To Path :" << filename;
                }
            }
            else
            {
                if ((_commRank == HEMI_A_LEAD) ||
                    (_commRank == HEMI_B_LEAD))
                {
                    ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Sending Reference "
                                                 << l
                                                 << " from Hemisphere A";

                    BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Sending Reference "
                                                 << l
                                                 << " from Hemisphere B";

                    MPI_Ssend_Large(&_model.ref(l)[0],
                                    2 * _model.ref(l).sizeFT(),
                                    TS_MPI_DOUBLE,
                                    MASTER_ID,
                                    l,
                                    MPI_COMM_WORLD);
                }
            }
        }
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
}

void Optimiser::saveFSC(const bool finished) const
{
    NT_MASTER return;

    char filename[FILE_NAME_LENGTH];

    if (finished)
        sprintf(filename, "%sFSC_Final.txt", _para.dstPrefix);
    else
        sprintf(filename, "%sFSC_Round_%03d.txt", _para.dstPrefix, _iter);

    FILE* file = fopen(filename, "w");

    for (int i = 1; i < _model.rU(); i++)
    {
        fprintf(file,
                "%05d   %10.6lf",
                i,
                1.0 / resP2A(i, _para.size, _para.pixelSize));
        for (int t = 0; t < _para.k; t++)
            fprintf(file,
                    "   %10.6lf",
                    (_model.fsc(t))(i));
        fprintf(file, "\n");
    }

    fclose(file);


    if(finished)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Final FSC File To Path: " << filename;
    }
    else
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Round FSC File To Path: " << filename;
    }

}

void Optimiser::saveClassInfo(const bool finished) const
{
    NT_MASTER return;

    char filename[FILE_NAME_LENGTH];

    if (finished)
        sprintf(filename, "%sClass_Info_Final.txt", _para.dstPrefix);
    else
        sprintf(filename, "%sClass_Info_Round_%03d.txt", _para.dstPrefix, _iter);

    FILE* file = fopen(filename, "w");

    for (int t = 0; t < _para.k; t++)
    {
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                t,
                1.0 / _model.resolutionA(t, _para.thresReportFSC),
                _cDistr(t));
    }

    fclose(file);
    if(finished)
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Class Information Final File To Path: " << filename;
    }
    else
    {
        MLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Class Information Round File To Path: " << filename;
    }
}

void Optimiser::saveSig() const
{
    if ((_commRank != HEMI_A_LEAD) &&
        (_commRank != HEMI_B_LEAD))
        return;

    char filename[FILE_NAME_LENGTH];

    if (_commRank == HEMI_A_LEAD)
        sprintf(filename, "%sSig_A_Round_%03d.txt", _para.dstPrefix, _iter);
    else
        sprintf(filename, "%sSig_B_Round_%03d.txt", _para.dstPrefix, _iter);

    FILE* file = fopen(filename, "w");

    for (int i = 0; i < maxR(); i++)
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                i,
                1.0 / resP2A(i, _para.size, _para.pixelSize),
                _sig(_groupID[0] - 1, i));

    fclose(file);
    if (_commRank == HEMI_A_LEAD)
    {
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Sig A Round File To Path: " << filename;
    }
    else
    {
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Sig B Round File To Path: " << filename;
    }

    if (_commRank == HEMI_A_LEAD)
        sprintf(filename, "%sSVD_A_Round_%03d.txt", _para.dstPrefix, _iter);
    else
        sprintf(filename, "%sSVD_B_Round_%03d.txt", _para.dstPrefix, _iter);

    file = fopen(filename, "w");

    for (int i = 0; i < maxR(); i++)
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                i,
                1.0 / resP2A(i, _para.size, _para.pixelSize),
                _svd(_groupID[0] - 1, i));

    fclose(file);

    if (_commRank == HEMI_A_LEAD)
    {
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving SVD A Round File To Path: " << filename;
    }
    else
    {
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving SVD B Round File To Path: " << filename;
    }
}

void Optimiser::saveTau() const
{
    if ((_commRank != HEMI_A_LEAD) &&
        (_commRank != HEMI_B_LEAD))
        return;

    char filename[FILE_NAME_LENGTH];

    if (_commRank == HEMI_A_LEAD)
        sprintf(filename, "%sTau_A_Round_%03d.txt", _para.dstPrefix, _iter);
    else if (_commRank == HEMI_B_LEAD)
        sprintf(filename, "%sTau_B_Round_%03d.txt", _para.dstPrefix, _iter);

    FILE* file = fopen(filename, "w");

    for (int i = 1; i < maxR() * _para.pf - 1; i++)
        fprintf(file,
                "%05d   %10.6lf   %10.6lf\n",
                i,
                1.0 / resP2A(i, _para.size * _para.pf, _para.pixelSize),
                _model.tau(0)(i));

    fclose(file);

    if (_commRank == HEMI_A_LEAD)
    {
        ALOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Tau A Round File To Path: " << filename;
    }
    else
    {
        BLOG(INFO, "LOGGER_ROUND") << "Round " << _iter << ", " << "Saving Tau B Round File To Path: " << filename;
    }}


void scaleDataVSPrior(vec& sXA,
                      vec& sAA,
                      const Image& dat,
                      const Image& pri,
                      const Image& ctf,
                      const RFLOAT rU,
                      const RFLOAT rL)
{
    RFLOAT rU2 = TSGSL_pow_2(rU);
    RFLOAT rL2 = TSGSL_pow_2(rL);

    for (int i = 0; i < rU; i++)
    {
        sXA(i) = 0;
        sAA(i) = 0;
    }

    // #pragma omp parallel for reduction(+:sXA) reduction(+:sAA) schedule(dynamic)
    #pragma omp parallel for schedule(dynamic)
    IMAGE_FOR_PIXEL_R_FT(CEIL(rU) + 1)
    {
        RFLOAT u = QUAD(i, j);

        if ((u < rU2) && (u >= rL2))
        {
            int v = AROUND(NORM(i, j));
            if ((v < rU) &&
                (v >= rL))
            {
                int index = dat.iFTHalf(i, j);

                RFLOAT XA = REAL(dat.iGetFT(index)
                               * pri.iGetFT(index)
                               * REAL(ctf.iGetFT(index)));

                RFLOAT AA = REAL(pri.iGetFT(index)
                               * pri.iGetFT(index)
                               * TSGSL_pow_2(REAL(ctf.iGetFT(index))));

                #pragma omp atomic
                sXA(v) += XA;

                #pragma omp atomic
                sAA(v) += AA;
            }
        }
    }
}
