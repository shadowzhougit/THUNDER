/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include <MLOptimiser.h>

MLOptimiser::MLOptimiser() {}

MLOptimiser::~MLOptimiser()
{
    clear();
}

void MLOptimiser::init()
{
    // set MPI environment of _model
    _model.setMPIEnv(_commSize, _commRank, _hemi);

    // append initial references into _model

    // apply low pass filter on initial references

    // read in images from hard disk

    // genereate corresponding CTF
    
    // estimate initial sigma values
    initSigma();

    //
    initParticles();
}

void MLOptimiser::initParticles()
{
    for (int i = 0; i < _model.K(); i++)
    {
        _par.push_back(Particle());
        _par.end()->init(_para.particleNum,
                         _para.maxX,
                         _para.maxY,
                         &_sym);
    }
}

double MLOptimiser::norm(const Image& imgA,
                         const Image& imgB,
                         const Image& ctf,
                         const vector<vec>& sig,
                         const int r) const
{
    // Todo
}

void MLOptimiser::expectation()
{
    Image image(_img[0].nColFT(),
                _img[0].nRowFT(),
                FT_SPACE);

    for (int i = 0; i < _img.size(); i++)
    {
        stringstream ss;
        ss << "Particle" << i << ".par";
        FILE* file = fopen(ss.str().c_str(), "w");
        ss.str("");

        for (int j = 0; j < _par[i].N(); j++)
        {
            Coordinate5D cor;
            _par[i].coord(cor, j);
            _model.proj(0).project(image, cor);

            double w = norm(image,
                            _img[i],
                            _ctf[i],
                            _sig,
                            _r);

            _par[i].mulW(w, j);
        }
        _par[i].normW();

        // Save particles
        vec4 q;
        vec2 t;
        for (int k = 0; k < _par[i].N(); k++)
        {
            _par[i].quaternion(q, k);
            _par[i].t(t, k);
            fprintf(file, "%f %f %f %f, %f %f, %f\n",
                          q(0),q(1),q(2),q(3),
                          t(0), t(1),
                          _par[i].w(k));
        }
        fclose(file);
    }
}

void MLOptimiser::maximization()
{
    /* generate sigma for the next iteration */
    allReduceSigma();

    /* reconstruct references */
    reconstructRef();
}

void MLOptimiser::run()
{
    init();

    for (int i = 0; i < _para.iterMax; i++)
    {
        expectation();

        maximization();

        /* calculate FSC */
        _model.BcastFSC();

        /* record current resolution */
        _res = _model.resolutionP();

        /* update the radius of frequency for computing */
        _model.updateR();
        _r = _model.r() / 2;
    }
}

void MLOptimiser::clear()
{
    _img.clear();
    _par.clear();
    _ctf.clear();
}

void MLOptimiser::allReduceN()
{
    if (_commRank == MASTER_ID) return;

    _N = _exp.nParticle();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, &_N, 1, MPI_INT, MPI_SUM, _hemi);

    MPI_Barrier(MPI_COMM_WORLD);
}

int MLOptimiser::size() const
{
    return _img[0].nColRL();
}

int MLOptimiser::maxR() const
{
    return size() / 2 - 1;
}

void MLOptimiser::initSigma()
{
    if (_commRank == MASTER_ID) return;

    /* calculate average image */

    Image avg = _img[0];
    for (int i = 1; i < _img.size(); i++)
        ADD_FT(avg, _img[i]);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  &avg[0],
                  avg.sizeFT(),
                  MPI_DOUBLE_COMPLEX,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(MPI_COMM_WORLD);

    SCALE_RL(avg, 1.0 / _N);

    /* calculate average power spectrum */

    double* avgPs = new double[maxR()];
    memset(avgPs, 0, maxR() * sizeof(double));

    double* ps = new double[maxR()];
    vec vps(ps, maxR(), false, true);
    for (int i = 0; i < _img.size(); i++)
    {
        powerSpectrum(vps, _img[i], maxR());
        cblas_daxpy(maxR(), 1, ps, 1, avgPs, 1);
    }
    delete[] ps;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE,
                  avgPs,
                  maxR(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(MPI_COMM_WORLD);

    vec vAvgPs(avgPs, maxR());
    delete[] avgPs;
    vAvgPs /= _N;

    vec vPsAvg(maxR());
    powerSpectrum(vPsAvg, avg, maxR());
    
    /* vAvgPs -> average power spectrum
     * vPsAvg -> power spectrum of average image */
    _sig[0] = (vAvgPs - vPsAvg) / 2;
    for (int i = 1; i < _sig.size(); i++)
        _sig[i] = _sig[0];
}

void MLOptimiser::allReduceSigma()
{
    // TODO
}

void MLOptimiser::reconstructRef()
{
    // TODO
}
