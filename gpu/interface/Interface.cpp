#include "Interface.h"

#include "cuthunder.h"

void readGPUPARA(char* gpuList,
                 std::vector<int>& iGPU,
                 int& nGPU)
{
    cuthunder::readGPUPARA(gpuList,
                           iGPU,
                           nGPU);
}

void gpuCheck(std::vector<void*>& stream,
              std::vector<int>& iGPU,
              int& nGPU)
{
    cuthunder::gpuCheck(stream,
                        iGPU,
                        nGPU);
}

void gpuEnvDestory(std::vector<void*>& stream,
                   std::vector<int>& iGPU,
                   int nGPU)
{
    cuthunder::gpuEnvDestory(stream,
                             iGPU,
                             nGPU);
}

void ExpectPreidx(int gpuIdx,
                  int** deviCol,
                  int** deviRow,
                  int* iCol,
                  int* iRow,
                  int npxl)
{
    cuthunder::expectPreidx(gpuIdx,
                            deviCol,
                            deviRow,
                            iCol,
                            iRow,
                            npxl);
}

void ExpectPrefre(int gpuIdx,
                  RFLOAT** devfreQ,
                  RFLOAT* freQ,
                  int npxl)
{
    cuthunder::expectPrefre(gpuIdx,
                            devfreQ,
                            freQ,
                            npxl);

}

void ExpectLocalIn(int gpuIdx,
                   RFLOAT** devdatPR,
                   RFLOAT** devdatPI,
                   RFLOAT** devctfP,
                   RFLOAT** devdefO,
                   RFLOAT** devsigP,
                   int nPxl,
                   int cpyNumL,
                   int searchType)
{
    cuthunder::expectLocalIn(gpuIdx,
                             devdatPR,
                             devdatPI,
                             devctfP,
                             devdefO,
                             devsigP,
                             nPxl,
                             cpyNumL,
                             searchType);

}

void ExpectLocalV2D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int dimSize)
{
    cuthunder::expectLocalV2D(gpuIdx,
                              mgr,
                              reinterpret_cast<cuthunder::Complex*>(volume),
                              dimSize);

}

void ExpectLocalV3D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int vdim)
{
    cuthunder::expectLocalV3D(gpuIdx,
                              mgr,
                              reinterpret_cast<cuthunder::Complex*>(volume),
                              vdim);

}

void ExpectLocalP(int gpuIdx,
                  RFLOAT* devdatPR,
                  RFLOAT* devdatPI,
                  RFLOAT* devctfP,
                  RFLOAT* devsigP,
                  RFLOAT* devdefO,
                  RFLOAT* datPR,
                  RFLOAT* datPI,
                  RFLOAT* ctfP,
                  RFLOAT* sigRcpP,
                  RFLOAT* defO,
                  int threadId,
                  //int imgId,
                  int npxl,
                  int cSearch)
{
    cuthunder::expectLocalP(gpuIdx,
                            devdatPR,
                            devdatPI,
                            devctfP,
                            devsigP,
                            devdefO,
                            datPR,
                            datPI,
                            ctfP,
                            sigRcpP,
                            defO,
                            threadId,
                            //imgId,
                            npxl,
                            cSearch);

}

void ExpectLocalHostA(int gpuIdx,
                      RFLOAT** wC,
                      RFLOAT** wR,
                      RFLOAT** wT,
                      RFLOAT** wD,
                      double** oldR,
                      double** oldT,
                      double** oldD,
                      double** trans,
                      double** rot,
                      double** dpara,
                      int mR,
                      int mT,
                      int mD,
                      int cSearch)
{
    cuthunder::expectLocalHostA(gpuIdx,
                                wC,
                                wR,
                                wT,
                                wD,
                                oldR,
                                oldT,
                                oldD,
                                trans,
                                rot,
                                dpara,
                                mR,
                                mT,
                                mD,
                                cSearch);
}

void ExpectLocalRTD(int gpuIdx,
                    ManagedCalPoint* mcp,
                    double* oldR,
                    double* oldT,
                    double* oldD,
                    double* trans,
                    double* rot,
                    double* dpara)
{
    cuthunder::expectLocalRTD(gpuIdx,
                              mcp,
                              oldR,
                              oldT,
                              oldD,
                              trans,
                              rot,
                              dpara);
}

void ExpectLocalPreI2D(int gpuIdx,
                       int datShift,
                       ManagedArrayTexture* mgr,
                       ManagedCalPoint* mcp,
                       RFLOAT* devdefO,
                       RFLOAT* devfreQ,
                       int *deviCol,
                       int *deviRow,
                       RFLOAT phaseShift,
                       RFLOAT conT,
                       RFLOAT k1,
                       RFLOAT k2,
                       int pf,
                       int idim,
                       int vdim,
                       int npxl,
                       int interp)
{
    cuthunder::expectLocalPreI2D(gpuIdx,
                                 datShift,
                                 mgr,
                                 mcp,
                                 devdefO,
                                 devfreQ,
                                 deviCol,
                                 deviRow,
                                 phaseShift,
                                 conT,
                                 k1,
                                 k2,
                                 pf,
                                 idim,
                                 vdim,
                                 npxl,
                                 interp);
}

void ExpectLocalPreI3D(int gpuIdx,
                       int datShift,
                       ManagedArrayTexture* mgr,
                       ManagedCalPoint* mcp,
                       RFLOAT* devdefO,
                       RFLOAT* devfreQ,
                       int *deviCol,
                       int *deviRow,
                       RFLOAT phaseShift,
                       RFLOAT conT,
                       RFLOAT k1,
                       RFLOAT k2,
                       int pf,
                       int idim,
                       int vdim,
                       int npxl,
                       int interp)
{
    cuthunder::expectLocalPreI3D(gpuIdx,
                                 datShift,
                                 mgr,
                                 mcp,
                                 devdefO,
                                 devfreQ,
                                 deviCol,
                                 deviRow,
                                 phaseShift,
                                 conT,
                                 k1,
                                 k2,
                                 pf,
                                 idim,
                                 vdim,
                                 npxl,
                                 interp);
}

void ExpectLocalM(int gpuIdx,
                  int datShift,
                  //int l,
                  //RFLOAT* dvpA,
                  //RFLOAT* baseL,
                  ManagedCalPoint* mcp,
                  RFLOAT* devdatPR,
                  RFLOAT* devdatPI,
                  RFLOAT* devctfP,
                  RFLOAT* devsigP,
                  RFLOAT* wC,
                  RFLOAT* wR,
                  RFLOAT* wT,
                  RFLOAT* wD,
                  double oldC,
                  int npxl)
{
    cuthunder::expectLocalM(gpuIdx,
                            datShift,
                            //l,
                            //dvpA,
                            //baseL,
                            mcp,
                            devdatPR,
                            devdatPI,
                            devctfP,
                            devsigP,
                            wC,
                            wR,
                            wT,
                            wD,
                            oldC,
                            npxl);
}

void ExpectLocalHostF(int gpuIdx,
                      RFLOAT** wC,
                      RFLOAT** wR,
                      RFLOAT** wT,
                      RFLOAT** wD,
                      double** oldR,
                      double** oldT,
                      double** oldD,
                      double** trans,
                      double** rot,
                      double** dpara,
                      int cSearch)
{
    cuthunder::expectLocalHostF(gpuIdx,
                                wC,
                                wR,
                                wT,
                                wD,
                                oldR,
                                oldT,
                                oldD,
                                trans,
                                rot,
                                dpara,
                                cSearch);
}

void ExpectLocalFin(int gpuIdx,
                    RFLOAT** devdatPR,
                    RFLOAT** devdatPI,
                    RFLOAT** devctfP,
                    RFLOAT** devdefO,
                    RFLOAT** devfreQ,
                    RFLOAT** devsigP,
                    int cSearch)
{
    cuthunder::expectLocalFin(gpuIdx,
                              devdatPR,
                              devdatPI,
                              devctfP,
                              devdefO,
                              devfreQ,
                              devsigP,
                              cSearch);
}

void ExpectFreeIdx(int gpuIdx,
                   int** deviCol,
                   int** deviRow)
{
    cuthunder::expectFreeIdx(gpuIdx,
                             deviCol,
                             deviRow);

}

void ExpectPrecal(vector<CTFAttr>& ctfAttr,
                  RFLOAT* def,
                  RFLOAT* k1,
                  RFLOAT* k2,
                  const int *iCol,
                  const int *iRow,
                  int idim,
                  int npxl,
                  int imgNum)
{
    LOG(INFO) << "Prepare Parameter for Expectation Pre-cal.";

    std::vector<cuthunder::CTFAttr*> ctfaData;
    for (int i = 0; i < imgNum; i++)
    {
        ctfaData.push_back(reinterpret_cast<cuthunder::CTFAttr*>(&ctfAttr[i]));
    }

    cuthunder::expectPrecal(ctfaData,
                            def,
                            k1,
                            k2,
                            iCol,
                            iRow,
                            idim,
                            npxl,
                            imgNum);
}

void ExpectRotran(std::vector<int>& iGPU,
                  std::vector<void*>& stream,
                  Complex** devrotP,
                  Complex** devtraP,
                  double** devRotMat,
                  double** devpR,
                  double** devpT,
                  double* trans,
                  double* rot,
                  double* pR,
                  double* pT,
                  int** deviCol,
                  int** deviRow,
                  int nR,
                  int nT,
                  int idim,
                  int npxl,
                  int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Expectation Rotation and Translate.";

    cuthunder::expectRotran(iGPU,
                            stream,
                            reinterpret_cast<cuthunder::Complex**>(devrotP),
                            reinterpret_cast<cuthunder::Complex**>(devtraP),
                            devRotMat,
                            devpR,
                            devpT,
                            trans,
                            rot,
                            pR,
                            pT,
                            deviCol,
                            deviRow,
                            nR,
                            nT,
                            idim,
                            npxl,
                            nGPU);
}

void ExpectRotran2D(std::vector<int>& iGPU,
                    std::vector<void*>& stream,
                    std::vector<void*>& symArray,
                    std::vector<void*>& texObject,
                    Complex* volume,
                    Complex** devtraP,
                    double** devnR,
                    double** devpR,
                    double** devpT,
                    double* trans,
                    double* rot,
                    double* pR,
                    double* pT,
                    int** deviCol,
                    int** deviRow,
                    int nK,
                    int nR,
                    int nT,
                    int idim,
                    int vdim,
                    int npxl,
                    int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Expectation Rotation and Translate.";

    cuthunder::expectRotran2D(iGPU,
                              stream,
                              symArray,
                              texObject,
                              reinterpret_cast<cuthunder::Complex*>(volume),
                              reinterpret_cast<cuthunder::Complex**>(devtraP),
                              devnR,
                              devpR,
                              devpT,
                              trans,
                              rot,
                              pR,
                              pT,
                              deviCol,
                              deviRow,
                              nK,
                              nR,
                              nT,
                              idim,
                              vdim,
                              npxl,
                              nGPU);
}

void ExpectProject(std::vector<int>& iGPU,
                   std::vector<void*>& stream,
                   Complex* volume,
                   Complex* rotP,
                   Complex** devrotP,
                   double** devRotMat,
                   int** deviCol,
                   int** deviRow,
                   int nR,
                   int pf,
                   int interp,
                   int vdim,
                   int npxl,
                   int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Expectation Projection.";

    cuthunder::expectProject(iGPU,
                             stream,
                             reinterpret_cast<cuthunder::Complex*>(volume),
                             reinterpret_cast<cuthunder::Complex*>(rotP),
                             reinterpret_cast<cuthunder::Complex**>(devrotP),
                             devRotMat,
                             deviCol,
                             deviRow,
                             nR,
                             pf,
                             interp,
                             vdim,
                             npxl,
                             nGPU);
}

void ExpectGlobal2D(std::vector<int>& iGPU,
                    std::vector<void*>& stream,
                    std::vector<void*>& texObject,
                    Complex** devtraP,
                    double** devnR,
                    double** devpR,
                    double** devpT,
                    RFLOAT* pglk_datPR,
                    RFLOAT* pglk_datPI,
                    RFLOAT* pglk_ctfP,
                    RFLOAT* pglk_sigRcpP,
                    RFLOAT* wC,
                    RFLOAT* wR,
                    RFLOAT* wT,
                    int** deviCol,
                    int** deviRow,
                    int nK,
                    int nR,
                    int nT,
                    int pf,
                    int interp,
                    int idim,
                    int vdim,
                    int npxl,
                    int imgNum,
                    int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Expectation Global.";

    cuthunder::expectGlobal2D(iGPU,
                              stream,
                              texObject,
                              reinterpret_cast<cuthunder::Complex**>(devtraP),
                              devnR,
                              devpR,
                              devpT,
                              pglk_datPR,
                              pglk_datPI,
                              pglk_ctfP,
                              pglk_sigRcpP,
                              wC,
                              wR,
                              wT,
                              deviCol,
                              deviRow,
                              nK,
                              nR,
                              nT,
                              pf,
                              interp,
                              idim,
                              vdim,
                              npxl,
                              imgNum,
                              nGPU);
}

void ExpectGlobal3D(std::vector<int>& iGPU,
                    std::vector<void*>& stream,
                    Complex** devrotP,
                    Complex** devtraP,
                    Complex* rotP,
                    double** devpR,
                    double** devpT,
                    RFLOAT* pglk_datPR,
                    RFLOAT* pglk_datPI,
                    RFLOAT* pglk_ctfP,
                    RFLOAT* pglk_sigRcpP,
                    RFLOAT* wC,
                    RFLOAT* wR,
                    RFLOAT* wT,
                    RFLOAT* baseL,
                    int kIdx,
                    int nK,
                    int nR,
                    int nT,
                    int npxl,
                    int imgNum,
                    int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Expectation Global.";

    cuthunder::expectGlobal3D(iGPU,
                              stream,
                              reinterpret_cast<cuthunder::Complex**>(devrotP),
                              reinterpret_cast<cuthunder::Complex**>(devtraP),
                              reinterpret_cast<cuthunder::Complex*>(rotP),
                              devpR,
                              devpT,
                              pglk_datPR,
                              pglk_datPI,
                              pglk_ctfP,
                              pglk_sigRcpP,
                              wC,
                              wR,
                              wT,
                              baseL,
                              kIdx,
                              nK,
                              nR,
                              nT,
                              npxl,
                              imgNum,
                              nGPU);
}

void freeRotran2D(std::vector<int>& iGPU,
                  std::vector<void*>& symArray,
                  std::vector<void*>& texObject,
                  Complex** devtraP,
                  double** devnR,
                  double** devpR,
                  double** devpT,
                  int nK,
                  int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Expectation Global.";

    cuthunder::freeRotran2D(iGPU,
                            symArray,
                            texObject,
                            reinterpret_cast<cuthunder::Complex**>(devtraP),
                            devnR,
                            devpR,
                            devpT,
                            nK,
                            nGPU);
}

void freeRotran(std::vector<int>& iGPU,
                Complex** devrotP,
                Complex** devtraP,
                double** devRotMat,
                double** devpR,
                double** devpT,
                int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Expectation Global.";

    cuthunder::freeRotran(iGPU,
                          reinterpret_cast<cuthunder::Complex**>(devrotP),
                          reinterpret_cast<cuthunder::Complex**>(devtraP),
                          devRotMat,
                          devpR,
                          devpT,
                          nGPU);
}

void allocFTO(std::vector<int>& iGPU,
              std::vector<void*>& stream,
              Complex* volumeF,
              Complex** dev_F,
              RFLOAT* volumeT,
              RFLOAT** dev_T,
              RFLOAT* arrayTau,
              RFLOAT** devTau,
              double* arrayO,
              double** dev_O,
              int* arrayC,
              int** dev_C,
              const int* iCol,
              int** deviCol,
              const int* iRow,
              int** deviRow,
              const int* iSig,
              int** deviSig,
              bool mode,
              int nk,
              int tauSize,
              int vdim,
              int npxl,
              int nGPU)
{

    cuthunder::allocFTO(iGPU,
                        stream,
                        reinterpret_cast<cuthunder::Complex*>(volumeF),
                        reinterpret_cast<cuthunder::Complex**>(dev_F),
                        volumeT,
                        dev_T,
                        arrayTau,
                        devTau,
                        arrayO,
                        dev_O,
                        arrayC,
                        dev_C,
                        iCol,
                        deviCol,
                        iRow,
                        deviRow,
                        iSig,
                        deviSig,
                        mode,
                        nk,
                        tauSize,
                        vdim,
                        npxl,
                        nGPU);

}

void volumeCopy2D(std::vector<int>& iGPU,
                  Complex* volumeF,
                  Complex** dev_F,
                  RFLOAT* volumeT,
                  RFLOAT** dev_T,
                  int nk,
                  int vdim,
                  int nGPU)
{

    cuthunder::volumeCopy2D(iGPU,
                            reinterpret_cast<cuthunder::Complex*>(volumeF),
                            reinterpret_cast<cuthunder::Complex**>(dev_F),
                            volumeT,
                            dev_T,
                            nk,
                            vdim,
                            nGPU);

}

void volumeCopy3D(std::vector<int>& iGPU,
                  Complex* volumeF,
                  Complex** dev_F,
                  RFLOAT* volumeT,
                  RFLOAT** dev_T,
                  int vdim,
                  int nGPU)
{

    cuthunder::volumeCopy3D(iGPU,
                            reinterpret_cast<cuthunder::Complex*>(volumeF),
                            reinterpret_cast<cuthunder::Complex**>(dev_F),
                            volumeT,
                            dev_T,
                            vdim,
                            nGPU);

}

void InsertI2D(std::vector<int>& iGPU,
               std::vector<void*>& stream,
               Complex** dev_F,
               RFLOAT** dev_T,
               RFLOAT** devTau,
               double** dev_O,
               int** dev_C,
               RFLOAT* pglk_datPR,
               RFLOAT* pglk_datPI,
               RFLOAT* pglk_ctfP,
               RFLOAT* pglk_sigRcpP,
               RFLOAT* w,
               double* offS,
               double *nR,
               double *nT,
               double *nD,
               int* nC,
               CTFAttr *ctfaData,
               int** deviCol,
               int** deviRow,
               int** deviSig,
               RFLOAT pixelSize,
               bool cSearch,
               int nk,
               int opf,
               int npxl,
               int mReco,
               int tauSize,
               int idim,
               int vdim,
               int imgNum,
               int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    cuthunder::InsertI2D(iGPU,
                         stream,
                         reinterpret_cast<cuthunder::Complex**>(dev_F),
                         dev_T,
                         devTau,
                         dev_O,
                         dev_C,
                         pglk_datPR,
                         pglk_datPI,
                         pglk_ctfP,
                         pglk_sigRcpP,
                         w,
                         offS,
                         nR,
                         nT,
                         nD,
                         nC,
                         reinterpret_cast<cuthunder::CTFAttr*>(ctfaData),
                         deviCol,
                         deviRow,
                         deviSig,
                         pixelSize,
                         cSearch,
                         nk,
                         opf,
                         npxl,
                         mReco,
                         tauSize,
                         idim,
                         vdim,
                         imgNum,
                         nGPU);
}

void InsertFT(std::vector<int>& iGPU,
              std::vector<void*>& stream,
              Complex** dev_F,
              RFLOAT** dev_T,
              RFLOAT** devTau,
              double** dev_O,
              int** dev_C,
              RFLOAT* pglk_datPR,
              RFLOAT* pglk_datPI,
              RFLOAT* pglk_ctfP,
              RFLOAT* pglk_sigRcpP,
              RFLOAT* w,
              double* offS,
              double* nR,
              double* nT,
              double* nD,
              int* nC,
              CTFAttr* ctfaData,
              int** deviCol,
              int** deviRow,
              int** deviSig,
              RFLOAT pixelSize,
              bool cSearch,
              int kIdx,
              int opf,
              int npxl,
              int mReco,
              int tauSize,
              int imgNum,
              int idim,
              int vdim,
              int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    cuthunder::InsertFT(iGPU,
                        stream,
                        reinterpret_cast<cuthunder::Complex**>(dev_F),
                        dev_T,
                        devTau,
                        dev_O,
                        dev_C,
                        pglk_datPR,
                        pglk_datPI,
                        pglk_ctfP,
                        pglk_sigRcpP,
                        w,
                        offS,
                        nR,
                        nT,
                        nD,
                        nC,
                        reinterpret_cast<cuthunder::CTFAttr*>(ctfaData),
                        deviCol,
                        deviRow,
                        deviSig,
                        pixelSize,
                        cSearch,
                        kIdx,
                        opf,
                        npxl,
                        mReco,
                        tauSize,
                        imgNum,
                        idim,
                        vdim,
                        nGPU);
}

void InsertFT(std::vector<int>& iGPU,
              std::vector<void*>& stream,
              Complex** dev_F,
              RFLOAT** dev_T,
              RFLOAT** devTau,
              double** dev_O,
              int** dev_C,
              RFLOAT* pglk_datPR,
              RFLOAT* pglk_datPI,
              RFLOAT* pglk_ctfP,
              RFLOAT* pglk_sigRcpP,
              RFLOAT* w,
              double* offS,
              double* nR,
              double* nT,
              double* nD,
              CTFAttr* ctfaData,
              int** deviCol,
              int** deviRow,
              int** deviSig,
              RFLOAT pixelSize,
              bool cSearch,
              int opf,
              int npxl,
              int mReco,
              int tauSize,
              int imgNum,
              int idim,
              int vdim,
              int nGPU)
{
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    cuthunder::InsertFT(iGPU,
                        stream,
                        reinterpret_cast<cuthunder::Complex**>(dev_F),
                        dev_T,
                        devTau,
                        dev_O,
                        dev_C,
                        pglk_datPR,
                        pglk_datPI,
                        pglk_ctfP,
                        pglk_sigRcpP,
                        w,
                        offS,
                        nR,
                        nT,
                        nD,
                        reinterpret_cast<cuthunder::CTFAttr*>(ctfaData),
                        deviCol,
                        deviRow,
                        deviSig,
                        pixelSize,
                        cSearch,
                        opf,
                        npxl,
                        mReco,
                        tauSize,
                        imgNum,
                        idim,
                        vdim,
                        nGPU);
}

void allReduceFTO(std::vector<int>& iGPU,
                  std::vector<void*>& stream,
                  Complex* volumeF,
                  Complex** dev_F,
                  RFLOAT* volumeT,
                  RFLOAT** dev_T,
                  RFLOAT* arrayTau,
                  RFLOAT** devTau,
                  double* arrayO,
                  double** dev_O,
                  int* arrayC,
                  int** dev_C,
                  MPI_Comm& hemi,
                  bool mode,
                  int kIdx,
                  int nk,
                  int tauSize,
                  int vdim,
                  int nGPU)
{
    cuthunder::allReduceFTO(iGPU,
                            stream,
                            reinterpret_cast<cuthunder::Complex*>(volumeF),
                            reinterpret_cast<cuthunder::Complex**>(dev_F),
                            volumeT,
                            dev_T,
                            arrayTau,
                            devTau,
                            arrayO,
                            dev_O,
                            arrayC,
                            dev_C,
                            hemi,
                            mode,
                            kIdx,
                            nk,
                            tauSize,
                            vdim,
                            nGPU);
}

void freeFTO(std::vector<int>& iGPU,
             Complex* volumeF,
             Complex** dev_F,
             RFLOAT* volumeT,
             RFLOAT** dev_T,
             RFLOAT* arrayTau,
             RFLOAT** devTau,
             double* arrayO,
             double** dev_O,
             int* arrayC,
             int** dev_C,
             int** deviCol,
             int** deviRow,
             int** deviSig,
             int nGPU)
{
    cuthunder::freeFTO(iGPU,
                       reinterpret_cast<cuthunder::Complex*>(volumeF),
                       reinterpret_cast<cuthunder::Complex**>(dev_F),
                       volumeT,
                       dev_T,
                       arrayTau,
                       devTau,
                       arrayO,
                       dev_O,
                       arrayC,
                       dev_C,
                       deviCol,
                       deviRow,
                       deviSig,
                       nGPU);

}

void normalizeTF(std::vector<int>& iGPU,
                 std::vector<void*>& stream,
                 Volume& F3D,
	             Volume& T3D,
                 int nGPU)
{
	LOG(INFO) << "Step1: Prepare Parameter for NormalizeT.";

	RFLOAT sf = 1.0 / REAL(T3D[0]);
    int dim = T3D.nSlcFT();
    int dimSize = dim * dim * (dim / 2 + 1);

    Complex *comF3D = &F3D[0];
	Complex *comT3D = &T3D[0];
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(comT3D[i]);
	}

    LOG(INFO) << "Step2: Start PrepareTF...";
    cuthunder::normalizeTF(iGPU,
                           stream,
                           reinterpret_cast<cuthunder::Complex*>(comF3D),
                           douT3D,
                           sf,
                           nGPU,
                           dim);

    for(int i = 0; i < dimSize; i++)
	{
        comT3D[i] = COMPLEX(douT3D[i], 0);
	}

    delete[]douT3D;
}

void symetrizeTF(std::vector<int>& iGPU,
                 std::vector<void*>& stream,
                 Volume& F3D,
	             Volume& T3D,
	             double* symMat,
                 int nGPU,
                 int nSymmetryElement,
                 int maxRadius,
	             int pf)
{
	LOG(INFO) << "Step1: Prepare Parameter for SymetrizeT.";

    int dim = T3D.nSlcFT();
    int dimSize = dim * dim * (dim / 2 + 1);
    int r = (maxRadius * pf + 1) * (maxRadius * pf + 1);

    Complex *comF3D = &F3D[0];
	Complex *comT3D = &T3D[0];
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(comT3D[i]);
	}

    LOG(INFO) << "Step2: Start SymetrizeTF...";
    cuthunder::symmetrizeTF(iGPU,
                            stream,
                            reinterpret_cast<cuthunder::Complex*>(comF3D),
                            douT3D,
                            symMat,
                            nGPU,
                            nSymmetryElement,
                            LINEAR_INTERP,
                            dim,
                            r);

    for(int i = 0; i < dimSize; i++)
	{
        comT3D[i] = COMPLEX(douT3D[i], 0);
	}

    delete[]douT3D;
}

void reconstructG2D(std::vector<int>& iGPU,
                    std::vector<void*>& stream,
                    TabFunction& kernelRL,
                    Complex* ref,
                    Complex* modelF,
                    RFLOAT* modelT,
                    RFLOAT* fscMat,
                    RFLOAT nf,
                    bool map,
                    bool gridCorr,
	                bool joinHalf,
                    int fscMatSize,
	                int maxRadius,
	                int pf,
                    int _N,
                    int vdim,
                    int kbatch,
                    int nThread,
                    int nGPU)
{
    int dimSize = vdim * (vdim / 2 + 1);
    int padSize = _N * pf;
    int r = (maxRadius * pf) * (maxRadius * pf);
    RFLOAT *dev_T[nGPU];
    RFLOAT *dev_W[nGPU];

    cuthunder::allocVolume(iGPU,
                           dev_T,
                           dev_W,
                           nGPU,
                           kbatch * dimSize);

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
    if (map)
    {
        cuthunder::CalculateT2D(stream,
                                iGPU,
                                modelT,
                                dev_T,
                                fscMat,
                                joinHalf,
                                fscMatSize,
                                maxRadius,
                                WIENER_FACTOR_MIN_R,
                                vdim,
                                pf,
                                kbatch,
                                nGPU);
    }
#endif
    if (gridCorr)
    {
        RFLOAT *tabData = kernelRL.getData();
        RFLOAT begin = kernelRL.getBegin();
        RFLOAT stop = kernelRL.getStop();
        RFLOAT step = kernelRL.getStep();
        int tabSize = kernelRL.getTabSize();
        
        Complex* devFourC[nGPU];
        RFLOAT* devRealC[nGPU];
        RFLOAT* dev_tab[nGPU];
        RFLOAT* devDiff[nGPU];
        RFLOAT* devMax[nGPU];
        int* devCount[nGPU];
        std::vector<void*> planC2R;
        std::vector<void*> planR2C;
        
        cuthunder::allocDevicePoint2D(stream,
                                      iGPU,
                                      planC2R,
                                      planR2C,
                                      reinterpret_cast<cuthunder::Complex**>(devFourC),
                                      devRealC,
                                      dev_T,
                                      dev_tab,
                                      devDiff,
                                      devMax,
                                      tabData,
                                      devCount,
                                      tabSize,
                                      kbatch,
                                      vdim,
                                      nGPU);
        
        int gpuIdx = 0;
        int streamIdx = 0;
        #pragma omp parallel for num_threads(nGPU)
        for (int t = 0; t < kbatch; t++)
        {
            gpuIdx = t % nGPU;
            streamIdx = gpuIdx * (stream.size() / nGPU) 
                      + (t / nGPU); 
            
            cuthunder::CalculateW2D(iGPU[gpuIdx],
                                    streamIdx,
                                    stream[streamIdx],
                                    planC2R[streamIdx],
                                    planR2C[streamIdx],
                                    reinterpret_cast<cuthunder::Complex**>(devFourC),
                                    devRealC,
                                    dev_T,
                                    dev_W,
                                    dev_tab,
                                    devDiff,
                                    devMax,
                                    modelT,
                                    tabData,
                                    devCount,
                                    begin,
                                    stop,
                                    step,
                                    nf,
                                    DIFF_C_THRES,
                                    DIFF_C_DECREASE_THRES,
                                    map,
                                    t,
                                    tabSize,
                                    vdim,
                                    r,
                                    MAX_N_ITER_BALANCE,
                                    MIN_N_ITER_BALANCE,
                                    N_DIFF_C_NO_DECREASE,
                                    padSize,
                                    kbatch,
                                    nGPU);
        }
            
        cuthunder::freePoint2D(iGPU,
                               planC2R,
                               planR2C,
                               reinterpret_cast<cuthunder::Complex**>(devFourC),
                               devRealC,
                               dev_T,
                               dev_tab,
                               devDiff,
                               devMax,
                               devCount,
                               nGPU);
    }
    else
    {
        cuthunder::CalculateW2D(stream,
                                iGPU,
                                dev_T,
                                dev_W,
                                modelT,
                                kbatch,
                                vdim,
                                r,
                                nGPU);
    }
   
    RFLOAT* padDstR[kbatch];
    int pdim = _N * pf;
    int pImgSizeRL = pdim * pdim; 

    for (int t = 0; t < kbatch; t++)
    {
        padDstR[t] = (RFLOAT*)malloc(sizeof(RFLOAT) * pImgSizeRL);
    }

    cuthunder::CalculateF2D(stream,
                            iGPU,
                            padDstR,
                            dev_W,
                            reinterpret_cast<cuthunder::Complex*>(modelF),
                            kbatch,
                            r,
                            pdim,
                            vdim,
                            nGPU);
    
    RFLOAT* img[kbatch];
    int tempDim = AROUND((1.0 / pf) * pdim);
    int tempSize = tempDim * tempDim;
    for (int t = 0; t < kbatch; t++)
    {
        img[t] = (RFLOAT*)malloc(sizeof(RFLOAT) * tempSize);
    }

    int idxI, idxP;
    for (int t = 0; t < kbatch; t++)
    {
        #pragma omp parallel for num_threads(nThread)
        for (int j = -tempDim / 2; j < tempDim / 2; j++)
            for (int i = -tempDim / 2; i < tempDim / 2; i++)
            {    
                idxI = (j >= 0 ? j : j + tempDim) * tempDim
                     + (i >= 0 ? i : i + tempDim);
                
                idxP = (j >= 0 ? j : j + pdim) * pdim
                     + (i >= 0 ? i : i + pdim);
                
                img[t][idxI] = padDstR[t][idxP];
            }
    }

    RFLOAT* imgDst[kbatch];
    for (int t = 0; t < kbatch; t++)
    {
        imgDst[t] = (RFLOAT*)malloc(sizeof(RFLOAT) * _N * _N);
    }
    
    for (int t = 0; t < kbatch; t++)
    {
        #pragma omp parallel for num_threads(nThread)
        for (int j = -tempDim / 2; j < tempDim / 2; j++)
            for (int i = -tempDim / 2; i < tempDim / 2; i++)
            {    
                idxI = (j >= 0 ? j : j + tempDim) * tempDim
                     + (i >= 0 ? i : i + tempDim);
                
                idxP = (j >= 0 ? j : j + _N) * _N
                     + (i >= 0 ? i : i + _N);
                
                imgDst[t][idxP] = img[t][idxI];
            }
    }
    
    for (int t = 0; t < kbatch; t++)
    {
        free(img[t]);
    }
    
    RFLOAT *mkbRL = new RFLOAT[(_N / 2 + 1) * (_N / 2 + 1)];
    
    #pragma omp parallel for num_threads(nThread)
    for (int j = 0; j <= _N / 2; j++)
    { 
        for (int i = 0; i <= _N / 2; i++) 
        {
            size_t index = j * (_N / 2 + 1) + i;
#ifdef RECONSTRUCTOR_MKB_KERNEL
                mkbRL[index] = MKB_RL(NORM(i, j) / padSize,
                                  _a * _pf,
                                  _alpha);
#endif
#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                mkbRL[index] = TIK_RL(NORM(i, j) / padSize);
#endif
        }
    }
    
    cuthunder::CorrSoftMaskF2D(stream,
                               iGPU,
                               reinterpret_cast<cuthunder::Complex*>(ref),
                               imgDst,
                               mkbRL,
                               nf,
                               kbatch,
                               _N,
                               nGPU);
   
    for (int t = 0; t < kbatch; t++)
    {
        free(imgDst[t]);
    }
    
    delete[] mkbRL;
}

void allocVolume(std::vector<int>& iGPU,
                 RFLOAT** dev_T,
                 RFLOAT** dev_W,
                 int nGPU,
                 size_t dimSize)
{
    cuthunder::allocVolume(iGPU,
                           dev_T,
                           dev_W,
                           nGPU,
                           dimSize);
}

void ExposePT(std::vector<void*>& stream,
              std::vector<int>& iGPU,
              RFLOAT* T3D,
              RFLOAT** dev_T,
	          vec FSC,
	          int nGPU,
	          int maxRadius,
	          int pf,
              int dim,
	          bool joinHalf,
              const int wienerF)
{
	LOG(INFO) << "Step1: Prepare Parameter for T.";

    int fscMatsize = FSC.size();
    RFLOAT *FSCmat = new RFLOAT[fscMatsize];
    Map<vec>(FSCmat, FSC.rows(), FSC.cols()) = FSC;

    LOG(INFO) << "Step2: Start CalculateT...";

    cuthunder::CalculateT(stream,
                          iGPU,
                          T3D,
                          dev_T,
                          FSCmat,
                          fscMatsize,
                          joinHalf,
                          maxRadius,
                          wienerF,
                          nGPU,
                          dim,
                          pf);

    delete[]FSCmat;
}

void gridCorrection(std::vector<void*>& stream,
                    std::vector<int>& iGPU,
                    Volume& C3D,
                    RFLOAT* volumeT,
                    RFLOAT** dev_W,
                    RFLOAT** dev_T,
                    TabFunction& kernelRL,
                    FFT& fft,
                    RFLOAT nf,
                    int nGPU,
                    int maxRadius,
                    int pf,
                    int _N,
                    bool map,
                    int nThread)
{
    Complex *comC3D;
    RFLOAT *realC3D;
    int buffNum = stream.size();
    int dim = C3D.nSlcRL();
    size_t dimSize = dim * dim * (dim / 2 + 1);
    Complex* devPartC[buffNum];
    RFLOAT* dev_tab[nGPU];
    RFLOAT* devDiff[nGPU];
    RFLOAT* devMax[nGPU];
    int* devCount[nGPU];
    
    RFLOAT* volumeC = (RFLOAT*)malloc(dimSize * sizeof(RFLOAT));
    RFLOAT begin = kernelRL.getBegin();
    RFLOAT stop = kernelRL.getStop();
    RFLOAT step = kernelRL.getStep();
    RFLOAT diffC = TS_MAX_RFLOAT_VALUE;
    RFLOAT diffCPrev = TS_MAX_RFLOAT_VALUE;
    int r = (maxRadius * pf) * (maxRadius * pf);
    int padSize = pf * _N;
    int tabSize = kernelRL.getTabSize();

    cuthunder::allocDevicePoint(iGPU,
                                reinterpret_cast<cuthunder::Complex**>(devPartC),
                                dev_tab,
                                devDiff,
                                devMax,
                                devCount,
                                tabSize,
                                dim,
                                nGPU);

    cuthunder::hostDeviceInit(iGPU,
                              stream,
                              volumeC,
                              volumeT,
                              kernelRL.getData(),
                              dev_W,
                              dev_T,
                              dev_tab,
                              nGPU,
                              tabSize,
                              r,
                              map,
                              dim);
            
    int m = 0;
    int nDiffCNoDecrease = 0;
    for (m = 0; m < MAX_N_ITER_BALANCE; m++)
    {
        cuthunder::CalculateC(iGPU,
                              stream,
                              volumeC,
                              reinterpret_cast<cuthunder::Complex**>(devPartC),
                              dev_T,
                              dev_W,
                              nGPU,
                              dim);
        
        #pragma omp parallel for num_threads(nThread)
        for(size_t i = 0; i < dimSize; i++)
	    {
            C3D[i].dat[0] = volumeC[i];
            C3D[i].dat[1] = 0;
	    }
        
        fft.bwExecutePlan(C3D, nThread);
        
        realC3D = &C3D(0);
        cuthunder::ConvoluteC(iGPU,
                              stream,
                              realC3D,
                              reinterpret_cast<cuthunder::Complex**>(devPartC),
                              dev_tab,
                              begin,
                              stop,
                              step,
                              nf,
                              nGPU,
                              tabSize,
                              padSize,
                              dim);
        
        fft.fwExecutePlan(C3D);
        
        comC3D = &C3D[0];
        diffCPrev = diffC;
        
        cuthunder::UpdateWC(iGPU,
                            stream,
                            reinterpret_cast<cuthunder::Complex*>(comC3D),
                            reinterpret_cast<cuthunder::Complex**>(devPartC),
                            dev_W,
                            devDiff,
                            devMax,
                            devCount,
                            diffC,
                            nGPU,
                            r,
                            dim);
        
        if (diffC > diffCPrev * DIFF_C_DECREASE_THRES)
            nDiffCNoDecrease += 1;
        else
            nDiffCNoDecrease = 0;

        if ((diffC < DIFF_C_THRES) ||
            ((m >= MIN_N_ITER_BALANCE) &&
            (nDiffCNoDecrease == N_DIFF_C_NO_DECREASE))) 
            break;
    }
            
    cuthunder::freeDevHostPoint(iGPU,
                                reinterpret_cast<cuthunder::Complex**>(devPartC),
                                dev_T,
                                dev_tab,
                                devDiff,
                                devMax,
                                devCount,
                                volumeC,
                                nGPU,
                                dim);

    free(volumeC);
}

void ExposeWT(std::vector<void*>& stream,
              std::vector<int>& iGPU,
              RFLOAT* T3D,
              RFLOAT** dev_W,
              RFLOAT** dev_T,
              int nGPU,
              int maxRadius,
              int pf,
              bool map,
              int dim)
{
    LOG(INFO) << "Step1: Prepare Parameter for InitialW.";

    int r = (maxRadius * pf) * (maxRadius * pf);

    LOG(INFO) << "Step2: Start Calculate W...";

    cuthunder::CalculateW(stream,
                          iGPU,
                          T3D,
                          dev_W,
                          dev_T,
                          nGPU,
                          dim,
                          map,
                          r);
}

void ExposePFW(std::vector<void*>& stream,
               std::vector<int>& iGPU,
               Volume& padDst,
               Volume& F3D,
               RFLOAT** dev_W,
               int nGPU,
               int maxRadius,
               int pf)
{
    LOG(INFO) << "Step1: Prepare Parameter for pad.";

    int dim = F3D.nSlcFT();
    int pdim = padDst.nSlcFT();
    int r = (maxRadius * pf) * (maxRadius * pf);

    Complex *comPAD = &padDst[0];
    Complex *comF3D = &F3D[0];

    LOG(INFO) << "Step2: Start PrepareF...";

    cuthunder::CalculateFW(stream,
                           iGPU,
                           reinterpret_cast<cuthunder::Complex*>(comPAD),
                           reinterpret_cast<cuthunder::Complex*>(comF3D),
                           dev_W,
                           nGPU,
                           r,
                           pdim,
                           dim);
}

void ExposePF(Volume& padDst,
              Volume& padDstR,
              Volume& F3D,
              RFLOAT* W3D,
              int maxRadius,
              int pf)
{
    LOG(INFO) << "Step1: Prepare Parameter for pad.";

    Complex *comPAD = &padDst[0];
    Complex *comF3D = &F3D[0];
    RFLOAT *comPADR = &padDstR(0);

    LOG(INFO) << "Step2: Prepare Paramete for CalculateFW.";

    int dim = F3D.nSlcFT();
    int pdim = padDst.nSlcFT();
    int r = (maxRadius * pf) * (maxRadius * pf);

    LOG(INFO) << "Step3: Start PrepareF...";

    cuthunder::CalculateF(reinterpret_cast<cuthunder::Complex*>(comPAD),
                          reinterpret_cast<cuthunder::Complex*>(comF3D),
                          comPADR,
                          W3D,
                          r,
                          pdim,
                          dim);
}

void ExposeCorrF(std::vector<void*>& stream,
                 std::vector<int>& iGPU,
                 Volume& dst,
                 RFLOAT* mkbRL,
                 RFLOAT nf,
                 int nGPU)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";
    RFLOAT *comDst = &dst(0);
    int dim = dst.nSlcRL();
    
    LOG(INFO) << "Step2: Start CorrSoftMaskF...";
    cuthunder::CorrSoftMaskF(stream,
                             iGPU,
                             comDst,
                             mkbRL,
                             nf,
                             nGPU,
                             dim);

}

void ExposeCorrF(Volume& dstN,
                 Volume& dst,
                 RFLOAT* mkbRL,
                 RFLOAT nf)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";

    RFLOAT *comDstN = &dstN(0);
    Complex *comDst = &dst[0];
    int dim = dstN.nSlcRL();

    LOG(INFO) << "Step2: Start CorrSoftMaskF...";

    cuthunder::CorrSoftMaskF(reinterpret_cast<cuthunder::Complex*>(comDst),
                             comDstN,
                             mkbRL,
                             nf,
                             dim);

}

void TranslateI2D(std::vector<void*>& stream,
                  std::vector<int>& iGPU,
                  Complex* img,
                  RFLOAT* ox,
                  RFLOAT* oy,
                  int kbatch,
                  int r,
                  int dim,
                  int nGPU)
{
    LOG(INFO) << "Step1: Prepare Parameter for TransImg.";

    cuthunder::TranslateI2D(stream,
                            iGPU,
                            reinterpret_cast<cuthunder::Complex*>(img),
                            ox,
                            oy,
                            kbatch,
                            r,
                            dim,
                            nGPU);
}

void TranslateI(std::vector<int>& iGPU,
                std::vector<void*>& stream,
                Volume& ref,
                double ox,
                double oy,
                double oz,
                int nGPU,
                int r)
{
    LOG(INFO) << "Step1: Prepare Parameter for TransImg.";

    Complex *comRef = &ref[0];
    int dim = ref.nSlcRL();

    LOG(INFO) << "Step4: Start PrepareF...";

    cuthunder::TranslateI(iGPU,
                          stream,
                          reinterpret_cast<cuthunder::Complex*>(comRef),
                          (RFLOAT)ox,
                          (RFLOAT)oy,
                          (RFLOAT)oz,
                          nGPU,
                          r,
                          dim);
}

void hostRegister(Complex* img,
                  int totalNum)
{
    cuthunder::hostRegister(reinterpret_cast<cuthunder::Complex*>(img),
                            totalNum);
}

void hostRegister(RFLOAT* data,
                  int totalNum)
{
    cuthunder::hostRegister(data,
                            totalNum);
}

void hostFree(Complex* img)
{
    cuthunder::hostFree(reinterpret_cast<cuthunder::Complex*>(img));
}

void hostFree(RFLOAT* data)
{
    cuthunder::hostFree(data);
}

void reMask(std::vector<void*>& stream,
            std::vector<int>& iGPU,
            Complex* img,
            RFLOAT maskRadius,
            RFLOAT pixelSize,
            RFLOAT ew,
            int idim,
            int imgNum,
            int nGPU)
{
    LOG(INFO) << "Step1: Prepare Parameter for Remask.";

    cuthunder::reMask(stream,
                      iGPU,
                      reinterpret_cast<cuthunder::Complex*>(img),
                      maskRadius,
                      pixelSize,
                      ew,
                      idim,
                      imgNum,
                      nGPU);
}

void GCTFinit(std::vector<void*>& stream,
              std::vector<int>& iGPU,
              Complex* ctf,
              vector<CTFAttr>& ctfAttr,
              RFLOAT pixelSize,
              int idim,
              int shift,
              int imgNum,
              int nGPU)
{
    LOG(INFO) << "Step1: Prepare Parameter for CTF calculation";

    std::vector<cuthunder::CTFAttr*> ctfaData;

    for (int i = 0; i < imgNum; i++)
    {
        ctfaData.push_back(reinterpret_cast<cuthunder::CTFAttr*>(&ctfAttr[shift + i]));
    }

    cuthunder::GCTF(stream,
                    iGPU,
                    reinterpret_cast<cuthunder::Complex*>(ctf),
                    ctfaData,
                    pixelSize,
                    idim,
                    imgNum,
                    nGPU);

}
