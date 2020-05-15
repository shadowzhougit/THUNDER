#ifndef INTERFACE_H
#define INTERFACE_H

#include "mpi.h"
#include "omp.h"
#include "Image.h"
#include "ImageFunctions.h"
#include "Particle.h"
#include "Volume.h"
#include "Symmetry.h"
#include "Database.h"
#include "Reconstructor.h"
#include "Typedef.h"
#include "cuthunder.h"
#include "ManagedArrayTexture.h"
#include "ManagedCalPoint.h"

void readGPUPARA(char* gpuList,
                 std::vector<int>& iGPU,
                 int& nGPU);

void gpuCheck(std::vector<void*>& stream,
              std::vector<int>& iGPU,
              int& nGPU);

void gpuMemoryCheck(std::vector<int>& iGPU,
                    int rankId,
                    int nGPU);

void gpuEnvDestory(std::vector<void*>& stream,
                   std::vector<int>& iGPU,
                   int nGPU);

void ExpectPreidx(int gpuIdx,
                  int** deviCol,
                  int** deviRow,
                  int* iCol,
                  int* iRow,
                  int npxl);


void ExpectPrefre(int gpuIdx,
                  RFLOAT** devfreQ,
                  RFLOAT* freQ,
                  int npxl);

void ExpectLocalIn(int gpuIdx,
                   RFLOAT** devdatPR,
                   RFLOAT** devdatPI,
                   RFLOAT** devctfP,
                   RFLOAT** devdefO,
                   RFLOAT** devsigP,
                   int nPxl,
                   int cpyNumL,
                   int searchType);

void ExpectLocalV2D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int dimSize);

void ExpectLocalV3D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int vdim);

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
                  int cSearch);

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
                      int cSearch);

void ExpectLocalRTD(int gpuIdx,
                    ManagedCalPoint* mcp,
                    double* oldR,
                    double* oldT,
                    double* oldD,
                    double* trans,
                    double* rot,
                    double* dpara);

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
                       int interp);

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
                       int interp);

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
                  int npxl);

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
                      int cSearch);

void ExpectLocalFin(int gpuIdx,
                    RFLOAT** devdatPR,
                    RFLOAT** devdatPI,
                    RFLOAT** devctfP,
                    RFLOAT** devdefO,
                    RFLOAT** devfreQ,
                    RFLOAT** devsigP,
                    int cSearch);

void ExpectFreeIdx(int gpuIdx,
                   int** deviCol,
                   int** deviRow);

void ExpectPrecal(vector<CTFAttr>& ctfAttr,
                  RFLOAT* def,
                  RFLOAT* k1,
                  RFLOAT* k2,
                  const int *iCol,
                  const int *iRow,
                  int idim,
                  int npxl,
                  int imgNum);

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
                  int nGPU);

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
                    int nGPU);

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
                   int nGPU);

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
                    int nGPU);

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
                    int nGPU);

void freeRotran2D(std::vector<int>& iGPU,
                  std::vector<void*>& symArray,
                  std::vector<void*>& texObject,
                  Complex** devtraP,
                  double** devnR,
                  double** devpR,
                  double** devpT,
                  int nK,
                  int nGPU);

void freeRotran(std::vector<int>& iGPU,
                Complex** devrotP,
                Complex** devtraP,
                double** devRotMat,
                double** devpR,
                double** devpT,
                int nGPU);

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
              int nGPU);

void volumeCopy2D(std::vector<int>& iGPU,
                  Complex* volumeF,
                  Complex** dev_F,
                  RFLOAT* volumeT,
                  RFLOAT** dev_T,
                  int nk,
                  int vdim,
                  int nGPU);

void volumeCopy3D(std::vector<int>& iGPU,
                  Complex* volumeF,
                  Complex** dev_F,
                  RFLOAT* volumeT,
                  RFLOAT** dev_T,
                  int vdim,
                  int nGPU);

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
               int nGPU);

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
              int nGPU);

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
              int nGPU);

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
                  int nGPU);

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
             int nGPU);

void normalizeTF(std::vector<int>& iGPU,
                 std::vector<void*>& stream,
                 Volume& F3D,
	             Volume& T3D,
                 int nGPU);

void symetrizeTF(std::vector<int>& iGPU,
                 std::vector<void*>& stream,
                 Volume& F3D,
	             Volume& T3D,
	             double* symMat,
                 int nGPU,
                 int nSymmetryElement,
                 int maxRadius,
	             int pf);

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
                    int nGPU);

void allocVolume(std::vector<int>& iGPU,
                 RFLOAT** dev_T,
                 RFLOAT** dev_W,
                 int nGPU,
                 size_t dimSize);

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
              const int wienerF);

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
                    int nThread);

void ExposeWT(std::vector<void*>& stream,
              std::vector<int>& iGPU,
              RFLOAT* T3D,
              RFLOAT** dev_W,
              RFLOAT** dev_T,
              int nGPU,
              int maxRadius,
              int pf,
              bool map,
              int dim);

void ExposePFW(std::vector<void*>& stream,
               std::vector<int>& iGPU,
               Volume& padDst,
               Volume& F3D,
               RFLOAT** dev_W,
               int nGPU,
               int maxRadius,
               int pf);

void ExposePF(Volume& padDst,
              Volume& padDstR,
              Volume& F3D,
              RFLOAT* W3D,
              int maxRadius,
              int pf);

void ExposeCorrF(std::vector<void*>& stream,
                 std::vector<int>& iGPU,
                 Volume& dst,
                 RFLOAT* mkbRL,
                 RFLOAT nf,
                 int nGPU);

void ExposeCorrF(Volume& dstN,
                 Volume& dst,
                 RFLOAT* mkbRL,
                 RFLOAT nf);

void TranslateI2D(std::vector<void*>& stream,
                  std::vector<int>& iGPU,
                  Complex* img,
                  RFLOAT* ox,
                  RFLOAT* oy,
                  int kbatch,
                  int r,
                  int dim,
                  int nGPU);

void TranslateI(std::vector<int>& iGPU,
                std::vector<void*>& stream,
                Volume& ref,
                double ox,
                double oy,
                double oz,
                int nGPU,
                int r);

void hostRegister(Complex* img,
                  int totalNum);

void hostRegister(RFLOAT* data,
                  int totalNum);

void hostFree(Complex* img);

void hostFree(RFLOAT* data);

void reMask(std::vector<void*>& stream,
            std::vector<int>& iGPU,
            Complex* img,
            RFLOAT maskRadius,
            RFLOAT pixelSize,
            RFLOAT ew,
            int idim,
            int imgNum,
            int nGPU);

void GCTFinit(std::vector<void*>& stream,
              std::vector<int>& iGPU,
              Complex* ctf,
              vector<CTFAttr>& ctfAttr,
              RFLOAT pixelSize,
              int idim,
              int shift,
              int imgNum,
              int nGPU);

#endif
