/**************************************************************
 * FileName: cuthunder.h
 * Author  : Kunpeng WANG, Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#ifndef CUTHUNDER_H
#define CUTHUNDER_H

#include "easylogging++.h"

#include "Config.h"
#include "Macro.h"
#include "Precision.h"
#include "MemoryBazaar.h"

#include "ManagedArrayTexture.h"
#include "ManagedCalPoint.h"

#include <mpi.h>
#include <vector>
#include <unistd.h>

namespace cuthunder {

using std::vector;

///////////////////////////////////////////////////////////////

class Complex;
class CTFAttr;

/* Volume create kind */
typedef enum {
    DEFAULT     = 0,
    HOST_ONLY   = 1,
    DEVICE_ONLY = 2,
    HD_SYNC     = 4,
    HD_BOTH     = HOST_ONLY | DEVICE_ONLY
} VolCrtKind;

/**
 * Test routines.
 *
 * ...
 */
void addTest();

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void getAviDevice(vector<int>& gpus);

/**
 * @brief  GPU environment check.
 *
 * @param
 * @param
 */
void readGPUPARA(char* gpuList,
                 vector<int>& iGPU,
                 int& nGPU);

/**
 * @brief  GPU environment check.
 *
 * @param
 * @param
 */
void gpuCheck(vector<void*>& stream,
              vector<int>& iGPU,
              int& nGPU);

void gpuMemoryCheck(vector<int>& iGPU,
                    int rankId,
                    int nGPU);

/**
 * @brief  GPU stream destory.
 *
 * @param
 * @param
 */
void gpuEnvDestory(vector<void*>& stream,
                   vector<int>& iGPU,
                   int nGPU);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectPreidx(int gpuIdx,
                  int** deviCol,
                  int** deviRow,
                  int* iCol,
                  int* iRow,
                  int npxl);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectPrefre(int gpuIdx,
                  RFLOAT** devfreQ,
                  RFLOAT* freQ,
                  int npxl);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalIn(int gpuIdx,
                   RFLOAT** devdatPR,
                   RFLOAT** devdatPI,
                   RFLOAT** devctfP,
                   RFLOAT** devdefO,
                   RFLOAT** devsigP,
                   int npxl,
                   int cpyNum,
                   int cSearch);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalP(int gpuIdx,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalV2D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int dimSize);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalV3D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int vdim);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalHostA(int gpuIdx,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalRTD(int gpuIdx,
                    ManagedCalPoint* mcp,
                    double* oldR,
                    double* oldT,
                    double* oldD,
                    double* trans,
                    double* rot,
                    double* dpara);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalPreI2D(int gpuIdx,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalPreI3D(int gpuIdx,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalM(int gpuIdx,
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
                  RFLOAT* baseLine,
                  int npxl);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalHostF(int gpuIdx,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalFin(int gpuIdx,
                    RFLOAT** devdatPR,
                    RFLOAT** devdatPI,
                    RFLOAT** devctfP,
                    RFLOAT** devdefO,
                    RFLOAT** devfreQ,
                    RFLOAT** devsigP,
                    int cSearch);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectFreeIdx(int gpuIdx,
                   int** deviCol,
                   int** deviRow);

/**
 * @brief Pre-calculation in expectation.
 *
 * @param
 * @param
 */
void expectPrecal(vector<CTFAttr*>& ctfaData,
                  RFLOAT* def,
                  RFLOAT* k1,
                  RFLOAT* k2,
                  const int *iCol,
                  const int *iRow,
                  int idim,
                  int npxl,
                  int imgNum);

/**
 * @brief  Expectation Rotran.
 *
 * @param
 * @param
 */
void expectRotran(vector<int>& iGPU,
                  vector<void*>& stream,
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

/**
 * @brief  Expectation Rotran.
 *
 * @param
 * @param
 */
void expectRotran2D(vector<int>& iGPU,
                    vector<void*>& stream,
                    vector<void*>& symArray,
                    vector<void*>& texObject,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectProject(vector<int>& iGPU,
                   vector<void*>& stream,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectGlobal2D(vector<int>& iGPU,
                    vector<void*>& stream,
                    vector<void*>& texObject,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectGlobal3D(vector<int>& iGPU,
                    vector<void*>& stream,
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
                    int nR,
                    int nT,
                    int npxl,
                    int imgNum,
                    int nGPU);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void freeRotran2D(vector<int>& iGPU,
                  vector<void*>& symArray,
                  vector<void*>& texObject,
                  Complex** devtraP,
                  double** devnR,
                  double** devpR,
                  double** devpT,
                  int nK,
                  int nGPU);

/**
 * @brief  free Rotran space.
 *
 * @param
 * @param
 */
void freeRotran(vector<int>& iGPU,
                Complex** devrotP,
                Complex** devtraP,
                double** devRotMat,
                double** devpR,
                double** devpT,
                int nGPU);

/**
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void allocFTO(vector<int>& iGPU,
              vector<void*>& stream,
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

/**
 * @brief Copy images' volume to Device.
 *
 * @param
 * @param
 */
void volumeCopy2D(vector<int>& iGPU,
                  Complex* volumeF,
                  Complex** dev_F,
                  RFLOAT* volumeT,
                  RFLOAT** dev_T,
                  int nk,
                  int vdim,
                  int nGPU);

/**
 * @brief Copy images' volume to Device.
 *
 * @param
 * @param
 */
void volumeCopy3D(vector<int>& iGPU,
                  Complex* volumeF,
                  Complex** dev_F,
                  RFLOAT* volumeT,
                  RFLOAT** dev_T,
                  int vdim,
                  int nGPU);

/**
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void InsertI2D(vector<int>& iGPU,
               vector<void*>& stream,
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

/**
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void InsertFT(vector<int>& iGPU,
              vector<void*>& stream,
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
              int *nC,
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

/**
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void InsertFT(vector<int>& iGPU,
              vector<void*>& stream,
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

/**
 * @brief AllReduce FTO.
 *
 * @param
 * @param
 */
void allReduceFTO(vector<int>& iGPU,
                  vector<void*>& stream,
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

/**
 * @brief Free Volume FTO.
 *
 * @param
 * @param
 */
void freeFTO(vector<int>& iGPU,
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

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void normalizeTF(vector<int>& iGPU,
                 vector<void*>& stream,
                 Complex *F3D,
                 RFLOAT *T3D,
                 const RFLOAT sf,
                 const int nGPU,
                 const int dim);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void symmetrizeTF(vector<int>& iGPU,
                  vector<void*>& stream,
                  Complex *F3D,
                  RFLOAT *T3D,
                  const double *symMat,
                  const int nGPU,
                  const int nSymmetryElement,
                  const int interp,
                  const int dim,
                  const int r);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void allocVolume(vector<int>& iGPU,
                 RFLOAT** dev_T,
                 RFLOAT** dev_W,
                 int nGPU,
                 size_t dimSize);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CalculateT2D(vector<void*>& stream,
                  vector<int>& iGPU,
                  RFLOAT* T2D,
                  RFLOAT** dev_T,
                  RFLOAT* fscMat,
                  bool joinHalf,
                  int fscMatsize,
                  int maxRadius,
                  int wienerF,
                  int dim,
                  int pf,
                  int kbatch,
                  int nGPU);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CalculateT(vector<void*>& stream,
                vector<int>& iGPU,
                RFLOAT* T3D,
                RFLOAT** dev_T,
                RFLOAT* FSC,
                int fscMatsize,
                bool joinHalf,
                int maxRadius,
                int wienerF,
                int nGPU,
                int dim,
                int pf);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW2D(vector<void*>& stream,
                  vector<int>& iGPU,
                  RFLOAT** dev_T,
                  RFLOAT** dev_W,
                  RFLOAT* T2D,
                  int kbatch,
                  int dim,
                  int r,
                  int nGPU);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW(vector<void*>& stream,
                vector<int>& iGPU,
                RFLOAT *T3D,
                RFLOAT** dev_W,
                RFLOAT** dev_T,
                int nGPU,
                int dim,
                bool map,
                int r);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDevicePoint2D(vector<void*>& stream,
                        vector<int>& iGPU,
                        vector<void*>& planC2R,
                        vector<void*>& planR2C,
                        Complex** devFourC,
                        RFLOAT** devRealC,
                        RFLOAT** dev_T,
                        RFLOAT** dev_tab,
                        RFLOAT** devDiff,
                        RFLOAT** devMax,
                        RFLOAT* tab,
                        int** devCount,
                        int tabSize,
                        int kbatch,
                        int dim,
                        int nGPU);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW2D(int gpuIdx,
                  int streamIdx,
                  void* stream,
                  void* planC2R,
                  void* planR2C,
                  Complex** devFourC,
                  RFLOAT** devRealC,
                  RFLOAT** dev_T,
                  RFLOAT** dev_W,
                  RFLOAT** dev_tab,
                  RFLOAT** devDiff,
                  RFLOAT** devMax,
                  RFLOAT* modelT,
                  RFLOAT* tabdata,
                  int** devCount,
                  RFLOAT begin,
                  RFLOAT end,
                  RFLOAT step,
                  RFLOAT nf,
                  RFLOAT diffC_Thres,
                  RFLOAT diffC_DThres,
                  bool map,
                  int kIdx,
                  int tabsize,
                  int dim,
                  int r,
                  int maxIter,
                  int minIter,
                  int noDiffC,
                  int padSize,
                  int kbatch,
                  int nGPU);

/**
 *
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void freePoint2D(vector<int>& iGPU,
                 vector<void*>& planC2R,
                 vector<void*>& planR2C,
                 Complex** devFourC,
                 RFLOAT** devRealC,
                 RFLOAT** dev_T,
                 RFLOAT** dev_tab,
                 RFLOAT** devDiff,
                 RFLOAT** devMax,
                 int** devCount,
                 int nGPU);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDevicePoint(vector<int>& iGPU,
                      Complex** devPartC,
                      RFLOAT** dev_tab,
                      RFLOAT** devDiff,
                      RFLOAT** devMax,
                      int** devCount,
                      int tabSize,
                      int dim,
                      int nGPU);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void hostDeviceInit(vector<int>& iGPU,
                    vector<void*>& stream,
                    RFLOAT* volumeC,
                    RFLOAT* T3D,
                    RFLOAT* tab,
                    RFLOAT** dev_W,
                    RFLOAT** dev_T,
                    RFLOAT** dev_tab,
                    int nGPU,
                    int tabSize,
                    int r,
                    bool map,
                    int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateC(vector<int>& iGPU,
                vector<void*>& stream,
                RFLOAT* volumeC,
                Complex** devPartC,
                RFLOAT** dev_T,
                RFLOAT** dev_W,
                int nGPU,
                int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void ConvoluteC(vector<int>& iGPU,
                vector<void*>& stream,
                RFLOAT* C3D,
                Complex** devPartC,
                RFLOAT** dev_tab,
                RFLOAT begin,
                RFLOAT end,
                RFLOAT step,
                RFLOAT nf,
                int nGPU,
                int tabsize,
                int padSize,
                const int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void UpdateWC(vector<int>& iGPU,
              vector<void*>& stream,
              Complex* C3D,
              Complex** devPartC,
              RFLOAT** dev_W,
              RFLOAT** devDiff,
              RFLOAT** devMax,
              int** devCount,
              RFLOAT &diffC,
              int nGPU,
              int r,
              int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void freeDevHostPoint(vector<int>& iGPU,
                      Complex** devPartC,
                      RFLOAT** dev_T,
                      RFLOAT** dev_tab,
                      RFLOAT** devDiff,
                      RFLOAT** devMax,
                      int** devCount,
                      RFLOAT* volumeC,
                      int nGPU,
                      int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW2D(RFLOAT *T2D,
                  RFLOAT *W2D,
                  RFLOAT *tabdata,
                  RFLOAT begin,
                  RFLOAT end,
                  RFLOAT step,
                  int tabsize,
                  const int dim,
                  const int r,
                  const RFLOAT nf,
                  const int maxIter,
                  const int minIter,
                  const int padSize);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW(RFLOAT *T3D,
                RFLOAT *W3D,
                RFLOAT *tabdata,
                RFLOAT begin,
                RFLOAT end,
                RFLOAT step,
                int tabsize,
                const int dim,
                const int r,
                const RFLOAT nf,
                const int maxIter,
                const int minIter,
                const int padSize);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CalculateF2D(vector<void*>& stream,
                  vector<int>& iGPU,
                  RFLOAT** padDstR,
                  RFLOAT** dev_W,
                  Complex* F2D,
                  int kbatch,
                  int r,
                  int pdim,
                  int fdim,
                  int nGPU);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CalculateFW(vector<void*>& stream,
                 vector<int>& iGPU,
                 Complex* padDst,
                 Complex* F3D,
                 RFLOAT** dev_W,
                 int nGPU,
                 const int r,
                 const int pdim,
                 const int fdim);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CalculateF(Complex *padDst,
                Complex *F3D,
                RFLOAT *padDstR,
                RFLOAT *W3D,
                const int r,
                const int pdim,
                const int fdim);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CorrSoftMaskF2D(vector<void*>& stream,
                     vector<int>& iGPU,
                     Complex* ref, 
                     RFLOAT** imgDst,
                     RFLOAT* mkbRL,
                     RFLOAT nf,
                     int kbatch,
                     int dim,
                     int nGPU);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CorrSoftMaskF(vector<void*>& stream,
                   vector<int>& iGPU,
                   RFLOAT *dstN,
                   RFLOAT *mkbRL,
                   RFLOAT nf,
                   int nGPU,
                   const int dim);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CorrSoftMaskF(Complex *dst,
                   RFLOAT *dstN,
                   RFLOAT *mkbRL,
                   RFLOAT nf,
                   const int dim);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void TranslateI2D(vector<void*>& stream,
                  vector<int>& iGPU,
                  Complex* src,
                  RFLOAT* ox,
                  RFLOAT* oy,
                  int kbatch,
                  int r,
                  int dim,
                  int nGPU);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void TranslateI(vector<int>& iGPU,
                vector<void*>& stream,
                Complex* ref,
                RFLOAT ox,
                RFLOAT oy,
                RFLOAT oz,
                int nGPU,
                int r,
                int dim);

/**
 * @brief .
 *
 * @param
 * @param
 */
void hostRegister(Complex* img,
                  int totalNum);

/**
 * @brief .
 *
 * @param
 * @param
 */
void hostRegister(RFLOAT* data,
                  int totalNum);

/**
 * @brief .
 *
 * @param
 * @param
 */
void hostFree(Complex* img);

/**
 * @brief .
 *
 * @param
 * @param
 */
void hostFree(RFLOAT* data);

/**
 * @brief ReMask.
 *
 * @param
 * @param
 */
void reMask(vector<void*>& stream,
            vector<int>& iGPU,
            Complex* imgData,
            RFLOAT maskRadius,
            RFLOAT pixelSize,
            RFLOAT ew,
            int idim,
            int imgNum,
            int nGPU);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void GCTF(vector<void*>& stream,
          vector<int>& iGPU,
          Complex* ctf,
          vector<CTFAttr*>& ctfaData,
          RFLOAT* dpara,
          RFLOAT pixelSize,
          int ndim,
          int nImg,
          int nGPU);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CorrSoftMaskF(RFLOAT *dst,
                   RFLOAT *mkbRL,
                   RFLOAT nf,
                   const int dim,
                   const int size,
                   const int edgeWidth);

///////////////////////////////////////////////////////////////

} // end namespace cunthunder

///////////////////////////////////////////////////////////////
#endif
