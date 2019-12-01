#ifndef INTERFACE_H
#define INTERFACE_H

#include "mpi.h"
#include "Image.h"
#include "ImageFunctions.h"
#include "Particle.h"
#include "Volume.h"
#include "Symmetry.h"
#include "Database.h"
#include "Typedef.h"
#include "cuthunder.h"
#include "ManagedArrayTexture.h"
#include "ManagedCalPoint.h"
#include "MemoryBazaar.h"

void getAviDevice(std::vector<int>& gpus);

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

void ExpectGlobal2D(Complex* vol,
                    MemoryBazaar<RFLOAT, BaseType, 4>& datPR,
                    MemoryBazaar<RFLOAT, BaseType, 4>& datPI,
                    MemoryBazaar<RFLOAT, BaseType, 4>& ctfP,
                    MemoryBazaar<RFLOAT, BaseType, 4>& sigRcpP,
                    double* trans,
                    RFLOAT* wC,
                    RFLOAT* wR,
                    RFLOAT* wT,
                    double* pR,
                    double* pT,
                    double* rot,
                    const int *iCol,
                    const int *iRow,
                    int nK,
                    int nR,
                    int nT,
                    int pf,
                    int interp,
                    int idim,
                    int vdim,
                    int npxl,
                    int imgNum);

void ExpectRotran(Complex* traP,
                  double* trans,
                  double* rot,
                  double* rotMat,
                  const int *iCol,
                  const int *iRow,
                  int nR,
                  int nT,
                  int idim,
                  int npxl);

void ExpectProject(Complex* volume,
                   Complex* rotP,
                   double* rotMat,
                   const int *iCol,
                   const int *iRow,
                   int nR,
                   int pf,
                   int interp,
                   int vdim,
                   int npxl);

void ExpectGlobal3D(Complex* rotP,
                    Complex* traP,
                    MemoryBazaar<RFLOAT, BaseType, 4>& datPR,
                    MemoryBazaar<RFLOAT, BaseType, 4>& datPI,
                    MemoryBazaar<RFLOAT, BaseType, 4>& ctfP,
                    MemoryBazaar<RFLOAT, BaseType, 4>& sigRcpP,
                    RFLOAT* wC,
                    RFLOAT* wR,
                    RFLOAT* wT,
                    double* pR,
                    double* pT,
                    RFLOAT* baseL,
                    int kIdx,
                    int nK,
                    int nR,
                    int nT,
                    int npxl,
                    int imgNum);

void InsertI2D(Complex* F2D,
               RFLOAT* T2D,
               double* O2D,
               int* counter,
               MPI_Comm& hemi,
               MPI_Comm& slav,
               MemoryBazaar<RFLOAT, BaseType, 4>& datPR,
               MemoryBazaar<RFLOAT, BaseType, 4>& datPI,
               MemoryBazaar<RFLOAT, BaseType, 4>& ctfP,
               MemoryBazaar<RFLOAT, BaseType, 4>& sigRcpP,
               RFLOAT *w,
               double *offS,
               int *nC,
               double *nR,
               double *nT,
               double *nD,
               CTFAttr* ctfaData,
               const int *iCol,
               const int *iRow,
               RFLOAT pixelSize,
               bool cSearch,
               int nk,
               int opf,
               int npxl,
               int mReco,
               int idim,
               int vdim,
               int imgNum);

void InsertFT(Volume& F3D,
              Volume& T3D,
              double* O3D,
              int* counter,
              MPI_Comm& hemi,
              MPI_Comm& slav,
              MemoryBazaar<RFLOAT, BaseType, 4>& datPR,
              MemoryBazaar<RFLOAT, BaseType, 4>& datPI,
              MemoryBazaar<RFLOAT, BaseType, 4>& ctfP,
              MemoryBazaar<RFLOAT, BaseType, 4>& sigRcpP,
              CTFAttr* ctfaData,
              double *offS,
              RFLOAT *w,
              double *nR,
              double *nT,
              double *nD,
              int *nC,
              const int *iCol,
              const int *iRow,
              RFLOAT pixelSize,
              bool cSearch,
              int opf,
              int npxl,
              int mReco,
              int idim,
              int dimSize,
              int imgNum);

void InsertFT(Volume& F3D,
              Volume& T3D,
              double* O3D,
              int* counter,
              MPI_Comm& hemi,
              MPI_Comm& slav,
              MemoryBazaar<RFLOAT, BaseType, 4>& datPR,
              MemoryBazaar<RFLOAT, BaseType, 4>& datPI,
              MemoryBazaar<RFLOAT, BaseType, 4>& ctfP,
              MemoryBazaar<RFLOAT, BaseType, 4>& sigRcpP,
              CTFAttr* ctfaData,
              double *offS,
              RFLOAT *w,
              double *nR,
              double *nT,
              double *nD,
              const int *iCol,
              const int *iRow,
              RFLOAT pixelSize,
              bool cSearch,
              int opf,
              int npxl,
              int mReco,
              int idim,
              int dimSize,
              int imgNum);

void PrepareTF(int gpuIdx,
               Volume& F3D,
	           Volume& T3D,
	           double* symMat,
               int nSymmetryElement,
	           int maxRadius,
	           int pf);

void ExposePT2D(int gpuIdx,
                RFLOAT* T2D,
	            int maxRadius,
	            int pf,
                int dim,
	            vec FSC,
	            bool joinHalf,
                const int wienerF);

void ExposePT(int gpuIdx,
              RFLOAT* T3D,
	          int maxRadius,
	          int pf,
              int dim,
	          vec FSC,
	          bool joinHalf,
              const int wienerF);

void ExposeWT2D(int gpuIdx,
                RFLOAT* T2D,
                RFLOAT* W2D,
                TabFunction& kernelRL,
                RFLOAT nf,
                int maxRadius,
                int pf,
                int dim,
                int maxIter,
                int minIter,
                int size);

void AllocDevicePoint(int gpuIdx,
                      Complex** dev_C,
                      RFLOAT** dev_W,
                      RFLOAT** dev_T,
                      RFLOAT** dev_tab,
                      RFLOAT** devDiff,
                      RFLOAT** devMax,
                      int** devCount,
                      void** stream,
                      int streamNum,
                      int tabSize,
                      int dim);

void HostDeviceInit(int gpuIdx,
                    Volume& C3D,
                    RFLOAT* W3D,
                    RFLOAT* T3D,
                    RFLOAT* tab,
                    RFLOAT* dev_W,
                    RFLOAT* dev_T,
                    RFLOAT* dev_tab,
                    void** stream,
                    int streamNum,
                    int tabSize,
                    int maxRadius,
                    int pf,
                    int dim);

void ExposeC(int gpuIdx,
             Volume& C3D,
             Complex* dev_C,
             RFLOAT* dev_T,
             RFLOAT* dev_W,
             void** stream,
             int streamNum,
             int dim);

void ExposeForConvC(int gpuIdx,
                    Volume& C3D,
                    Complex* dev_C,
                    RFLOAT* dev_tab,
                    void** stream,
                    TabFunction& kernelRL,
                    RFLOAT nf,
                    int streamNum,
                    int tabSize,
                    int pf,
                    int size);

void ExposeWC(int gpuIdx,
              Volume& C3D,
              Complex* dev_C,
              RFLOAT* diff,
              RFLOAT* cmax,
              RFLOAT* dev_W,
              RFLOAT* devDiff,
              RFLOAT* devMax,
              int *devCount,
              int* counter,
              void** stream,
              RFLOAT& diffC,
              int streamNum,
              int maxRadius,
              int pf);

void FreeDevHostPoint(int gpuIdx,
                      Complex** dev_C,
                      RFLOAT** dev_W,
                      RFLOAT** dev_T,
                      RFLOAT** dev_tab,
                      RFLOAT** devDiff,
                      RFLOAT** devMax,
                      int** devCount,
                      void** stream,
                      Volume& C3D,
                      RFLOAT* volumeW,
                      RFLOAT* volumeT,
                      int streamNum,
                      int dim);

void ExposeWT(int gpuIdx,
              RFLOAT* T3D,
              RFLOAT* W3D,
              TabFunction& kernelRL,
              RFLOAT nf,
              int maxRadius,
              int pf,
              int dim,
              int maxIter,
              int minIter,
              int size);

void ExposeWT2D(int gpuIdx,
                RFLOAT* T2D,
                RFLOAT* W2D,
                int maxRadius,
                int pf,
                int dim);

void ExposeWT(int gpuIdx,
              RFLOAT* T3D,
              RFLOAT* W3D,
              int maxRadius,
              int pf,
              int dim);

void ExposePF2D(int gpuIdx,
                Image& padDst,
                Image& padDstR,
                Image& F2D,
                RFLOAT* W2D,
                int maxRadius,
                int pf);

void ExposePFW(int gpuIdx,
               Volume& padDst,
               Volume& F3D,
               RFLOAT* W3D,
               int maxRadius,
               int pf);

void ExposePF(int gpuIdx,
              Volume& padDst,
              Volume& padDstR,
              Volume& F3D,
              RFLOAT* W3D,
              int maxRadius,
              int pf);

void ExposeCorrF2D(int gpuIdx,
                   Image& imgDst,
                   Volume& dst,
                   RFLOAT* mkbRL,
                   RFLOAT nf);

void ExposeCorrF(int gpuIdx,
                 Volume& dst,
                 RFLOAT* mkbRL,
                 RFLOAT nf);

void ExposeCorrF(int gpuIdx,
                 Volume& dstN,
                 Volume& dst,
                 RFLOAT* mkbRL,
                 RFLOAT nf);

void TranslateI2D(int gpuIdx,
                  Image& img,
                  double ox,
                  double oy,
                  int r);

void TranslateI(int gpuIdx,
                Volume& ref,
                double ox,
                double oy,
                double oz,
                int r);

void hostRegister(Complex* img,
                  int totalNum);

void hostFree(Complex* img);

void reMask(Complex* img,
            RFLOAT maskRadius,
            RFLOAT pixelSize,
            RFLOAT ew,
            int idim,
            int imgNum);

void GCTFinit(Complex* ctf,
              vector<CTFAttr>& ctfAttr,
              RFLOAT pixelSize,
              int idim,
              int shift,
              int imgNum);

#endif
