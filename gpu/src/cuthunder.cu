/*
 * FileName: cuthunder.cu
 * Author  : Kunpeng WANG，Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 ***************************************************************/
#include "cuthunder.h"

#include "Device.cuh"
#include "Complex.cuh"
#include "Image.cuh"
#include "Volume.cuh"
#include "TabFunction.cuh"
#include "Device.cuh"
#include "Constructor.cuh"
#include "Kernel.cuh"

#include "cufft.h"
#include "nccl.h"
#include <cuda.h>
#include <cfloat>
#include <cuda_profiler_api.h>

/* Index for two stream buffer. */
#define NUM_STREAM_PER_DEVICE 3
#define A 0
#define B 1

#define THREAD_PER_BLOCK 512
#define VOLUME_BATCH_2D 1024
#define VOLUME_BATCH_3D 8
#define TRANS_BATCH 32
#define ROT_BATCH 256
#define SHARE_LIMIT 256
#define IMAGE_BATCH 4096

/* Perf */
//#define PERF_SYNC_STREAM
//#define PERF_CALLBACK

namespace cuthunder {

////////////////////////////////////////////////////////////////

/**
 * Test rountines.
 *
 * ...
 */
void allocDeviceVolume(Volume& vol, int nCol, int nRow, int nSlc)
{
    vol.init(nCol, nRow, nSlc);

    Complex *dat;
    cudaMalloc((void**)&dat, vol.nSize() * sizeof(Complex));
    //cudaCheckErrors("Allocate device volume data.");

    vol.devPtr(dat);
}

__global__ void adder(Volume v1, Volume v2, Volume v3)
{
    int i = threadIdx.x - 4;
    int j = threadIdx.y - 4;
    int k = threadIdx.z - 4;

    Volume v(v2);

    Complex c = v1.getFT(i, j, k) + v2.getFT(i, j, k);
    v3.setFT(c, i, j, k);
}

void addTest()
{
    Volume v1, v2, v3;

    allocDeviceVolume(v1, 8, 8, 8);
    allocDeviceVolume(v2, 8, 8, 8);
    allocDeviceVolume(v3, 8, 8, 8);

    cudaSetDevice(0);
    //cudaCheckErrors("Set device error.");

    dim3 block(8, 8, 8);
    adder<<<1, block>>>(v1, v2, v3);
    //cudaCheckErrors("Lanch kernel adder.");

    cudaFree(v1.devPtr());
    //cudaCheckErrors("Free device volume memory 1.");
    cudaFree(v2.devPtr());
    //cudaCheckErrors("Free device volume memory 2.");
    cudaFree(v3.devPtr());
    //cudaCheckErrors("Free device volume memory 3.");
}


////////////////////////////////////////////////////////////////
//                     COMMON SUBROUTINES
//
//   The following routines are used by interface rountines to
// manipulate the data allocation, synchronization or transfer
// between host and device.
//

void readGPUPARA(char* gpuList,
                 vector<int>& iGPU,
                 int& nGPU)
{
    nGPU = 0;
    iGPU.clear();

    if (strlen(gpuList) == 0)
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        cudaCheckErrors("get devices num.");

        for (int i = 0; i < deviceCount; i++)
        {
            cudaDeviceProp deviceProperties;

            if (cudaGetDeviceProperties(&deviceProperties, i) == cudaSuccess)
            {
                if ((deviceProperties.major >= CUDA_MAJOR_MIN) &&
                    (deviceProperties.minor >= CUDA_MINOR_MIN))
                {
                    CLOG(INFO, "LOGGER_GPU") << "DEVICE #" << i
                                             << ", NAME : " << deviceProperties.name;
                    CLOG(INFO, "LOGGER_GPU") << "DEVICE #" << i
                                             << ", MEMORY : " << deviceProperties.totalGlobalMem / MEGABYTE << "MB";
                    CLOG(INFO, "LOGGER_GPU") << "DEVICE #" << i
                                             << ", CUDA CAPABILITY : " << deviceProperties.major << "." << deviceProperties.minor;

                    iGPU.push_back(i);
                    nGPU += 1;
                }
            }
        }
    }
    else
    {
        const char* split = ",";
        char* p = strtok(gpuList, split);
        if (p == NULL)
        {
            printf("json GPU format wrong\n");
            abort();
        }

        int gpuId;
        while (p != NULL)
        {
            sscanf(p, "%d", &gpuId);
            
            cudaDeviceProp deviceProperties;
            if (cudaGetDeviceProperties(&deviceProperties, gpuId) == cudaSuccess)
            {
                if ((deviceProperties.major >= CUDA_MAJOR_MIN) &&
                    (deviceProperties.minor >= CUDA_MINOR_MIN))
                {
                    CLOG(INFO, "LOGGER_GPU") << "DEVICE #" << gpuId
                                             << ", NAME : " << deviceProperties.name;
                    CLOG(INFO, "LOGGER_GPU") << "DEVICE #" << gpuId
                                             << ", MEMORY : " << deviceProperties.totalGlobalMem / MEGABYTE << "MB";
                    CLOG(INFO, "LOGGER_GPU") << "DEVICE #" << gpuId
                                             << ", CUDA CAPABILITY : " << deviceProperties.major << "." << deviceProperties.minor;

                    iGPU.push_back(gpuId);
                    nGPU += 1;
                }
            }

            p = strtok(NULL, split);
        }

        if (iGPU.size() == 0)
        {
            printf("No GPU availble!\n");
            abort();
        }
    }
    
}

void gpuCheck(vector<void*>& stream,
              vector<int>& iGPU,
              int& nGPU)
{
    CLOG(INFO, "LOGGER_GPU") << "NUMBER OF DEVICE FOR COMPUTING : " << nGPU;
    
    void* gpuStream[nGPU * NUM_STREAM_PER_DEVICE];
    int baseS;
    for (int n = 0; n < nGPU; n++)
    { 
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("stream create.");
        
        for(int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            gpuStream[i + baseS] = (cudaStream_t*)malloc(sizeof(cudaStream_t));
            cudaStreamCreate((cudaStream_t*)gpuStream[i + baseS]);
            cudaCheckErrors("stream create.");
            
            stream.push_back(gpuStream[i + baseS]);
        }
    }
}

void gpuEnvDestory(vector<void*>& stream,
                   vector<int>& iGPU,
                   int nGPU)
{
    int baseS;
    for (int n = 0; n < nGPU; n++)
    { 
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("stream create.");
        
        for(int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamDestroy(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Destroy stream.");
        }
    }

    stream.clear();
    iGPU.clear();
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void createVolume(Volume& vol,
                  int ndim,
                  VolCrtKind type,
                  const Complex *data = NULL)
{
    Complex *ptr = NULL;
    vol.init(ndim, ndim, ndim);

    if (type & HOST_ONLY){
        cudaHostAlloc((void**)&ptr,
                      vol.nSize() * sizeof(Complex),
                      cudaHostAllocPortable|cudaHostAllocWriteCombined);
        //cudaCheckErrors("Allocate page-lock volume data.");

        vol.hostPtr(ptr);

        if (data)
            memcpy(ptr, data, vol.nSize() * sizeof(Complex));
    }

    if (type & DEVICE_ONLY) {
        cudaMalloc((void**)&ptr, vol.nSize() * sizeof(Complex));
        //cudaCheckErrors("Allocate device volume data.");

        vol.devPtr(ptr);
    }

    if ((type & HD_SYNC) && (type & DEVICE_ONLY)) {
        if (data == NULL) return;

        cudaMemcpy((void*)vol.devPtr(),
                   (void*)data,
                   vol.nSize() * sizeof(Complex),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("Copy src volume data to device.");
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void freeVolume(Volume& vol)
{
    Complex *ptr;

    if (ptr = vol.hostPtr()) {
        cudaFreeHost(ptr);
        //cudaCheckErrors("Free host page-lock memory.");
    }

    if (ptr = vol.devPtr()) {
        cudaFree(ptr);
        //cudaCheckErrors("Free device memory.");
    }

    vol.clear();
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void setupSurfaceF(cudaArray *symArray,
                   cudaResourceDesc& resDesc,
                   cudaSurfaceObject_t& surfObject,
                   Complex *volume,
                   int dim)
{
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)volume, (dim / 2 + 1) * sizeof(int4), dim / 2 + 1, dim);
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    //cudaCheckErrors("copy F3D to device.");

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = symArray;

    cudaCreateSurfaceObject(&surfObject, &resDesc);
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void setupSurfaceT(cudaArray *symArray,
                   cudaResourceDesc& resDesc,
                   cudaSurfaceObject_t& surfObject,
                   double *volume,
                   int dim)
{
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)volume, (dim / 2 + 1) * sizeof(int2), dim / 2 + 1, dim);
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    //cudaCheckErrors("copy T3D to device.");

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = symArray;

    cudaCreateSurfaceObject(&surfObject, &resDesc);
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void reduceT3D(cudaArray* symArrayT[],
               cudaStream_t* stream,
               ncclComm_t* comm,
               double* T3D,
               int dimSize,
               int aviDevs,
               int nranks,
               int dim)
{
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);

    double *__device__T[aviDevs];

    for (int i = 0; i < aviDevs; i++)
    {
        cudaSetDevice(i);
        cudaMalloc((void**)&__device__T[i], dimSize * sizeof(double));
        //cudaCheckErrors("malloc normal __device__T.");

        cudaMemcpy3DParms copyParamsT = {0};
        copyParamsT.dstPtr   = make_cudaPitchedPtr((void*)__device__T[i], (dim / 2 + 1) * sizeof(int2), dim / 2 + 1, dim);
        copyParamsT.srcArray = symArrayT[i];
        copyParamsT.extent   = extent;
        copyParamsT.kind     = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParamsT);
        //cudaCheckErrors("copy T3D from array to normal.");
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < aviDevs; i++)
    {
        cudaSetDevice(i);
        NCCLCHECK(ncclReduce((const void*)__device__T[i],
                             (void*)__device__T[0],
                             dimSize,
                             ncclDouble,
                             ncclSum,
                             0,
                             comm[i],
                             stream[0 + i * NUM_STREAM_PER_DEVICE]));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int n = 0; n < aviDevs; ++n)
    {
        int baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(n);
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
            cudaStreamSynchronize(stream[i + baseS]);
    }

    if (nranks == 0)
    {
        cudaSetDevice(0);
        cudaMemcpy(T3D,
                   __device__T[0],
                   dimSize * sizeof(double),
                   cudaMemcpyDeviceToHost);
        //cudaCheckErrors("copy F3D from device to host.");
    }

    //free device buffers
    for (int i = 0; i < aviDevs; ++i)
    {
        cudaSetDevice(i);
        cudaFree(__device__T[i]);
    }

}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void reduceF3D(cudaArray* symArrayF[],
               cudaStream_t* stream,
               ncclComm_t* comm,
               Complex* F3D,
               int dimSize,
               int aviDevs,
               int nranks,
               int dim)
{
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);

    Complex *__device__F[aviDevs];

    for (int i = 0; i < aviDevs; i++)
    {
        cudaSetDevice(i);
        cudaMalloc((void**)&__device__F[i], dimSize * sizeof(Complex));
        //cudaCheckErrors("malloc normal __device__F.");

        cudaMemcpy3DParms copyParamsF = {0};
        copyParamsF.dstPtr   = make_cudaPitchedPtr((void*)__device__F[i], (dim / 2 + 1) * sizeof(int4), dim / 2 + 1, dim);
        copyParamsF.srcArray = symArrayF[i];
        copyParamsF.extent   = extent;
        copyParamsF.kind     = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParamsF);
        //cudaCheckErrors("copy T3D from array to normal.");
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < aviDevs; i++)
    {
        cudaSetDevice(i);
        NCCLCHECK(ncclReduce((const void*)__device__F[i],
                             (void*)__device__F[0],
                             dimSize * 2,
                             ncclDouble,
                             ncclSum,
                             0,
                             comm[i],
                             stream[0 + i * NUM_STREAM_PER_DEVICE]));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int n = 0; n < aviDevs; ++n)
    {
        int baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(n);
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
            cudaStreamSynchronize(stream[i + baseS]);
    }

    if (nranks == 0)
    {
        cudaSetDevice(0);
        cudaMemcpy(F3D,
                   __device__F[0],
                   dimSize * sizeof(double),
                   cudaMemcpyDeviceToHost);
        //cudaCheckErrors("copy F3D from device to host.");
    }

    //free device buffers
    for (int i = 0; i < aviDevs; ++i)
    {
        cudaSetDevice(i);
        cudaFree(__device__F[i]);
    }

}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocPGLKImagesBuffer(Complex **pglkptr, int ndim, int length)
{
    size_t size = length * (ndim / 2 + 1) * ndim;

    cudaHostAlloc((void**)pglkptr,
                  size * sizeof(Complex),
                  cudaHostAllocPortable|cudaHostAllocWriteCombined);
    //cudaCheckErrors("Alloc page-lock memory of batch size images.");
}

void allocPGLKRFLOATBuffer(RFLOAT **pglkptr, int length)
{
    cudaHostAlloc((void**)pglkptr,
                  length * sizeof(RFLOAT),
                  cudaHostAllocPortable|cudaHostAllocWriteCombined);
    cudaCheckErrors("Alloc page-lock memory of batch CTFAttr.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocPGLKFFTImagesBuffer(Complex **pglkptr, int nRow, int nCol, int length)
{
    size_t size = length * (nCol / 2 + 1) * nRow;

    cudaHostAlloc((void**)pglkptr,
                  size * sizeof(Complex),
                  cudaHostAllocPortable|cudaHostAllocWriteCombined);
    //cudaCheckErrors("Alloc page-lock memory of batch size images.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocPGLKCTFAttrBuffer(CTFAttr **pglkptr, int length)
{
    cudaHostAlloc((void**)pglkptr,
                  length * sizeof(CTFAttr),
                  cudaHostAllocPortable|cudaHostAllocWriteCombined);
    //cudaCheckErrors("Alloc page-lock memory of batch CTFAttr.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void updatePGLKImagesBuffer(Complex *pglkptr,
                            const vector<Complex*>& images,
                            int ndim,
                            int basePos,
                            int nImgBatch)
{
    size_t imageSize = (ndim / 2 + 1) * ndim;

    for (int i = 0; i < nImgBatch; i++) {
        memcpy((void*)(pglkptr + i * imageSize),
               (void*)(images[basePos + i]),
               imageSize * sizeof(Complex));
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void updatePGLKCTFAttrsBuffer(CTFAttr *pglkptr,
                              const vector<CTFAttr*>& ctfas,
                              int basePos,
                              int nImgBatch)
{
    for (int i = 0; i < nImgBatch; i++) {
        memcpy((void*)(pglkptr + i),
               (void*)(ctfas[basePos + i]),
               sizeof(CTFAttr));
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
typedef struct {
    Complex *pglkptr;
    vector<Complex*> *images;
    int imageSize;
    int nImgBatch;
    int basePos;
} CB_UPIB_t;

void CUDART_CB cbUpdatePGLKImagesBuffer(cudaStream_t stream,
                                        cudaError_t status,
                                        void *data)
{
    CB_UPIB_t *args = (CB_UPIB_t *)data;
    long long shift = 0;

    for (int i = 0; i < args->nImgBatch; i++) {
        shift = (long long)i * args->imageSize;
        memcpy((void*)(args->pglkptr + shift),
               (void*)(*args->images)[args->basePos + i],
               args->imageSize * sizeof(Complex));
    }
}

void CUDART_CB cbUpdateImagesBuffer(cudaStream_t stream,
                                    cudaError_t status,
                                    void *data)
{
    CB_UPIB_t *args = (CB_UPIB_t *)data;
    long long shift = 0;

    for (int i = 0; i < args->nImgBatch; i++) {
        shift = (long long)i * args->imageSize;
        memcpy((void*)(*args->images)[args->basePos + i],
               (void*)(args->pglkptr + shift),
               args->imageSize * sizeof(Complex));
    }
}

typedef struct {
    CTFAttr *pglkptr;
    vector<CTFAttr*> *ctfa;
    int nImgBatch;
    int basePos;
} CB_UPIB_ta;

void CUDART_CB cbUpdatePGLKCTFABuffer(cudaStream_t stream,
                                      cudaError_t status,
                                      void *data)
{
    CB_UPIB_ta *args = (CB_UPIB_ta *)data;

    for (int i = 0; i < args->nImgBatch; i++) {
        memcpy((void*)(args->pglkptr + i),
               (void*)(*args->ctfa)[args->basePos + i],
               sizeof(CTFAttr));
    }
}

typedef struct {
    RFLOAT *pglkptr;
    MemoryBazaar<RFLOAT, BaseType, 4> *data;
    int imageSize;
    int nImgBatch;
    int basePos;
} CB_UPIB_m;

void CUDART_CB cbUpdatePGLKRFLOAT(cudaStream_t stream,
                                  cudaError_t status,
                                  void *data)
{
    CB_UPIB_m *args = (CB_UPIB_m *)data;
    size_t oShift = 0;
    size_t nShift = 0;

    RFLOAT* temp;
    oShift = args->basePos * args->imageSize;
    for (int i = 0; i < args->nImgBatch; i++) {
        nShift = i * args->imageSize;
        temp = &((*(args->data))[oShift + nShift]);
        //temp = &((*(args->data))[2*args->imageSize]);
        memcpy((void*)(args->pglkptr + nShift),
               (void*)temp,
               args->imageSize * sizeof(RFLOAT));
    }

    //if (args->basePos == 0)
    //{
    //    RFLOAT *pglk_temp;
    //    for (int i = 0; i < args->nImgBatch; i++)
    //    {
    //        nShift = (long long)i * args->imageSize;
    //        oShift = (long long)args->basePos * args->imageSize;
    //        temp = &((*(args->data))[oShift + nShift]);
    //        pglk_temp = args->pglkptr + nShift;
    //        int j = 0;
    //        for (j = 0; j < args->imageSize; j++)
    //        {
    //            if (temp[j] - pglk_temp[j] >= 1e-5)
    //            {
    //                printf("j:%d, origin:%.6f, pglk:%.6f\n",j,temp[j],pglk_temp[j]);
    //                break;
    //            } 
    //        }
    //        if (j == args->imageSize)
    //            printf("successMemcpy:%d\n", i);
    //    }
    //}
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceCTFAttrBuffer(CTFAttr **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(CTFAttr));
    //cudaCheckErrors("Alloc device memory of batch CTF.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceComplexBuffer(Complex **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(Complex));
    //cudaCheckErrors("Alloc device memory of batch param.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceParamBuffer(RFLOAT **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(RFLOAT));
    //cudaCheckErrors("Alloc device memory of batch param.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceParamBufferD(double **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(double));
    //cudaCheckErrors("Alloc device memory of batch param.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceParamBufferI(int **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(int));
    //cudaCheckErrors("Alloc device memory of batch param.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceRandomBuffer(int **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(int));
    //cudaCheckErrors("Alloc device memory of batch random num.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void uploadTabFunction(TabFunction& tabfunc, const RFLOAT *tab)
{
    RFLOAT *devptr;

    cudaMalloc((void**)&devptr, tabfunc.size() * sizeof(RFLOAT));
    //cudaCheckErrors("Alloc device memory for tabfunction.");

    cudaMemcpy((void*)devptr,
               (void*)tab,
               tabfunc.size() * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    //cudaCheckErrors("Copy tabfunction to device.");

    tabfunc.devPtr(devptr);
}

////////////////////////////////////////////////////////////////
//                    RECONSTRUCTION ROUTINES
//
//   Below are interface rountines implemented to accelerate the
// reconstruction process.
//

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
                  int nImg)
{
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    //cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    int nStream = aviDevs * NUM_STREAM_PER_DEVICE;
    CTFAttr *pglk_ctfas_buf[nStream];

    CTFAttr *dev_ctfas_buf[nStream];
    RFLOAT *dev_def_buf[nStream];
    RFLOAT *dev_k1_buf[nStream];
    RFLOAT *dev_k2_buf[nStream];

    CB_UPIB_ta cbArgsA[nStream];

    int *deviCol[aviDevs];
    int *deviRow[aviDevs];

    LOG(INFO) << "Step1: alloc Memory.";

    cudaStream_t stream[nStream];

    int baseS;
    const int BATCH_SIZE = IMAGE_BUFF;

    for (int n = 0; n < aviDevs; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;

        cudaSetDevice(gpus[n]);

        cudaMalloc((void**)&deviCol[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iCol data.");

        cudaMalloc((void**)&deviRow[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iRow data.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            allocPGLKCTFAttrBuffer(&pglk_ctfas_buf[i + baseS], BATCH_SIZE);
            allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&dev_def_buf[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&dev_k1_buf[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&dev_k2_buf[i + baseS], BATCH_SIZE);

            cudaStreamCreate(&stream[i + baseS]);

        }
    }

    LOG(INFO) << "alloc memory done, begin to cpy...";

    for (int n = 0; n < aviDevs; ++n)
    {
        cudaSetDevice(gpus[n]);

        cudaMemcpy(deviCol[n],
                   iCol,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iCol.");

        cudaMemcpy(deviRow[n],
                   iRow,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iRow.");
    }

    LOG(INFO) << "Volume memcpy done...";

    int nImgBatch = 0, smidx = 0;

    for (int i = 0; i < nImg;)
    {
        for (int n = 0; n < aviDevs; ++n)
        {
            if (i >= nImg)
                break;

            baseS = n * NUM_STREAM_PER_DEVICE;
            nImgBatch = (i + BATCH_SIZE < nImg) ? BATCH_SIZE : (nImg - i);

            cudaSetDevice(gpus[n]);

            cbArgsA[smidx + baseS].pglkptr = pglk_ctfas_buf[smidx + baseS];
            cbArgsA[smidx + baseS].ctfa = &ctfaData;
            cbArgsA[smidx + baseS].nImgBatch = nImgBatch;
            cbArgsA[smidx + baseS].basePos = i;
            cudaStreamAddCallback(stream[smidx + baseS], cbUpdatePGLKCTFABuffer, (void*)&cbArgsA[smidx + baseS], 0);

            cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                            pglk_ctfas_buf[smidx + baseS],
                            nImgBatch * sizeof(CTFAttr),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy CTFAttr to device.");

            kernel_ExpectPrectf<<<nImgBatch,
                                  512,
                                  0,
                                  stream[smidx + baseS]>>>(dev_ctfas_buf[smidx + baseS],
                                                           dev_def_buf[smidx + baseS],
                                                           dev_k1_buf[smidx + baseS],
                                                           dev_k2_buf[smidx + baseS],
                                                           deviCol[n],
                                                           deviRow[n],
                                                           npxl);
            //cudaCheckErrors("kernel expectPrectf error.");

            cudaMemcpyAsync(def + i * npxl,
                            dev_def_buf[smidx + baseS],
                            nImgBatch * npxl * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy def to host.");

            cudaMemcpyAsync(k1 + i,
                            dev_k1_buf[smidx + baseS],
                            nImgBatch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy k1 to host.");

            cudaMemcpyAsync(k2 + i,
                            dev_k2_buf[smidx + baseS],
                            nImgBatch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy k2 to host.");

            i += nImgBatch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < aviDevs; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(gpus[n]);
        //cudaDeviceSynchronize();

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(stream[i + baseS]);
            //cudaCheckErrors("Stream synchronize.");

            cudaFreeHost(pglk_ctfas_buf[i + baseS]);
            cudaFree(dev_ctfas_buf[i + baseS]);
            cudaFree(dev_def_buf[i + baseS]);
            cudaFree(dev_k1_buf[i + baseS]);
            cudaFree(dev_k2_buf[i + baseS]);
        }
    }

    //free device buffers
    for (int n = 0; n < aviDevs; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(gpus[n]);

        cudaFree(deviCol[n]);
        cudaFree(deviRow[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
            cudaStreamDestroy(stream[i + baseS]);
    }

    delete[] gpus;
    LOG(INFO) << "expectationPre done.";
}

/**
 * @brief  Expectation GLobal.
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
                  int nGPU)
{
    LOG(INFO) << "expectation Rotation and Translate begin.";

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;
    double* dev_trans[nStream];
    double* devnR[nStream];
    
    int baseS = 0;
    for (int n = 0; n < nGPU; n++)
    { 
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("stream create.");
        
        cudaMalloc((void**)&devtraP[n], 
                   (long long)nT * npxl * sizeof(Complex));
        cudaCheckErrors("Allocate traP data.");

        cudaMalloc((void**)&devRotMat[n], 
                   nR * 9 * sizeof(double));
        cudaCheckErrors("Allocate rot data.");

        cudaMalloc((void**)&devpR[n], 
                   nR * sizeof(double));
        cudaCheckErrors("Allocate pR data.");

        cudaMalloc((void**)&devpT[n], 
                   nR * sizeof(double));
        cudaCheckErrors("Allocate pT data.");

        cudaMemcpy(devpR[n],
                   pR,
                   nR * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy pR.");

        cudaMemcpy(devpT[n],
                   pT,
                   nT * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy pT.");
        
        baseS = n * NUM_STREAM_PER_DEVICE;
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaMalloc((void**)&dev_trans[baseS + i], 
                       TRANS_BATCH * 2 * sizeof(double));
            cudaCheckErrors("Allocate trans data.");

            cudaMalloc((void**)&devnR[baseS + i], 
                       ROT_BATCH * 4 * sizeof(double));
            cudaCheckErrors("Allocate nR data.");

            cudaMalloc((void**)&devrotP[baseS + i], 
                       ROT_BATCH * npxl * sizeof(Complex));
            cudaCheckErrors("Allocate nR data.");
        }
    }

    int smidx = 0;
    int baseC = 0;
    int batch = 0;
    int block = 0;
    int thread = THREAD_PER_BLOCK;
    for (size_t i = 0; i < nT;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= nT)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + TRANS_BATCH > nT) 
                    ? (nT - i) : TRANS_BATCH;
            block = ((batch * npxl) % thread == 0)
                    ? (batch * npxl) / thread
                    : (batch * npxl) / thread + 1;

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(dev_trans[smidx + baseS],
                            trans + i * 2,
                            batch * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy trans.");

            kernel_Translate<<<block,
                               thread,
                               0,
                               *((cudaStream_t*)stream[smidx + baseS])>>>(devtraP[n] + i * npxl,
                                                                          dev_trans[smidx + baseS],
                                                                          deviCol[n],
                                                                          deviRow[n],
                                                                          idim,
                                                                          batch,
                                                                          npxl);
            cudaCheckErrors("kernel trans.");

            for (int card = 0; card < nGPU; card++)
            {
                if (card != n)
                {
                    baseC = card * NUM_STREAM_PER_DEVICE;
                    cudaSetDevice(iGPU[card]);
                    cudaCheckErrors("set device.");

                    cudaMemsetAsync(devtraP[card] + i * npxl,
                                    0.0,
                                    batch * npxl * sizeof(Complex),
                                    *((cudaStream_t*)stream[smidx + baseC]));
                    cudaCheckErrors("for memcpy.");
                }
            }

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    smidx = 0;
    thread = SHARE_LIMIT;
    for (size_t i = 0; i < nR;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= nR)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + ROT_BATCH > nR) 
                    ? (nR - i) : ROT_BATCH;
            block = (batch % thread == 0)
                    ? batch / thread
                    : batch / thread + 1;

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(devnR[smidx + baseS],
                            rot + i * 4,
                            batch * 4 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseC]));
            cudaCheckErrors("memcpy rot to device.");

            kernel_getRotMat<<<block,
                               thread,
                               thread * 18 * sizeof(double),
                               *((cudaStream_t*)stream[smidx + baseS])>>>(devRotMat[n] + i * 9,
                                                                          devnR[smidx + baseS],
                                                                          batch);
            cudaCheckErrors("getRotMat3D kernel.");

            for (int card = 0; card < nGPU; card++)
            {
                if (card != n)
                {
                    baseC = card * NUM_STREAM_PER_DEVICE;
                    cudaSetDevice(iGPU[card]);
                    cudaCheckErrors("set device.");
                    
                    cudaMemsetAsync(devRotMat[card] + i * 9,
                                    0.0,
                                    batch * 9 * sizeof(double),
                                    *((cudaStream_t*)stream[smidx + baseC]));
                    cudaCheckErrors("for memcpy.");
                }
            }

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    ncclComm_t commNccl[nGPU];
    int* gpus = new int[nGPU];
    for (int n = 0; n < nGPU; n++)
        gpus[n] = iGPU[n];
    NCCLCHECK(ncclCommInitAll(commNccl, nGPU, gpus));

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)devtraP[i],
                                (void*)devtraP[i],
                                nT * npxl,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commNccl[i],
                                *((cudaStream_t*)stream[i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)devRotMat[i],
                                (void*)devRotMat[i],
                                nR * 9,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commNccl[i],
                                *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    //finalizing NCCL
    for (int i = 0; i < nGPU; i++)
    {
        ncclCommDestroy(commNccl[i]);
    }

    delete[] gpus;
    
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(devnR[baseS + i]);
            cudaFree(dev_trans[baseS + i]);
            cudaCheckErrors("free RotMat.");
        }
    }

    LOG(INFO) << "expect Rotation and Translate done.";
}

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
                    int nGPU)
{
    LOG(INFO) << "expectation Rotation and Translate begin.";

    int dimSize = (vdim / 2 + 1) * vdim;
    cudaHostRegister(volume, 
                     nK * dimSize * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register volume data.");

    cudaHostRegister(rot, 
                     nR * 2 * sizeof(double), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register rot data.");

#ifdef SINGLE_PRECISION
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0,cudaChannelFormatKindFloat);
#else
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindSigned);
#endif
    struct cudaResourceDesc resDesc[nGPU * nK];
    void* symVoid[nGPU * nK];
    void* texVoid[nGPU * nK];

    double* dev_trans[nGPU];

    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        for (int k = 0; k < nK; k++)
        {
            symVoid[k + n * nK] = (cudaArray**)malloc(sizeof(cudaArray*));
            cudaMallocArray((cudaArray**)symVoid[k + n * nK], 
                            &channelDesc, 
                            vdim / 2 + 1, 
                            vdim);
            cudaCheckErrors("Allocate symArray data.");
            symArray.push_back(symVoid[k + n * nK]);
        }

        cudaMalloc((void**)&devtraP[n], 
                   nT * npxl * sizeof(Complex));
        cudaCheckErrors("Allocate traP data.");

        cudaMalloc((void**)&dev_trans[n], 
                   nT * 2 * sizeof(double));
        cudaCheckErrors("Allocate trans data.");

        cudaMalloc((void**)&devnR[n], 
                   nR * 2 * sizeof(double));
        cudaCheckErrors("Allocate nR data.");

        cudaMalloc((void**)&devpR[n], 
                   nR * sizeof(double));
        cudaCheckErrors("Allocate pR data.");

        cudaMalloc((void**)&devpT[n], 
                   nT * sizeof(double));
        cudaCheckErrors("Allocate pT data.");
    }
     
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMemcpy(dev_trans[n],
                   trans,
                   nT * 2 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy trans.");

        cudaMemcpy(devpR[n],
                   pR,
                   nR * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy pR.");

        cudaMemcpy(devpT[n],
                   pT,
                   nT * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy pT.");

    }

    int thread = THREAD_PER_BLOCK;
    int block = ((nT * npxl) % thread == 0)
              ? (nT * npxl) / thread
              : (nT * npxl) / thread + 1;
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        cudaMemcpyAsync(devnR[n],
                        rot,
                        nR * 2 * sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)stream[0 + baseS]));
        cudaCheckErrors("memcpy rot to device.");

        for (int k = 0; k < nK; k++)
        {
            cudaMemcpyToArrayAsync(*((cudaArray**)symVoid[k + n * nK]),
                                   0,
                                   0,
                                   (void*)(volume + k * dimSize),
                                   sizeof(Complex) * dimSize,
                                   cudaMemcpyHostToDevice,
                                   *((cudaStream_t*)stream[1 + baseS]));
            cudaCheckErrors("memcpy array error");
        }
        
        kernel_Translate<<<block,
                           thread,
                           0,
                           *((cudaStream_t*)stream[2 + baseS])>>>(devtraP[n],
                                                                  dev_trans[n],
                                                                  deviCol[n],
                                                                  deviRow[n],
                                                                  idim,
                                                                  nT,
                                                                  npxl);
        cudaCheckErrors("kernel trans.");
    }

    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;

    for (int n = 0; n < nGPU; ++n)
    {
        for (int k = 0; k < nK; k++)
        {
            memset(&resDesc[k + n * nK], 0, sizeof(resDesc[0]));
            resDesc[k + n * nK].resType = cudaResourceTypeArray;
            resDesc[k + n * nK].res.array.array = *((cudaArray**)symVoid[k + n * nK]);

            cudaSetDevice(iGPU[n]);
            texVoid[k + n * nK] = (cudaTextureObject_t*)malloc(sizeof(cudaTextureObject_t));
            cudaCreateTextureObject((cudaTextureObject_t*)texVoid[k + n * nK], 
                                    &resDesc[k + n * nK], 
                                    &td, 
                                    NULL);
            cudaCheckErrors("create TexObject.");
            texObject.push_back(texVoid[k + n * nK]);
        }
    }

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("device synchronize.");
        }

        cudaFree(dev_trans[n]);
        cudaCheckErrors("Free tran.");
    }

    cudaHostUnregister(volume);
    cudaCheckErrors("Unregister vol.");
    cudaHostUnregister(rot);
    cudaCheckErrors("Unregister rot.");

    LOG(INFO) << "expect Rotation and Translate done.";
}

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
                   int nGPU)
{
    LOG(INFO) << "expectation Projection begin.";

    cudaHostRegister(rotP, 
                     (long long)nR * npxl * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register volume data.");

#ifdef SINGLE_PRECISION
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
#else
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
#endif
    cudaExtent extent = make_cudaExtent(vdim / 2 + 1, vdim, vdim);
    cudaArray *symArray[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        
        cudaMalloc3DArray(&symArray[n], 
                          &channelDesc, 
                          extent);
        cudaCheckErrors("malloc error\n.");
    }

    cudaMemcpy3DParms tempP[nGPU];
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        
        memset(&tempP[n], 0, sizeof(tempP[n]));
#ifdef SINGLE_PRECISION
        tempP[n].srcPtr   = make_cudaPitchedPtr((void*)volume, 
                                                (vdim / 2 + 1) * sizeof(float2), 
                                                vdim / 2 + 1, 
                                                vdim);
#else
        tempP[n].srcPtr   = make_cudaPitchedPtr((void*)volume, 
                                                (vdim / 2 + 1) * sizeof(int4), 
                                                vdim / 2 + 1, 
                                                vdim);
#endif
        tempP[n].dstArray = symArray[n];
        tempP[n].extent   = extent;
        tempP[n].kind     = cudaMemcpyHostToDevice;

        cudaMemcpy3DAsync(&tempP[n],
                          *((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("memcpy error\n.");
    }

    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;

    struct cudaResourceDesc resDesc[nGPU];
    cudaTextureObject_t texObject[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        
        memset(&resDesc[n], 0, sizeof(resDesc[n]));
        resDesc[n].resType = cudaResourceTypeArray;
        resDesc[n].res.array.array = symArray[n];

        cudaCreateTextureObject(&texObject[n], 
                                &resDesc[n], 
                                &td, 
                                NULL);
        cudaCheckErrors("create Texture object\n.");
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    LOG(INFO) << "Projection begin.";

    int smidx = 0;
    int batch = 0;
    int block = 0;
    int thread = THREAD_PER_BLOCK;
    for (size_t i = 0; i < nR;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= nR)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + ROT_BATCH > nR) 
                    ? (nR - i) : ROT_BATCH;
            block = ((batch * npxl) % thread == 0)
                    ? (batch * npxl) / thread
                    : (batch * npxl) / thread + 1;

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            kernel_Project3D<<<block,
                               thread,
                               0,
                               *((cudaStream_t*)stream[smidx + baseS])>>>(devrotP[smidx + baseS],
                                                                          devRotMat[n] + i * 9,
                                                                          deviCol[n],
                                                                          deviRow[n],
                                                                          batch,
                                                                          pf,
                                                                          vdim,
                                                                          npxl,
                                                                          interp,
                                                                          texObject[n]);
            cudaCheckErrors("Project3D kernel.");

            cudaMemcpyAsync(rotP + (long long)i * npxl,
                            devrotP[smidx + baseS],
                            batch * npxl * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy rotP to host.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaDestroyTextureObject(texObject[n]);
        cudaFreeArray(symArray[n]);
        cudaCheckErrors("Free device memory SymArray.");
    }

    LOG(INFO) << "Projection done.";

    //unregister pglk_memory
    cudaHostUnregister(rotP);
    cudaCheckErrors("unregister rotP.");
}

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
                    int nGPU)
{
    LOG(INFO) << "expectation Global begin.";

    const int BATCH_SIZE = IMAGE_BUFF;
    int nStream = nGPU * NUM_STREAM_PER_DEVICE;

    cudaHostRegister(wC, 
                     imgNum * nK * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register wC data.");

    cudaHostRegister(wR, 
                     imgNum * nK * nR * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register wR data.");

    cudaHostRegister(wT, 
                     imgNum * nK * nT * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register wT data.");

    RFLOAT* devdatPR[nStream];
    RFLOAT* devdatPI[nStream];
    Complex* priRotP[nStream];
    RFLOAT* devctfP[nStream];
    RFLOAT* devsigP[nStream];
    RFLOAT* devDvp[nStream];
    RFLOAT* devbaseL[nStream];
    RFLOAT* devwC[nStream];
    RFLOAT* devwR[nStream];
    RFLOAT* devwT[nStream];
    RFLOAT* devcomP[nStream];

    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            allocDeviceComplexBuffer(&priRotP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devdatPR[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devdatPI[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devsigP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devDvp[i + baseS], BATCH_SIZE * nR * nT);
            allocDeviceParamBuffer(&devbaseL[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&devwC[i + baseS], BATCH_SIZE * nK);
            allocDeviceParamBuffer(&devwR[i + baseS], BATCH_SIZE * nK * nR);
            allocDeviceParamBuffer(&devwT[i + baseS], BATCH_SIZE * nK * nT);

            if (nK != 1)
            {
                allocDeviceParamBuffer(&devcomP[i + baseS], BATCH_SIZE);
            }
        }
    }

    //RFLOAT *pglk_datPR = (RFLOAT*)malloc(IMAGE_BATCH * npxl * sizeof(RFLOAT));
    //RFLOAT *pglk_datPI = (RFLOAT*)malloc(IMAGE_BATCH * npxl * sizeof(RFLOAT));
    //RFLOAT *pglk_ctfP = (RFLOAT*)malloc(IMAGE_BATCH * npxl * sizeof(RFLOAT));
    //RFLOAT *pglk_sigRcpP = (RFLOAT*)malloc(IMAGE_BATCH * npxl * sizeof(RFLOAT));
    //
    //cudaHostRegister(pglk_datPR, IMAGE_BATCH * npxl * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register datP data.");

    //cudaHostRegister(pglk_datPI, IMAGE_BATCH * npxl * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register datP data.");

    //cudaHostRegister(pglk_ctfP, IMAGE_BATCH * npxl * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register ctfP data.");

    //cudaHostRegister(pglk_sigRcpP, IMAGE_BATCH * npxl * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register sigRcpP data.");
    
    int thread = THREAD_PER_BLOCK;
    //TODO: 49152 change to special GPU device & threadS should not be 0
    int threadS = (49152 / (sizeof(RFLOAT)) / (nT + 1));
    int block = ((nT * npxl) % thread == 0)
              ? (nT * npxl) / thread
              : (nT * npxl) / thread + 1;
    int batch = 0, rbatch = 0, smidx = 0;
    //int imgBatch = 0;
    //for (int l = 0; l < imgNum;)
    //{
    //    if (l >= imgNum)
    //        break;

    //    imgBatch = (l + IMAGE_BATCH < imgNum)
    //             ? IMAGE_BATCH : (imgNum - l);

    //    RFLOAT *temp_datPR;
    //    RFLOAT *temp_datPI;
    //    RFLOAT *temp_ctfP;
    //    RFLOAT *temp_sigP;
    //    
    //    for (int i = 0; i < imgBatch; i++) 
    //    {
    //        temp_datPR = &datPR[(l + i) * npxl];
    //        temp_datPI = &datPI[(l + i) * npxl];
    //        temp_ctfP = &ctfP[(l + i) * npxl];
    //        temp_sigP = &sigRcpP[(l + i) * npxl];
    //        memcpy((void*)(pglk_datPR + i * npxl),
    //               (void*)temp_datPR,
    //               npxl * sizeof(RFLOAT));
    //        memcpy((void*)(pglk_datPI + i * npxl),
    //               (void*)temp_datPI,
    //               npxl * sizeof(RFLOAT));
    //        memcpy((void*)(pglk_ctfP + i * npxl),
    //               (void*)temp_ctfP,
    //               npxl * sizeof(RFLOAT));
    //        memcpy((void*)(pglk_sigRcpP + i * npxl),
    //               (void*)temp_sigP,
    //               npxl * sizeof(RFLOAT));
    //    }

        smidx = 0;
        for (int i = 0; i < imgNum;)
        {
            for (int n = 0; n < nGPU; ++n)
            {
                if (i >= imgNum)
                    break;

                baseS = n * NUM_STREAM_PER_DEVICE;
                batch = (i + BATCH_SIZE < imgNum) 
                      ? BATCH_SIZE : (imgNum - i);

                cudaSetDevice(iGPU[n]);

                cudaMemcpyAsync(devdatPR[smidx + baseS],
                                pglk_datPR + i * npxl,
                                batch * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy datP to device.");

                cudaMemcpyAsync(devdatPI[smidx + baseS],
                                pglk_datPI + i * npxl,
                                batch * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy datP to device.");

                cudaMemcpyAsync(devctfP[smidx + baseS],
                                pglk_ctfP + i * npxl,
                                batch * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy ctfP to device.");

                cudaMemcpyAsync(devsigP[smidx + baseS],
                                pglk_sigRcpP + i * npxl,
                                batch * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy sigP to device.");

                cudaMemsetAsync(devwC[smidx + baseS],
                                0.0,
                                batch * nK * sizeof(RFLOAT),
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wC.");

                cudaMemsetAsync(devwR[smidx + baseS],
                                0.0,
                                batch * nK * nR * sizeof(RFLOAT),
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wR.");

                cudaMemsetAsync(devwT[smidx + baseS],
                                0.0,
                                batch * nK * nT * sizeof(RFLOAT),
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wT.");

                for (int k = 0; k < nK; k++)
                {
                    for (int r = 0; r < nR;)
                    {
                        rbatch = (r + ROT_BATCH < nR) ? ROT_BATCH : (nR - r);

                        block = ((rbatch * npxl) % thread == 0)
                              ? (rbatch * npxl) / thread
                              : (rbatch * npxl) / thread + 1;
                        
                        kernel_Project2D<<<block,
                                           thread,
                                           0,
                                           *((cudaStream_t*)stream[smidx + baseS])>>>(priRotP[smidx + baseS],
                                                                                      devnR[n] + r * 2,
                                                                                      deviCol[n],
                                                                                      deviRow[n],
                                                                                      rbatch,
                                                                                      pf,
                                                                                      vdim,
                                                                                      npxl,
                                                                                      interp,
                                                                                      *((cudaTextureObject_t*)texObject[k + n * nK]));
                        cudaCheckErrors("Project2D error.");

                        kernel_logDataVS<<<rbatch * batch * nT,
                                           64,
                                           64 * sizeof(RFLOAT),
                                           *((cudaStream_t*)stream[smidx + baseS])>>>(devdatPR[smidx + baseS],
                                                                                      devdatPI[smidx + baseS],
                                                                                      priRotP[smidx + baseS],
                                                                                      devtraP[n],
                                                                                      devctfP[smidx + baseS],
                                                                                      devsigP[smidx + baseS],
                                                                                      devDvp[smidx + baseS],
                                                                                      r,
                                                                                      nR,
                                                                                      nT,
                                                                                      rbatch,
                                                                                      npxl);
                        cudaCheckErrors("logDataVS error.");

                        r += rbatch;
                    }

                    if (k == 0)
                    {
                        kernel_getMaxBase<<<batch,
                                            thread,
                                            thread * sizeof(RFLOAT),
                                            *((cudaStream_t*)stream[smidx + baseS])>>>(devbaseL[smidx + baseS],
                                                                                       devDvp[smidx + baseS],
                                                                                       nR * nT);
                        cudaCheckErrors("getMaxBase error.");
                    }
                    else
                    {
                        kernel_getMaxBase<<<batch,
                                            thread,
                                            thread * sizeof(RFLOAT),
                                            *((cudaStream_t*)stream[smidx + baseS])>>>(devcomP[smidx + baseS],
                                                                                       devDvp[smidx + baseS],
                                                                                       nR * nT);
                        cudaCheckErrors("getMaxBase error.");

                        kernel_setBaseLine<<<batch,
                                             thread,
                                             0,
                                             *((cudaStream_t*)stream[smidx + baseS])>>>(devcomP[smidx + baseS],
                                                                                        devbaseL[smidx + baseS],
                                                                                        devwC[smidx + baseS],
                                                                                        devwR[smidx + baseS],
                                                                                        devwT[smidx + baseS],
                                                                                        nK,
                                                                                        nR,
                                                                                        nT,
                                                                                        nK * (1 + nR + nT));
                        cudaCheckErrors("setBaseLine error.");
                    }

                    //printf("batch:%d, threadS:%d, nT:%d, share:%d\n", batch, threadS, nT, (nT + 1) * threadS);
                    kernel_UpdateW<<<batch,
                                     threadS,
                                     (nT + 1) * threadS * sizeof(RFLOAT),
                                     *((cudaStream_t*)stream[smidx + baseS])>>>(devDvp[smidx + baseS],
                                                                                devbaseL[smidx + baseS],
                                                                                devwC[smidx + baseS],
                                                                                devwR[smidx + baseS],
                                                                                devwT[smidx + baseS],
                                                                                devpR[n],
                                                                                devpT[n],
                                                                                k,
                                                                                nK,
                                                                                nR,
                                                                                nT,
                                                                                nR * nT);
                    cudaCheckErrors("Update error.");

                }

                cudaMemcpyAsync(wC + i * nK,
                                devwC[smidx + baseS],
                                batch * nK * sizeof(RFLOAT),
                                cudaMemcpyDeviceToHost,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy wC to host.");

                cudaMemcpyAsync(wR + (long long)i * nK * nR,
                                devwR[smidx + baseS],
                                batch * nR  * nK * sizeof(RFLOAT),
                                cudaMemcpyDeviceToHost,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy wR to host.");

                cudaMemcpyAsync(wT + (long long)i * nK * nT,
                                devwT[smidx + baseS],
                                batch * nT * nK * sizeof(RFLOAT),
                                cudaMemcpyDeviceToHost,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy wT to host.");

                i += batch;
            }

            smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
        }
    
    //    //synchronizing on CUDA streams
    //    for (int n = 0; n < nGPU; ++n)
    //    {
    //        baseS = n * NUM_STREAM_PER_DEVICE;
    //        cudaSetDevice(iGPU[n]);

    //        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
    //        {
    //            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
    //            cudaCheckErrors("Stream synchronize after.");
    //        }
    //    }

    //    l += imgBatch;
    //}

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");

            cudaFree(devdatPR[i + baseS]);
            cudaFree(devdatPI[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(devsigP[i + baseS]);
            cudaFree(priRotP[i + baseS]);
            cudaFree(devDvp[i + baseS]);
            cudaFree(devbaseL[i + baseS]);
            cudaFree(devwC[i + baseS]);
            cudaFree(devwR[i + baseS]);
            cudaFree(devwT[i + baseS]);
            if (nK != 1)
            {
                cudaFree(devcomP[i + baseS]);
            }
            cudaCheckErrors("cuda Free error.");
        }
    }

    //unregister pglk_memory
    //cudaHostUnregister(pglk_datPR);
    //cudaHostUnregister(pglk_datPI);
    //cudaHostUnregister(pglk_ctfP);
    //cudaHostUnregister(pglk_sigRcpP);
    //free(pglk_datPR);
    //free(pglk_datPI);
    //free(pglk_ctfP);
    //free(pglk_sigRcpP);
    cudaHostUnregister(wC);
    cudaHostUnregister(wR);
    cudaHostUnregister(wT);
    cudaCheckErrors("cuda Host Unregister error.");

    LOG(INFO) << "expectation Global done.";
}

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
                    int kIdx,
                    int nK,
                    int nR,
                    int nT,
                    int npxl,
                    int imgNum,
                    int nGPU)
{
    LOG(INFO) << "expectation Global begin.";

    const int BATCH_SIZE = IMAGE_BUFF;

    cudaHostRegister(rotP, 
                     (long long)nR * npxl * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register datP data.");

    cudaHostRegister(baseL, 
                     imgNum * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register wC data.");

    cudaHostRegister(wC, 
                     imgNum * nK * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register wC data.");

    cudaHostRegister(wR, 
                     (long long)imgNum * nK * nR * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register wR data.");

    cudaHostRegister(wT, 
                     (long long)imgNum * nK * nT * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register wT data.");

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;

    RFLOAT* devdatPR[nStream];
    RFLOAT* devdatPI[nStream];
    RFLOAT* devctfP[nStream];
    RFLOAT* devsigP[nStream];
    RFLOAT* devDvp[nStream];
    RFLOAT* devbaseL[nStream];
    RFLOAT* devwC[nStream];
    RFLOAT* devwR[nStream];
    RFLOAT* devwT[nStream];
    RFLOAT* devcomP[nStream];

    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            allocDeviceParamBuffer(&devdatPR[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devdatPI[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devsigP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devDvp[i + baseS], BATCH_SIZE * nR * nT);
            allocDeviceParamBuffer(&devbaseL[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&devwC[i + baseS], BATCH_SIZE * nK);
            allocDeviceParamBuffer(&devwR[i + baseS], BATCH_SIZE * nK * nR);
            allocDeviceParamBuffer(&devwT[i + baseS], BATCH_SIZE * nK * nT);

            if (kIdx != 0)
            {
                allocDeviceParamBuffer(&devcomP[i + baseS], BATCH_SIZE);
            }
        }
    }

    int batch = 0, rbatch = 0, smidx = 0;
    int thread = THREAD_PER_BLOCK;
    //int threadS = SHARE_LIMIT;
    int threadS = (49152 / (sizeof(RFLOAT)) / (nT + 1));
    for (int i = 0; i < imgNum;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= imgNum)
                break;

            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + BATCH_SIZE < imgNum) 
                  ? BATCH_SIZE : (imgNum - i);

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(devdatPR[smidx + baseS],
                            pglk_datPR + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy datP to device.");

            cudaMemcpyAsync(devdatPI[smidx + baseS],
                            pglk_datPI + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy datP to device.");
            
            cudaMemcpyAsync(devctfP[smidx + baseS],
                            pglk_ctfP + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy ctfP to device.");

            cudaMemcpyAsync(devsigP[smidx + baseS],
                            pglk_sigRcpP + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy sigP to device.");

            if (kIdx == 0)
            {
                cudaMemsetAsync(devwC[smidx + baseS],
                                0.0,
                                batch * nK * sizeof(RFLOAT),
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wC.");

                cudaMemsetAsync(devwR[smidx + baseS],
                                0.0,
                                batch * nK * nR * sizeof(RFLOAT),
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wR.");

                cudaMemsetAsync(devwT[smidx + baseS],
                                0.0,
                                batch * nK * nT * sizeof(RFLOAT),
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wT.");
            }
            else
            {
                cudaMemcpyAsync(devbaseL[smidx + baseS],
                                baseL + i,
                                batch * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset baseL.");

                cudaMemcpyAsync(devwC[smidx + baseS],
                                wC + i * nK,
                                batch * nK * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wC.");

                cudaMemcpyAsync(devwR[smidx + baseS],
                                wR + (long long)i * nR,
                                batch * nK * nR * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wR.");

                cudaMemcpyAsync(devwT[smidx + baseS],
                                wT + (long long)i * nT,
                                batch * nK * nT * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memset wT.");
            }

            for (int r = 0; r < nR;)
            {
                rbatch = (r + ROT_BATCH < nR) ? ROT_BATCH : (nR - r);

                cudaMemcpyAsync(devrotP[smidx + baseS],
                                rotP + (long long)r * npxl,
                                rbatch * npxl * sizeof(Complex),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy datP to device.");

                kernel_logDataVS<<<rbatch * batch * nT,
                                   64,
                                   64 * sizeof(RFLOAT),
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(devdatPR[smidx + baseS],
                                                                              devdatPI[smidx + baseS],
                                                                              devrotP[smidx + baseS],
                                                                              devtraP[n],
                                                                              devctfP[smidx + baseS],
                                                                              devsigP[smidx + baseS],
                                                                              devDvp[smidx + baseS],
                                                                              r,
                                                                              nR,
                                                                              nT,
                                                                              rbatch,
                                                                              npxl);
                cudaCheckErrors("logDataVS error.");

                r += rbatch;
            }

            if (kIdx == 0)
            {
                kernel_getMaxBase<<<batch,
                                    thread,
                                    thread * sizeof(RFLOAT),
                                    *((cudaStream_t*)stream[smidx + baseS])>>>(devbaseL[smidx + baseS],
                                                                               devDvp[smidx + baseS],
                                                                               nR * nT);
                cudaCheckErrors("getMaxBase error.");
            }
            else
            {
                kernel_getMaxBase<<<batch,
                                    thread,
                                    thread * sizeof(RFLOAT),
                                    *((cudaStream_t*)stream[smidx + baseS])>>>(devcomP[smidx + baseS],
                                                                               devDvp[smidx + baseS],
                                                                               nR * nT);
                cudaCheckErrors("getMaxBase error.");

                kernel_setBaseLine<<<batch,
                                     thread,
                                     0,
                                     *((cudaStream_t*)stream[smidx + baseS])>>>(devcomP[smidx + baseS],
                                                                                devbaseL[smidx + baseS],
                                                                                devwC[smidx + baseS],
                                                                                devwR[smidx + baseS],
                                                                                devwT[smidx + baseS],
                                                                                nK,
                                                                                nR,
                                                                                nT,
                                                                                nK * (1 + nR + nT));
                cudaCheckErrors("setBaseLine error.");
            }

            kernel_UpdateW<<<batch,
                             threadS,
                             (nT + 1) * threadS * sizeof(RFLOAT),
                             *((cudaStream_t*)stream[smidx + baseS])>>>(devDvp[smidx + baseS],
                                                                        devbaseL[smidx + baseS],
                                                                        devwC[smidx + baseS],
                                                                        devwR[smidx + baseS],
                                                                        devwT[smidx + baseS],
                                                                        devpR[n],
                                                                        devpT[n],
                                                                        kIdx,
                                                                        nK,
                                                                        nR,
                                                                        nT,
                                                                        nR * nT);
            cudaCheckErrors("UpdateW error.");

            cudaMemcpyAsync(baseL + i,
                            devbaseL[smidx + baseS],
                            batch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy baseL to host.");

            cudaMemcpyAsync(wC + i * nK,
                            devwC[smidx + baseS],
                            batch * nK * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy wC to host.");

            cudaMemcpyAsync(wR + (long long)i * nK * nR,
                            devwR[smidx + baseS],
                            batch * nK * nR * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy wR to host.");

            cudaMemcpyAsync(wT + (long long)i * nK * nT,
                            devwT[smidx + baseS],
                            batch * nK * nT * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy wT to host.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }
    
    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");

            if (kIdx != 0)
            {
                cudaFree(devcomP[i + baseS]);
            }
            cudaFree(devdatPR[i + baseS]);
            cudaFree(devdatPI[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(devsigP[i + baseS]);
            cudaFree(devDvp[i + baseS]);
            cudaFree(devbaseL[i + baseS]);
            cudaFree(devwC[i + baseS]);
            cudaFree(devwR[i + baseS]);
            cudaFree(devwT[i + baseS]);
        }
    }

    //unregister pglk_memory
    cudaHostUnregister(rotP);
    cudaHostUnregister(baseL);
    cudaHostUnregister(wC);
    cudaHostUnregister(wR);
    cudaHostUnregister(wT);
    cudaCheckErrors("unregister rot.");

    LOG(INFO) << "expectation Global done.";
}

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
                  int nGPU)
{
    LOG(INFO) << "expectation Rotation and Translate begin.";

    for (int n = 0; n < nGPU; n++)
    { 
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("stream create.");
        
        cudaFree(devtraP[n]);
        cudaCheckErrors("free traP data.");

        cudaFree(devnR[n]);
        cudaCheckErrors("free RotMat data.");

        cudaFree(devpR[n]);
        cudaCheckErrors("free pR.");
        
        cudaFree(devpT[n]);
        cudaCheckErrors("free pT.");
        
        for (int k = 0; k < nK; k++)
        {
            cudaFreeArray(*((cudaArray**)symArray[k + n * nK]));
            cudaDestroyTextureObject(*((cudaTextureObject_t*)texObject[k + n * nK]));
            cudaCheckErrors("cuda Destory texobject error.");
        }
    }

    symArray.clear();
    texObject.clear();
}

/**
 * @brief  Expectation GLobal.
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
                int nGPU)
{
    LOG(INFO) << "expectation Rotation and Translate begin.";

    int baseS = 0;
    for (int n = 0; n < nGPU; n++)
    { 
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("stream create.");
        
        cudaFree(devtraP[n]);
        cudaCheckErrors("free traP data.");

        cudaFree(devRotMat[n]);
        cudaCheckErrors("free RotMat data.");

        cudaFree(devpR[n]);
        cudaCheckErrors("free pR.");
        
        cudaFree(devpT[n]);
        cudaCheckErrors("free pT.");
        
        baseS = n * NUM_STREAM_PER_DEVICE;
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(devrotP[baseS + i]);
            cudaCheckErrors("free rotP data.");
        }
    }

}

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
                  int npxl)
{
    cudaSetDevice(gpuIdx);

    cudaMalloc((void**)deviCol, npxl * sizeof(int));
    cudaCheckErrors("Allocate iCol data.");

    cudaMalloc((void**)deviRow, npxl * sizeof(int));
    cudaCheckErrors("Allocate iRow data.");

    cudaMemcpy(*deviCol,
               iCol,
               npxl * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy iCol.");

    cudaMemcpy(*deviRow,
               iRow,
               npxl * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy iRow.");
}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectPrefre(int gpuIdx,
                  RFLOAT** devfreQ,
                  RFLOAT* freQ,
                  int npxl)
{
    cudaSetDevice(gpuIdx);

    cudaMalloc((void**)devfreQ, npxl * sizeof(RFLOAT));
    cudaCheckErrors("Allocate freQ data.");

    cudaMemcpy(*devfreQ,
               freQ,
               npxl * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy freQ.");
}

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
                   int cSearch)
{
    cudaSetDevice(gpuIdx);

    cudaMalloc((void**)devdatPR, cpyNum * npxl * sizeof(Complex));
    cudaCheckErrors("Allocate datP data.");

    cudaMalloc((void**)devdatPI, cpyNum * npxl * sizeof(Complex));
    cudaCheckErrors("Allocate datP data.");

    if (cSearch != 2)
    {
        cudaMalloc((void**)devctfP, cpyNum * npxl * sizeof(RFLOAT));
        cudaCheckErrors("Allocate ctfP data.");
    }
    else
    {
        cudaMalloc((void**)devdefO, cpyNum * npxl * sizeof(RFLOAT));
        cudaCheckErrors("Allocate defocus data.");
    }

    cudaMalloc((void**)devsigP, cpyNum * npxl * sizeof(RFLOAT));
    cudaCheckErrors("Allocate sigP data.");
}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalV2D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int dimSize)
{
    cudaSetDevice((mgr->getDeviceId()));
    cudaCheckErrors("set Device error");

    cudaArray* symArray = static_cast<cudaArray*>(mgr->GetArray());

    cudaMemcpyToArray(symArray,
                      0,
                      0,
                      (void*)(volume),
                      sizeof(Complex) * dimSize,
                      cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy array error");
}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalV3D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int vdim)
{

    cudaSetDevice((mgr->getDeviceId()));
    cudaCheckErrors("set Device error");

    cudaExtent extent = make_cudaExtent(vdim / 2 + 1, vdim, vdim);
    cudaArray* symArray = static_cast<cudaArray*>(mgr->GetArray());

    cudaMemcpy3DParms copyParams;
    copyParams = {0};
#ifdef SINGLE_PRECISION
    copyParams.srcPtr = make_cudaPitchedPtr((void*)volume,
                                            (vdim / 2 + 1) * sizeof(float2),
                                            vdim / 2 + 1,
                                            vdim);
#else
    copyParams.srcPtr = make_cudaPitchedPtr((void*)volume,
                                            (vdim / 2 + 1) * sizeof(int4),
                                            vdim / 2 + 1,
                                            vdim);
#endif
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    cudaCheckErrors("memcpy array error");
}

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
                      int cSearch)
{
    cudaSetDevice(gpuIdx);

    cudaHostAlloc((void**)wC, sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Host Alloc wC data.");

    cudaHostAlloc((void**)wR, mR * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldR data.");

    cudaHostAlloc((void**)wT, mT * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldT data.");

    cudaHostAlloc((void**)wD, mD * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldD data.");

    cudaHostAlloc((void**)oldR, mR * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldR data.");

    cudaHostAlloc((void**)oldT, mT * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldT data.");

    cudaHostAlloc((void**)trans, mT * 2 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register trans data.");

    cudaHostAlloc((void**)rot, mR * 4 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register rot data.");

    if (cSearch == 2)
    {
        cudaHostAlloc((void**)dpara, mD * sizeof(double), cudaHostRegisterDefault);
        cudaCheckErrors("Register dpara data.");

        cudaHostAlloc((void**)oldD, mD * sizeof(double), cudaHostRegisterDefault);
        cudaCheckErrors("Register oldD data.");
    }
    else
    {
        cudaHostAlloc((void**)oldD, sizeof(double), cudaHostRegisterDefault);
        cudaCheckErrors("Register oldD data.");
    }
}

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
                  int cSearch)
{
    cudaSetDevice(gpuIdx);

    cudaMemcpy(devdatPR + threadId * npxl,
               datPR,
               npxl * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy datP.");

    cudaMemcpy(devdatPI + threadId * npxl,
               datPI,
               npxl * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy datP.");

    if (cSearch != 2)
    {
        cudaMemcpy(devctfP + threadId * npxl,
                   ctfP,
                   npxl * sizeof(RFLOAT),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy ctfP.");
    }
    else
    {
        cudaMemcpy(devdefO + threadId * npxl,
                   defO,
                   npxl * sizeof(RFLOAT),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy defO.");
    }

    cudaMemcpy(devsigP + threadId * npxl,
               sigRcpP,
               npxl * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy sigP.");
}

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
                    double* dpara)
{
    cudaSetDevice(gpuIdx);

    cudaMemcpyAsync(mcp->getDevR(),
                    oldR,
                    mcp->getNR() * sizeof(double),
                    cudaMemcpyHostToDevice,
                    *((cudaStream_t*)mcp->getStream()));
    cudaCheckErrors("memcpy oldR to device memory.");

    cudaMemcpyAsync(mcp->getDevT(),
                    oldT,
                    mcp->getNT() * sizeof(double),
                    cudaMemcpyHostToDevice,
                    *((cudaStream_t*)mcp->getStream()));
    cudaCheckErrors("memcpy oldT to device memory.");

    if (mcp->getMode() == 1)
    {
        cudaMemcpyAsync(mcp->getDevnR(),
                        rot,
                        mcp->getNR() * 4 * sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy trans to device memory.");
    }
    else
    {
        cudaMemcpyAsync(mcp->getDevnR(),
                        rot,
                        mcp->getNR() * 2 * sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy trans to device memory.");
    }

    cudaMemcpyAsync(mcp->getDevnT(),
                    trans,
                    mcp->getNT() * 2 * sizeof(double),
                    cudaMemcpyHostToDevice,
                    *((cudaStream_t*)mcp->getStream()));
    cudaCheckErrors("memcpy trans to device memory.");

    if (mcp->getCSearch() == 2)
    {
        cudaMemcpyAsync(mcp->getDevdP(),
                        dpara,
                        mcp->getMD() * sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("for memcpy iRow.");

        cudaMemcpyAsync(mcp->getDevD(),
                        oldD,
                        mcp->getMD() * sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy oldD to device memory.");
    }
    else
    {
        cudaMemcpyAsync(mcp->getDevD(),
                        oldD,
                        sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy oldD to device memory.");
    }
    //cudaStreamSynchronize(*((cudaStream_t*)mcp->getStream()));
    //cudaCheckErrors("stream synchronize rtd.");

}

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
                       int interp)
{
    cudaSetDevice(gpuIdx);

    Complex* traP = reinterpret_cast<Complex*>(mcp->getDevtraP());
    Complex* rotP = reinterpret_cast<Complex*>(mcp->getPriRotP());

    int thread = THREAD_PER_BLOCK;
    int block = ((mcp->getNT() * npxl) % thread == 0)
              ? (mcp->getNT() * npxl) / thread
              : (mcp->getNT() * npxl) / thread + 1;
    
    kernel_Translate<<<block,
                       thread,
                       0,
                       *((cudaStream_t*)mcp->getStream())>>>(traP,
                                                             mcp->getDevnT(),
                                                             deviCol,
                                                             deviRow,
                                                             idim,
                                                             mcp->getNT(),
                                                             npxl);
    cudaCheckErrors("kernel trans.");

    if (mcp->getCSearch() == 2)
    {
        block = ((mcp->getMD() * npxl) % thread == 0)
              ? (mcp->getMD() * npxl) / thread
              : (mcp->getMD() * npxl) / thread + 1;
        
        kernel_CalCTFL<<<block,
                         thread,
                         0,
                         *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevctfD(),
                                                               devdefO + datShift * npxl,
                                                               devfreQ,
                                                               mcp->getDevdP(),
                                                               phaseShift,
                                                               conT,
                                                               k1,
                                                               k2,
                                                               mcp->getMD(),
                                                               npxl);
        cudaCheckErrors("kernel ctf.");
    }

    block = ((mcp->getNR() * npxl) % thread == 0)
          ? (mcp->getNR() * npxl) / thread
          : (mcp->getNR() * npxl) / thread + 1;

    kernel_Project2D<<<block,
                       thread,
                       0,
                       *((cudaStream_t*)mcp->getStream())>>>(rotP,
                                                             mcp->getDevnR(),
                                                             deviCol,
                                                             deviRow,
                                                             mcp->getNR(),
                                                             pf,
                                                             vdim,
                                                             npxl,
                                                             interp,
                                                             *static_cast<cudaTextureObject_t*>(mgr->GetTextureObject()));
    cudaCheckErrors("kernel Project.");
}

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
                       int interp)
{
    cudaSetDevice(gpuIdx);

    Complex* traP = reinterpret_cast<Complex*>(mcp->getDevtraP());
    Complex* rotP = reinterpret_cast<Complex*>(mcp->getPriRotP());

    int thread = THREAD_PER_BLOCK;
    int block = ((mcp->getNT() * npxl) % thread == 0)
                ? (mcp->getNT() * npxl) / thread
                : (mcp->getNT() * npxl) / thread + 1;
    
    kernel_Translate<<<block,
                       thread,
                       0,
                       *((cudaStream_t*)mcp->getStream())>>>(traP,
                                                             mcp->getDevnT(),
                                                             deviCol,
                                                             deviRow,
                                                             idim,
                                                             mcp->getNT(),
                                                             npxl);
    cudaCheckErrors("kernel trans.");

    if (mcp->getCSearch() == 2)
    {
        block = ((mcp->getMD() * npxl) % thread == 0)
                ? (mcp->getMD() * npxl) / thread
                : (mcp->getMD() * npxl) / thread + 1;
        kernel_CalCTFL<<<mcp->getMD(),
                         thread,
                         0,
                         *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevctfD(),
                                                               devdefO + datShift * npxl,
                                                               devfreQ,
                                                               mcp->getDevdP(),
                                                               phaseShift,
                                                               conT,
                                                               k1,
                                                               k2,
                                                               mcp->getMD(),
                                                               npxl);
        cudaCheckErrors("kernel ctf.");
    }

    kernel_getRotMat<<<1,
                       mcp->getNR(),
                       mcp->getNR() * 18 * sizeof(double),
                       *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevRotm(),
                                                             mcp->getDevnR(),
                                                             mcp->getNR());
    cudaCheckErrors("getRotMat3D kernel.");

    block = ((mcp->getNR() * npxl) % thread == 0)
            ? (mcp->getNR() * npxl) / thread
            : (mcp->getNR() * npxl) / thread + 1;
    
    kernel_Project3D<<<block,
                       thread,
                       0,
                       *((cudaStream_t*)mcp->getStream())>>>(rotP,
                                                             mcp->getDevRotm(),
                                                             deviCol,
                                                             deviRow,
                                                             mcp->getNR(),
                                                             pf,
                                                             vdim,
                                                             npxl,
                                                             interp,
                                                             *static_cast<cudaTextureObject_t*>(mgr->GetTextureObject()));
    cudaCheckErrors("kernel Project.");

}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalM(int gpuIdx,
                  int datShift,
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
    cudaSetDevice(gpuIdx);

    Complex* traP = reinterpret_cast<Complex*>(mcp->getDevtraP());
    Complex* rotP = reinterpret_cast<Complex*>(mcp->getPriRotP());

    int thread = (THREAD_PER_BLOCK < mcp->getNR() * mcp->getNT())
               ? THREAD_PER_BLOCK
               : mcp->getNR() * mcp->getNT();
    if (mcp->getCSearch() != 2)
    {
        kernel_logDataVSL<<<mcp->getNR() * mcp->getNT(),
                            64,
                            64 * sizeof(RFLOAT),
                            *((cudaStream_t*)mcp->getStream())>>>(rotP,
                                                                  traP,
                                                                  devdatPR + datShift * npxl,
                                                                  devdatPI + datShift * npxl,
                                                                  devctfP + datShift * npxl,
                                                                  devsigP + datShift * npxl,
                                                                  mcp->getDevDvp(),
                                                                  mcp->getNT(),
                                                                  npxl);
        cudaCheckErrors("logDataVSL kernel.");
        
        kernel_getMaxBase<<<1,
                            thread,
                            thread * sizeof(RFLOAT),
                            *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevBaseL(),
                                                                  mcp->getDevDvp(),
                                                                  mcp->getNR() * mcp->getNT());
        cudaCheckErrors("getMaxBaseL kernel.");

        kernel_UpdateWL<<<1,
                          mcp->getNR(),
                          mcp->getNR() * (mcp->getNT() + 3) * sizeof(RFLOAT),
                          *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevDvp(),
                                                                mcp->getDevBaseL(),
                                                                mcp->getDevwC(),
                                                                mcp->getDevwR(),
                                                                mcp->getDevwT(),
                                                                mcp->getDevwD(),
                                                                mcp->getDevR(),
                                                                mcp->getDevT(),
                                                                mcp->getDevD(),
                                                                oldC,
                                                                mcp->getNT());
        cudaCheckErrors("UpdateWL kernel.");

        cudaMemcpyAsync(wC,
                        mcp->getDevwC(),
                        sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wC to host.");

        cudaMemcpyAsync(wR,
                        mcp->getDevwR(),
                        mcp->getNR() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wR to host.");

        cudaMemcpyAsync(wT,
                        mcp->getDevwT(),
                        mcp->getNT() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wT to host.");

        cudaMemcpyAsync(wD,
                        mcp->getDevwD(),
                        mcp->getMD() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wD to host.");
    }
    else
    {
        kernel_logDataVSLC<<<mcp->getNR() * mcp->getNT() * mcp->getMD(),
                             64,
                             64 * sizeof(RFLOAT),
                             *((cudaStream_t*)mcp->getStream())>>>(rotP,
                                                                   traP,
                                                                   devdatPR + datShift * npxl,
                                                                   devdatPI + datShift * npxl,
                                                                   mcp->getDevctfD(),
                                                                   devsigP + datShift * npxl,
                                                                   mcp->getDevDvp(),
                                                                   mcp->getNT(),
                                                                   mcp->getMD(),
                                                                   npxl);
        cudaCheckErrors("logDataVSLC kernel.");

        kernel_getMaxBase<<<1,
                            thread,
                            thread * sizeof(RFLOAT),
                            *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevBaseL(),
                                                                  mcp->getDevDvp(),
                                                                  mcp->getNR() * mcp->getNT()
                                                                               * mcp->getMD());
        cudaCheckErrors("getMaxBaseLC kernel.");

        cudaMemsetAsync(mcp->getDevtT(),
                        0.0,
                        mcp->getNR() * mcp->getNT() * sizeof(RFLOAT),
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("for memset tT.");

        cudaMemsetAsync(mcp->getDevtD(),
                        0.0,
                        mcp->getNR() * mcp->getMD() * sizeof(RFLOAT),
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("for memset tD.");

        kernel_UpdateWLC<<<1,
                           mcp->getNR(),
                           2 * mcp->getNR() * sizeof(RFLOAT),
                           *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevDvp(),
                                                                 mcp->getDevBaseL(),
                                                                 mcp->getDevwC(),
                                                                 mcp->getDevwR(),
                                                                 mcp->getDevtT(),
                                                                 mcp->getDevtD(),
                                                                 mcp->getDevR(),
                                                                 mcp->getDevT(),
                                                                 mcp->getDevD(),
                                                                 oldC,
                                                                 mcp->getNT(),
                                                                 mcp->getMD());
        cudaCheckErrors("UpdateWL kernel.");

        kernel_ReduceW<<<mcp->getNT(),
                         mcp->getNR(),
                         mcp->getNR() * sizeof(RFLOAT),
                         *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevwT(),
                                                               mcp->getDevtT());
        cudaCheckErrors("ReduceWT kernel.");

        kernel_ReduceW<<<mcp->getMD(),
                         mcp->getNR(),
                         mcp->getNR() * sizeof(RFLOAT),
                         *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevwD(),
                                                               mcp->getDevtD());
        cudaCheckErrors("ReduceWD kernel.");

        cudaMemcpyAsync(wC,
                        mcp->getDevwC(),
                        sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wC to host.");

        cudaMemcpyAsync(wR,
                        mcp->getDevwR(),
                        mcp->getNR() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wR to host.");

        cudaMemcpyAsync(wT,
                        mcp->getDevwT(),
                        mcp->getNT() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wT to host.");

        cudaMemcpyAsync(wD,
                        mcp->getDevwD(),
                        mcp->getMD() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wD to host.");
    }

    cudaStreamSynchronize(*((cudaStream_t*)mcp->getStream()));
    cudaCheckErrors("stream synchronize.");
}

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
                      int cSearch)
{
    cudaSetDevice(gpuIdx);

    cudaFreeHost(*wC);
    cudaFreeHost(*wR);
    cudaFreeHost(*wT);
    cudaFreeHost(*wD);
    cudaFreeHost(*oldR);
    cudaFreeHost(*oldT);
    cudaFreeHost(*oldD);
    cudaFreeHost(*trans);
    cudaFreeHost(*rot);
    if (cSearch == 2)
    {
        cudaFreeHost(*dpara);
    }
    cudaCheckErrors("Free host page-lock memory.");
}

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
                    int cSearch)
{
    cudaSetDevice(gpuIdx);

    cudaFree(*devdatPR);
    cudaFree(*devdatPI);
    cudaFree(*devsigP);

    if (cSearch != 2)
    {
        cudaFree(*devctfP);
    }
    else
    {
        cudaFree(*devdefO);
        cudaFree(*devfreQ);
    }
    cudaCheckErrors("Free host Image data memory.");
}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectFreeIdx(int gpuIdx,
                   int** deviCol,
                   int** deviRow)
{
    cudaSetDevice(gpuIdx);

    cudaFree(*deviCol);
    cudaFree(*deviRow);
    cudaCheckErrors("Free host Pre iCol & iRow memory.");
}

/**
 * @brief Alloc Insert images' volume.
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
              int nGPU)
{
    int volumeSize = 0;
    int oSize = 0; 
    int cSize = 0; 
    
    if (mode)
    {
        volumeSize = (vdim / 2 + 1) * vdim * vdim;
        oSize = nk * 3;
        cSize = nk;
    }
    else
    {
        volumeSize = (vdim / 2 + 1) * vdim * nk;
        oSize = nk * 2;
        cSize = nk;
    }
    
    //register pglk_memory
    cudaHostRegister(volumeF, 
                     volumeSize * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register volumeF data.");

    cudaHostRegister(volumeT, 
                     volumeSize * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register volumeT data.");

    cudaHostRegister(arrayTau, 
                     nk * tauSize * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register Tau data.");

    cudaHostRegister(arrayO, 
                     oSize * sizeof(double), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register O3D data.");

    cudaHostRegister(arrayC, 
                     cSize * sizeof(int), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register C3D data.");

    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        cudaMalloc((void**)&dev_F[n], 
                   volumeSize * sizeof(Complex));
        cudaCheckErrors("Allocate __device__F data.");

        cudaMalloc((void**)&dev_T[n], 
                   volumeSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate __device__T data.");

        cudaMalloc((void**)&devTau[n], 
                   nk * tauSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate __device__Tau data.");

        cudaMalloc((void**)&dev_O[n], 
                   oSize * sizeof(double));
        cudaCheckErrors("Allocate __device__O data.");

        cudaMalloc((void**)&dev_C[n], 
                   cSize * sizeof(int));
        cudaCheckErrors("Allocate __device__C data.");

        cudaMalloc((void**)&deviCol[n], 
                   npxl * sizeof(int));
        cudaCheckErrors("FAIL TO ALLOCATE ICOL IN DEVICE");

        cudaMalloc((void**)&deviRow[n], 
                   npxl * sizeof(int));
        cudaCheckErrors("FAIL TO ALLOCATE IROW IN DEVICE");
        
        cudaMalloc((void**)&deviSig[n], 
                   npxl * sizeof(int));
        cudaCheckErrors("FAIL TO ALLOCATE IROW IN DEVICE");
        
        cudaMemcpy(deviCol[n],
                   iCol,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy iCol.");

        cudaMemcpy(deviRow[n],
                   iRow,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy iRow.");
        
        cudaMemcpy(deviSig[n],
                   iSig,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy iSig.");
    }
    
    cudaSetDevice(iGPU[0]);

    cudaMemcpy(devTau[0],
               arrayTau,
               nk * tauSize * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy Tau.");

    cudaMemcpy(dev_O[0],
               arrayO,
               oSize * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy O2D.");

    cudaMemcpy(dev_C[0],
               arrayC,
               cSize * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy O2D.");

    for (int n = 1; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMemset(devTau[n],
                   0.0,
                   nk * tauSize * sizeof(RFLOAT));
        cudaCheckErrors("for memset Tau.");

        cudaMemset(dev_O[n],
                   0.0,
                   oSize * sizeof(double));
        cudaCheckErrors("for memset O2D.");

        cudaMemset(dev_C[n],
                   0.0,
                   cSize * sizeof(int));
        cudaCheckErrors("for memset O2D.");
    }

}

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
                  int nGPU)
{
    int volumeSize = (vdim / 2 + 1) * vdim * nk;
    
    cudaSetDevice(iGPU[0]);

    cudaMemcpy(dev_F[0],
               volumeF,
               volumeSize * sizeof(Complex),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy F2D.");

    cudaMemcpy(dev_T[0],
               volumeT,
               volumeSize * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy T2D.");
    
    for (int n = 1; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        cudaMemset(dev_F[n],
                   0.0,
                   volumeSize * sizeof(Complex));
        cudaCheckErrors("for memset F2D.");

        cudaMemset(dev_T[n],
                   0.0,
                   volumeSize * sizeof(RFLOAT));
        cudaCheckErrors("for memset T2D."); 
    }
}

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
                  int nGPU)
{
    int volumeSize = (vdim / 2 + 1) * vdim * vdim;
    
    cudaSetDevice(iGPU[0]);

    cudaMemcpy(dev_F[0],
               volumeF,
               volumeSize * sizeof(Complex),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy F3D.");

    cudaMemcpy(dev_T[0],
               volumeT,
               volumeSize * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy T3D.");
    
    for (int n = 1; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        cudaMemset(dev_F[n],
                   0.0,
                   volumeSize * sizeof(Complex));
        cudaCheckErrors("for memset F3D.");

        cudaMemset(dev_T[n],
                   0.0,
                   volumeSize * sizeof(RFLOAT));
        cudaCheckErrors("for memset T3D."); 
    }
}

/**
 * @brief Insert reales into volume.
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
               int nGPU)
{
    const int BATCH_SIZE = IMAGE_BUFF;
    RFLOAT pixel = pixelSize * idim;
    int dimSize = (vdim / 2 + 1) * vdim;
    int nStream = nGPU * NUM_STREAM_PER_DEVICE;

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    RFLOAT *devsigRcpP[nStream];
#endif
    CTFAttr *dev_ctfas_buf[nStream];
    Complex *devtranP[nStream];
    RFLOAT *devdatPR[nStream];
    RFLOAT *devdatPI[nStream];
    RFLOAT *devctfP[nStream];
    double *devnR[nStream];
    double *devnT[nStream];
    double *dev_nd_buf[nStream];
    double *dev_offs_buf[nStream];
    int *devnC[nStream];

    LOG(INFO) << "Step1: Insert Image.";

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostRegister(offS, 
                     imgNum * 2 * sizeof(double), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register offset data.");
#endif

    cudaHostRegister(w, 
                     imgNum * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register w data.");

    cudaHostRegister(nC, 
                     mReco * imgNum * sizeof(int), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register nR data.");

    cudaHostRegister(nR, 
                     mReco * imgNum * 2 * sizeof(double), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register nR data.");

    cudaHostRegister(nT, 
                     mReco * imgNum * 2 * sizeof(double), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register nT data.");

    if (cSearch)
    {
        cudaHostRegister(nD, 
                         mReco * imgNum * sizeof(double), 
                         cudaHostRegisterDefault);
        cudaCheckErrors("Register nT data.");

        cudaHostRegister(ctfaData, 
                         imgNum * sizeof(CTFAttr), 
                         cudaHostRegisterDefault);
        cudaCheckErrors("Register ctfAdata data.");
    }

    //cudaEvent_t start[nStream], stop[nStream];

    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            if (cSearch)
            {
                allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
                allocDeviceParamBufferD(&dev_nd_buf[i + baseS], BATCH_SIZE * mReco);
            }

            allocDeviceComplexBuffer(&devtranP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devdatPR[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devdatPI[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBufferI(&devnC[i + baseS], BATCH_SIZE * mReco);
            allocDeviceParamBufferD(&devnR[i + baseS], BATCH_SIZE * mReco * 2);
            allocDeviceParamBufferD(&devnT[i + baseS], BATCH_SIZE * mReco * 2);
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            allocDeviceParamBufferD(&dev_offs_buf[i + baseS], BATCH_SIZE * 2);
#endif
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            allocDeviceParamBuffer(&devsigRcpP[i + baseS], BATCH_SIZE * npxl);
            cudaCheckErrors("Allocate sigRcp data.");
#endif
        }
    }

    int thread = THREAD_PER_BLOCK;
    int block;
    int batch = 0, smidx = 0;
   for (int i = 0; i < imgNum;)
   {
       for (int n = 0; n < nGPU; ++n)
       {
           if (i >= imgNum)
               break;

           baseS = n * NUM_STREAM_PER_DEVICE;
           batch = (i + BATCH_SIZE < imgNum) 
                 ? BATCH_SIZE : (imgNum - i);

           cudaSetDevice(iGPU[n]);

           cudaMemcpyAsync(devnR[smidx + baseS],
                           nR + i * mReco * 2,
                           batch * mReco * 2 * sizeof(double),
                           cudaMemcpyHostToDevice,
                           *((cudaStream_t*)stream[smidx + baseS]));
           cudaCheckErrors("memcpy nr to device.");

           cudaMemcpyAsync(devnT[smidx + baseS],
                           nT + i * mReco * 2,
                           batch * mReco * 2 * sizeof(double),
                           cudaMemcpyHostToDevice,
                           *((cudaStream_t*)stream[smidx + baseS]));
           cudaCheckErrors("memcpy nt to device.");

           cudaMemcpyAsync(devdatPR[smidx + baseS],
                           pglk_datPR + i * npxl,
                           batch * npxl * sizeof(RFLOAT),
                           cudaMemcpyHostToDevice,
                           *((cudaStream_t*)stream[smidx + baseS]));
           cudaCheckErrors("memcpy reale to device.");

           cudaMemcpyAsync(devdatPI[smidx + baseS],
                           pglk_datPI + i * npxl,
                           batch * npxl * sizeof(RFLOAT),
                           cudaMemcpyHostToDevice,
                           *((cudaStream_t*)stream[smidx + baseS]));
           cudaCheckErrors("memcpy reale to device.");
           
           if (cSearch)
           {
               cudaMemcpyAsync(dev_nd_buf[smidx + baseS],
                               nD + i * mReco,
                               batch * mReco * sizeof(double),
                               cudaMemcpyHostToDevice,
                               *((cudaStream_t*)stream[smidx + baseS]));
               cudaCheckErrors("memcpy nt to device.");

               cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                               ctfaData + i,
                               batch * sizeof(CTFAttr),
                               cudaMemcpyHostToDevice,
                               *((cudaStream_t*)stream[smidx + baseS]));
               cudaCheckErrors("memcpy CTFAttr to device.");
           }
           else
           {
               cudaMemcpyAsync(devctfP[smidx + baseS],
                               pglk_ctfP + i * npxl,
                               batch * npxl * sizeof(RFLOAT),
                               cudaMemcpyHostToDevice,
                               *((cudaStream_t*)stream[smidx + baseS]));
               cudaCheckErrors("memcpy ctf to device.");
           }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
           cudaMemcpyAsync(devsigRcpP[smidx + baseS],
                           pglk_sigRcpP + i * npxl,
                           batch * npxl * sizeof(RFLOAT),
                           cudaMemcpyHostToDevice,
                           *((cudaStream_t*)stream[smidx + baseS]));
           cudaCheckErrors("for memcpy sigRcp.");
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
           cudaMemcpyAsync(dev_offs_buf[smidx + baseS],
                           offS + 2 * i,
                           batch * 2 * sizeof(double),
                           cudaMemcpyHostToDevice,
                           *((cudaStream_t*)stream[smidx + baseS]));
           cudaCheckErrors("memcpy offset to device.");
#endif

           cudaMemcpyToSymbolAsync(dev_ws_data,
                                   w + i,
                                   batch * sizeof(RFLOAT),
                                   smidx * batch * sizeof(RFLOAT),
                                   cudaMemcpyHostToDevice,
                                   *((cudaStream_t*)stream[smidx + baseS]));
           cudaCheckErrors("memcpy w to device constant memory.");

           cudaMemcpyAsync(devnC[smidx + baseS],
                           nC + i * mReco,
                           batch * mReco * sizeof(int),
                           cudaMemcpyHostToDevice,
                           *((cudaStream_t*)stream[smidx + baseS]));
           cudaCheckErrors("memcpy nr to device.");

               //cudaEventRecord(start[smidx + baseS], stream[smidx + baseS]);

           for (int m = 0; m < mReco; m++)
           {
               block = ((batch * npxl) % thread == 0) 
                     ? (batch * npxl) / thread 
                     : (batch * npxl) / thread + 1; 
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
               kernel_Translate<<<block,
                                  thread,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(devdatPR[smidx + baseS],
                                                                             devdatPI[smidx + baseS],
                                                                             devtranP[smidx + baseS],
                                                                             dev_offs_buf[smidx + baseS],
                                                                             devnT[smidx + baseS],
                                                                             deviCol[n],
                                                                             deviRow[n],
                                                                             batch,
                                                                             m,
                                                                             opf,
                                                                             npxl,
                                                                             mReco,
                                                                             idim);

               cudaCheckErrors("translate kernel.");

               kernel_InsertO2D<<<1,
                                  batch,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_O[n],
                                                                             dev_C[n],
                                                                             devnR[smidx + baseS],
                                                                             devnT[smidx + baseS],
                                                                             dev_offs_buf[smidx + baseS],
                                                                             devnC[smidx + baseS],
                                                                             m,
                                                                             mReco);

               cudaCheckErrors("InsertO kernel.");
#else
               kernel_Translate<<<block,
                                  thread,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(devdatPR[smidx + baseS],
                                                                             devdatPI[smidx + baseS],
                                                                             devtranP[smidx + baseS],
                                                                             devnT[smidx + baseS],
                                                                             deviCol[n],
                                                                             deviRow[n],
                                                                             batch,
                                                                             m,
                                                                             opf,
                                                                             npxl,
                                                                             mReco,
                                                                             idim);

               cudaCheckErrors("translate kernel.");

               kernel_InsertO2D<<<1,
                                  batch,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_O[n],
                                                                             dev_C[n],
                                                                             devnR[smidx + baseS],
                                                                             devnT[smidx + baseS],
                                                                             devnC[smidx + baseS],
                                                                             m,
                                                                             mReco);

               cudaCheckErrors("InsertO kernel.");
#endif

               if (cSearch)
               {
                   kernel_CalculateCTF<<<block,
                                         thread,
                                         0,
                                         *((cudaStream_t*)stream[smidx + baseS])>>>(devctfP[smidx + baseS],
                                                                                    dev_ctfas_buf[smidx + baseS],
                                                                                    dev_nd_buf[smidx + baseS],
                                                                                    deviCol[n],
                                                                                    deviRow[n],
                                                                                    pixel,
                                                                                    batch,
                                                                                    m,
                                                                                    opf,
                                                                                    npxl,
                                                                                    mReco);

                   cudaCheckErrors("calculateCTF kernel.");
               }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
               kernel_InsertT2D<<<block,
                                  thread,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n],
                                                                             devctfP[smidx + baseS],
                                                                             devsigRcpP[smidx + baseS],
                                                                             devTau[n],
                                                                             devnR[smidx + baseS],
                                                                             devnC[smidx + baseS],
                                                                             deviCol[n],
                                                                             deviRow[n],
                                                                             batch,
                                                                             m,
                                                                             tauSize,
                                                                             npxl,
                                                                             mReco,
                                                                             vdim,
                                                                             dimSize,
                                                                             smidx);
               cudaCheckErrors("InsertT error.");

               kernel_InsertF2D<<<block,
                                  thread,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_F[n],
                                                                             devtranP[smidx + baseS],
                                                                             devctfP[smidx + baseS],
                                                                             devsigRcpP[smidx + baseS],
                                                                             devnR[smidx + baseS],
                                                                             devnC[smidx + baseS],
                                                                             deviCol[n],
                                                                             deviRow[n],
                                                                             deviSig[n],
                                                                             batch,
                                                                             m,
                                                                             npxl,
                                                                             mReco,
                                                                             vdim,
                                                                             dimSize,
                                                                             smidx);
               cudaCheckErrors("InsertF error.");
#else
               kernel_InsertT2D<<<block,
                                  thread,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n],
                                                                             devctfP[smidx + baseS],
                                                                             devTau[n],
                                                                             devnR[smidx + baseS],
                                                                             devnC[smidx + baseS],
                                                                             deviCol[n],
                                                                             deviRow[n],
                                                                             deviSig[n],
                                                                             batch,
                                                                             m,
                                                                             tauSize,
                                                                             npxl,
                                                                             mReco,
                                                                             vdim,
                                                                             dimSize,
                                                                             smidx);
               cudaCheckErrors("InsertT error.");

               kernel_InsertF2D<<<block,
                                  thread,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_F[n],
                                                                             devtranP[smidx + baseS],
                                                                             devctfP[smidx + baseS],
                                                                             devnR[smidx + baseS],
                                                                             devnC[smidx + baseS],
                                                                             deviCol[n],
                                                                             deviRow[n],
                                                                             batch,
                                                                             m,
                                                                             npxl,
                                                                             mReco,
                                                                             vdim,
                                                                             dimSize,
                                                                             smidx);
               cudaCheckErrors("InsertF error.");
#endif
            }
            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams to wait for start of NCCL operation
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {

            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize.");
            
            if (cSearch)
            {
                cudaFree(dev_ctfas_buf[i + baseS]);
                cudaFree(dev_nd_buf[i + baseS]);
            }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaFree(devsigRcpP[i + baseS]);
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaFree(dev_offs_buf[i + baseS]);
#endif
            cudaFree(devtranP[i + baseS]);
            cudaFree(devdatPR[i + baseS]);
            cudaFree(devdatPI[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(devnC[i + baseS]);
            cudaFree(devnR[i + baseS]);
            cudaFree(devnT[i + baseS]);
            cudaCheckErrors("cuda Free error.");
        }
    }

    //unregister pglk_memory
    cudaHostUnregister(w);
    cudaHostUnregister(nC);
    cudaHostUnregister(nR);
    cudaHostUnregister(nT);
    if (cSearch)
    {
        cudaHostUnregister(nD);
        cudaHostUnregister(ctfaData);
    }
    cudaCheckErrors("cuda Host Unregister error.");

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostUnregister(offS);
    cudaCheckErrors("cuda Host Unregister error.");
#endif

    LOG(INFO) << "Insert done.";
}

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
    const int BATCH_SIZE = IMAGE_BUFF;
    RFLOAT pixel = pixelSize * idim; // boxsize of image in Angstrom

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;
    Complex *devtranP[nStream];
    RFLOAT *devdatPR[nStream];
    RFLOAT *devdatPI[nStream];
    RFLOAT *devctfP[nStream];
    double *devnR[nStream];
    double *devnT[nStream];
    double *dev_nd_buf[nStream];
    double *dev_offs_buf[nStream];
    int *devnC[nStream];

    CTFAttr *dev_ctfas_buf[nStream];
    double *dev_mat_buf[nStream];
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    RFLOAT *devsigRcpP[nStream];
#endif

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostRegister(offS,
                     imgNum * 2 * sizeof(double),
                     cudaHostRegisterDefault);
#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF OFFS");
#endif

#endif

    cudaHostRegister(w,
                     imgNum * sizeof(RFLOAT),
                     cudaHostRegisterDefault);
#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF W");
#endif

    cudaHostRegister(nC,
                     imgNum * sizeof(int),
                     cudaHostRegisterDefault);
#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF NC");
#endif

    cudaHostRegister(nR,
                     mReco * imgNum * 4 * sizeof(double),
                     cudaHostRegisterDefault);
#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF NR");
#endif

    cudaHostRegister(nT,
                     mReco * imgNum * 2 * sizeof(double),
                     cudaHostRegisterDefault);
#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF NT");
#endif

    if (cSearch)
    {
        cudaHostRegister(nD,
                         mReco * imgNum * sizeof(double),
                         cudaHostRegisterDefault);
#ifdef GPU_ERROR_CHECK
        cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF ND");
#endif

        cudaHostRegister(ctfaData,
                         imgNum * sizeof(CTFAttr),
                         cudaHostRegisterDefault);
#ifdef GPU_ERROR_CHECK
        cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF CTFADATA");
#endif
    }
    
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            if (cSearch)
            {
                allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
                allocDeviceParamBufferD(&dev_nd_buf[i + baseS], BATCH_SIZE * mReco);
            }

            allocDeviceParamBuffer(&devdatPR[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devdatPI[i + baseS], BATCH_SIZE * npxl);
            allocDeviceComplexBuffer(&devtranP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBufferI(&devnC[i + baseS], BATCH_SIZE);
            allocDeviceParamBufferD(&devnR[i + baseS], BATCH_SIZE * mReco * 4);
            allocDeviceParamBufferD(&devnT[i + baseS], BATCH_SIZE * mReco * 2);
            allocDeviceParamBufferD(&dev_mat_buf[i + baseS], BATCH_SIZE * mReco * 9);
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            allocDeviceParamBufferD(&dev_offs_buf[i + baseS], BATCH_SIZE * 2);
#endif
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            allocDeviceParamBuffer(&devsigRcpP[i + baseS], BATCH_SIZE * npxl);
            cudaCheckErrors("Allocate sigRcp data.");
#endif
        }
    }

    int batch = 0;
    int smidx = 0;
    int thread = (npxl > THREAD_PER_BLOCK) 
               ? THREAD_PER_BLOCK : npxl; 
    for (int i = 0; i < imgNum;)
    {
        for (int n = 0; n < nGPU; n++)
        {
            if (i >= imgNum) 
                break;

            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + BATCH_SIZE < imgNum) 
                  ? BATCH_SIZE : (imgNum - i);

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(devnC[smidx + baseS],
                            nC + i,
                            batch * sizeof(int),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("FAIL TO COPY A BATCH OF NC FROM HOST TO DEVICE");

            cudaMemcpyAsync(devnR[smidx + baseS],
                            nR + i * mReco * 4,
                            batch * mReco * 4 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("FAIL TO COPY A BATCH OF NR FROM HOST TO DEVICE");

            kernel_getRandomR<<<batch,
                                mReco,
                                mReco * 18 * sizeof(double),
                                *((cudaStream_t*)stream[smidx + baseS])>>>(dev_mat_buf[smidx + baseS],
                                                                           devnR[smidx + baseS],
                                                                           devnC[smidx + baseS]);
            cudaCheckErrors("FAIL TO CALCULATE ROTATION MARTICES FROM NR");

            cudaMemcpyAsync(devnT[smidx + baseS],
                            nT + i * mReco * 2,
                            batch * mReco * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("FAIL TO COPY A BATCH OF NT FROM HOST TO DEVICE");

            cudaMemcpyAsync(devdatPR[smidx + baseS],
                            pglk_datPR + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("FAIL TO COPY A BATCH OF IMAGES FROM HOST TO DEVICE");

            cudaMemcpyAsync(devdatPI[smidx + baseS],
                            pglk_datPI + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("FAIL TO COPY A BATCH OF IMAGES FROM HOST TO DEVICE");

            if (cSearch)
            {
                cudaMemcpyAsync(dev_nd_buf[smidx + baseS],
                                nD + i * mReco,
                                batch * mReco * sizeof(double),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy nt to device.");

                cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                                ctfaData + i,
                                batch * sizeof(CTFAttr),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy CTFAttr to device.");
            }
            else
            {
                cudaMemcpyAsync(devctfP[smidx + baseS],
                                pglk_ctfP + i * npxl,
                                batch * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy ctf to device.");
            }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaMemcpyAsync(devsigRcpP[smidx + baseS],
                            pglk_sigRcpP + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy sigRcp.");
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaMemcpyAsync(dev_offs_buf[smidx + baseS],
                            offS + 2 * i,
                            batch * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy offset to device.");
#endif

            cudaMemcpyToSymbolAsync(dev_ws_data,
                                    w + i,
                                    batch * sizeof(RFLOAT),
                                    smidx * batch * sizeof(RFLOAT),
                                    cudaMemcpyHostToDevice,
                                    *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy w to device constant memory.");

            //cudaEventRecord(start[smidx + baseS], stream[smidx + baseS]);

            for (int m = 0; m < mReco; m++)
            {
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                kernel_Translate<<<batch,
                                   thread,
                                   0,
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(devdatPR[smidx + baseS],
                                                                              devdatPI[smidx + baseS],
                                                                              devtranP[smidx + baseS],
                                                                              dev_offs_buf[smidx + baseS],
                                                                              devnT[smidx + baseS],
                                                                              devnC[smidx + baseS],
                                                                              deviCol[n],
                                                                              deviRow[n],
                                                                              batch,
                                                                              m,
                                                                              opf,
                                                                              npxl,
                                                                              mReco,
                                                                              idim);

                cudaCheckErrors("translate kernel.");

                kernel_InsertO3D<<<1,
                                   thread,
                                   3 * thread * sizeof(double)
                                     + thread * sizeof(int),
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(dev_O[n] + kIdx * 3,
                                                                              dev_C[n] + kIdx,
                                                                              dev_mat_buf[smidx + baseS],
                                                                              devnT[smidx + baseS],
                                                                              dev_offs_buf[smidx + baseS],
                                                                              devnC[smidx + baseS],
                                                                              m,
                                                                              mReco,
                                                                              batch);

                cudaCheckErrors("InsertO kernel.");
#else
                kernel_Translate<<<batch,
                                   thread,
                                   0,
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(devdatPR[smidx + baseS],
                                                                              devdatPI[smidx + baseS],
                                                                              devtranP[smidx + baseS],
                                                                              devnT[smidx + baseS],
                                                                              devnC[smidx + baseS],
                                                                              deviCol[n],
                                                                              deviRow[n],
                                                                              batch,
                                                                              m,
                                                                              opf,
                                                                              npxl,
                                                                              mReco,
                                                                              idim);

                cudaCheckErrors("translate kernel.");

                kernel_InsertO3D<<<1,
                                   thread,
                                   3 * thread * sizeof(double)
                                     + thread * sizeof(int),
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(dev_O[n] + kIdx * 3,
                                                                              dev_C[n] + kIdx,
                                                                              dev_mat_buf[smidx + baseS],
                                                                              devnT[smidx + baseS],
                                                                              devnC[smidx + baseS],
                                                                              m,
                                                                              mReco,
                                                                              batch);

                cudaCheckErrors("InsertO kernel.");
#endif

                if (cSearch)
                {
                    kernel_CalculateCTF<<<batch,
                                          thread,
                                          0,
                                          *((cudaStream_t*)stream[smidx + baseS])>>>(devctfP[smidx + baseS],
                                                                                     dev_ctfas_buf[smidx + baseS],
                                                                                     dev_nd_buf[smidx + baseS],
                                                                                     devnC[smidx + baseS],
                                                                                     deviCol[n],
                                                                                     deviRow[n],
                                                                                     pixel,
                                                                                     batch,
                                                                                     m,
                                                                                     opf,
                                                                                     npxl,
                                                                                     mReco);

                    cudaCheckErrors("calculateCTF kernel.");
                }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                kernel_InsertT<<<batch,
                                 thread,
                                 thread * 9 * sizeof(double),
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n],
                                                                            devctfP[smidx + baseS],
                                                                            devsigRcpP[smidx + baseS],
                                                                            devTau[n] + kIdx * tauSize,
                                                                            dev_mat_buf[smidx + baseS],
                                                                            devnC[smidx + baseS],
                                                                            deviCol[n],
                                                                            deviRow[n],
                                                                            deviSig[n],
                                                                            batch,
                                                                            m,
                                                                            npxl,
                                                                            mReco,
                                                                            vdim,
                                                                            smidx);
                cudaCheckErrors("InsertT error.");

                kernel_InsertF<<<batch,
                                 thread,
                                 thread * 9 * sizeof(double),
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_F[n],
                                                                            devtranP[smidx + baseS],
                                                                            //__device__batch__datP[smidx + baseS],
                                                                            devctfP[smidx + baseS],
                                                                            devsigRcpP[smidx + baseS],
                                                                            dev_mat_buf[smidx + baseS],
                                                                            devnC[smidx + baseS],
                                                                            deviCol[n],
                                                                            deviRow[n],
                                                                            batch,
                                                                            m,
                                                                            npxl,
                                                                            mReco,
                                                                            vdim,
                                                                            smidx);
                cudaCheckErrors("InsertF error.");
#else
                kernel_InsertT<<<batch,
                                 thread,
                                 thread * 9 * sizeof(double),
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n],
                                                                            devctfP[smidx + baseS],
                                                                            devTau[n] + kIdx * tauSize,
                                                                            dev_mat_buf[smidx + baseS],
                                                                            devnC[smidx + baseS],
                                                                            deviCol[n],
                                                                            deviRow[n],
                                                                            deviSig[n],
                                                                            batch,
                                                                            m,
                                                                            npxl,
                                                                            mReco,
                                                                            vdim,
                                                                            smidx);
                cudaCheckErrors("InsertT error.");

                kernel_InsertF<<<batch,
                                 thread,
                                 thread * 9 * sizeof(double),
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_F[n],
                                                                            devtranP[smidx + baseS],
                                                                            //__device__batch__datP[smidx + baseS],
                                                                            devctfP[smidx + baseS],
                                                                            dev_mat_buf[smidx + baseS],
                                                                            devnC[smidx + baseS],
                                                                            deviCol[n],
                                                                            deviRow[n],
                                                                            batch,
                                                                            m,
                                                                            npxl,
                                                                            mReco,
                                                                            vdim,
                                                                            smidx);
                cudaCheckErrors("InsertF error.");

#endif
            }
            i += batch;
        }
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams to wait for start of NCCL operation
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
#ifdef GPU_ERROR_CHECK
            cudaCheckErrors("FAIL TO SYNCHRONIZE STREAM");
#endif
            if (cSearch)
            {
                cudaFree(dev_ctfas_buf[i + baseS]);
                cudaFree(dev_nd_buf[i + baseS]);
            }
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaFree(devsigRcpP[i + baseS]);
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaFree(dev_offs_buf[i + baseS]);
#endif
            cudaFree(devdatPR[i + baseS]);
            cudaFree(devdatPI[i + baseS]);
            cudaFree(devtranP[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(devnC[i + baseS]);
            cudaFree(devnR[i + baseS]);
            cudaFree(devnT[i + baseS]);
            cudaFree(dev_mat_buf[i + baseS]);
            cudaCheckErrors("cuda Free error.");
        }
    }

    //unregister pglk_memory
    cudaHostUnregister(w);
    cudaHostUnregister(nC);
    cudaHostUnregister(nR);
    cudaHostUnregister(nT);

    if (cSearch)
    {
        cudaHostUnregister(nD);
        cudaHostUnregister(ctfaData);
    }
    cudaCheckErrors("cuda Host Unregister error.");

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostUnregister(offS);
    cudaCheckErrors("cuda Host Unregister error.");
#endif
}

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
              int nGPU)
{
    const int BATCH_SIZE = IMAGE_BUFF;
    RFLOAT pixel = pixelSize * idim;
    int nStream = nGPU * NUM_STREAM_PER_DEVICE;

    Complex *devtranP[nStream];
    RFLOAT *devdatPR[nStream];
    RFLOAT *devdatPI[nStream];
    RFLOAT *devctfP[nStream];
    double *devnR[nStream];
    double *devnT[nStream];
    double *dev_nd_buf[nStream];
    double *dev_offs_buf[nStream];

    CTFAttr *dev_ctfas_buf[nStream];
    double *dev_mat_buf[nStream];

    LOG(INFO) << "Step1: Insert Image.";

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    RFLOAT *devsigRcpP[nStream];
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostRegister(offS, 
                     imgNum * 2 * sizeof(double), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register offset data.");
#endif

    if (cSearch)
    {
        cudaHostRegister(nD, 
                         mReco * imgNum * sizeof(double), 
                         cudaHostRegisterDefault);
        cudaCheckErrors("Register nT data.");

        cudaHostRegister(ctfaData, 
                         imgNum * sizeof(CTFAttr), 
                         cudaHostRegisterDefault);
        cudaCheckErrors("Register ctfAdata data.");
    }

    cudaHostRegister(w, 
                     imgNum * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register w data.");

    cudaHostRegister(nR, 
                     mReco * imgNum * 4 * sizeof(double), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register nR data.");

    cudaHostRegister(nT, 
                     mReco * imgNum * 2 * sizeof(double), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register nT data.");

    //cudaEvent_t start[nStream], stop[nStream];

    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            if (cSearch)
            {
                allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
                allocDeviceParamBufferD(&dev_nd_buf[i + baseS], BATCH_SIZE * mReco);
            }

            allocDeviceComplexBuffer(&devtranP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devdatPR[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devdatPI[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBufferD(&devnR[i + baseS], BATCH_SIZE * mReco * 4);
            allocDeviceParamBufferD(&devnT[i + baseS], BATCH_SIZE * mReco * 2);
            allocDeviceParamBufferD(&dev_mat_buf[i + baseS], BATCH_SIZE * mReco * 9);
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            allocDeviceParamBufferD(&dev_offs_buf[i + baseS], BATCH_SIZE * 2);
#endif
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            allocDeviceParamBuffer(&devsigRcpP[i + baseS], BATCH_SIZE * npxl);
            cudaCheckErrors("Allocate sigRcp data.");
#endif
        }
    }

    int thread = THREAD_PER_BLOCK;
    int block;
    int batch = 0, smidx = 0;
    for (int i = 0; i < imgNum;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= imgNum)
                break;

            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + BATCH_SIZE < imgNum) 
                  ? BATCH_SIZE : (imgNum - i);

            cudaSetDevice(iGPU[n]);

            long long mrShift  = (long long)i * mReco;

            cudaMemcpyAsync(devnR[smidx + baseS],
                            nR + mrShift * 4,
                            batch * mReco * 4 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy nr to device.");

            kernel_getRandomR<<<batch,
                                mReco,
                                mReco * 18 * sizeof(double),
                                *((cudaStream_t*)stream[smidx + baseS])>>>(dev_mat_buf[smidx + baseS],
                                                                           devnR[smidx + baseS]);
            cudaCheckErrors("getrandomR kernel.");

            cudaMemcpyAsync(devnT[smidx + baseS],
                            nT + mrShift * 2,
                            batch * mReco * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy nt to device.");

            cudaMemcpyAsync(devdatPR[smidx + baseS],
                            pglk_datPR + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy image to device.");

            cudaMemcpyAsync(devdatPI[smidx + baseS],
                            pglk_datPI + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy image to device.");

            if (cSearch)
            {
                cudaMemcpyAsync(dev_nd_buf[smidx + baseS],
                                nD + mrShift,
                                batch * mReco * sizeof(double),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy nt to device.");

                cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                                ctfaData + i,
                                batch * sizeof(CTFAttr),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy CTFAttr to device.");
            }
            else
            {
                cudaMemcpyAsync(devctfP[smidx + baseS],
                                pglk_ctfP + i * npxl,
                                batch * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("memcpy ctf to device.");
            }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaMemcpyAsync(devsigRcpP[smidx + baseS],
                            pglk_sigRcpP + i * npxl,
                            batch * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy sigRcp.");
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaMemcpyAsync(dev_offs_buf[smidx + baseS],
                            offS + 2 * i,
                            batch * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy offset to device.");
#endif

            cudaMemcpyToSymbolAsync(dev_ws_data,
                                    w + i,
                                    batch * sizeof(RFLOAT),
                                    smidx * batch * sizeof(RFLOAT),
                                    cudaMemcpyHostToDevice,
                                    *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy w to device constant memory.");

            //cudaEventRecord(start[smidx + baseS], stream[smidx + baseS]);

            for (int m = 0; m < mReco; m++)
            {
                block = ((batch * npxl) % thread == 0) 
                      ? (batch * npxl) / thread 
                      : (batch * npxl) / thread + 1; 
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                kernel_Translate<<<block,
                                   thread,
                                   0,
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(devdatPR[smidx + baseS],
                                                                              devdatPI[smidx + baseS],
                                                                              devtranP[smidx + baseS],
                                                                              dev_offs_buf[smidx + baseS],
                                                                              devnT[smidx + baseS],
                                                                              deviCol[n],
                                                                              deviRow[n],
                                                                              batch,
                                                                              m,
                                                                              opf,
                                                                              npxl,
                                                                              mReco,
                                                                              idim);
                cudaCheckErrors("translate kernel.");

                kernel_InsertO3D<<<1,
                                   thread,
                                   3 * thread * sizeof(double)
                                     + thread * sizeof(int),
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(dev_O[n],
                                                                              dev_C[n],
                                                                              dev_mat_buf[smidx + baseS],
                                                                              devnT[smidx + baseS],
                                                                              dev_offs_buf[smidx + baseS],
                                                                              m,
                                                                              mReco,
                                                                              batch);

                cudaCheckErrors("InsertO kernel.");
#else
                kernel_Translate<<<block,
                                   thread,
                                   0,
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(devdatPR[smidx + baseS],
                                                                              devdatPI[smidx + baseS],
                                                                              devtranP[smidx + baseS],
                                                                              devnT[smidx + baseS],
                                                                              deviCol[n],
                                                                              deviRow[n],
                                                                              batch,
                                                                              m,
                                                                              opf,
                                                                              npxl,
                                                                              mReco,
                                                                              idim);

                cudaCheckErrors("translate kernel.");

                kernel_InsertO3D<<<1,
                                   thread,
                                   3 * thread * sizeof(double)
                                     + thread * sizeof(int),
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(dev_O[n],
                                                                              dev_C[n],
                                                                              dev_mat_buf[smidx + baseS],
                                                                              devnT[smidx + baseS],
                                                                              m,
                                                                              mReco,
                                                                              batch);

                cudaCheckErrors("InsertO kernel.");
#endif
                if (cSearch)
                {
                    kernel_CalculateCTF<<<block,
                                          thread,
                                          0,
                                          *((cudaStream_t*)stream[smidx + baseS])>>>(devctfP[smidx + baseS],
                                                                                     dev_ctfas_buf[smidx + baseS],
                                                                                     dev_nd_buf[smidx + baseS],
                                                                                     deviCol[n],
                                                                                     deviRow[n],
                                                                                     pixel,
                                                                                     batch,
                                                                                     m,
                                                                                     opf,
                                                                                     npxl,
                                                                                     mReco);

                    cudaCheckErrors("calculateCTF kernel.");
                }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                kernel_InsertT<<<block,
                                 thread,
                                 thread * 9 * sizeof(double),
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n],
                                                                            devctfP[smidx + baseS],
                                                                            devsigRcpP[smidx + baseS],
                                                                            devTau[n],
                                                                            dev_mat_buf[smidx + baseS],
                                                                            deviCol[n],
                                                                            deviRow[n],
                                                                            deviSig[n],
                                                                            //batch,
                                                                            m,
                                                                            npxl,
                                                                            mReco,
                                                                            vdim,
                                                                            smidx);
                cudaCheckErrors("InsertT error.");

                kernel_InsertF<<<block,
                                 thread,
                                 thread * 9 * sizeof(double),
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_F[n],
                                                                            devtranP[smidx + baseS],
                                                                            devctfP[smidx + baseS],
                                                                            devsigRcpP[smidx + baseS],
                                                                            dev_mat_buf[smidx + baseS],
                                                                            deviCol[n],
                                                                            deviRow[n],
                                                                            batch,
                                                                            m,
                                                                            npxl,
                                                                            mReco,
                                                                            vdim,
                                                                            smidx);
                cudaCheckErrors("InsertF error.");
#else
                kernel_InsertT<<<block,
                                 thread,
                                 thread * 9 * sizeof(double),
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n],
                                                                            devctfP[smidx + baseS],
                                                                            devTau[n],
                                                                            dev_mat_buf[smidx + baseS],
                                                                            deviCol[n],
                                                                            deviRow[n],
                                                                            deviSig[n],
                                                                            batch,
                                                                            m,
                                                                            npxl,
                                                                            mReco,
                                                                            vdim,
                                                                            smidx);
                cudaCheckErrors("InsertT error.");

                kernel_InsertF<<<block,
                                 thread,
                                 thread * 9 * sizeof(double),
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_F[n],
                                                                            devtranP[smidx + baseS],
                                                                            devctfP[smidx + baseS],
                                                                            dev_mat_buf[smidx + baseS],
                                                                            deviCol[n],
                                                                            deviRow[n],
                                                                            batch,
                                                                            m,
                                                                            npxl,
                                                                            mReco,
                                                                            vdim,
                                                                            smidx);
                cudaCheckErrors("InsertF error.");
#endif
            }
            i += batch;
        }
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }
    
    //synchronizing on CUDA streams to wait for start of NCCL operation
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {

            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize.");
            //cudaEventSynchronize(stop[i + baseS]);
            //float elapsed_time;
            //cudaEventElapsedTime(&elapsed_time, start[i + baseS], stop[i + baseS]);
            //if (n == 0 && i == 0)
            //{
            //    printf("insertF:%f\n", elapsed_time);
            //}

            if (cSearch)
            {
                cudaFree(dev_ctfas_buf[i + baseS]);
                cudaFree(dev_nd_buf[i + baseS]);
            }

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaFree(devsigRcpP[i + baseS]);
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaFree(dev_offs_buf[i + baseS]);
#endif
            cudaFree(devdatPR[i + baseS]);
            cudaFree(devdatPI[i + baseS]);
            cudaFree(devtranP[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(devnR[i + baseS]);
            cudaFree(devnT[i + baseS]);
            cudaFree(dev_mat_buf[i + baseS]);
           /*
            cudaEventDestroy(start[i + baseS]);
            cudaEventDestroy(stop[i + baseS]);
            cudaCheckErrors("Event destory.");
            */
        }
    }

    //unregister pglk_memory
    cudaHostUnregister(w);
    cudaHostUnregister(nR);
    cudaHostUnregister(nT);
    if (cSearch)
    {
        cudaHostUnregister(nD);
        cudaHostUnregister(ctfaData);
    }

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostUnregister(offS);
#endif
    cudaCheckErrors("host memory unregister.");

    LOG(INFO) << "Insert done.";
}

/**
 * @brief AllReduce 3D volume.
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
                  int nGPU)
{
    ncclUniqueId commIdF;
    ncclComm_t commF[nGPU];
    ncclComm_t commO[nGPU];

    int rankF, sizeF;
    MPI_Comm_size(hemi, &sizeF);
    MPI_Comm_rank(hemi, &rankF);

    if (rankF == 0)
        NCCLCHECK(ncclGetUniqueId(&commIdF));
    
    MPI_Bcast(&commIdF, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, hemi);

    int* gpus = new int[nGPU];
    for (int n = 0; n < nGPU; n++)
        gpus[n] = iGPU[n];
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclCommInitRank(commF + i,
                                   sizeF * nGPU,
                                   commIdF,
                                   rankF * nGPU + i));
    }
    NCCLCHECK(ncclGroupEnd());

    NCCLCHECK(ncclCommInitAll(commO, nGPU, gpus));

    MPI_Barrier(hemi);

    LOG(INFO) << "rank" << rankF << ": Step1: Reduce Volume.";

    int reduceSize = 0;
    int dimSize = 0;

    if (mode)
    {
        dimSize = (vdim / 2 + 1) * vdim * vdim;
        reduceSize = dimSize;
    }
    else
    {
        dimSize = (vdim / 2 + 1) * vdim;
        reduceSize = nk * dimSize;
    }
    
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)dev_F[i],
                                (void*)dev_F[i],
                                reduceSize * 2,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commF[i],
                                *((cudaStream_t*)stream[0 + i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)dev_T[i],
                                (void*)dev_T[i],
                                reduceSize,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commF[i],
                                *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("Set device.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("stream synchronize.");
        }
    }

    MPI_Barrier(hemi);

#ifdef RECONSTRUCTOR_NORMALISE_T_F
    RFLOAT* sf[nGPU];
#endif
    if (mode)
    {
#ifdef RECONSTRUCTOR_NORMALISE_T_F
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            cudaCheckErrors("Set device.");
            
            cudaMalloc((void**)&sf[i], sizeof(RFLOAT));
            cudaCheckErrors("Allocate sf data.");
        
            kernel_SetSF<<<1, 1>>>(dev_T[i],
                                   sf[i]);
            cudaCheckErrors("kernel sf.");
        }
#endif

        int nImgBatch = VOLUME_BATCH_3D * vdim * (vdim / 2 + 1);
#ifdef RECONSTRUCTOR_NORMALISE_T_F
        int thread = (vdim / 2 + 1 > THREAD_PER_BLOCK) 
                   ? THREAD_PER_BLOCK : vdim / 2 + 1;
#endif
        int smidx = 0;
        int batch = 0;
        for (int i = 0; i < dimSize;)
        {
            for (int n = 0; n < nGPU; ++n)
            {
                if (i >= dimSize)
                    break;
                
                baseS = n * NUM_STREAM_PER_DEVICE;
                batch = (i + nImgBatch > dimSize) 
                      ? (dimSize - i) : nImgBatch;

                cudaSetDevice(iGPU[n]);

#ifdef RECONSTRUCTOR_NORMALISE_T_F
                kernel_NormalizeTF<<<vdim * VOLUME_BATCH_3D,
                                     thread,
                                     0,
                                     *((cudaStream_t*)stream[smidx + baseS])>>>(dev_F[n] + i,
                                                                                dev_T[n] + i,
                                                                                sf[n],
                                                                                batch);
                cudaCheckErrors("normalTF.");
#endif
                cudaMemcpyAsync(volumeT + i,
                                dev_T[n] + i,
                                batch * sizeof(RFLOAT),
                                cudaMemcpyDeviceToHost,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memcpy T.");

                cudaMemcpyAsync(volumeF + i,
                                dev_F[n] + i,
                                batch * sizeof(Complex),
                                cudaMemcpyDeviceToHost,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memcpy F.");

                i += batch;
            }
            smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
        }
    }
    else
    {
#ifdef RECONSTRUCTOR_NORMALISE_T_F
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            cudaCheckErrors("Set device.");
            
            cudaMalloc((void**)&sf[i], nk * sizeof(RFLOAT));
            cudaCheckErrors("Allocate sf data.");
        
            kernel_SetSF2D<<<1, nk>>>(dev_T[i],
                                      sf[i],
                                      dimSize);
            cudaCheckErrors("kernel sf.");
        }
#endif

#ifdef RECONSTRUCTOR_NORMALISE_T_F
        int thread = (vdim / 2 + 1 > THREAD_PER_BLOCK) 
                   ? THREAD_PER_BLOCK : vdim / 2 + 1;
#endif
        int smidx = 0;
        int gIdx = 0;
        for(int i = 0; i < nk; i++)
        {
            baseS = gIdx * NUM_STREAM_PER_DEVICE;
            cudaSetDevice(iGPU[gIdx]);
            cudaCheckErrors("Set device.");
            
#ifdef RECONSTRUCTOR_NORMALISE_T_F
            kernel_NormalizeTF2D<<<vdim,
                                   thread,
                                   0,
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(dev_F[gIdx] + i * dimSize,
                                                                              dev_T[gIdx] + i * dimSize,
                                                                              sf[gIdx],
                                                                              i,
                                                                              dimSize);
            cudaCheckErrors("normalize TF2D error.");
#endif

            cudaMemcpyAsync(volumeT + i * dimSize,
                            dev_T[gIdx] + i * dimSize,
                            dimSize * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("copy T3D from device to host.");

            cudaMemcpyAsync(volumeF + i * dimSize,
                            dev_F[gIdx] + i * dimSize,
                            dimSize * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("copy F3D from device to host.");

            gIdx  = (gIdx + 1) % nGPU;
            
            if (gIdx == 0)
                smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
        }
    }

    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("Set device.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("stream synchronize.");
        }
    }

#ifdef RECONSTRUCTOR_NORMALISE_T_F
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        cudaCheckErrors("Set device.");
        
        cudaFree(sf[i]);
        cudaCheckErrors("Free sf.");
    }
#endif

    MPI_Barrier(hemi);
    //MPI_Barrier(slav);

    int reduceO, reduceC;
    if (mode)
    {
        reduceO = 3;
        reduceC = 1;
    }
    else
    {
        reduceO = nk * 2;
        reduceC = nk;
    }
    
    if (mode && nk != 1)
    {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            NCCLCHECK(ncclAllReduce((const void*)(dev_O[i] + kIdx * 3),
                                    (void*)(dev_O[i] + kIdx * 3),
                                    reduceO,
                                    ncclDouble,
                                    ncclSum,
                                    commO[i],
                                    *((cudaStream_t*)stream[0 + i * NUM_STREAM_PER_DEVICE])));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            NCCLCHECK(ncclAllReduce((const void*)(dev_C[i] + kIdx),
                                    (void*)(dev_C[i] + kIdx),
                                    reduceC,
                                    ncclInt,
                                    ncclSum,
                                    commO[i],
                                    *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            NCCLCHECK(ncclAllReduce((const void*)(devTau[i] + kIdx * tauSize),
                                    (void*)(devTau[i] + kIdx * tauSize),
                                    tauSize,
#ifdef SINGLE_PRECISION
                                    ncclFloat,
#else
                                    ncclDouble,
#endif
                                    ncclSum,
                                    commO[i],
                                    *((cudaStream_t*)stream[2 + i * NUM_STREAM_PER_DEVICE])));
        }
        NCCLCHECK(ncclGroupEnd());
    }
    else
    {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            NCCLCHECK(ncclAllReduce((const void*)dev_O[i],
                                    (void*)dev_O[i],
                                    reduceO,
                                    ncclDouble,
                                    ncclSum,
                                    commO[i],
                                    *((cudaStream_t*)stream[0 + i * NUM_STREAM_PER_DEVICE])));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            NCCLCHECK(ncclAllReduce((const void*)dev_C[i],
                                    (void*)dev_C[i],
                                    reduceC,
                                    ncclInt,
                                    ncclSum,
                                    commO[i],
                                    *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
        }
        NCCLCHECK(ncclGroupEnd());
        
        int reduceTau;
        if (mode)
            reduceTau = tauSize;
        else
            reduceTau = nk * tauSize;

        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            NCCLCHECK(ncclAllReduce((const void*)(devTau[i]),
                                    (void*)(devTau[i]),
                                    reduceTau,
#ifdef SINGLE_PRECISION
                                    ncclFloat,
#else
                                    ncclDouble,
#endif
                                    ncclSum,
                                    commO[i],
                                    *((cudaStream_t*)stream[2 + i * NUM_STREAM_PER_DEVICE])));
        }
        NCCLCHECK(ncclGroupEnd());
    }

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device error.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("stream synchronize error.");
        }
    }

    if (mode && nk != 1)
    {
        cudaSetDevice(iGPU[0]);
        cudaMemcpy(arrayO + kIdx * 3,
                   dev_O[0] + kIdx * 3,
                   reduceO * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaCheckErrors("copy O2D from device to host.");

        cudaMemcpy(arrayC + kIdx,
                   dev_C[0] + kIdx,
                   reduceC * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaCheckErrors("copy C2D from device to host.");
        
        cudaMemcpy(arrayTau + kIdx * tauSize,
                   devTau[0] + kIdx * tauSize,
                   tauSize * sizeof(RFLOAT),
                   cudaMemcpyDeviceToHost);
        cudaCheckErrors("copy tau from device to host.");
    }
    else
    {
        cudaSetDevice(iGPU[0]);
        cudaMemcpy(arrayO,
                   dev_O[0],
                   reduceO * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaCheckErrors("copy O2D from device to host.");

        cudaMemcpy(arrayC,
                   dev_C[0],
                   reduceC * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaCheckErrors("copy O2D from device to host.");
        
        int reduceTau;
        if (mode)
            reduceTau = tauSize;
        else
            reduceTau = nk * tauSize;
        
        cudaMemcpy(arrayTau,
                   devTau[0],
                   reduceTau * sizeof(RFLOAT),
                   cudaMemcpyDeviceToHost);
        cudaCheckErrors("copy tau from device to host.");

    }
    
    MPI_Barrier(hemi);
    
    LOG(INFO) << "rank" << rankF << ":Step3: Copy done, free Volume and Nccl object.";

    //finalizing NCCL
    for (int i = 0; i < nGPU; i++)
    {
        ncclCommDestroy(commF[i]);
        ncclCommDestroy(commO[i]);
    }
    
    delete[] gpus;
}

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
             int nGPU)
{
    cudaHostUnregister(volumeF);
    cudaHostUnregister(volumeT);
    cudaHostUnregister(arrayTau);
    cudaHostUnregister(arrayO);
    cudaHostUnregister(arrayC);
    cudaCheckErrors("Unregister memory.");

    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        
        cudaFree(dev_F[n]);
        cudaFree(dev_T[n]);
        cudaFree(devTau[n]);
        cudaFree(dev_O[n]);
        cudaFree(dev_C[n]);
        cudaFree(deviCol[n]);
        cudaFree(deviRow[n]);
        cudaFree(deviSig[n]);
        cudaCheckErrors("free device memory.");
    }
}

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
                 const int dim)
{
    int dimSize = (dim / 2 + 1) * dim * dim;
    int nImgBatch = VOLUME_BATCH_3D * dim * (dim / 2 + 1);
#ifdef RECONSTRUCTOR_NORMALISE_T_F
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim / 2 + 1;
#endif
    
    cudaHostRegister(T3D, 
                     dimSize * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");

    cudaHostRegister(F3D, 
                     dimSize * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");

    LOG(INFO) << "Step1: NormalizeT.";

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;
    Complex* device_F[nStream];
    RFLOAT* device_T[nStream];
    RFLOAT* devSF[nGPU];

    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devSF[n], 
                   sizeof(RFLOAT));
        cudaCheckErrors("Allocate __device__F data.");

        cudaMemcpy(devSF[n],
                   &sf,
                   sizeof(RFLOAT),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaMalloc((void**)&device_F[i + baseS], 
                       nImgBatch * sizeof(Complex));
            cudaCheckErrors("Allocate __device__F data.");

            cudaMalloc((void**)&device_T[i + baseS], 
                       nImgBatch * sizeof(RFLOAT));
            cudaCheckErrors("Allocate __device__T data.");
        }
    }

    int batch;
    int smidx = 0;
    for (int i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(device_F[smidx + baseS],
                            F3D + i,
                            batch * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            cudaMemcpyAsync(device_T[smidx + baseS],
                            T3D + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

#ifdef RECONSTRUCTOR_NORMALISE_T_F
            kernel_NormalizeTF<<<dim * VOLUME_BATCH_3D,
                                 threadInBlock,
                                 0,
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(device_F[smidx + baseS],
                                                                            device_T[smidx + baseS],
                                                                            devSF[n],
                                                                            batch);
            cudaCheckErrors("normalTF.");
#endif

            cudaMemcpyAsync(F3D + i,
                            device_F[smidx + baseS],
                            batch * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy F.");

            cudaMemcpyAsync(T3D + i,
                            device_T[smidx + baseS],
                            batch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy T.");

            i += batch;
        }
        
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    cudaHostUnregister(F3D);
    cudaHostUnregister(T3D);

    LOG(INFO) << "Step2: Clean up the streams and memory.";

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(device_F[i + baseS]);
            cudaCheckErrors("Free device memory device_F.");
            cudaFree(device_T[i + baseS]);
            cudaCheckErrors("Free device memory device_T.");
        }
    }
}

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
                  const int r)
{
    int symMatsize = nSymmetryElement * 9;
    size_t dimSize = (dim / 2 + 1) * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * (dim / 2 + 1);
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim / 2 + 1;
    
    cudaHostRegister(T3D, 
                     dimSize * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");

    cudaHostRegister(F3D, 
                     dimSize * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;
    Complex* device_F[nStream];
    RFLOAT* device_T[nStream];
    double *devSymmat[nGPU];

    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devSymmat[n], 
                   symMatsize * sizeof(double));
        cudaCheckErrors("Allocate devSymmat data.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaMalloc((void**)&device_F[i + baseS], 
                       nImgBatch * sizeof(Complex));
            cudaCheckErrors("Allocate __device__F data.");

            cudaMalloc((void**)&device_T[i + baseS], 
                       nImgBatch * sizeof(RFLOAT));
            cudaCheckErrors("Allocate __device__T data.");
        }
    }

    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaMemcpyAsync(devSymmat[n],
                        symMat,
                        symMatsize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("copy symmat for memcpy.");

    }

    LOG(INFO) << "Step1: Symmetrize F.";

    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);

    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;

#ifdef SINGLE_PRECISION
    cudaChannelFormatDesc channelDescF =cudaCreateChannelDesc(32, 32, 0, 0,cudaChannelFormatKindFloat);
#else
    cudaChannelFormatDesc channelDescF =cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindSigned);
#endif
    cudaArray *symArrayF[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        
        cudaMalloc3DArray(&symArrayF[n], 
                          &channelDescF, 
                          extent);
        cudaCheckErrors("malloc error\n.");
    }

    cudaMemcpy3DParms tempP[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        
        memset(&tempP[n], 0, sizeof(tempP[n]));
#ifdef SINGLE_PRECISION
        tempP[n].srcPtr   = make_cudaPitchedPtr((void*)F3D, (dim / 2 + 1) * sizeof(float2), dim / 2 + 1, dim);
#else
        tempP[n].srcPtr   = make_cudaPitchedPtr((void*)F3D, (dim / 2 + 1) * sizeof(int4), dim / 2 + 1, dim);
#endif
        tempP[n].dstArray = symArrayF[n];
        tempP[n].extent   = extent;
        tempP[n].kind     = cudaMemcpyHostToDevice;

        cudaMemcpy3DAsync(&tempP[n],
                          *((cudaStream_t*)stream[baseS + 1]));
        cudaCheckErrors("memcpy error\n.");
    }

    struct cudaResourceDesc resDescF[nGPU];
    cudaTextureObject_t texObjectF[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        
        memset(&resDescF[n], 0, sizeof(resDescF[n]));
        resDescF[n].resType = cudaResourceTypeArray;
        resDescF[n].res.array.array = symArrayF[n];

        cudaCreateTextureObject(&texObjectF[n], 
                                &resDescF[n], 
                                &td, 
                                NULL);
        cudaCheckErrors("create Texture object\n.");
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    int smidx = 0;
    size_t batch;
    for (size_t i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) 
                    ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(device_F[smidx + baseS],
                            F3D + i,
                            batch * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            kernel_SymmetrizeF<<<dim * VOLUME_BATCH_3D,
                                 threadInBlock,
                                 0,
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(device_F[smidx + baseS],
                                                                            devSymmat[n],
                                                                            nSymmetryElement,
                                                                            r,
                                                                            interp,
                                                                            i,
                                                                            dim,
                                                                            batch,
                                                                            texObjectF[n]);
            cudaCheckErrors("symT for stream 0");

            cudaMemcpyAsync(F3D + i,
                            device_F[smidx + baseS],
                            batch * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
   
    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaDestroyTextureObject(texObjectF[n]);
        cudaFreeArray(symArrayF[n]);
        cudaCheckErrors("Free device memory SymArrayF.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(device_F[i + baseS]);
            cudaCheckErrors("Free device memory device_F.");
        }
    }

    LOG(INFO) << "Step2: Symmetrize T.";

#ifdef SINGLE_PRECISION
    cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
#else
    cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc(32, 32, 0, 0,cudaChannelFormatKindSigned);
#endif
    cudaArray *symArrayT[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        
        cudaMalloc3DArray(&symArrayT[n], 
                          &channelDescT, 
                          extent);
        cudaCheckErrors("memcpy error\n.");
    }

    cudaMemcpy3DParms copyParamsT[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        memset(&copyParamsT[n], 0, sizeof(copyParamsT[n]));
#ifdef SINGLE_PRECISION
        copyParamsT[n].srcPtr   = make_cudaPitchedPtr((void*)T3D, (dim / 2 + 1) * sizeof(float), dim / 2 + 1, dim);
#else
        copyParamsT[n].srcPtr   = make_cudaPitchedPtr((void*)T3D, (dim / 2 + 1) * sizeof(int2), dim / 2 + 1, dim);
#endif
        copyParamsT[n].dstArray = symArrayT[n];
        copyParamsT[n].extent   = extent;
        copyParamsT[n].kind     = cudaMemcpyHostToDevice;

        cudaMemcpy3DAsync(&copyParamsT[n],
                          *((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("memcpy error\n.");
    }

    struct cudaResourceDesc resDescT[nGPU];
    cudaTextureObject_t texObjectT[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        
        memset(&resDescT[n], 0, sizeof(resDescT[n]));
        resDescT[n].resType = cudaResourceTypeArray;
        resDescT[n].res.array.array = symArrayT[n];

        cudaCreateTextureObject(&texObjectT[n], 
                                &resDescT[n], 
                                &td, 
                                NULL);

    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    LOG(INFO) << "Step3: SymmetrizeT.";

    smidx = 0;
    for (size_t i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(device_T[smidx + baseS],
                            T3D + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            kernel_SymmetrizeT<<<dim * VOLUME_BATCH_3D,
                                 threadInBlock,
                                 0,
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(device_T[smidx + baseS],
                                                                            devSymmat[n],
                                                                            nSymmetryElement,
                                                                            r,
                                                                            interp,
                                                                            i,
                                                                            dim,
                                                                            batch,
                                                                            texObjectT[n]);
            cudaCheckErrors("symT for stream 0");

            cudaMemcpyAsync(T3D + i,
                            device_T[smidx + baseS],
                            batch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    LOG(INFO) << "Step3: Clean up the streams and memory.";

    cudaHostUnregister(F3D);
    cudaHostUnregister(T3D);
    
    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaDestroyTextureObject(texObjectT[n]);

        cudaFreeArray(symArrayT[n]);
        cudaCheckErrors("Free device memory SymArrayT.");
        cudaFree(devSymmat[n]);
        cudaCheckErrors("Free device memory devSymmat.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(device_T[i + baseS]);
            cudaCheckErrors("Free device memory device_T.");
        }
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocVolume(vector<int>& iGPU,
                 RFLOAT** dev_T,
                 RFLOAT** dev_W,
                 int nGPU,
                 size_t dimSize)
{
    for (int n = 0; n < nGPU; n++)
    { 
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("stream create.");
        
        cudaMalloc((void**)&dev_W[n], 
                   dimSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devDataW data.");

        cudaMalloc((void**)&dev_T[n], 
                   dimSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devDataT data.");
    }
}

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
                  int fscMatSize,
                  int maxRadius,
                  int wienerF,
                  int dim,
                  int pf,
                  int kbatch,
                  int nGPU)
{
    int wiener = pow(wienerF * pf, 2);
    int r = pow(maxRadius * pf, 2);
    int dimSize = (dim / 2 + 1) * dim;
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                      ? THREAD_PER_BLOCK : dim / 2 + 1;

    RFLOAT *devFSC[nGPU];
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devFSC[n], 
                   kbatch * fscMatSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devFSC data.");
    }

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    LOG(INFO) << "Step1: ShellAverage.";
    int vecSize = maxRadius * pf + 1;
    int gridSize, mergeThread;
    int avgTemp = (vecSize - 2) * dim;
    int avgThread = (dim > THREAD_PER_BLOCK) 
                  ? THREAD_PER_BLOCK : dim;
    
    if (vecSize - 2 > THREAD_PER_BLOCK)
    {
        mergeThread = THREAD_PER_BLOCK;
        if ((vecSize - 2) % mergeThread == 0) 
            gridSize = (vecSize - 2) / mergeThread;
        else
            gridSize = (vecSize - 2) / mergeThread + 1;
    }
    else
    {
        mergeThread = vecSize - 2;
        gridSize = 1;
    }

    RFLOAT *devAvg[nGPU];
    RFLOAT *devAvg2D[nGPU];
    int *devCount2D[nGPU];
    int *devCount[nGPU];
    
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devAvg[n], 
                   kbatch * vecSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate deviceAvg data.");
        
        cudaMalloc((void**)&devAvg2D[n], 
                   kbatch * avgTemp * sizeof(RFLOAT));
        cudaCheckErrors("Allocate deviceAvg2D data.");
        
        cudaMalloc((void**)&devCount[n], 
                   kbatch * vecSize * sizeof(int));
        cudaCheckErrors("Allocate deviceCount data.");
        
        cudaMalloc((void**)&devCount2D[n], 
                   kbatch * avgTemp * sizeof(int));
        cudaCheckErrors("Allocate deviceCount2D data.");
    }
#endif

    int smidx = 0;
    for (int t = 0; t < kbatch;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (t >= kbatch)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(dev_T[n] + t * dimSize,
                            T2D + t * dimSize,
                            dimSize * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            cudaMemcpyAsync(devFSC[n] + t * fscMatSize,
                            fscMat + t * fscMatSize,
                            fscMatSize * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
            kernel_ShellAverage2D<<<dim,
                                    threadInBlock,
                                    (vecSize - 2)
                                     * (sizeof(RFLOAT)
                                        + sizeof(int)),
                                    *((cudaStream_t*)stream[smidx + baseS])>>>(devAvg2D[n] + t * avgTemp,
                                                                               devCount2D[n] + t * avgTemp,
                                                                               dev_T[n] + t * dimSize,
                                                                               dim,
                                                                               vecSize - 2);
            cudaCheckErrors("Shell for stream default.");

            kernel_CalculateAvg<<<vecSize - 2,
                                  avgThread,
                                  avgThread * (sizeof(RFLOAT) 
                                               + sizeof(int)),
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(devAvg2D[n] + t * avgTemp,
                                                                             devCount2D[n] + t * avgTemp,
                                                                             devAvg[n] + t * vecSize,
                                                                             devCount[n] + t * vecSize,
                                                                             dim);
            cudaCheckErrors("calAvg for stream default.");
            
            kernel_mergeAvgCount<<<gridSize,
                                   mergeThread,
                                   0,
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(devAvg[n] + t * vecSize,
                                                                              devCount[n] + t * vecSize,
                                                                              vecSize - 2);
            cudaCheckErrors("Merge Avg for stream.");
           
            kernel_CalculateFSC2D<<<dim,
                                    threadInBlock,
                                    0,
                                    *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n] + t * dimSize,
                                                                               devFSC[n] + t * fscMatSize,
                                                                               devAvg[n] + t * vecSize,
                                                                               joinHalf,
                                                                               fscMatSize,
                                                                               wiener,
                                                                               dim,
                                                                               pf,
                                                                               r);
            cudaCheckErrors("calFSC for stream 0.");

#else
            kernel_CalculateFSC2D<<<dim,
                                    threadInBlock,
                                    0,
                                    *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n] + t * dimSize,
                                                                               devFSC[n] + t * fscMatSize,
                                                                               joinHalf,
                                                                               fscMatSize,
                                                                               wiener,
                                                                               dim,
                                                                               pf,
                                                                               r);
            cudaCheckErrors("calFSC for stream 0.");
#endif
            cudaMemcpyAsync(T2D + t * dimSize,
                            dev_T[n] + t * dimSize,
                            dimSize * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("out for memcpy 0.");

            t++;
        }
        
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
        cudaFree(devAvg[n]);
        cudaCheckErrors("Free device memory devFSC.");
        cudaFree(devAvg2D[n]);
        cudaCheckErrors("Free device memory devAvg2D.");
        cudaFree(devCount2D[n]);
        cudaCheckErrors("Free device memory devCount2D.");
        cudaFree(devCount[n]);
        cudaCheckErrors("Free device memory devCount2D.");
#endif
        cudaFree(devFSC[n]);
        cudaCheckErrors("Free device memory devFSC.");
    }
}

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
                int pf)
{
    size_t dimSize = (dim / 2 + 1) * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * (dim / 2 + 1);
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim / 2 + 1;

    cudaHostRegister(T3D, 
                     dimSize * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");

    ncclComm_t commAvg[nGPU];
    int* gpus = new int[nGPU];
    for (int n = 0; n < nGPU; n++)
        gpus[n] = iGPU[n];
    NCCLCHECK(ncclCommInitAll(commAvg, nGPU, gpus));

    size_t batch;
    int smidx = 0;
    int baseS;
    RFLOAT *devFSC[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devFSC[n], 
                   fscMatsize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devFSC data.");
    }

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    LOG(INFO) << "Step1: ShellAverage.";

    int vecSize = maxRadius * pf + 1;
    int avgTemp = (vecSize - 2) * dim * NUM_STREAM_PER_DEVICE;
    RFLOAT* devAvg[nGPU];
    RFLOAT *devAvg2D[nGPU];
    int *devCount2D[nGPU];
    int* devCount[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devAvg[n], 
                   vecSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate deviceAvg data.");
        
        cudaMalloc((void**)&devAvg2D[n], 
                   avgTemp * sizeof(RFLOAT));
        cudaCheckErrors("Allocate deviceAvg2D data.");
        
        cudaMemset(devAvg2D[n], 
                   0.0, 
                   avgTemp * sizeof(RFLOAT));
        cudaCheckErrors("Memset devAVG2D data.");

        cudaMalloc((void**)&devCount[n], 
                   vecSize * sizeof(int));
        cudaCheckErrors("Allocate deviceCount data.");
        
        cudaMalloc((void**)&devCount2D[n], 
                   avgTemp * sizeof(int));
        cudaCheckErrors("Allocate deviceCount2D data.");
        
        cudaMemset(devCount2D[n], 
                   0.0, 
                   avgTemp * sizeof(int));
        cudaCheckErrors("Memset devCount2D data.");
    }

    int lineSize = dim * NUM_STREAM_PER_DEVICE;
    int smidxShift = smidx * dim;

    for (size_t i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) 
                    ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(dev_T[n] + i,
                            T3D + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            smidxShift = smidx * dim;
            kernel_ShellAverage<<<dim,
                                  threadInBlock,
                                  (vecSize - 2) * (sizeof(RFLOAT)
                                                + sizeof(int)),
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(devAvg2D[n],
                                                                             devCount2D[n],
                                                                             dev_T[n] + i,
                                                                             i,
                                                                             smidxShift,
                                                                             lineSize,
                                                                             vecSize - 2,
                                                                             dim,
                                                                             batch);
            cudaCheckErrors("Shell for stream smidx.");

            i += batch;
        }
        
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    threadInBlock = (lineSize > THREAD_PER_BLOCK) 
                    ? THREAD_PER_BLOCK : lineSize;
    for (int i = 0; i < nGPU; i++)
    {
        baseS = i * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[i]);
        
        kernel_CalculateAvg<<<vecSize - 2,
                              threadInBlock,
                              threadInBlock * (sizeof(RFLOAT) 
                                               + sizeof(int)),
                              *((cudaStream_t*)stream[smidx + baseS])>>>(devAvg2D[i],
                                                                         devCount2D[i],
                                                                         devAvg[i],
                                                                         devCount[i],
                                                                         lineSize);
        cudaCheckErrors("calAvg for stream.");
        
        cudaMemcpyAsync(devFSC[i],
                        FSC,
                        fscMatsize * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)stream[1 + baseS]));
        cudaCheckErrors("copy FSC for memcpy.");
    }
    
    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaStreamSynchronize(*((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("Stream synchronize after.");
        
        cudaStreamSynchronize(*((cudaStream_t*)stream[1 + baseS]));
        cudaCheckErrors("Stream synchronize after.");
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)devAvg[i],
                                (void*)devAvg[i],
                                vecSize - 2,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commAvg[i],
                                *((cudaStream_t*)stream[i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)devCount[i],
                                (void*)devCount[i],
                                vecSize - 2,
                                ncclInt,
                                ncclSum,
                                commAvg[i],
                                *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaStreamSynchronize(*((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("Stream synchronize after.");
        cudaStreamSynchronize(*((cudaStream_t*)stream[1 + baseS]));
        cudaCheckErrors("Stream synchronize after.");
    }

    int gridSize;
    if (vecSize - 2 > THREAD_PER_BLOCK)
    {
        threadInBlock = THREAD_PER_BLOCK;
        if ((vecSize - 2) % threadInBlock == 0) 
            gridSize = (vecSize - 2) / threadInBlock;
        else
            gridSize = (vecSize - 2) / threadInBlock + 1;
    }
    else
    {
        threadInBlock = vecSize - 2;
        gridSize = 1;
    }
    for (int i = 0; i < nGPU; i++)
    {
        baseS = i * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[i]);
        
        kernel_mergeAvgCount<<<gridSize,
                               threadInBlock,
                               0,
                               *((cudaStream_t*)stream[baseS])>>>(devAvg[i],
                                                                  devCount[i],
                                                                  vecSize - 2);
        cudaCheckErrors("Merge Avg for stream.");
    }
    
    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaFree(devAvg2D[n]);
        cudaCheckErrors("Free device memory devAvg2D.");
        cudaFree(devCount2D[n]);
        cudaCheckErrors("Free device memory devCount2D.");
        cudaFree(devCount[n]);
        cudaCheckErrors("Free device memory devCount2D.");
    }
#endif

    LOG(INFO) << "Step2: Calculate WIENER_FILTER.";

#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    for (int i = 0; i < nGPU; i++)
    {
        baseS = i * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[i]);
        
        cudaMemcpyAsync(devFSC[i], 
                        FSC, 
                        fscMatsize * sizeof(RFLOAT), 
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("Copy FSC to device.");
    }
#endif
    
    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                    ? THREAD_PER_BLOCK : dim / 2 + 1;
    
    int wiener = wienerF * wienerF * pf * pf;
    int r = maxRadius * maxRadius * pf * pf;
    smidx = 0;
    int baseC;
    for (size_t i = 0; i < dimSize; )
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) 
                    ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
            kernel_CalculateFSC<<<dim * VOLUME_BATCH_3D,
                                  threadInBlock,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n] + i,
                                                                             devFSC[n],
                                                                             devAvg[n],
                                                                             i,
                                                                             joinHalf,
                                                                             fscMatsize,
                                                                             wiener,
                                                                             r,
                                                                             pf,
                                                                             dim,
                                                                             batch);
            cudaCheckErrors("calFSC for stream smidx.");
#else
            cudaMemcpyAsync(dev_T[n] + i,
                            T3D + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            kernel_CalculateFSC<<<dim * VOLUME_BATCH_3D,
                                  threadInBlock,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_T[n] + i,
                                                                             devFSC[n],
                                                                             i,
                                                                             joinHalf,
                                                                             fscMatsize,
                                                                             wiener,
                                                                             r,
                                                                             pf,
                                                                             dim,
                                                                             batch);
            cudaCheckErrors("calFSC for stream smidx.");
#endif
            for (int card = 0; card < nGPU; card++)
            {
                if (card != n)
                {
                    baseC = card * NUM_STREAM_PER_DEVICE;
                    cudaSetDevice(iGPU[card]);
                    cudaCheckErrors("set device.");
                    
                    cudaMemsetAsync(dev_T[card] + i,
                                    0.0,
                                    batch * sizeof(RFLOAT),
                                    *((cudaStream_t*)stream[smidx + baseC]));
                    cudaCheckErrors("for memcpy.");
                }
            }

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(T3D + i,
                            dev_T[n] + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx+ baseS]));
            cudaCheckErrors("out for memcpy.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)dev_T[i],
                                (void*)dev_T[i],
                                dimSize,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commAvg[i],
                                *((cudaStream_t*)stream[i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    cudaHostUnregister(T3D);
    cudaCheckErrors("Unregister T3D.");

    LOG(INFO) << "Step3: Clean up the memory.";

    //finalizing NCCL
    for (int i = 0; i < nGPU; i++)
    {
        ncclCommDestroy(commAvg[i]);
    }

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
        cudaFree(devAvg[n]);
        cudaCheckErrors("Free device memory devFSC.");
#endif
        cudaFree(devFSC[n]);
        cudaCheckErrors("Free device memory devFSC.");
    }
    delete[] gpus;
}

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
                        int nGPU)
{
    int dimSize = dim * (dim / 2 + 1); 
    int dimSizeRL = dim * dim; 
    int baseS;

    cufftHandle *c2r[nGPU * NUM_STREAM_PER_DEVICE]; 
    cufftHandle *r2c[nGPU * NUM_STREAM_PER_DEVICE]; 

    for (int n = 0; n < nGPU; n++)
    { 
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");
#ifdef SINGLE_PRECISION
        cudaMalloc((void**)&devFourC[n], 
                   kbatch * dimSize * sizeof(cufftComplex));
        cudaCheckErrors("Allocate device memory for C.");
        
        cudaMalloc((void**)&devRealC[n], 
                   kbatch * dimSizeRL * sizeof(cufftReal));
        cudaCheckErrors("Allocate device memory for C.");
#else
        cudaMalloc((void**)&devFourC[n], 
                   kbatch * dimSize * sizeof(cufftDoubleComplex));
        cudaCheckErrors("Allocate device memory for C.");
        
        cudaMalloc((void**)&devRealC[n], 
                   kbatch * dimSizeRL * sizeof(cufftDoubleReal));
        cudaCheckErrors("Allocate device memory for C.");
#endif
        cudaMalloc((void**)&dev_tab[n], 
                   tabSize * sizeof(RFLOAT));
        cudaCheckErrors("Alloc device memory for tabfunction.");
    
        cudaMemcpy(dev_tab[n],
                   tab,
                   tabSize * sizeof(RFLOAT),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("Copy tabfunction to device.");

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
        cudaMalloc((void**)&devDiff[n], 
                   dim * kbatch
                       * sizeof(RFLOAT));
        cudaCheckErrors("Allocate device memory for devDiff.");
        cudaMalloc((void**)&devCount[n], 
                   dim * kbatch
                       * sizeof(int));
        cudaCheckErrors("Allocate device memory for devcount.");
#endif
#ifdef RECONSTRUCTOR_CHECK_C_MAX
        cudaMalloc((void**)&devMax[n], 
                   dim * kbatch
                       * sizeof(RFLOAT));
        cudaCheckErrors("Allocate device memory for devMax.");
#endif
        baseS = n * NUM_STREAM_PER_DEVICE;
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            c2r[i + baseS] = (cufftHandle*)malloc(sizeof(cufftHandle));
            r2c[i + baseS] = (cufftHandle*)malloc(sizeof(cufftHandle));
#ifdef SINGLE_PRECISION
            CUFFTCHECK(cufftPlan2d(c2r[i + baseS], 
                                   dim, 
                                   dim, 
                                   CUFFT_C2R));
            CUFFTCHECK(cufftPlan2d(r2c[i + baseS], 
                                   dim, 
                                   dim, 
                                   CUFFT_R2C));
#else
            CUFFTCHECK(cufftPlan2d(c2r[i + baseS], 
                                   dim, 
                                   dim, 
                                   CUFFT_Z2D));
            CUFFTCHECK(cufftPlan2d(r2c[i + baseS], 
                                   dim, 
                                   dim, 
                                   CUFFT_D2Z));
#endif
            cufftSetStream(*c2r[i + baseS], 
                           *((cudaStream_t*)stream[i + baseS]));
            cufftSetStream(*r2c[i + baseS], 
                           *((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Set plan stream.");

            planC2R.push_back(c2r[i + baseS]);
            planR2C.push_back(r2c[i + baseS]);
        }
    }
}

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
                  int nGPU)
{
    cudaSetDevice(gpuIdx);
    cudaCheckErrors("set device.");
    
    int dimSize = (dim / 2 + 1) * dim;
    int dimSizeRL = dim * dim;
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                      ? THREAD_PER_BLOCK : dim / 2 + 1;

    /* Upload tabfunction to device */
    TabFunction tabfunc;
    tabfunc.init(begin,
                 end,
                 step,
                 NULL,
                 tabsize);
    tabfunc.devPtr(dev_tab[gpuIdx]);

    if (!map)
    {
        cudaMemcpyAsync(dev_T[gpuIdx] + kIdx * dimSize,
                        modelT + kIdx * dimSize,
                        dimSize * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)stream));
        cudaCheckErrors("for memcpy.");
    }

    kernel_InitialW2D<<<dim,
                        threadInBlock,
                        0,
                        *((cudaStream_t*)stream)>>>(dev_W[gpuIdx] + kIdx * dimSize,
                                                    r,
                                                    dim);
    cudaCheckErrors("Init W2D error.");

    cudaStreamSynchronize(*((cudaStream_t*)stream));
    cudaCheckErrors("Stream synchronize after.");

    RFLOAT diffC;
    RFLOAT diffCPrev;
#ifdef SINGLE_PRECISION
    diffC = FLT_MAX;
    diffCPrev = FLT_MAX;
#else
    diffC = DBL_MAX;
    diffCPrev = DBL_MAX;
#endif

    int nDiffCNoDecrease = 0;
    for(int m = 0; m < maxIter; m++)
    {
        kernel_DeterminingC2D<<<dim,
                                threadInBlock,
                                0,
                                *((cudaStream_t*)stream)>>>(devFourC[gpuIdx] + kIdx * dimSize,
                                                            dev_T[gpuIdx] + kIdx * dimSize,
                                                            dev_W[gpuIdx] + kIdx * dimSize,
                                                            dimSize);
        cudaCheckErrors("kernel determining C.");

#ifdef SINGLE_PRECISION
        cufftExecC2R(*((cufftHandle*)planC2R), 
                     (cufftComplex*)devFourC[gpuIdx] + kIdx * dimSize, 
                     (cufftReal*)devRealC[gpuIdx] + kIdx * dimSizeRL);

        kernel_convoluteC2D<<<dim,
                              threadInBlock,
                              0,
                              *((cudaStream_t*)stream)>>>(devRealC[gpuIdx] + kIdx * dimSizeRL,
                                                          tabfunc,
                                                          nf,
                                                          padSize,
                                                          dim,
                                                          dimSizeRL);
        cudaCheckErrors("kernel convoluteC.");

        cufftExecR2C(*((cufftHandle*)planR2C), 
                     (cufftReal*)devRealC[gpuIdx] + kIdx * dimSizeRL, 
                     (cufftComplex*)devFourC[gpuIdx] + kIdx * dimSize);
#else
        cufftExecZ2D(*((cufftHandle*)planC2R), 
                     (cufftDoubleComplex*)devFourC[gpuIdx] + kIdx * dimSize, 
                     (cufftDoubleReal*)devRealC[gpuIdx] + kIdx * dimSizeRL);

        kernel_convoluteC2D<<<dim,
                              threadInBlock,
                              0,
                              *((cudaStream_t*)stream)>>>(deRealC[gpuIdx] + kIdx * dimSizeRL,
                                                          tabfunc,
                                                          nf,
                                                          padSize,
                                                          dim,
                                                          dimSizeRL);

        cufftExecD2Z(*((cufftHandle*)planR2C), 
                     (cufftDoubleReal*)devRealC[gpuIdx] + kIdx * dimSizeRL, 
                     (cufftDoubleComplex*)devFourC[gpuIdx] + kIdx * dimSize);
#endif

        kernel_RecalculateW2D<<<dim,
                                threadInBlock,
                                0,
                                *((cudaStream_t*)stream)>>>(dev_W[gpuIdx] + kIdx * dimSize,
                                                            devFourC[gpuIdx] + kIdx * dimSize,
                                                            r,
                                                            dim);
        cudaCheckErrors("kernel recalculateW.");

        diffCPrev = diffC;

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
        kernel_CheckCAVG2D<<<dim,
                             threadInBlock,
                             threadInBlock * (sizeof(RFLOAT)
                                           + sizeof(int)),
                             *((cudaStream_t*)stream)>>>(devDiff[gpuIdx] + kIdx * dim,
                                                         devCount[gpuIdx] + kIdx * dim,
                                                         devFourC[gpuIdx] + kIdx * dimSize,
                                                         r,
                                                         dim);
        cudaCheckErrors("kernel checkCAVG.");
        
        kernel_mergeCAvg<<<1,
                           threadInBlock,
                           threadInBlock * (sizeof(RFLOAT)
                                            + sizeof(int)),
                           *((cudaStream_t*)stream)>>>(devDiff[gpuIdx] + kIdx * dim,
                                                       devCount[gpuIdx] + kIdx * dim,
                                                       dim);
        cudaCheckErrors("Merge C Avg for stream.");
        
        RFLOAT tempD = 0;
        int tempC = 0;

        cudaMemcpyAsync(&tempD,
                        devDiff[gpuIdx] + kIdx * dim,
                        sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)stream));
        cudaCheckErrors("Copy devDiff array to host.");
        
        cudaMemcpyAsync(&tempC,
                        devCount[gpuIdx] + kIdx * dim,
                        sizeof(int),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)stream));
        cudaCheckErrors("Copy devDiff array to host.");
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
        kernel_CheckCMAX2D<<<dim,
                             threadInBlock,
                             threadInBlock * sizeof(RFLOAT),
                             *((cudaStream_t*)stream)>>>(devMax[gpuIdx] + kIdx * dim,
                                                         devFourC[gpuIdx] + kIdx * dimSize,
                                                         r,
                                                         dim);
        cudaCheckErrors("kernel checkCMAX.");
        
        kernel_mergeCMax<<<1,
                           threadInBlock,
                           threadInBlock * sizeof(RFLOAT),
                           *((cudaStream_t*)stream)>>>(devMax[gpuIdx] + kIdx * dim,
                                                       dim);
        cudaCheckErrors("Merge C Max for stream.");
        
        cudaMemcpyAsync(&diffC,
                        devMax[gpuIdx] + kIdx * dim,
                        sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)stream));
        cudaCheckErrors("Copy devMax array to host.");
#endif

        cudaStreamSynchronize(*((cudaStream_t*)stream));
        cudaCheckErrors("Stream synchronize after.");

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
        diffC = tempD / tempC;
#endif
        
        if (diffC > diffCPrev * diffC_DThres)
            nDiffCNoDecrease += 1;
        else
            nDiffCNoDecrease = 0;

        if ((diffC < diffC_Thres) ||
            ((m >= minIter) &&
            (nDiffCNoDecrease == noDiffC))) break;
    }
}

/**
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
                 int nGPU)
{
    int shift = 0;
    for (int n = 0; n < nGPU; n++)
    { 
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");
        
        cudaFree(devFourC[n]);
        cudaCheckErrors("Free devFourC data.");
        cudaFree(devRealC[n]);
        cudaCheckErrors("Free devRealC data.");
        cudaFree(dev_T[n]);
        cudaCheckErrors("Free devDataT data.");
        cudaFree(dev_tab[n]);
        cudaCheckErrors("Free dev_tab data.");
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
        cudaFree(devDiff[n]);
        cudaCheckErrors("Free devdiff data.");
        cudaFree(devCount[n]);
        cudaCheckErrors("Free devcount data.");
#endif
#ifdef RECONSTRUCTOR_CHECK_C_MAX
        cudaFree(devMax[n]);
        cudaCheckErrors("Free devcount data.");
#endif
        
        shift = n * NUM_STREAM_PER_DEVICE;
        for(int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cufftDestroy(*((cufftHandle*)planC2R[i + shift]));
            cudaCheckErrors("DestroyPlan planc2r.");

            cufftDestroy(*((cufftHandle*)planR2C[i + shift]));
            cudaCheckErrors("DestroyPlan planr2c.");
        }
    }
    
    planC2R.clear();
    planR2C.clear();
}

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
                      int nGPU)
{
    size_t nImgBatch = VOLUME_BATCH_3D * dim 
                                       * (dim / 2 + 1);

    int shift = 0;
    for (int n = 0; n < nGPU; n++)
    { 
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("stream create.");
        
        cudaMalloc((void**)&dev_tab[n], 
                   tabSize * sizeof(RFLOAT));
        cudaCheckErrors("Alloc device memory for tabfunction.");
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
        cudaMalloc((void**)&devDiff[n], 
                   dim * NUM_STREAM_PER_DEVICE
                       * sizeof(RFLOAT));
        cudaCheckErrors("Allocate device memory for devDiff.");
        cudaMalloc((void**)&devCount[n], 
                   dim * NUM_STREAM_PER_DEVICE
                       * sizeof(int));
        cudaCheckErrors("Allocate device memory for devcount.");
#endif
#ifdef RECONSTRUCTOR_CHECK_C_MAX
        cudaMalloc((void**)&devMax[n], 
                   dim * NUM_STREAM_PER_DEVICE
                       * sizeof(RFLOAT));
        cudaCheckErrors("Allocate device memory for devMax.");
#endif
        
        shift = n * NUM_STREAM_PER_DEVICE;
        for(int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaMalloc((void**)&devPartC[shift + i], 
                       nImgBatch * sizeof(Complex));
            cudaCheckErrors("Allocate devDataC data.");
        }
    }
}

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
                    int dim)
{
    size_t dimSize = (dim / 2 + 1) * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * (dim / 2 + 1);
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim / 2 + 1;

    cudaHostRegister(volumeC, 
                     dimSize * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register C3D data.");

#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC
    cudaHostRegister(T3D, 
                     dimSize * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");
#else
    if (!map)
    {
        cudaHostRegister(T3D, 
                         dimSize * sizeof(RFLOAT), 
                         cudaHostRegisterDefault);
        cudaCheckErrors("Register T3D data.");
    }
#endif
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");
        
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaMemcpyAsync(dev_tab[n],
                        tab,
                        tabSize * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("Copy tabfunction to device.");
    }
    
    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    size_t batch;
    int smidx = 0;
    int baseC;
    for (size_t i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) 
                    ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC
            cudaMemcpyAsync(dev_T[n] + i,
                            T3D + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");
#else
            if (!map)
            {
                cudaMemcpyAsync(dev_T[n] + i,
                                T3D + i,
                                batch * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("for memcpy.");
            }
#endif
            for (int card = 0; card < nGPU; card++)
            {
                if (card != n)
                {
                    baseC = card * NUM_STREAM_PER_DEVICE;
                    cudaSetDevice(iGPU[card]);
                    cudaCheckErrors("set device.");
                    
#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC
                    cudaMemsetAsync(dev_T[card] + i,
                                    0.0,
                                    batch * sizeof(RFLOAT),
                                    *((cudaStream_t*)stream[smidx + baseC]));
                    cudaCheckErrors("for memcpy.");
#else
                    if (!map)
                    {
                        cudaMemsetAsync(dev_T[card] + i,
                                        0.0,
                                        batch * sizeof(RFLOAT),
                                        *((cudaStream_t*)stream[smidx + baseC]));
                        cudaCheckErrors("for memcpy.");
                    }
#endif
                    cudaMemsetAsync(dev_W[card] + i,
                                    0.0,
                                    batch * sizeof(RFLOAT),
                                    *((cudaStream_t*)stream[smidx + baseC]));
                    cudaCheckErrors("for memcpy.");
                }
            }

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            kernel_InitialW<<<dim * VOLUME_BATCH_3D,
                              threadInBlock,
                              0,
                              *((cudaStream_t*)stream[smidx + baseS])>>>(dev_W[n] + i,
                                                                         r,
                                                                         i,
                                                                         dim,
                                                                         batch);
            cudaCheckErrors("Kernel Init W.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC
    cudaHostUnregister(T3D);
    cudaCheckErrors("T3D host Unregister.");
#else
    if (!map)
    {
        cudaHostUnregister(T3D);
        cudaCheckErrors("T3D host Unregister.");
    }
#endif
   
    ncclComm_t commNccl[nGPU];
    int* gpus = new int[nGPU];
    for (int n = 0; n < nGPU; n++)
        gpus[n] = iGPU[n];
    NCCLCHECK(ncclCommInitAll(commNccl, nGPU, gpus));

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)dev_W[i],
                                (void*)dev_W[i],
                                dimSize,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commNccl[i],
                                *((cudaStream_t*)stream[i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)dev_T[i],
                                (void*)dev_T[i],
                                dimSize,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commNccl[i],
                                *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());
#else
    if (!map)
    {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < nGPU; i++)
        {
            cudaSetDevice(iGPU[i]);
            NCCLCHECK(ncclAllReduce((const void*)dev_T[i],
                                    (void*)dev_T[i],
                                    dimSize,
#ifdef SINGLE_PRECISION
                                    ncclFloat,
#else
                                    ncclDouble,
#endif
                                    ncclSum,
                                    commNccl[i],
                                    *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
        }
        NCCLCHECK(ncclGroupEnd());
    }
#endif

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    //finalizing NCCL
    for (int i = 0; i < nGPU; i++)
    {
        ncclCommDestroy(commNccl[i]);
    }

    delete[] gpus;
}

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
                int dim)
{
    size_t dimSize = (dim / 2 + 1) * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * (dim / 2 + 1);
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim / 2 + 1;

    size_t batch;
    int smidx = 0;
    int baseS;
    for (size_t i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) 
                    ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            kernel_DeterminingC<<<dim * VOLUME_BATCH_3D,
                                  threadInBlock,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>((RFLOAT*)devPartC[smidx + baseS],
                                                                             dev_T[n] + i,
                                                                             dev_W[n] + i,
                                                                             batch);
            cudaCheckErrors("kernel DeterminingC error.");

            cudaMemcpyAsync(volumeC + i,
                            (RFLOAT*)devPartC[smidx + baseS],
                            batch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
}

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
                int dim)
{
    size_t dimSizeRL = dim * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * dim;
    int threadInBlock = (dim > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim;

    cudaHostRegister(C3D, 
                     dimSizeRL * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register C3D data.");

    /* Upload tabfunction to device */
    TabFunction tabfunc[nGPU];
    for (int n = 0; n < nGPU; n++)
    {
        tabfunc[n].init(begin,
                        end,
                        step,
                        NULL,
                        tabsize);
        tabfunc[n].devPtr(dev_tab[n]);
    }

    size_t batch;
    int smidx = 0;
    int baseS;
    for (size_t i = 0; i < dimSizeRL;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSizeRL)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSizeRL) 
                    ? (dimSizeRL - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync((RFLOAT*)devPartC[smidx + baseS],
                            C3D + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            kernel_ConvoluteC<<<dim * VOLUME_BATCH_3D,
                                threadInBlock,
                                0,
                                *((cudaStream_t*)stream[smidx + baseS])>>>((RFLOAT*)devPartC[smidx + baseS],
                                                                           tabfunc[n],
                                                                           nf,
                                                                           dim,
                                                                           i,
                                                                           padSize,
                                                                           batch);
            cudaCheckErrors("kernel DeterminingC error.");

            cudaMemcpyAsync(C3D + i,
                            (RFLOAT*)devPartC[smidx + baseS],
                            batch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    cudaHostUnregister(C3D);
    cudaCheckErrors("C3D host Unregister.");
    
    //if (iter == 1)
    //{
    //    RFLOAT* c3d = new RFLOAT[dimSizeRL];
    //    FILE* pfile;
    //    pfile = fopen("c3d.txt", "rb");
    //    if (pfile == NULL)
    //        printf("open c3d error!\n");
    //    if (fread(c3d, sizeof(RFLOAT), dimSizeRL, pfile) != dimSizeRL)
    //        printf("read c3d error!\n");
    //    fclose(pfile);
    //    printf("i:%d,c3d:%.8lf,C3D:%.8lf\n",0,c3d[0], C3D[0]);
    //    int t;
    //    for (t = 0; t < dimSizeRL; t++){
    //        if (fabs(C3D[t] - c3d[t]) >= 1e-4){
    //            printf("i:%d,c3d:%.8lf,volumeC:%.8lf\n",t,c3d[t],C3D[t]);
    //            break;
    //        }
    //    }
    //    if (t == dimSizeRL)
    //        printf("successC:%d\n", dimSizeRL); 
    //}
}

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
              int dim)
{
    size_t dimSize = (dim / 2 + 1) * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * (dim / 2 + 1);
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim / 2 + 1;

    cudaHostRegister(C3D, 
                     dimSize * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register C3D data.");

    size_t batch;
    int smidx = 0;
    int baseS;
    int baseC;
    for (int card = 0; card < nGPU; card++)
    {
        cudaSetDevice(iGPU[card]);
        cudaCheckErrors("set device.");

        baseC = card * NUM_STREAM_PER_DEVICE;
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
        cudaMemsetAsync(devDiff[card],
                        0.0,
                        dim * NUM_STREAM_PER_DEVICE * sizeof(RFLOAT),
                        *((cudaStream_t*)stream[smidx + baseC]));
        cudaCheckErrors("for memcpy.");
        
        cudaMemsetAsync(devCount[card],
                        0,
                        dim * NUM_STREAM_PER_DEVICE * sizeof(int),
                        *((cudaStream_t*)stream[smidx + baseC]));
        cudaCheckErrors("for memcpy.");
#endif
#ifdef RECONSTRUCTOR_CHECK_C_MAX
        cudaMemsetAsync(devMax[card],
                        0.0,
                        dim * NUM_STREAM_PER_DEVICE * sizeof(RFLOAT),
                        *((cudaStream_t*)stream[smidx + baseC]));
        cudaCheckErrors("for memcpy.");
#endif
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    for (size_t i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) 
                    ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(devPartC[smidx + baseS],
                            C3D + i,
                            batch * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            for (int card = 0; card < nGPU; card++)
            {
                if (card != n)
                {
                    baseC = card * NUM_STREAM_PER_DEVICE;
                    cudaSetDevice(iGPU[card]);
                    cudaCheckErrors("set device.");
                    
                    cudaMemsetAsync(dev_W[card] + i,
                                    0.0,
                                    batch * sizeof(RFLOAT),
                                    *((cudaStream_t*)stream[smidx + baseC]));
                    cudaCheckErrors("for memcpy.");
                }
            }

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            kernel_RecalculateW<<<dim * VOLUME_BATCH_3D,
                                  threadInBlock,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(devPartC[smidx + baseS],
                                                                             dev_W[n],
                                                                             r,
                                                                             i,
                                                                             dim,
                                                                             batch);
            cudaCheckErrors("kernel ReCalculateW error.");

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
            kernel_CheckCAVG<<<dim,
                               threadInBlock,
                               threadInBlock * (sizeof(RFLOAT)
                                                + sizeof(int)),
                               *((cudaStream_t*)stream[smidx + baseS])>>>(devDiff[n] + smidx * dim,
                                                                          devCount[n] + smidx * dim,
                                                                          devPartC[smidx + baseS],
                                                                          i,
                                                                          r,
                                                                          dim,
                                                                          batch);
            cudaCheckErrors("Copy devcount array to host.");
#endif
#ifdef RECONSTRUCTOR_CHECK_C_MAX
            kernel_CheckCMAX<<<dim,
                               threadInBlock,
                               threadInBlock * sizeof(RFLOAT),
                               *((cudaStream_t*)stream[smidx + baseS])>>>(devMax[n] + smidx * dim,
                                                                          devPartC[smidx + baseS],
                                                                          i,
                                                                          r,
                                                                          dim,
                                                                          batch);
            cudaCheckErrors("Copy devMax array to host.");
#endif
            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    cudaHostUnregister(C3D);
    cudaCheckErrors("C3D host Unregister.");

    threadInBlock = (dim * NUM_STREAM_PER_DEVICE > THREAD_PER_BLOCK) 
                    ? THREAD_PER_BLOCK : dim * NUM_STREAM_PER_DEVICE;
    for (int i = 0; i < nGPU; i++)
    {
        baseS = i * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[i]);
        
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
        kernel_mergeCAvg<<<1,
                           threadInBlock,
                           threadInBlock * (sizeof(RFLOAT)
                                            + sizeof(int)),
                           *((cudaStream_t*)stream[baseS])>>>(devDiff[i],
                                                              devCount[i],
                                                              dim * NUM_STREAM_PER_DEVICE);
        cudaCheckErrors("Merge C Avg for stream.");
#endif
#ifdef RECONSTRUCTOR_CHECK_C_MAX
        kernel_mergeCMax<<<1,
                           threadInBlock,
                           threadInBlock * sizeof(RFLOAT),
                           *((cudaStream_t*)stream[baseS])>>>(devMax[i],
                                                              dim * NUM_STREAM_PER_DEVICE);
        cudaCheckErrors("Merge C Max for stream.");
#endif
    }
    
    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaStreamSynchronize(*((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("Stream synchronize after.");
    }
    
    ncclComm_t commNccl[nGPU];
    int* gpus = new int[nGPU];
    for (int n = 0; n < nGPU; n++)
        gpus[n] = iGPU[n];
    NCCLCHECK(ncclCommInitAll(commNccl, nGPU, gpus));
    
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)dev_W[i],
                                (void*)dev_W[i],
                                dimSize,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commNccl[i],
                                *((cudaStream_t*)stream[i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclReduce((const void*)devDiff[i],
                             (void*)devDiff[0],
                             1,
#ifdef SINGLE_PRECISION
                             ncclFloat,
#else
                             ncclDouble,
#endif
                             ncclSum,
                             0,
                             commNccl[i],
                             *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclReduce((const void*)devCount[i],
                             (void*)devCount[0],
                             1,
#ifdef SINGLE_PRECISION
                             ncclFloat,
#else
                             ncclDouble,
#endif
                             ncclSum,
                             0,
                             commNccl[i],
                             *((cudaStream_t*)stream[1 + i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclReduce((const void*)devMax[i],
                             (void*)devMax[0],
                             1,
#ifdef SINGLE_PRECISION
                             ncclFloat,
#else
                             ncclDouble,
#endif
                             ncclMax,
                             0,
                             commNccl[i],
                             *((cudaStream_t*)stream[2 + i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());
#endif
    
    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    //finalizing NCCL
    for (int i = 0; i < nGPU; i++)
    {
        ncclCommDestroy(commNccl[i]);
    }

    delete[] gpus;
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
    RFLOAT tempD = 0;
    int tempC = 0;

    cudaSetDevice(iGPU[0]);
    cudaMemcpy(&tempD,
               devDiff[0],
               sizeof(RFLOAT),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("Copy devDiff array to host.");
    
    cudaMemcpy(&tempC,
               devCount[0],
               sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("Copy devDiff array to host.");
    
    diffC = tempD / tempC;
#endif
#ifdef RECONSTRUCTOR_CHECK_C_MAX
    cudaSetDevice(iGPU[0]);
    cudaMemcpy(&diffC,
               devMax[0],
               sizeof(RFLOAT),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("Copy devMax array to host.");
#endif
}

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
                      int dim)
{
    cudaHostUnregister(volumeC);
    cudaCheckErrors("host Unregister.");
    
    int shift = 0;
    for (int n = 0; n < nGPU; n++)
    { 
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");
        
        cudaFree(dev_T[n]);
        cudaCheckErrors("Free devDataT data.");
        cudaFree(dev_tab[n]);
        cudaCheckErrors("Free dev_tab data.");
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
        cudaFree(devDiff[n]);
        cudaCheckErrors("Free devdiff data.");
        cudaFree(devCount[n]);
        cudaCheckErrors("Free devcount data.");
#endif
#ifdef RECONSTRUCTOR_CHECK_C_MAX
        cudaFree(devMax[n]);
        cudaCheckErrors("Free devcount data.");
#endif
        
        shift = n * NUM_STREAM_PER_DEVICE;
        for(int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(devPartC[shift + i]);
            cudaCheckErrors("Free devcount data.");
        }
    }
}

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
                  int nGPU)
{
    int dimSize = (dim / 2 + 1) * dim;
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                      ? THREAD_PER_BLOCK : dim / 2 + 1;

    LOG(INFO) << "Step1: CalculateW.";

    int baseS;
    int smidx = 0;
    for (int t = 0; t < kbatch;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (t >= kbatch)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(dev_T[n] + t * dimSize,
                            T2D + t * dimSize,
                            dimSize * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            kernel_CalculateW2D<<<dim,
                                  threadInBlock,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(dev_W[n] + t * dimSize,
                                                                             dev_T[n] + t * dimSize,
                                                                             dim,
                                                                             r);
            cudaCheckErrors("kernel calculateW2D.");
            
            t++;
        }
        
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
}

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
                int r)
{
    size_t dimSize = (dim / 2 + 1) * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * (dim / 2 + 1);
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim / 2 + 1;

#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC
    cudaHostRegister(T3D, 
                     dimSize * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");
#else
    if (!map)
    {
        cudaHostRegister(T3D, 
                         dimSize * sizeof(RFLOAT), 
                         cudaHostRegisterDefault);
        cudaCheckErrors("Register T3D data.");
    }
#endif

    LOG(INFO) << "Step1: CalculateW.";

    size_t batch;
    int baseS, baseC;
    int smidx = 0;
    for (size_t i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC
            cudaMemcpyAsync(dev_T[n] + i,
                            T3D + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("Copy dev_T volume to device.");
#else
            if (!map)
            {
                cudaMemcpyAsync(dev_T[n] + i,
                                T3D + i,
                                batch * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                *((cudaStream_t*)stream[smidx + baseS]));
                cudaCheckErrors("Copy dev_T volume to device.");
            }
#endif
            kernel_CalculateW<<<dim * VOLUME_BATCH_3D,
                                threadInBlock,
                                0,
                                *((cudaStream_t*)stream[smidx + baseS])>>>(dev_W[n] + i,
                                                                           dev_T[n] + i,
                                                                           i,
                                                                           r,
                                                                           dim,
                                                                           batch);
            cudaCheckErrors("CalculateW error.");

            for (int card = 0; card < nGPU; card++)
            {
                if (card != n)
                {
                    baseC = card * NUM_STREAM_PER_DEVICE;
                    cudaSetDevice(iGPU[card]);
                    cudaCheckErrors("set device.");
                    
                    cudaMemsetAsync(dev_W[card] + i,
                                    0.0,
                                    batch * sizeof(RFLOAT),
                                    *((cudaStream_t*)stream[smidx + baseC]));
                    cudaCheckErrors("for memcpy.");
                }
            }

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    LOG(INFO) << "Step3: Clean up the memory.";
    
    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

#ifndef RECONSTRUCTOR_WIENER_FILTER_FSC
    cudaHostUnregister(T3D);
    cudaCheckErrors("unregister T3D.");
#else
    if (!map)
    {
        cudaHostUnregister(T3D);
        cudaCheckErrors("unregister T3D.");
    }
#endif

    ncclComm_t commNccl[nGPU];
    int* gpus = new int[nGPU];
    for (int n = 0; n < nGPU; n++)
        gpus[n] = iGPU[n];
    NCCLCHECK(ncclCommInitAll(commNccl, nGPU, gpus));

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclAllReduce((const void*)dev_W[i],
                                (void*)dev_W[i],
                                dimSize,
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commNccl[i],
                                *((cudaStream_t*)stream[i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    //finalizing NCCL
    for (int i = 0; i < nGPU; i++)
    {
        ncclCommDestroy(commNccl[i]);
    }

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaFree(dev_T[n]);
        cudaCheckErrors("Free device memory device_T.");
    }
    
    delete[] gpus;
}

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
                  int nGPU)
{
    int streamNum = nGPU * NUM_STREAM_PER_DEVICE;
    int dimSizePRL = pdim * pdim;
    int dimSizeP = (pdim / 2 + 1) * pdim;
    int dimSizeF = (fdim / 2 + 1) * fdim;
    int pthreadInBlock = (pdim / 2 + 1 > THREAD_PER_BLOCK) 
                       ? THREAD_PER_BLOCK : pdim / 2 + 1;
    int fthreadInBlock = (fdim / 2 + 1 > THREAD_PER_BLOCK) 
                       ? THREAD_PER_BLOCK : fdim / 2 + 1;

#ifdef SINGLE_PRECISION
    cufftReal *devDstR[nGPU];
#else
    cufftDoubleReal *devDstR[nGPU];
#endif
    Complex *devDst[nGPU];
    Complex *dev_F[nGPU];
    cufftHandle planc2r[streamNum];
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&dev_F[n], 
                   kbatch * dimSizeF * sizeof(Complex));
        cudaCheckErrors("Allocate devDataW data.");

#ifdef SINGLE_PRECISION
        cudaMalloc((void**)&devDstR[n], 
                   kbatch * dimSizePRL * sizeof(cufftReal));
        cudaCheckErrors("Allocate device memory for devDstR.");
#else
        cudaMalloc((void**)&devDstR[n], 
                   kbatch * dimSizePRL * sizeof(cufftDoubleReal));
        cudaCheckErrors("Allocate device memory for devDstR.");
#endif

        cudaMalloc((void**)&devDst[n], 
                   kbatch * dimSizeP * sizeof(Complex));
        cudaCheckErrors("Allocate devDst data.");

        cudaMemset(devDst[n], 
                   0.0, 
                   kbatch * dimSizeP * sizeof(Complex));
        cudaCheckErrors("Memset devDst data.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
#ifdef SINGLE_PRECISION
            CUFFTCHECK(cufftPlan2d(&planc2r[i + baseS], pdim, pdim, CUFFT_C2R));
#else
            CUFFTCHECK(cufftPlan2d(&planc2r[i + baseS], pdim, pdim, CUFFT_Z2D));
#endif
            cufftSetStream(planc2r[i + baseS], 
                           *((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Set plan stream.");
        }
    }

    LOG(INFO) << "Step1: CalculateFW.";

    int smidx = 0;
    for (int t = 0; t < kbatch;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (t >= kbatch)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(dev_F[n] + t * dimSizeF,
                            F2D + t * dimSizeF,
                            dimSizeF * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            kernel_NormalizeFW2D<<<fdim,
                                   fthreadInBlock,
                                   0,
                                   *((cudaStream_t*)stream[smidx + baseS])>>>(devDst[n] + t * dimSizeP,
                                                                              dev_F[n] + t * dimSizeF,
                                                                              dev_W[n] + t * dimSizeF,
                                                                              r,
                                                                              pdim,
                                                                              fdim);
            cudaCheckErrors("kernel_normlaizeFW2D.");

#ifdef SINGLE_PRECISION
            cufftExecC2R(planc2r[smidx + baseS], 
                         (cufftComplex*)devDst[n] + t * dimSizeP, 
                         devDstR[n] + t * dimSizePRL);
#else
            cufftExecZ2D(planc2r[smidx + baseS], 
                         (cufftDoubleComplex*)devDst[n] + t * dimSizeP, 
                         devDstR[n] + t * dimSizePRL);
#endif
            
            kernel_NormalizeP2D<<<pdim,
                                  pthreadInBlock,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(devDstR[n] + t * dimSizePRL,
                                                                             dimSizePRL);
            cudaCheckErrors("kernel_normlaizeP2D.");

            cudaMemcpyAsync(padDstR[t],
                            devDstR[n] + t * dimSizePRL,
                            dimSizePRL * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("out for memcpy 0.");

            t++;
        }
        
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    //free GPU memory
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaFree(dev_F[n]);
        cudaCheckErrors("Free device memory dev_F.");
        cudaFree(devDstR[n]);
        cudaCheckErrors("Free device memory devDstR.");
        cudaFree(devDst[n]);
        cudaCheckErrors("Free device memory devDst.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cufftDestroy(planc2r[i + baseS]);
            cudaCheckErrors("DestroyPlan planc2r.");
        }
    }
}

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
                 const int fdim)
{
    size_t dimSizeP = (pdim / 2 + 1) * pdim * pdim;
    size_t dimSizeF = (fdim / 2 + 1) * fdim * fdim;
    size_t nImgBatch = VOLUME_BATCH_3D * fdim * (fdim / 2 + 1);
    int fthreadInBlock = (fdim / 2 + 1 > THREAD_PER_BLOCK) 
                         ? THREAD_PER_BLOCK : fdim / 2 + 1;

    cudaHostRegister(F3D, 
                     dimSizeF * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register F3D data.");

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;
    Complex *devDst[nGPU];
    Complex* devPartF[nStream];

    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devDst[n], 
                   dimSizeP * sizeof(Complex));
        cudaCheckErrors("Allocate devDst data.");

        cudaMemset(devDst[n],
                   0.0,
                   dimSizeP * sizeof(Complex));
        cudaCheckErrors("for memcpy.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaMalloc((void**)&devPartF[i + baseS], 
                       nImgBatch * sizeof(Complex));
            cudaCheckErrors("Allocate __device__W data.");
        }
    }

    LOG(INFO) << "Step1: CalculateFW.";

    int smidx = 0;
    size_t batch;
    for (size_t i = 0; i < dimSizeF;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSizeF)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSizeF) 
                    ? (dimSizeF - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);
            
            cudaMemcpyAsync(devPartF[smidx + baseS],
                            F3D + i,
                            batch * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy.");

            kernel_NormalizeFW<<<fdim * VOLUME_BATCH_3D,
                                 fthreadInBlock,
                                 0,
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(devDst[n],
                                                                            devPartF[smidx + baseS],
                                                                            dev_W[n] + i,
                                                                            batch,
                                                                            i,
                                                                            r,
                                                                            pdim,
                                                                            fdim);
            cudaCheckErrors("kernel NormalizFW error.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    cudaHostUnregister(F3D);
    cudaCheckErrors("Unregister F3D.");

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaFree(dev_W[n]);
        cudaCheckErrors("Free device memory device_W.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(devPartF[i + baseS]);
            cudaCheckErrors("Free device memory device_F.");
            
        }
    }

    ncclComm_t commNccl[nGPU];
    int* gpus = new int[nGPU];
    for (int n = 0; n < nGPU; n++)
        gpus[n] = iGPU[n];
    NCCLCHECK(ncclCommInitAll(commNccl, nGPU, gpus));

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPU; i++)
    {
        cudaSetDevice(iGPU[i]);
        NCCLCHECK(ncclReduce((const void*)devDst[i],
                             (void*)devDst[0],
                             2 * dimSizeP,
#ifdef SINGLE_PRECISION
                             ncclFloat,
#else
                             ncclDouble,
#endif
                             ncclSum,
                             0,
                             commNccl[i],
                             *((cudaStream_t*)stream[i * NUM_STREAM_PER_DEVICE])));
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    cudaSetDevice(iGPU[0]);
    cudaCheckErrors("set device.");
    
    cudaMemcpy(padDst,
               devDst[0],
               dimSizeP * sizeof(Complex),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("for memcpy.");

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");
        
        cudaFree(devDst[n]);
        cudaCheckErrors("Free device memory Dst.");

    }
    
    //finalizing NCCL
    for (int i = 0; i < nGPU; i++)
    {
        ncclCommDestroy(commNccl[i]);
    }

    delete[] gpus;
}

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
                const int fdim)
{
    cudaSetDevice(0);

    size_t dimSizeP = (pdim / 2 + 1) * pdim * pdim;
    size_t dimSizePRL = pdim * pdim * pdim;
    size_t dimSizeF = (fdim / 2 + 1) * fdim * fdim;

    size_t nImgBatch = VOLUME_BATCH_3D * fdim * fdim;
    int fthreadInBlock = (fdim / 2 + 1 > THREAD_PER_BLOCK) ? THREAD_PER_BLOCK : fdim / 2 + 1;
    int pthreadInBlock = (pdim / 2 + 1 > THREAD_PER_BLOCK) ? THREAD_PER_BLOCK : pdim / 2 + 1;

    cudaHostRegister(F3D, dimSizeF * sizeof(Complex), cudaHostRegisterDefault);

#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF F3D");
#endif

    cudaHostRegister(W3D, dimSizeF * sizeof(RFLOAT), cudaHostRegisterDefault);

#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO REGISTER PINNED MEMORY OF W3D");
#endif

    Complex *devDst;
    cudaMalloc((void**)&devDst, dimSizeP * sizeof(Complex));

#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO ALLOCATE DST IN DEVICE");
#endif

    cudaMemset(devDst, 0.0, dimSizeP * sizeof(Complex));

#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO SET DST TO 0 IN DEVICE");
#endif

    Complex *devPartF[3];
    RFLOAT *devPartW[3];
    for (int i = 0; i < 3; i++)
    {
        cudaMalloc((void**)&devPartF[i], nImgBatch * sizeof(Complex));
        cudaCheckErrors("Allocate devDataW data.");

        cudaMalloc((void**)&devPartW[i], nImgBatch * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devDataW data.");
    }

    cudaStream_t stream[NUM_STREAM_PER_DEVICE];

    for(int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
    {
        cudaStreamCreate(&stream[i]);

#ifdef GPU_ERROR_CHECK
        cudaCheckErrors("FAIL TO CREATE STREAMS");
#endif
    }

    LOG(INFO) << "Step1: CalculateFW.";

    int smidx = 0;
    size_t batch;

    for (size_t i = 0; i < dimSizeF;)
    {
        batch = (i + nImgBatch > dimSizeF) ? (dimSizeF - i) : nImgBatch;

        cudaMemcpyAsync(devPartF[smidx],
                        F3D + i,
                        batch * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

        cudaMemcpyAsync(devPartW[smidx],
                        W3D + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

        kernel_NormalizeFW<<<fdim,
                             fthreadInBlock,
                             0,
                             stream[smidx]>>>(devDst,
                                              devPartF[smidx],
                                              devPartW[smidx],
                                              batch,
                                              i,
                                              r,
                                              pdim,
                                              fdim);

        i += batch;
        smidx = (smidx + 1) % 3;
    }

    for(int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
    {
        cudaStreamSynchronize(stream[i]);

#ifdef GPU_ERROR_CHECK
        cudaCheckErrors("FAIL TO SYNCHRONIZE CUDA STREAM");
#endif
    }

    for (int i = 0; i < 3; i++)
    {
        cudaFree(devPartW[i]);
        cudaCheckErrors("Free device memory of W");
        cudaFree(devPartF[i]);
        cudaCheckErrors("Free device memory __device__F.");
    }

    cudaHostUnregister(F3D);
    cudaHostUnregister(W3D);

#ifdef SINGLE_PRECISION
    cufftReal *devDstR;
    cudaMalloc((void**)&devDstR, dimSizePRL * sizeof(cufftReal));
    cudaCheckErrors("Allocate device memory for devDstR.");
#else
    cufftDoubleReal *devDstR;
    cudaMalloc((void**)&devDstR, dimSizePRL * sizeof(cufftDoubleReal));
    cudaCheckErrors("Allocate device memory for devDstR.");
#endif

    cufftHandle planc2r;
#ifdef SINGLE_PRECISION
    CUFFTCHECK(cufftPlan3d(&planc2r, pdim, pdim, pdim, CUFFT_C2R));
#else
    CUFFTCHECK(cufftPlan3d(&planc2r, pdim, pdim, pdim, CUFFT_Z2D));
#endif

#ifdef SINGLE_PRECISION
    cufftExecC2R(planc2r, (cufftComplex*)devDst, devDstR);
#else
    cufftExecZ2D(planc2r, (cufftDoubleComplex*)devDst, devDstR);
#endif

    cufftDestroy(planc2r);

#ifdef GPU_ERROR_CHECK
    cudaCheckErrors("FAIL TO DESTROY CUDA FFTW PLAN");
#endif

    cudaFree(devDst);
    cudaCheckErrors("Free device memory devDst.");

    size_t pnImgBatch = VOLUME_BATCH_3D * pdim * pdim;
    smidx = 0;
    for (size_t i = 0; i < dimSizePRL;)
    {
        batch = (i + pnImgBatch > dimSizePRL) ? (dimSizePRL - i) : pnImgBatch;

        kernel_NormalizeP<<<pdim,
                            pthreadInBlock,
                            0,
                            stream[smidx]>>>(devDstR,
                                             batch,
                                             i,
                                             dimSizePRL);
        cudaCheckErrors("kernel normalizeP launch.");

        cudaMemcpyAsync(padDstR + i,
                        devDstR + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

        i += batch;
        smidx = (smidx + 1) % 3;
    }

    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");

    LOG(INFO) << "Step 5: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

    cudaFree(devDstR);
    cudaCheckErrors("Free device memory devDst.");
}

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
                     int nGPU)
{
    int streamNum = nGPU * NUM_STREAM_PER_DEVICE;
    int dimSize = (dim / 2 + 1) * dim;
    int dimSizeRL = dim * dim;
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                      ? THREAD_PER_BLOCK : dim / 2 + 1;

    RFLOAT *devDstI[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devDstI[n], 
                   kbatch * dimSizeRL * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devDst data.");
    }

#ifdef SINGLE_PRECISION
    cufftComplex *devDstC[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        cudaMalloc((void**)&devDstC[n], 
                   kbatch * dimSize * sizeof(cufftComplex));
        cudaCheckErrors("Allocate device memory for devDst.");
    }
#else
    cufftDoubleComplex *devDstC[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);
        cudaMalloc((void**)&devDstC[n], 
                   kbatch * dimSize * sizeof(cufftDoubleComplex));
        cudaCheckErrors("Allocate device memory for devDst.");
    }
#endif

    cufftHandle planr2c[streamNum];
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
#ifdef SINGLE_PRECISION
            CUFFTCHECK(cufftPlan2d(&planr2c[i + baseS], dim, dim, CUFFT_R2C));
#else                                                 
            CUFFTCHECK(cufftPlan2d(&planr2c[i + baseS], dim, dim, CUFFT_D2Z));
#endif
            cufftSetStream(planr2c[i + baseS], 
                           *((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Set plan stream.");
        }
    }
    
    LOG(INFO) << "Step2: Correcting Convolution Kernel.";
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devMkb[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devMkb[n], 
                   mkbSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devMkb data.");
    }

    for (int n = 0; n < nGPU; ++n)
    { 
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        cudaMemcpyAsync(devMkb[n], 
                        mkbRL, 
                        mkbSize * sizeof(RFLOAT), 
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("Copy devMkb to device.");
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }
    
    int smidx = 0;
    for (int t = 0; t < kbatch;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (t >= kbatch)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(devDstI[n] + t * dimSizeRL,
                            imgDst[t],
                            dimSizeRL * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_MKB_KERNEL
            kernel_CorrectF2D<<<dim,
                                threadInBlock,
                                0,
                                *((cudaStream_t*)stream[smidx + baseS])>>>(devDstI[n] + t * dimSizeRL,
                                                                           devMkb[n],
                                                                           nf,
                                                                           dim);
            cudaCheckErrors("correctF error.");
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            kernel_CorrectF2D<<<dim,
                                threadInBlock,
                                0,
                                *((cudaStream_t*)stream[smidx + baseS])>>>(devDstI[n] + t * dimSizeRL,
                                                                           devMkb[n],
                                                                           dim);
            cudaCheckErrors("correctF error.");
#endif

#ifdef SINGLE_PRECISION
            cufftExecR2C(planr2c[smidx + baseS], 
                         (cufftReal*)devDstI[n] + t * dimSizeRL, 
                         devDstC[n] + t * dimSize);
#else
            cufftExecD2Z(planr2c[smidx + baseS], 
                         (cufftDoubleReal*)devDstI[n] + t * dimSizeRL, 
                         devDstC[n] + t * dimSize);
#endif

            cudaMemcpyAsync(ref + t * dimSize,
                            devDstC[n] + t * dimSize,
                            dimSize * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("out for memcpy 0.");

            t++;
        }
        
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    LOG(INFO) << "Step 3: Clean up the streams and memory.";

    //free GPU memory
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaFree(devMkb[n]);
        cudaCheckErrors("Free device memory devDst.");

        cudaFree(devDstI[n]);
        cudaCheckErrors("Free device memory devDst.");

        cudaFree(devDstC[n]);
        cudaCheckErrors("Free device memory devDst.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cufftDestroy(planr2c[i + baseS]);
            cudaCheckErrors("DestroyPlan planc2r.");
        }
    }
}

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
                   const int dim)
{
    size_t dimSizeRL = dim * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * dim;
    int threadInBlock = (dim > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim;

    cudaHostRegister(dstN, 
                     dimSizeRL * sizeof(RFLOAT), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register dst data.");

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;
    size_t mkbSize = (dim / 2 + 1) * (dim / 2 + 1) 
                                   * (dim / 2 + 1);
    RFLOAT *devDstN[nStream];
    RFLOAT *devMkb[nGPU];
    
    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devMkb[n], 
                   mkbSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devMkb data.");
        
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaMalloc((void**)&devDstN[i + baseS], 
                       nImgBatch * sizeof(RFLOAT));
            cudaCheckErrors("Allocate devDstN data.");
        }
    }
    
    for (int n = 0; n < nGPU; n++)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaMemcpyAsync(devMkb[n], 
                        mkbRL, 
                        mkbSize * sizeof(RFLOAT), 
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)stream[baseS]));
        cudaCheckErrors("Copy devMkb to device.");
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    int smidx = 0;
    size_t batch;
    for (size_t i = 0; i < dimSizeRL;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSizeRL)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSizeRL) 
                    ? (dimSizeRL - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(devDstN[smidx + baseS],
                            dstN + i,
                            batch * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_MKB_KERNEL
            kernel_CorrectF<<<dim * VOLUME_BATCH_3D,
                              threadInBlock,
                              0,
                              *((cudaStream_t*)stream[smidx + baseS])>>>(devDstN[smidx + baseS],
                                                                         devMkb[n],
                                                                         nf,
                                                                         dim,
                                                                         batch,
                                                                         i);
#endif
#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            kernel_CorrectF<<<dim * VOLUME_BATCH_3D,
                              threadInBlock,
                              0,
                              *((cudaStream_t*)stream[smidx + baseS])>>>(devDstN[smidx + baseS],
                                                                         devMkb[n],
                                                                         dim,
                                                                         batch,
                                                                         i);
#endif

            cudaMemcpyAsync(dstN + i,
                            devDstN[smidx + baseS],
                            batch * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("out for memcpy.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    cudaHostUnregister(dstN);

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaFree(devMkb[n]);
        cudaCheckErrors("Free device memory Mkb.");
            
        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(devDstN[i + baseS]);
            cudaCheckErrors("Free device memory DstN.");
        }
    }
}

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
                   const int dim)
{
    cudaSetDevice(0);

    size_t dimSizeRL = dim * dim * dim;
    size_t dimSize = (dim / 2 + 1) * dim * dim;
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) ? THREAD_PER_BLOCK : dim / 2 + 1;

    cudaHostRegister(dstN, dimSizeRL * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register dst data.");

    RFLOAT *devDstN;
    cudaMalloc((void**)&devDstN, dimSizeRL * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDst data.");

    LOG(INFO) << "Step2: Correcting Convolution Kernel.";

#ifdef RECONSTRUCTOR_MKB_KERNEL

    size_t mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devMkb;
    cudaMalloc((void**)&devMkb, mkbSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devMkb data.");
    cudaMemcpy(devMkb, mkbRL, mkbSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devMkb to device.");

#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL

    size_t mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devTik;
    cudaMalloc((void**)&devTik, mkbSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devTik data.");
    cudaMemcpy(devTik, mkbRL, mkbSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devTik to device.");
#endif

    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);

    size_t nImgBatch = VOLUME_BATCH_3D * dim * dim;

    int smidx = 0;
    size_t batch;
    for (size_t i = 0; i < dimSizeRL;)
    {
        batch = (i + nImgBatch > dimSizeRL) ? (dimSizeRL - i) : nImgBatch;

        cudaMemcpyAsync(devDstN + i,
                        dstN + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_MKB_KERNEL
        kernel_CorrectF<<<dim,
                          threadInBlock,
                          0,
                          stream[smidx]>>>(devDstN,
                                           devMkb,
                                           nf,
                                           dim,
                                           batch,
                                           i);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        kernel_CorrectF<<<dim,
                          threadInBlock,
                          0,
                          stream[smidx]>>>(devDstN,
                                           devTik,
                                           dim,
                                           batch,
                                           i);
#endif

        i += batch;
        smidx = (smidx + 1) % 3;
    }

    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");

    cudaHostUnregister(dstN);

#ifdef SINGLE_PRECISION
    cufftComplex *devDst;
    cudaMalloc((void**)&devDst, dimSize * sizeof(cufftComplex));
    cudaCheckErrors("Allocate device memory for devDst.");
#else
    cufftDoubleComplex *devDst;
    cudaMalloc((void**)&devDst, dimSize * sizeof(cufftDoubleComplex));
    cudaCheckErrors("Allocate device memory for devDst.");
#endif

    cufftHandle planr2c;
#ifdef SINGLE_PRECISION
    CUFFTCHECK(cufftPlan3d(&planr2c, dim, dim, dim, CUFFT_R2C));
#else
    CUFFTCHECK(cufftPlan3d(&planr2c, dim, dim, dim, CUFFT_D2Z));
#endif

#ifdef SINGLE_PRECISION
    cufftExecR2C(planr2c, (cufftReal*)devDstN, devDst);
#else
    cufftExecD2Z(planr2c, (cufftDoubleReal*)devDstN, devDst);
#endif

    cudaMemcpy(dst,
               devDst,
               dimSize * sizeof(Complex),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("out for memcpy 1.");

    LOG(INFO) << "Step 3: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

    cufftDestroy(planr2c);
    cudaCheckErrors("DestroyPlan planr2c.");

#ifdef RECONSTRUCTOR_MKB_KERNEL
    cudaFree(devMkb);
    cudaCheckErrors("Free device memory devDst.");
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    cudaFree(devTik);
    cudaCheckErrors("Free device memory devDst.");
#endif
    cudaFree(devDstN);
    cudaCheckErrors("Free device memory devDst.");

    cudaFree(devDst);
    cudaCheckErrors("Free device memory devDst.");
}

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
                  int nGPU)
{
    int dimSize = (dim / 2 + 1) * dim;
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                      ? THREAD_PER_BLOCK : dim / 2 + 1;

    cudaHostRegister(src, 
                     kbatch * dimSize * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register src data.");

    Complex *devSrc[nGPU];
    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devSrc[n], 
                   kbatch * dimSize * sizeof(Complex));
        cudaCheckErrors("Allocate devSrc data.");
    }

    int smidx = 0;
    int baseS;
    for (int t = 0; t < kbatch;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (t >= kbatch)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            cudaSetDevice(iGPU[n]);
            cudaCheckErrors("set device.");

            cudaMemcpyAsync(devSrc[n] + t * dimSize,
                            src + t * dimSize,
                            dimSize * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy 0.");

            kernel_TranslateI2D<<<dim,
                                  threadInBlock,
                                  0,
                                  *((cudaStream_t*)stream[smidx + baseS])>>>(devSrc[n] + t * dimSize,
                                                                             ox[t],
                                                                             oy[t],
                                                                             r,
                                                                             dim,
                                                                             dimSize);
            cudaCheckErrors("TranslateI2D error.");

            cudaMemcpyAsync(src + t * dimSize,
                            devSrc[n] + t * dimSize,
                            dimSize * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy 0.");

            t++;
        }
        
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    cudaHostUnregister(src);
    cudaCheckErrors("src host unregister.");

    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaFree(devSrc[n]);
        cudaCheckErrors("Free device memory devSrc.");
    }
}

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
                int dim)
{
    size_t dimSize = (dim / 2 + 1) * dim * dim;
    size_t nImgBatch = VOLUME_BATCH_3D * dim * (dim / 2 + 1);
    int threadInBlock = (dim / 2 + 1 > THREAD_PER_BLOCK) 
                        ? THREAD_PER_BLOCK : dim / 2 + 1;

    cudaHostRegister(ref, 
                     dimSize * sizeof(Complex), 
                     cudaHostRegisterDefault);
    cudaCheckErrors("Register src data.");

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;
    Complex* devRef[nStream];

    int baseS;
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaMalloc((void**)&devRef[i + baseS], 
                       nImgBatch * sizeof(Complex));
            cudaCheckErrors("Allocate devRef data.");
        }
    }

    size_t batch;
    int smidx = 0;
    for (size_t i = 0; i < dimSize;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= dimSize)
                break;
            
            baseS = n * NUM_STREAM_PER_DEVICE;
            batch = (i + nImgBatch > dimSize) 
                    ? (dimSize - i) : nImgBatch;

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(devRef[smidx + baseS],
                            ref + i,
                            batch * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy 0.");

            kernel_TranslateI<<<dim * VOLUME_BATCH_3D,
                                threadInBlock,
                                0,
                                *((cudaStream_t*)stream[smidx + baseS])>>>(devRef[smidx + baseS],
                                                                           ox,
                                                                           oy,
                                                                           oz,
                                                                           r,
                                                                           i,
                                                                           dim,
                                                                           batch);

            cudaMemcpyAsync(ref + i,
                            devRef[smidx + baseS],
                            batch * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("for memcpy 0.");

            i += batch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize after.");
        }
    }

    cudaHostUnregister(ref);

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaFree(devRef[i + baseS]);
            cudaCheckErrors("Free device memory Ref.");
        }
    }
}

/**
 * @brief .
 *
 * @param
 * @param
 */
void hostRegister(Complex* img,
                  int totalNum)
{
    cudaHostRegister(img, totalNum * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register img data.");
}

/**
 * @brief .
 *
 * @param
 * @param
 */
void hostRegister(RFLOAT* data,
                  int totalNum)
{
    cudaHostRegister(data, totalNum * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register data.");
}

/**
 * @brief .
 *
 * @param
 * @param
 */
void hostFree(Complex* img)
{
    cudaHostUnregister(img);
    cudaCheckErrors("Free img data.");
}

/**
 * @brief .
 *
 * @param
 * @param
 */
void hostFree(RFLOAT* data)
{
    cudaHostUnregister(data);
    cudaCheckErrors("Free data.");
}

/**
 * @brief Pre-calculation in expectation.
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
            int nImg,
            int nGPU)
{
    LOG(INFO) << "ReMask begin.";

    RFLOAT *devMask[nGPU];
    RFLOAT r = maskRadius / pixelSize;
    size_t imgSize = idim * (idim / 2 + 1);
    size_t imgSizeRL = idim * idim;
    int threadInBlock = (idim / 2 + 1 > THREAD_PER_BLOCK) 
                      ? THREAD_PER_BLOCK : idim / 2 + 1;

    for (int n = 0; n < nGPU; ++n)
    {
        cudaSetDevice(iGPU[n]);

        cudaMalloc((void**)&devMask[n], imgSizeRL * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devMask data.");

        kernel_SoftMask<<<idim,
                          threadInBlock>>>(devMask[n],
                                           r,
                                           ew,
                                           idim,
                                           imgSizeRL);
        cudaCheckErrors("kernel SoftMask error.");
    }

    int nStream = nGPU * NUM_STREAM_PER_DEVICE;

    Complex *dev_image_buf[nStream];
    RFLOAT *dev_imageF_buf[nStream];

    LOG(INFO) << "alloc Memory.";

    cufftHandle* planc2r = (cufftHandle*)malloc(sizeof(cufftHandle) * nStream);
    cufftHandle* planr2c = (cufftHandle*)malloc(sizeof(cufftHandle) * nStream);

    int baseS;
    const int BATCH_SIZE = IMAGE_BUFF;

    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            allocDeviceComplexBuffer(&dev_image_buf[i + baseS], BATCH_SIZE * imgSize);
            allocDeviceParamBuffer(&dev_imageF_buf[i + baseS], BATCH_SIZE * imgSizeRL);

#ifdef SINGLE_PRECISION
            CUFFTCHECK(cufftPlan2d(&planc2r[i + baseS], idim, idim, CUFFT_C2R));
            CUFFTCHECK(cufftPlan2d(&planr2c[i + baseS], idim, idim, CUFFT_R2C));
#else
            CUFFTCHECK(cufftPlan2d(&planc2r[i + baseS], idim, idim, CUFFT_Z2D));
            CUFFTCHECK(cufftPlan2d(&planr2c[i + baseS], idim, idim, CUFFT_D2Z));
#endif
            cufftSetStream(planc2r[i + baseS], *((cudaStream_t*)stream[i + baseS]));
            cufftSetStream(planr2c[i + baseS], *((cudaStream_t*)stream[i + baseS]));
        }
    }

    LOG(INFO) << "alloc memory done, begin to calculate...";

    int nImgBatch = 0, smidx = 0;
    smidx = 0;
    for (int i = 0; i < nImg;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= nImg)
                break;

            baseS = n * NUM_STREAM_PER_DEVICE;
            nImgBatch = (i + BATCH_SIZE < nImg) ? BATCH_SIZE : (nImg - i);

            cudaSetDevice(iGPU[n]);

            cudaMemcpyAsync(dev_image_buf[smidx + baseS],
                            imgData + i * imgSize,
                            nImgBatch * imgSize * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy image to device.");

            for (int r = 0; r < nImgBatch; r++)
            {
                long long shift = (long long)r * imgSize;
                long long shiftRL = (long long)r * imgSizeRL;

#ifdef SINGLE_PRECISION
                cufftExecC2R(planc2r[smidx + baseS],
                             (cufftComplex *)dev_image_buf[smidx + baseS] + shift,
                             dev_imageF_buf[smidx + baseS] + shiftRL);
#else
                cufftExecZ2D(planc2r[smidx + baseS],
                             (cufftComplex *)dev_image_buf[smidx + baseS] + shift,
                             dev_imageF_buf[smidx + baseS] + shiftRL);
#endif

                kernel_MulMask<<<idim,
                                 threadInBlock,
                                 0,
                                 *((cudaStream_t*)stream[smidx + baseS])>>>(dev_imageF_buf[smidx + baseS],
                                                                            devMask[n],
                                                                            r,
                                                                            idim,
                                                                            imgSizeRL);
                cudaCheckErrors("kernel MulMask error.");

#ifdef SINGLE_PRECISION
                cufftExecR2C(planr2c[smidx + baseS],
                             dev_imageF_buf[smidx + baseS] + shiftRL,
                             (cufftComplex *)dev_image_buf[smidx + baseS] + shift);
#else
                cufftExecD2Z(planr2c[smidx + baseS],
                             dev_imageF_buf[smidx + baseS] + shiftRL,
                             (cufftComplex *)dev_image_buf[smidx + baseS] + shift);
#endif
            }

            cudaMemcpyAsync(imgData + i * imgSize,
                            dev_image_buf[smidx + baseS],
                            nImgBatch * imgSize * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("memcpy image to host.");

            i += nImgBatch;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize.");

            cudaFree(dev_image_buf[i + baseS]);
            cudaFree(dev_imageF_buf[i + baseS]);
            cudaCheckErrors("Free error.");
        }
    }

    //free device buffers
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);
        cudaCheckErrors("set device.");

        cudaFree(devMask[n]);
        cudaCheckErrors("free device mask.");

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cufftDestroy(planc2r[i + baseS]);
            cudaCheckErrors("DestroyPlan planc2r.");

            cufftDestroy(planr2c[i + baseS]);
            cudaCheckErrors("DestroyPlan planr2c.");
        }
    }

    LOG(INFO) << "ReMask done.";
}

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
          RFLOAT pixelSize,
          int ndim,
          int nImg,
          int nGPU)
{
    int nRow = ndim;
    int nCol = ndim;
    size_t imgSizeFT = nRow * (nCol / 2 + 1);
    int threadInBlock = (nCol / 2 + 1 > THREAD_PER_BLOCK) ? THREAD_PER_BLOCK : nCol / 2 + 1;
    int nStream = nGPU * NUM_STREAM_PER_DEVICE;

    CTFAttr *pglk_ctfattr_buf[nStream];
    CTFAttr *dev_ctfattr_buf[nStream];
    Complex *dev_image_buf[nStream];
    vector<CB_UPIB_ta> cbArgsA;

    LOG(INFO) << "Allocate Memory.";

    int baseS;
    const int BATCH_SIZE = IMAGE_BUFF;

    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            allocDeviceComplexBuffer(&dev_image_buf[i + baseS], BATCH_SIZE * imgSizeFT);
            allocPGLKCTFAttrBuffer(&pglk_ctfattr_buf[i + baseS], BATCH_SIZE);
            allocDeviceCTFAttrBuffer(&dev_ctfattr_buf[i + baseS], BATCH_SIZE);
        }
    }

    LOG(INFO) << "Allocate memory done, begin to calculate...";

    int nImgBatch = 0, smidx = 0;
    int index = 0;
    for (int i = 0; i < nImg;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= nImg)
                break;

            baseS = n * NUM_STREAM_PER_DEVICE;
            nImgBatch = (i + BATCH_SIZE < nImg) ? BATCH_SIZE : (nImg - i);

            cudaSetDevice(iGPU[n]);

            CB_UPIB_ta cb_A;
            cb_A.pglkptr = pglk_ctfattr_buf[smidx + baseS];
            cb_A.ctfa = &ctfaData;
            cb_A.nImgBatch = nImgBatch;
            cb_A.basePos = i;

            cbArgsA.push_back(cb_A);
            
            i += nImgBatch;
            index++;
        }
        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    smidx = 0;
    index = 0;
    for (int i = 0; i < nImg;)
    {
        for (int n = 0; n < nGPU; ++n)
        {
            if (i >= nImg)
                break;

            baseS = n * NUM_STREAM_PER_DEVICE;
            nImgBatch = (i + BATCH_SIZE < nImg) ? BATCH_SIZE : (nImg - i);

            cudaSetDevice(iGPU[n]);

            cudaStreamAddCallback(*((cudaStream_t*)stream[smidx + baseS]),
                                  cbUpdatePGLKCTFABuffer,
                                  (void*)&cbArgsA[index],
                                  0);

            cudaMemcpyAsync(dev_ctfattr_buf[smidx + baseS],
                            pglk_ctfattr_buf[smidx + baseS],
                            nImgBatch * sizeof(CTFAttr),
                            cudaMemcpyHostToDevice,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("Memory copy CTFAttr to device.");


            kernel_CTF<<<nImgBatch,
                         threadInBlock,
                         0,
                         *((cudaStream_t*)stream[smidx + baseS])>>>(dev_image_buf[smidx + baseS],
                                                                    dev_ctfattr_buf[smidx + baseS],
                                                                    pixelSize,
                                                                    nRow,
                                                                    nCol,
                                                                    imgSizeFT);
            cudaCheckErrors("Kernel CTF calculation error.");

            cudaMemcpyAsync(ctf + i * imgSizeFT,
                            dev_image_buf[smidx + baseS],
                            nImgBatch * imgSizeFT * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            *((cudaStream_t*)stream[smidx + baseS]));
            cudaCheckErrors("Memory copy image to host.");

            i += nImgBatch;
            index++;
        }

        smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
    }

    //synchronizing on CUDA streams
    for (int n = 0; n < nGPU; ++n)
    {
        baseS = n * NUM_STREAM_PER_DEVICE;
        cudaSetDevice(iGPU[n]);

        for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
        {
            cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
            cudaCheckErrors("Stream synchronize.");

            cudaFreeHost(pglk_ctfattr_buf[i + baseS]);
            cudaFree(dev_image_buf[i + baseS]);
            cudaFree(dev_ctfattr_buf[i + baseS]);
        }
    }

    LOG(INFO) << "CTF calculation done.";
}

////////////////////////////////////////////////////////////////
// TODO cudarize more modules.
//

////////////////////////////////////////////////////////////////

} // end namespace cuthunder

////////////////////////////////////////////////////////////////
