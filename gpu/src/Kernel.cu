/**************************************************************
 * FileName: Kernel.cu
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#include "Kernel.cuh"

namespace cuthunder {

///////////////////////////////////////////////////////////////
//                     GLOBAL VARIABLES
//

/* Constant memory on device for rotation and symmetry matrix */
//__constant__ double dev_mat_data[3][DEV_CONST_MAT_SIZE * 9];
__constant__ RFLOAT dev_ws_data[3][DEV_CONST_MAT_SIZE];


///////////////////////////////////////////////////////////////
//                     KERNEL ROUTINES
//

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE __forceinline__ int getIndexHalf2D(const int i,
                                              const int j,
                                              const int dim)
{
   return (j >= 0 ? j : j + dim) * (dim / 2 + 1)
        + i;
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE __forceinline__ int getIndexHalf(const int i,
                                            const int j,
                                            const int k,
                                            const int dim)
{
   return (k >= 0 ? k : k + dim) * (dim / 2 + 1) * dim
        + (j >= 0 ? j : j + dim) * (dim / 2 + 1)
        + i;
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void addFTD2D(RFLOAT* devDataT,
                         RFLOAT value,
                         RFLOAT iCol,
                         RFLOAT iRow,
                         const int dim)
{
    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
    }

    RFLOAT w[2][2];
    int x0[2];
    RFLOAT x[2] = {iCol, iRow};
    int index;

    WG_BI_LINEAR_INTERPF(w, x0, x);

    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++)
        {
            index = getIndexHalf2D(x0[0] + i,
                                   x0[1] + j,
                                   dim);
            //if (index < 0 || index >= dim * dim * (dim / 2 + 1))
            //    printf("index error!\n");
            atomicAdd(&devDataT[index], value * w[j][i]);
        }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void addFTC2D(Complex* devDataF,
                         Complex& value,
                         RFLOAT iCol,
                         RFLOAT iRow,
                         const int dim)
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        conjug = true;
    }

    RFLOAT w[2][2];
    int x0[2];
    RFLOAT x[2] = {iCol, iRow};
    int index;

    WG_BI_LINEAR_INTERPF(w, x0, x);

    conjug ? value.conj() : value;

    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++)
        {
            index = getIndexHalf2D(x0[0] + i,
                                   x0[1] + j,
                                   dim);

            //if (index < 0 || index >= dim * dim * (dim / 2 + 1))
            //    printf("index error!\n");
            atomicAdd(devDataF[index].realAddr(), value.real() * w[j][i]);
            atomicAdd(devDataF[index].imagAddr(), value.imag() * w[j][i]);
        }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void addFTD(RFLOAT* devDataT,
                       RFLOAT value,
                       RFLOAT iCol,
                       RFLOAT iRow,
                       RFLOAT iSlc,
                       const int dim)
{
    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
    }

    RFLOAT w[2][2][2];
    int x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};
    int index;

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
                index = getIndexHalf(x0[0] + i,
                                     x0[1] + j,
                                     x0[2] + k,
                                     dim);
                //if (index < 0 || index >= dim * dim * (dim / 2 + 1))
                //    printf("index error!\n");
                atomicAdd(&devDataT[index], value * w[k][j][i]);
            }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void addFTC(Complex* devDataF,
                       Complex& value,
                       RFLOAT iCol,
                       RFLOAT iRow,
                       RFLOAT iSlc,
                       const int dim)
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
        conjug = true;
    }

    RFLOAT w[2][2][2];
    int x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};
    int index;

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    conjug ? value.conj() : value;

    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
                index = getIndexHalf(x0[0] + i,
                                     x0[1] + j,
                                     x0[2] + k,
                                     dim);

                //if (index < 0 || index >= dim * dim * (dim / 2 + 1))
                //    printf("index error!\n");
                atomicAdd(devDataF[index].realAddr(), value.real() * w[k][j][i]);
                atomicAdd(devDataF[index].imagAddr(), value.imag() * w[k][j][i]);
            }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE RFLOAT getTexture(RFLOAT iCol,
                             RFLOAT iRow,
                             RFLOAT iSlc,
                             const int dim,
                             cudaTextureObject_t texObject)
{
    if (iRow < 0) iRow += dim;
    if (iSlc < 0) iSlc += dim;

#ifdef SINGLE_PRECISION
    float dval = tex3D<float>(texObject, iCol, iRow, iSlc);
    return dval;
#else
    int2 dval = tex3D<int2>(texObject, iCol, iRow, iSlc);
    return __hiloint2double(dval.y, dval.x);
#endif    
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getTextureC(RFLOAT iCol,
                               RFLOAT iRow,
                               RFLOAT iSlc,
                               const int dim,
                               cudaTextureObject_t texObject)
{
    if (iRow < 0) iRow += dim;
    if (iSlc < 0) iSlc += dim;

#ifdef SINGLE_PRECISION
    float2 cval = tex3D<float2>(texObject, iCol, iRow, iSlc);

    return Complex(cval.x, cval.y);
#else
    int4 cval = tex3D<int4>(texObject, iCol, iRow, iSlc);

    return Complex(__hiloint2double(cval.y,cval.x),
                   __hiloint2double(cval.w,cval.z));
#endif
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getTextureC2D(RFLOAT iCol,
                                 RFLOAT iRow,
                                 const int dim,
                                 cudaTextureObject_t texObject)
{
    if (iRow < 0) iRow += dim;

#ifdef SINGLE_PRECISION
    float2 cval = tex2D<float2>(texObject, iCol, iRow);

    return Complex(cval.x, cval.y);
#else
    int4 cval = tex2D<int4>(texObject, iCol, iRow);

    return Complex(__hiloint2double(cval.y,cval.x),
                   __hiloint2double(cval.w,cval.z));
#endif
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getByInterp2D(RFLOAT iCol,
                                 RFLOAT iRow,
                                 const int interp,
                                 const int dim,
                                 cudaTextureObject_t texObject)
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        conjug = true;
    }

    if(interp == 0)
    {
        Complex result = getTextureC2D(iCol,
                                       iRow,
                                       dim,
                                       texObject);
        return conjug ? result.conj() : result;
    }

    RFLOAT w[2][2];
    int x0[2];
    RFLOAT x[2] = {iCol, iRow};

    WG_BI_LINEAR_INTERPF(w, x0, x);

    Complex result (0.0, 0.0);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++){

            result += getTextureC2D((RFLOAT)x0[0] + i,
                                    (RFLOAT)x0[1] + j,
                                    dim,
                                    texObject)
                   * w[j][i];
        }
    return conjug ? result.conj() : result;
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE RFLOAT getByInterpolationFT(RFLOAT iCol,
                                       RFLOAT iRow,
                                       RFLOAT iSlc,
                                       const int interp,
                                       const int dim,
                                       cudaTextureObject_t texObject)
{
    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
    }

    if(interp == 0)
    {
        RFLOAT result = getTexture(iCol, 
                                   iRow,
                                   iSlc,
                                   dim,
                                   texObject);
        return result;
    }

    RFLOAT w[2][2][2];
    int x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    RFLOAT result = 0.0;
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++){
                result += getTexture((RFLOAT)x0[0] + i, 
                                     (RFLOAT)x0[1] + j, 
                                     (RFLOAT)x0[2] + k, 
                                     dim, 
                                     texObject)
                        * w[k][j][i];
            } 
    return result;
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getByInterpolationFTC(RFLOAT iCol,
                                         RFLOAT iRow,
                                         RFLOAT iSlc,
                                         const int interp,
                                         const int dim,
                                         cudaTextureObject_t texObject)
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
        conjug = true;
    }

    if(interp == 0)
    {
        Complex result = getTextureC(iCol,
                                     iRow,
                                     iSlc,
                                     dim,
                                     texObject);
        return conjug ? result.conj() : result;
    }

    RFLOAT w[2][2][2];
    int x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    Complex result (0.0, 0.0);
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++){

                result += getTextureC((RFLOAT)x0[0] + i,
                                      (RFLOAT)x0[1] + j,
                                      (RFLOAT)x0[2] + k,
                                      dim,
                                      texObject)
                       * w[k][j][i];
            }
    return conjug ? result.conj() : result;
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ExpectPrectf(CTFAttr* dev_ctfa,
                                    RFLOAT* dev_def,
                                    RFLOAT* dev_k1,
                                    RFLOAT* dev_k2,
                                    int* deviCol,
                                    int* deviRow,
                                    int npxl)
{
    Constructor constructor;

    int tid = threadIdx.x;

    constructor.init(tid);

    constructor.expectPrectf(dev_ctfa,
                             dev_def,
                             dev_k1,
                             dev_k2,
                             deviCol,
                             deviRow,
                             blockIdx.x,
                             blockDim.x,
                             npxl);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(Complex* devtraP,
                                 double* dev_trans,
                                 int* deviCol,
                                 int* deviRow,
                                 int idim,
                                 int batch,
                                 int npxl)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tNum;
    int i, j;
    RFLOAT phase, col, row;
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        tNum = itr / npxl;
        i = deviCol[itr % npxl];
        j = deviRow[itr % npxl];

        col = (RFLOAT)(PI_2 * dev_trans[tNum * 2] / idim);
        row = (RFLOAT)(PI_2 * dev_trans[tNum * 2 + 1] / idim);

        phase = -1 * (i * col + j * row);
#ifdef SINGLE_PRECISION
        devtraP[itr].set(cosf(phase), sinf(phase));
#else
        devtraP[itr].set(cos(phase), sin(phase));
#endif
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalCTFL(RFLOAT* devctfP,
                               RFLOAT* devdefO,
                               RFLOAT* devfreQ,
                               double* devdP,
                               RFLOAT phaseShift,
                               RFLOAT conT,
                               RFLOAT k1,
                               RFLOAT k2,
                               int dSize,
                               int npxl)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    RFLOAT freq, ki;
    int dNum;
    int pIdx;
    for (int itr = tid; itr < npxl * dSize; itr += gridDim.x * blockDim.x)
    {
        dNum = itr / npxl;
        pIdx = itr % npxl;
        freq = devfreQ[pIdx] * devfreQ[pIdx];

        ki = k1
           * devdefO[pIdx]
           * devdP[dNum]
           * freq
           + k2
           * freq * freq
           - phaseShift;
#ifdef SINGLE_PRECISION
        devctfP[itr] = -sqrtf(1 - conT * conT)
                     * sinf(ki)
                     + conT
                     * cosf(ki);
#else
        devctfP[itr] = -sqrt(1 - conT * conT)
                     * sin(ki)
                     + conT
                     * cos(ki);
#endif
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRotMat(double* devRotMat,
                                 double* devnR,
                                 int batch)
{
    extern __shared__ double matRot[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < batch)
    {
        double *mat, *res;
        mat = matRot + threadIdx.x * 18;
        res = mat  + 9;

        mat[0] = 0; mat[4] = 0; mat[8] = 0;
        mat[5] = devnR[tid * 4 + 1];
        mat[6] = devnR[tid * 4 + 2];
        mat[1] = devnR[tid * 4 + 3];
        mat[7] = -mat[5];
        mat[2] = -mat[6];
        mat[3] = -mat[1];

        for(int i = 0; i < 9; i++)
            res[i] = 0;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    res[i + j * 3] += mat[i + k * 3] * mat[k + j * 3];

        double scale = 2 * devnR[tid * 4];
        for (int n = 0; n < 9; n++)
        {
            mat[n] *= scale;
            mat[n] += res[n] * 2;
        }

        mat[0] += 1;
        mat[4] += 1;
        mat[8] += 1;

        for (int n = 0; n < 9; n++)
        {
            devRotMat[tid * 9 + n] = mat[n];
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Project3D(Complex* priRotP,
                                 double* devRotMat,
                                 int* deviCol,
                                 int* deviRow,
                                 int batch,
                                 int pf,
                                 int vdim,
                                 int npxl,
                                 int interp,
                                 cudaTextureObject_t texObject)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int rNum;
    Mat33 mat;
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        rNum = itr / npxl;
        mat.init(&devRotMat[rNum * 9], 0);
        
        Vec3 newCor((double)(deviCol[itr % npxl] * pf),
                    (double)(deviRow[itr % npxl] * pf),
                    0);
        Vec3 oldCor = mat * newCor;

        priRotP[itr] = getByInterpolationFTC((RFLOAT)oldCor(0),
                                             (RFLOAT)oldCor(1),
                                             (RFLOAT)oldCor(2),
                                             interp,
                                             vdim,
                                             texObject);
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Project2D(Complex* priRotP,
                                 double* devnR,
                                 int* deviCol,
                                 int* deviRow,
                                 int batch,
                                 int pf,
                                 int vdim,
                                 int npxl,
                                 int interp,
                                 cudaTextureObject_t texObject)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int rNum;
    double i, j;
    double oldi, oldj;
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        rNum = itr / npxl;
        i = (double)(deviCol[itr % npxl] * pf);
        j = (double)(deviRow[itr % npxl] * pf);

        oldi = i * devnR[rNum * 2] - j * devnR[rNum * 2 + 1];
        oldj = i * devnR[rNum * 2 + 1] + j * devnR[rNum * 2];

        priRotP[itr] = getByInterp2D((RFLOAT)oldi,
                                     (RFLOAT)oldj,
                                     interp,
                                     vdim,
                                     texObject);
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_logDataVSL(Complex* priRotP,
                                  Complex* devtraP,
                                  RFLOAT* devdatPR,
                                  RFLOAT* devdatPI,
                                  RFLOAT* devctfP,
                                  RFLOAT* devsigP,
                                  RFLOAT* devDvp,
                                  //RFLOAT* devre,
                                  int nT,
                                  int npxl)
{
    extern __shared__ RFLOAT resL[];

    resL[threadIdx.x] = 0;

    int nrIdx = blockIdx.x / nT * npxl;
    int ntIdx = (blockIdx.x % nT) * npxl;

    Complex temp(0.0, 0.0);
    RFLOAT realC = 0;
    RFLOAT imagC = 0;
    RFLOAT tempD = 0;

    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        temp  = (devtraP[ntIdx + itr] * priRotP[nrIdx + itr])
              * devctfP[itr];
        realC = devdatPR[itr] - temp.real();
        imagC = devdatPI[itr] - temp.imag();
        tempD = realC * realC + imagC * imagC;
        resL[threadIdx.x] += tempD * devsigP[itr];
    }

    __syncthreads();

    int i = 32;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            resL[threadIdx.x] += resL[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        devDvp[blockIdx.x] = resL[0];
    }

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_logDataVSLC(Complex* priRotP,
                                   Complex* devtraP,
                                   RFLOAT* devdatPR,
                                   RFLOAT* devdatPI,
                                   RFLOAT* devctfP,
                                   RFLOAT* devsigP,
                                   RFLOAT* devDvp,
                                   int nT,
                                   int nD,
                                   int npxl)
{
    extern __shared__ RFLOAT result[];

    result[threadIdx.x] = 0;

    int nrIdx = blockIdx.x / (nT * nD);
    int ntIdx = (blockIdx.x % (nT * nD)) / nD;
    int ndIdx = (blockIdx.x % (nT * nD)) % nD;

    Complex temp(0.0, 0.0);
    RFLOAT realC = 0;
    RFLOAT imagC = 0;
    RFLOAT tempD = 0;

    nrIdx *= npxl;
    ntIdx *= npxl;
    ndIdx *= npxl;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        temp  = (devtraP[ntIdx + itr] * priRotP[nrIdx + itr])
              * devctfP[ndIdx + itr];
        realC = devdatPR[itr] - temp.real();
        imagC = devdatPI[itr] - temp.imag();
        tempD = realC * realC + imagC * imagC;
        result[threadIdx.x] += tempD * devsigP[itr];
    }

    __syncthreads();

    int i = 32;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            result[threadIdx.x] += result[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        devDvp[blockIdx.x] = result[0];
    }

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_logDataVS(RFLOAT* devdatPR,
                                 RFLOAT* devdatPI,
                                 Complex* priRotP,
                                 Complex* devtraP,
                                 RFLOAT* devctfP,
                                 RFLOAT* devsigP,
                                 RFLOAT* devDvp,
                                 int r,
                                 int nR,
                                 int nT,
                                 int rbatch,
                                 int npxl)
{
    extern __shared__ RFLOAT resDvp[];

    resDvp[threadIdx.x] = 0;

    /* One block handle one par:
     *    i: Range: ibatch
     *    j: Range: rbatch * nT
     *    blockId = i * rbatch * nT + j
     */
    int nrIdx = (blockIdx.x % (rbatch * nT)) / nT;
    int ntIdx = (blockIdx.x % (rbatch * nT)) % nT;
    int imgIdx = blockIdx.x / (rbatch * nT);
    int dvpIdx = imgIdx * nR * nT + (r + nrIdx) * nT + ntIdx;

    Complex temp(0.0, 0.0);
    RFLOAT realC = 0;
    RFLOAT imagC = 0;
    RFLOAT tempD = 0;

    nrIdx *= npxl;
    ntIdx *= npxl;
    imgIdx *= npxl;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        temp  = (devtraP[ntIdx + itr] * priRotP[nrIdx + itr]) 
              * devctfP[imgIdx + itr];
        realC = devdatPR[imgIdx + itr] - temp.real();
        imagC = devdatPI[imgIdx + itr] - temp.imag();
        tempD = realC * realC + imagC * imagC;
        resDvp[threadIdx.x] += tempD * devsigP[imgIdx + itr];
    }

    __syncthreads();

    int i = 32;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            resDvp[threadIdx.x] += resDvp[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        devDvp[dvpIdx] = resDvp[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getMaxBase(RFLOAT* devbaseL,
                                  RFLOAT* devDvp,
                                  int angleNum)
{
    extern __shared__ RFLOAT resD[];

    int imgIdx = blockIdx.x * angleNum;

    RFLOAT tempD;
    
    if (threadIdx.x < angleNum)
        tempD = devDvp[imgIdx + threadIdx.x];
    else
        tempD = 0.0;

    for (int itr = threadIdx.x; itr < angleNum; itr += blockDim.x)
    {
        if (devDvp[imgIdx + itr] > tempD)
            tempD = devDvp[imgIdx + itr];
    }

    if (threadIdx.x < angleNum)
        resD[threadIdx.x] = tempD;
    else
        resD[threadIdx.x] = 0.0;

    __syncthreads();

    int i;
    int size;
    bool flag;
    if (angleNum < blockDim.x)
        size = angleNum;
    else
        size = blockDim.x;

    if (blockDim.x % 2 == 0)
    {
        i = size / 2;
        flag = true;
    }
    else
    {
        i = size / 2 + 1;
        flag = false;
    }
    while (i != 0)
    {
        if (flag)
        {
            if (threadIdx.x < i)
            {
                if (resD[threadIdx.x] < resD[threadIdx.x + i])
                {
                    resD[threadIdx.x] = resD[threadIdx.x + i];
                }
            }

        }
        else
        {
            if (threadIdx.x < i - 1)
            {
                if (resD[threadIdx.x] < resD[threadIdx.x + i])
                {
                    resD[threadIdx.x] = resD[threadIdx.x + i];
                }
            }

        }

        __syncthreads();

        if(i % 2 != 0 && i != 1)
        {
            i++;
            flag = false;
        }
        else
            flag = true;

        i /= 2;
    }
    if (threadIdx.x == 0)
    {
        devbaseL[blockIdx.x] = resD[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_setBaseLine(RFLOAT* devcomP,
                                   RFLOAT* devbaseL,
                                   RFLOAT* devwC,
                                   RFLOAT* devwR,
                                   RFLOAT* devwT,
                                   int nK,
                                   int nR,
                                   int nT,
                                   int total)
{
    if (devcomP[blockIdx.x] > devbaseL[blockIdx.x])
    {
        RFLOAT offset, nf;
        offset = devcomP[blockIdx.x] - devbaseL[blockIdx.x];
#ifdef SINGLE_PRECISION
        nf = expf(-offset);
#else
        nf = exp(-offset);
#endif
        int shiftR = blockIdx.x * nK * nR;
        int shiftT = blockIdx.x * nK * nT;
        int kIdx, rtIdx;
        for (int itr = threadIdx.x; itr < total; itr += blockDim.x)
        {
            if (itr / nK < 1)
                devwC[blockIdx.x * nK + itr] *= nf;
            else
            {
                if (itr / nK < (1 + nR))
                {
                    kIdx = ((itr - nK) / nR) * nR;
                    rtIdx = (itr - nK) % nR;
                    devwR[shiftR + kIdx + rtIdx] *= nf;
                }
                else
                {
                    kIdx = ((itr - nK * (nR + 1)) / nT) * nT;
                    rtIdx = (itr - nK * (nR + 1)) % nT;
                    devwT[shiftT + kIdx + rtIdx] *= nf;
                }
            }
        } 

        if (threadIdx.x == 0)
            devbaseL[blockIdx.x] = devcomP[blockIdx.x];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateW2D(RFLOAT* devDvp,
                                 RFLOAT* devbaseL,
                                 RFLOAT* devwC,
                                 RFLOAT* devwR,
                                 RFLOAT* devwT,
                                 double* devpR,
                                 double* devpT,
                                 int kIdx,
                                 int nK,
                                 int nR,
                                 int nT,
                                 int rSize)
{
    extern __shared__ RFLOAT result[];
    int itr = 0;
    for (itr = threadIdx.x; itr < blockDim.x * (nT + 1); itr += blockDim.x)
        result[itr] = 0;

    __syncthreads();

    RFLOAT w;
    int tIdx;
    int index;
    int indexC = nT * blockDim.x;
    int shiftR = blockIdx.x * nK * nR + kIdx * nR;
    for (itr = threadIdx.x; itr < nR; itr += blockDim.x)
    {
        index = blockIdx.x * rSize + itr * nT;
        for (tIdx = 0; tIdx < nT; tIdx++)
        {
#ifdef SINGLE_PRECISION
            w = expf(devDvp[index + tIdx] - devbaseL[blockIdx.x]);
#else
            w = exp(devDvp[index + tIdx] - devbaseL[blockIdx.x]);
#endif
            result[indexC + threadIdx.x] += w * devpR[itr] * devpT[tIdx];
            result[tIdx * blockDim.x + threadIdx.x] += w * devpR[itr]; 
            devwR[shiftR + itr] += w * devpT[tIdx];
        } 
    }

    __syncthreads();

    int i;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        i = blockDim.x / 2;
        flag = true;
    }
    else
    {
        i = blockDim.x / 2 + 1;
        flag = false;
    }
    while (i != 0)
    {
        if (flag)
        {
            if (threadIdx.x < i)
            {
                for (itr = 0; itr < nT; itr++)
                {
                    index = itr * blockDim.x + threadIdx.x;
                    result[index] += result[index + i];
                } 
                result[indexC + threadIdx.x] += result[indexC + threadIdx.x + i];
            }
        }
        else
        {
            if (threadIdx.x < i - 1)
            {
                for (itr = 0; itr < nT; itr++)
                {
                    index = itr * blockDim.x + threadIdx.x;
                    result[index] += result[index + i];
                } 
                result[indexC + threadIdx.x] += result[indexC + threadIdx.x + i];
            }
        }
        
        __syncthreads();
        
        if(i % 2 != 0 && i != 1)
        {
            i++;
            flag = false;
        }
        else
            flag = true;

        i /= 2;
    }
    
    if (threadIdx.x == 0)
        devwC[blockIdx.x * nK + kIdx] = result[indexC];
    
    int shiftT = blockIdx.x * nK * nT + kIdx * nT;
    for (itr = threadIdx.x; itr < nT; itr += blockDim.x)
    {
        devwT[shiftT + itr] += result[itr * blockDim.x];
    } 
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateW3D(RFLOAT* devDvp,
                                 RFLOAT* devbaseL,
                                 RFLOAT* devwC,
                                 RFLOAT* devwR,
                                 RFLOAT* devwT,
                                 double* devpR,
                                 double* devpT,
                                 int nR,
                                 int nT,
                                 int rSize)
{
    extern __shared__ RFLOAT result[];
    int itr = 0;
    for (itr = threadIdx.x; itr < blockDim.x * (nT + 1); itr += blockDim.x)
        result[itr] = 0;

    __syncthreads();

    RFLOAT w;
    int tIdx;
    int index;
    int indexC = nT * blockDim.x;
    int shiftR = blockIdx.x * nR;
    for (itr = threadIdx.x; itr < nR; itr += blockDim.x)
    {
        index = blockIdx.x * rSize + itr * nT;
        for (tIdx = 0; tIdx < nT; tIdx++)
        {
#ifdef SINGLE_PRECISION
            w = expf(devDvp[index + tIdx] - devbaseL[blockIdx.x]);
#else
            w = exp(devDvp[index + tIdx] - devbaseL[blockIdx.x]);
#endif
            result[indexC + threadIdx.x] += w * devpR[itr] * devpT[tIdx];
            result[tIdx * blockDim.x + threadIdx.x] += w * devpR[itr]; 
            devwR[shiftR + itr] += w * devpT[tIdx];
        } 
    }

    __syncthreads();

    int i;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        i = blockDim.x / 2;
        flag = true;
    }
    else
    {
        i = blockDim.x / 2 + 1;
        flag = false;
    }
    while (i != 0)
    {
        if (flag)
        {
            if (threadIdx.x < i)
            {
                for (itr = 0; itr < nT; itr++)
                {
                    index = itr * blockDim.x + threadIdx.x;
                    result[index] += result[index + i];
                } 
                result[indexC + threadIdx.x] += result[indexC + threadIdx.x + i];
            }
        }
        else
        {
            if (threadIdx.x < i - 1)
            {
                for (itr = 0; itr < nT; itr++)
                {
                    index = itr * blockDim.x + threadIdx.x;
                    result[index] += result[index + i];
                } 
                result[indexC + threadIdx.x] += result[indexC + threadIdx.x + i];
            }
        }
        
        __syncthreads();
        
        if(i % 2 != 0 && i != 1)
        {
            i++;
            flag = false;
        }
        else
            flag = true;

        i /= 2;
    }
    
    if (threadIdx.x == 0)
        devwC[blockIdx.x] = result[indexC];
    
    int shiftT = blockIdx.x * nT;
    for (itr = threadIdx.x; itr < nT; itr += blockDim.x)
    {
        devwT[shiftT + itr] += result[itr * blockDim.x];
    } 
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateWL(RFLOAT* devDvp,
                                RFLOAT* devBaseL,
                                RFLOAT* devwC,
                                RFLOAT* devwR,
                                RFLOAT* devwT,
                                RFLOAT* devwD,
                                double* devR,
                                double* devT,
                                double* devD,
                                int nT)
{
    extern __shared__ RFLOAT resCRD[];

    RFLOAT* resC = resCRD;
    RFLOAT* resR = &resC[blockDim.x];
    RFLOAT* resD = &resR[blockDim.x];
    RFLOAT* resT = &resD[blockDim.x];
    resC[threadIdx.x] = 0;
    resR[threadIdx.x] = 0;
    resD[threadIdx.x] = 0;
    int itr;
    for (itr = threadIdx.x; itr < nT; itr += blockDim.x)
        devwT[itr] = 0;

    __syncthreads();

    RFLOAT w;
    int dvpIdx = threadIdx.x * nT;
    double rd = devR[threadIdx.x] * devD[0];

    for (itr = 0; itr < nT; itr++)
    {
#ifdef SINGLE_PRECISION
        w = expf(devDvp[dvpIdx + itr] - devBaseL[0]);
#else
        w = exp(devDvp[dvpIdx + itr] - devBaseL[0]);
#endif
        resC[threadIdx.x] += w * rd * devT[itr];
        resR[threadIdx.x] += w * devD[0] * devT[itr];
        resD[threadIdx.x] += w * devR[threadIdx.x] * devT[itr];
        resT[itr * blockDim.x + threadIdx.x] = w * rd; 
    }

    devwR[threadIdx.x] = resR[threadIdx.x];
    __syncthreads();

    int j;
    int index;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0)
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                for (itr = 0; itr < nT; itr++)
                {
                    index = itr * blockDim.x + threadIdx.x;
                    resT[index] += resT[index + j];
                } 
                resC[threadIdx.x] += resC[threadIdx.x + j];
                resD[threadIdx.x] += resD[threadIdx.x + j];
            }

        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                for (itr = 0; itr < nT; itr++)
                {
                    index = itr * blockDim.x + threadIdx.x;
                    resT[index] += resT[index + j];
                } 
                resC[threadIdx.x] += resC[threadIdx.x + j];
                resD[threadIdx.x] += resD[threadIdx.x + j];
            }

        }

        __syncthreads();

        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;

        j /= 2;
    }

    for (itr = threadIdx.x; itr < nT; itr += blockDim.x)
    {
        index = itr * blockDim.x;
        devwT[itr] = resT[index];
    } 
    
    if (threadIdx.x == 0)
    {
        devwC[blockIdx.x] = resC[0];
        devwD[blockIdx.x] = resD[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateWLC(RFLOAT* devDvp,
                                 RFLOAT* devBaseL,
                                 RFLOAT* devwC,
                                 RFLOAT* devwR,
                                 RFLOAT* devtT,
                                 RFLOAT* devtD,
                                 double* devR,
                                 double* devT,
                                 double* devD,
                                 int nT,
                                 int nD)
{
    extern __shared__ RFLOAT resCR[];
    RFLOAT* resC = resCR;
    RFLOAT* resR = &resC[blockDim.x];
    resC[threadIdx.x] = 0;
    resR[threadIdx.x] = 0;

    __syncthreads();

    int itr;
    RFLOAT w;
    int dvpIdx = threadIdx.x * nT * nD;
    double td;
    int tIdx, dIdx;

    for (itr = 0; itr < nT * nD; itr++)
    {
#ifdef SINGLE_PRECISION
        w = expf(devDvp[dvpIdx + itr] - devBaseL[0]);
#else
        w = exp(devDvp[dvpIdx + itr] - devBaseL[0]);
#endif
        tIdx = itr / nD;
        dIdx = itr % nD;
        td = devT[tIdx] * devD[dIdx];
        resC[threadIdx.x] += w * (td * devR[threadIdx.x]);
        resR[threadIdx.x] += w * td;
        devtT[threadIdx.x + tIdx * blockDim.x] += w * (devR[threadIdx.x] * devD[dIdx]);
        devtD[threadIdx.x + dIdx * blockDim.x] += w * (devR[threadIdx.x] * devT[tIdx]);
    }

    devwR[threadIdx.x] = resR[threadIdx.x];
    __syncthreads();

    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0)
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                resC[threadIdx.x] += resC[threadIdx.x + j];
            }

        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                resC[threadIdx.x] += resC[threadIdx.x + j];
            }

        }

        __syncthreads();

        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;

        j /= 2;
    }

    if (threadIdx.x == 0)
    {
        devwC[blockIdx.x] = resC[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ReduceW(RFLOAT* devw,
                               RFLOAT* devt)
{
    extern __shared__ RFLOAT resW[];

    resW[threadIdx.x] = devt[blockIdx.x * blockDim.x + threadIdx.x];

    __syncthreads();

    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0)
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                resW[threadIdx.x] += resW[threadIdx.x + j];
            }

        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                resW[threadIdx.x] += resW[threadIdx.x + j];
            }

        }

        __syncthreads();

        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;

        j /= 2;
    }

    if (threadIdx.x == 0)
    {
        devw[blockIdx.x] = resW[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRandomCTD(double* dev_nt,
                                    double* dev_tran,
                                    double* dev_nd,
                                    double* dev_ramD,
                                    double* dev_nr,
                                    double* dev_ramR,
                                    unsigned int out,
                                    int rSize,
                                    int tSize,
                                    int dSize
                                    )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float myrand;

    curandState s;
    curand_init(out, tid, 0, &s);

    //myrand = curand_uniform(&s);
    //myrand *= (0 - nC);
    //myrand += (nC - 0);
    //dev_ramC[tid] = (int)truncf(myrand);

    myrand = curand_uniform(&s);
    myrand *= (0 - tSize);
    myrand += (tSize - 0);
    int t = ((int)truncf(myrand) + blockIdx.x * tSize) * 2;
    //int t = (blockIdx.x * tSize) * 2;
    for (int n = 0; n < 2; n++)
    {
        dev_tran[tid * 2 + n] = dev_nt[t + n];
    }

    myrand = curand_uniform(&s);
    myrand *= (0 - rSize);
    myrand += (rSize - 0);
    int r = ((int)truncf(myrand) + blockIdx.x * rSize) * 4;
    //int r = (blockIdx.x + blockIdx.x * rSize) * 4;
    for (int n = 0; n < 4; n++)
    {
        dev_ramR[tid * 4 + n] = dev_nr[r + n];
    }

    myrand = curand_uniform(&s);
    myrand *= (0 - dSize);
    myrand += (dSize - 0);
    dev_ramD[tid] = dev_nd[blockIdx.x * dSize + (int)truncf(myrand)];
    //dev_ramD[tid] = dev_nd[blockIdx.x * dSize];
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRandomCTD(double* dev_nt,
                                    double* dev_tran,
                                    double* dev_nr,
                                    double* dev_ramR,
                                    unsigned int out,
                                    int rSize,
                                    int tSize
                                    )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float myrand;

    curandState s;
    curand_init(out, tid, 0, &s);

    //myrand = curand_uniform(&s);
    //myrand *= (0 - nC);
    //myrand += (nC - 0);
    //dev_ramC[tid] = (int)truncf(myrand);

    myrand = curand_uniform(&s);
    myrand *= (0 - tSize);
    myrand += (tSize - 0);
    int t = ((int)truncf(myrand) + blockIdx.x * tSize) * 2;
    //int t = (blockIdx.x * tSize) * 2;
    for (int n = 0; n < 2; n++)
    {
        dev_tran[tid * 2 + n] = dev_nt[t + n];
    }

    myrand = curand_uniform(&s);
    myrand *= (0 - rSize);
    myrand += (rSize - 0);
    int r = ((int)truncf(myrand) + blockIdx.x * rSize) * 4;
    //int r = (blockIdx.x + blockIdx.x * rSize) * 4;
    for (int n = 0; n < 4; n++)
    {
        dev_ramR[tid * 4 + n] = dev_nr[r + n];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRandomR(double* dev_mat,
                                  double* dev_ramR,
                                  int* dev_nc)
{
    // blockIdx.x -> index of each image
    // threadIdx.x -> index of each insertation of each image

    if (threadIdx.x < dev_nc[blockIdx.x]) // true if this image should be inserted
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        extern __shared__ double matR[];

        double *mat, *res;
        mat = matR + threadIdx.x * 18;
        res = mat  + 9;

        mat[0] = 0; mat[4] = 0; mat[8] = 0;
        mat[5] = dev_ramR[tid * 4 + 1];
        mat[6] = dev_ramR[tid * 4 + 2];
        mat[1] = dev_ramR[tid * 4 + 3];
        mat[7] = -mat[5];
        mat[2] = -mat[6];
        mat[3] = -mat[1];

        for(int i = 0; i < 9; i++)
            res[i] = 0;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    res[i + j * 3] += mat[i + k * 3] * mat[k + j * 3];

        double scale = 2 * dev_ramR[tid * 4];
        for (int n = 0; n < 9; n++)
        {
            mat[n] *= scale;
            mat[n] += res[n] * 2;
        }

        mat[0] += 1;
        mat[4] += 1;
        mat[8] += 1;

        for (int n = 0; n < 9; n++)
        {
            dev_mat[tid * 9 + n] = mat[n];
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRandomR(double* dev_mat,
                                  double* dev_ramR)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ double ramR[];

    double *mat, *res;
    mat = ramR + threadIdx.x * 18;
    res = mat  + 9;

    mat[0] = 0; mat[4] = 0; mat[8] = 0;
    mat[5] = dev_ramR[tid * 4 + 1];
    mat[6] = dev_ramR[tid * 4 + 2];
    mat[1] = dev_ramR[tid * 4 + 3];
    mat[7] = -mat[5];
    mat[2] = -mat[6];
    mat[3] = -mat[1];

    for(int i = 0; i < 9; i++)
        res[i] = 0;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                res[i + j * 3] += mat[i + k * 3] * mat[k + j * 3];

    double scale = 2 * dev_ramR[tid * 4];
    for (int n = 0; n < 9; n++)
    {
        mat[n] *= scale;
        mat[n] += res[n] * 2;
    }

    mat[0] += 1;
    mat[4] += 1;
    mat[8] += 1;

    for (int n = 0; n < 9; n++)
    {
        dev_mat[tid * 9 + n] = mat[n];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(RFLOAT* devdatPR,
                                 RFLOAT* devdatPI,
                                 Complex* devtranP,
                                 double* dev_tran,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int batch,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offNum, ntNum;
    int i, j;
    Complex imgTemp(0.0, 0.0);
    Complex datTemp(0.0, 0.0);
    RFLOAT phase, col, row;
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        offNum = itr / npxl;
        
        if (insertIdx < dev_nc[offNum])
        {
            ntNum = offNum * mReco + insertIdx;
            i = deviCol[itr % npxl] / opf;
            j = deviRow[itr % npxl] / opf;
            col = -(RFLOAT)(dev_tran[ntNum]) / idim;
            row = -(RFLOAT)(dev_tran[ntNum + 1]) / idim;
            phase = PI_2 * (i * col + j * row);
#ifdef SINGLE_PRECISION
            imgTemp.set(cosf(-phase), sinf(-phase));
#else
            imgTemp.set(cos(-phase), sin(-phase));
#endif
            datTemp.set(devdatPR[itr],
                        devdatPI[itr]);
            devtranP[itr] = datTemp * imgTemp;
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(RFLOAT* devdatPR,
                                 RFLOAT* devdatPI,
                                 Complex* devtranP,
                                 double* dev_offS,
                                 double* dev_tran,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int batch,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offNum, ntNum;
    int i, j;
    Complex imgTemp(0.0, 0.0);
    Complex datTemp(0.0, 0.0);
    RFLOAT phase, col, row;
    
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        offNum = itr / npxl;
        
        if (insertIdx < dev_nc[offNum])
        {
            ntNum = offNum * mReco + insertIdx;
            i = deviCol[itr % npxl] / opf;
            j = deviRow[itr % npxl] / opf;
            col = -(RFLOAT)(dev_tran[ntNum * 2] - dev_offS[offNum * 2]) / idim;
            row = -(RFLOAT)(dev_tran[ntNum * 2 + 1] - dev_offS[offNum * 2 + 1]) / idim;
            phase = PI_2 * (i * col + j * row);
#ifdef SINGLE_PRECISION
            imgTemp.set(cosf(-phase), sinf(-phase));
#else
            imgTemp.set(cos(-phase), sin(-phase));
#endif
            datTemp.set(devdatPR[itr],
                        devdatPI[itr]);
            devtranP[itr] = datTemp * imgTemp;
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(RFLOAT* devdatPR,
                                 RFLOAT* devdatPI,
                                 Complex* devtranP,
                                 double* dev_tran,
                                 int* deviCol,
                                 int* deviRow,
                                 int batch,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int ntNum;
    int i, j;
    Complex imgTemp(0.0, 0.0);
    Complex datTemp(0.0, 0.0);
    RFLOAT phase, col, row;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        i = deviCol[itr % npxl] / opf;
        j = deviRow[itr % npxl] / opf;

        ntNum = (itr / npxl) * mReco + insertIdx;
        col = -(RFLOAT)(dev_tran[ntNum]) / idim;
        row = -(RFLOAT)(dev_tran[ntNum + 1]) / idim;
        phase = PI_2 * (i * col + j * row);

#ifdef SINGLE_PRECISION
        imgTemp.set(cosf(-phase), sinf(-phase));
#else
        imgTemp.set(cos(-phase), sin(-phase));
#endif
        datTemp.set(devdatPR[itr],
                    devdatPI[itr]);
        devtranP[itr] = datTemp * imgTemp;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(RFLOAT* devdatPR,
                                 RFLOAT* devdatPI,
                                 Complex* devtranP,
                                 double* dev_offS,
                                 double* dev_tran,
                                 int* deviCol,
                                 int* deviRow,
                                 int batch,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j;
    int offNum, ntNum;
    Complex imgTemp(0.0, 0.0);
    Complex datTemp(0.0, 0.0);
    RFLOAT phase, col, row;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        i = deviCol[itr % npxl] / opf;
        j = deviRow[itr % npxl] / opf;

        offNum = itr / npxl;
        ntNum = (itr / npxl) * mReco + insertIdx;
        col = -(RFLOAT)(dev_tran[ntNum * 2] - dev_offS[offNum * 2]) / idim;
        row = -(RFLOAT)(dev_tran[ntNum * 2 + 1] - dev_offS[offNum * 2 + 1]) / idim;
        phase = PI_2 * (i * col + j * row);

#ifdef SINGLE_PRECISION
        imgTemp.set(cosf(-phase), sinf(-phase));
#else
        imgTemp.set(cos(-phase), sin(-phase));
#endif
        datTemp.set(devdatPR[itr],
                    devdatPI[itr]);
        devtranP[itr] = datTemp * imgTemp;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateCTF(RFLOAT* devctfP,
                                    CTFAttr* dev_ctfas,
                                    double* dev_ramD,
                                    int* dev_nc,
                                    int* deviCol,
                                    int* deviRow,
                                    RFLOAT pixel,
                                    int batch,
                                    int insertIdx,
                                    int opf,
                                    int npxl,
                                    int mReco)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j;
    int iNum, dNum;
    RFLOAT lambda, defocus, angle, k1, k2, ki, u, w1, w2;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        if (insertIdx < dev_nc[iNum])
        {
            dNum = iNum * mReco + insertIdx;
            i = deviCol[itr % npxl] / opf;
            j = deviRow[itr % npxl] / opf;
#ifdef SINGLE_PRECISION
            lambda = 12.2643247 / sqrtf(dev_ctfas[iNum].voltage
                                * (1 + dev_ctfas[iNum].voltage * 0.978466e-6));

            w1 = sqrtf(1 - dev_ctfas[iNum].amplitudeContrast * dev_ctfas[iNum].amplitudeContrast);
            w2 = dev_ctfas[iNum].amplitudeContrast;

            k1 = PI * lambda;
            k2 = DIVPI2 * dev_ctfas[iNum].Cs * lambda * lambda * lambda;
            u = sqrtf((i / pixel) * (i / pixel) + (j / pixel) * (j / pixel));
            angle = atan2f((float)j, (float)i) - dev_ctfas[iNum].defocusTheta;

            defocus = -(dev_ctfas[iNum].defocusU
                        + dev_ctfas[iNum].defocusV
                        + (dev_ctfas[iNum].defocusU - dev_ctfas[iNum].defocusV)
                        * cosf(2 * angle)) * (float)dev_ramD[dNum] / 2;

            ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[iNum].phaseShift;
            devctfP[itr] = -w1 * sinf(ki) + w2 * cosf(ki);
#else
            lambda = 12.2643247 / sqrt(dev_ctfas[iNum].voltage
                                * (1 + dev_ctfas[iNum].voltage * 0.978466e-6));

            w1 = sqrt(1 - dev_ctfas[iNum].amplitudeContrast * dev_ctfas[iNum].amplitudeContrast);
            w2 = dev_ctfas[iNum].amplitudeContrast;

            k1 = PI * lambda;
            k2 = DIVPI2 * dev_ctfas[iNum].Cs * lambda * lambda * lambda;
            u = sqrt((i / pixel) * (i / pixel) + (j / pixel) * (j / pixel));
            angle = atan2((double)j, (double)i) - dev_ctfas[iNum].defocusTheta;

            defocus = -(dev_ctfas[iNum].defocusU
                        + dev_ctfas[iNum].defocusV
                        + (dev_ctfas[iNum].defocusU - dev_ctfas[iNum].defocusV)
                        * cos(2 * angle)) * dev_ramD[dNum] / 2;

            ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[iNum].phaseShift;
            devctfP[itr] = -w1 * sin(ki) + w2 * cos(ki);
#endif
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateCTF(RFLOAT* devctfP,
                                    CTFAttr* dev_ctfas,
                                    double* dev_ramD,
                                    int* deviCol,
                                    int* deviRow,
                                    RFLOAT pixel,
                                    int batch,
                                    int insertIdx,
                                    int opf,
                                    int npxl,
                                    int mReco)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j;
    int iNum, dNum;
    RFLOAT lambda, defocus, angle, k1, k2, ki, u, w1, w2;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        i = deviCol[itr % npxl] / opf;
        j = deviRow[itr % npxl] / opf;
        iNum = itr / npxl;
        dNum = iNum * mReco + insertIdx;
#ifdef SINGLE_PRECISION
        lambda = 12.2643247 / sqrtf(dev_ctfas[iNum].voltage
                            * (1 + dev_ctfas[iNum].voltage * 0.978466e-6));

        w1 = sqrtf(1 - dev_ctfas[iNum].amplitudeContrast * dev_ctfas[iNum].amplitudeContrast);
        w2 = dev_ctfas[iNum].amplitudeContrast;

        k1 = PI * lambda;
        k2 = DIVPI2 * dev_ctfas[iNum].Cs * lambda * lambda * lambda;
        u = sqrtf((i / pixel) * (i / pixel) + (j / pixel) * (j / pixel));
        angle = atan2f((float)j, (float)i) - dev_ctfas[iNum].defocusTheta;

        defocus = -(dev_ctfas[iNum].defocusU
                    + dev_ctfas[iNum].defocusV
                    + (dev_ctfas[iNum].defocusU - dev_ctfas[iNum].defocusV)
                    * cosf(2 * angle)) * (float)dev_ramD[dNum] / 2;

        ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[iNum].phaseShift;
        devctfP[itr] = -w1 * sinf(ki) + w2 * cosf(ki);
#else
        lambda = 12.2643247 / sqrt(dev_ctfas[iNum].voltage
                            * (1 + dev_ctfas[iNum].voltage * 0.978466e-6));

        w1 = sqrt(1 - dev_ctfas[iNum].amplitudeContrast * dev_ctfas[iNum].amplitudeContrast);
        w2 = dev_ctfas[iNum].amplitudeContrast;

        k1 = PI * lambda;
        k2 = DIVPI2 * dev_ctfas[iNum].Cs * lambda * lambda * lambda;
        u = sqrt((i / pixel) * (i / pixel) + (j / pixel) * (j / pixel));
        angle = atan2((double)j, (double)i) - dev_ctfas[iNum].defocusTheta;

        defocus = -(dev_ctfas[iNum].defocusU
                    + dev_ctfas[iNum].defocusV
                    + (dev_ctfas[iNum].defocusU - dev_ctfas[iNum].defocusV)
                    * cos(2 * angle)) * dev_ramD[dNum] / 2;

        ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[iNum].phaseShift;
        devctfP[itr] = -w1 * sin(ki) + w2 * cos(ki);
#endif
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT2D(RFLOAT* devDataT,
                                 RFLOAT* devctfP,
                                 RFLOAT* devsigRcpP,
                                 RFLOAT* devTau,
                                 double* dev_nr,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int* deviSig,
                                 int batch,
                                 int insertIdx,
                                 int tauSize,
                                 int npxl,
                                 int mReco,
                                 int vdim,
                                 int vdimSize,
                                 int smidx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    int tauShift;
    RFLOAT ctfTemp;
    RFLOAT tempSW = 0.0;
    double oldCor0, oldCor1;
    
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        pNum = itr % npxl;
        cNum = iNum * mReco + insertIdx;

        oldCor0 = dev_nr[cNum * 2] * deviCol[pNum] 
                - dev_nr[cNum * 2 + 1] * deviRow[pNum];
        oldCor1 = dev_nr[cNum * 2 + 1] * deviCol[pNum] 
                + dev_nr[cNum * 2] * deviRow[pNum];

        tempSW = devsigRcpP[itr]
               * dev_ws_data[smidx][iNum];

        tauShift = dev_nc[cNum] * tauSize;
        atomicAdd(&devTau[tauShift + deviSig[pNum]], tempSW);
        
        ctfTemp = devctfP[itr]
                * devctfP[itr]
                * tempSW;

        addFTD2D(devDataT + dev_nc[cNum] * vdimSize,
                 ctfTemp,
                 (RFLOAT)oldCor0,
                 (RFLOAT)oldCor1,
                 vdim);
    }
}

__global__ void kernel_InsertF2D(Complex* devDataF,
                                 Complex* devtranP,
                                 RFLOAT* devctfP,
                                 RFLOAT* devsigRcpP,
                                 double* dev_nr,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int batch,
                                 int insertIdx,
                                 int npxl,
                                 int mReco,
                                 int vdim,
                                 int vdimSize,
                                 int smidx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    Complex tran(0.0, 0.0);
    double oldCor0, oldCor1;
    
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        pNum = itr % npxl;
        cNum = iNum * mReco + insertIdx;
        
        oldCor0 = dev_nr[cNum * 2] * deviCol[pNum] 
                - dev_nr[cNum * 2 + 1] * deviRow[pNum];
        oldCor1 = dev_nr[cNum * 2 + 1] * deviCol[pNum] 
                + dev_nr[cNum * 2] * deviRow[pNum];

        tran = devtranP[itr];
        tran *= (devctfP[itr]
                * devsigRcpP[itr]
                * dev_ws_data[smidx][iNum]);

        addFTC2D(devDataF + dev_nc[cNum] * vdimSize,
                 tran,
                 (RFLOAT)oldCor0,
                 (RFLOAT)oldCor1,
                 vdim);
    }

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT2D(RFLOAT* devDataT,
                                 RFLOAT* devctfP,
                                 RFLOAT* devTau,
                                 double* dev_nr,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int* deviSig,
                                 int batch,
                                 int insertIdx,
                                 int tauSize,
                                 int npxl,
                                 int mReco,
                                 int vdim,
                                 int vdimSize,
                                 int smidx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    int tauShift;
    RFLOAT ctfTemp;
    double oldCor0, oldCor1;
    
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        pNum = itr % npxl;
        cNum = iNum * mReco + insertIdx;
        
        oldCor0 = dev_nr[cNum * 2] * deviCol[pNum] 
                - dev_nr[cNum * 2 + 1] * deviRow[pNum];
        oldCor1 = dev_nr[cNum * 2 + 1] * deviCol[pNum] 
                + dev_nr[cNum * 2] * deviRow[pNum];

        tauShift = dev_nc[cNum] * tauSize;
        atomicAdd(&devTau[tauShift + deviSig[pNum]], dev_ws_data[smidx][iNum]);
        
        ctfTemp = devctfP[itr]
                * devctfP[itr]
                * dev_ws_data[smidx][iNum];

        addFTD2D(devDataT + dev_nc[cNum] * vdimSize,
                 ctfTemp,
                 (RFLOAT)oldCor0,
                 (RFLOAT)oldCor1,
                 vdim);
    }
}

__global__ void kernel_InsertF2D(Complex* devDataF,
                                 Complex* devtranP,
                                 RFLOAT* devctfP,
                                 double* dev_nr,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int batch,
                                 int insertIdx,
                                 int npxl,
                                 int mReco,
                                 int vdim,
                                 int vdimSize,
                                 int smidx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    Complex tran(0.0, 0.0);
    double oldCor0, oldCor1;
    
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        pNum = itr % npxl;
        cNum = iNum * mReco + insertIdx;
        
        oldCor0 = dev_nr[cNum * 2] * deviCol[pNum] 
                - dev_nr[cNum * 2 + 1] * deviRow[pNum];
        oldCor1 = dev_nr[cNum * 2 + 1] * deviCol[pNum] 
                + dev_nr[cNum * 2] * deviRow[pNum];

        tran = devtranP[itr];
        tran *= (devctfP[itr]
                * dev_ws_data[smidx][iNum]);

        addFTC2D(devDataF + dev_nc[cNum] * vdimSize,
                 tran,
                 (RFLOAT)oldCor0,
                 (RFLOAT)oldCor1,
                 vdim);
    }

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO2D(double* devO,
                                 int* devC,
                                 double* dev_nr,
                                 double* dev_nt,
                                 double* dev_offs,
                                 int* dev_nc,
                                 int insertIdx,
                                 int mReco)
{
    int ncIdx = threadIdx.x * mReco + insertIdx;
    int oshift = dev_nc[ncIdx];
    double resX = -1 * dev_nr[ncIdx * 2]
                     * (dev_nt[ncIdx * 2]
                        - dev_offs[threadIdx.x * 2])
                     + dev_nr[ncIdx * 2 + 1]
                     * (dev_nt[ncIdx * 2 + 1]
                        - dev_offs[threadIdx.x * 2 + 1]);

    double resY = -1 * (dev_nr[ncIdx * 2 + 1]
                        * (dev_nt[ncIdx * 2]
                           - dev_offs[threadIdx.x * 2])
                        + dev_nr[ncIdx * 2]
                        * (dev_nt[ncIdx * 2 + 1]
                           - dev_offs[threadIdx.x * 2 + 1]));

    atomicAdd(&devO[oshift * 2], resX);
    atomicAdd(&devO[oshift * 2 + 1], resY);
    atomicAdd(&devC[oshift], 1);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO2D(double* devO,
                                 int* devC,
                                 double* dev_nr,
                                 double* dev_nt,
                                 int* dev_nc,
                                 int insertIdx,
                                 int mReco)
{
    int ncIdx = threadIdx.x * mReco + insertIdx;
    int oshift = dev_nc[ncIdx];
    double resX = -1 * dev_nr[ncIdx * 2]
                     * dev_nt[ncIdx * 2]
                     + dev_nr[ncIdx * 2 + 1]
                     * dev_nt[ncIdx * 2 + 1];

    double resY = -1 * (dev_nr[ncIdx * 2 + 1]
                        * dev_nt[ncIdx * 2]
                        + dev_nr[ncIdx * 2]
                        * dev_nt[ncIdx * 2 + 1]);

    atomicAdd(&devO[oshift * 2], resX);
    atomicAdd(&devO[oshift * 2 + 1], resY);
    atomicAdd(&devC[oshift], 1);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(RFLOAT* devDataT,
                               RFLOAT* devctfP,
                               RFLOAT* devsigRcpP,
                               RFLOAT* devTau,
                               double* dev_mat,
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int* deviSig,
                               int batch,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    extern __shared__ double rotMat[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    RFLOAT ctfTemp;
    RFLOAT tempSW = 0.0;
    Mat33 mat;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        if (insertIdx < dev_nc[iNum])
        {
            pNum = itr % npxl;
            cNum = iNum * mReco + insertIdx;
            
            for (int r = 0; r < 9; r++)
                rotMat[threadIdx.x * 9 + r] = dev_mat[cNum * 9 + r];
            
            mat.init(rotMat, threadIdx.x);
        
            Vec3 newCor((double)deviCol[pNum], 
                        (double)deviRow[pNum], 
                        0);
            Vec3 oldCor = mat * newCor;
            
            tempSW = devsigRcpP[itr]
                   * dev_ws_data[smidx][iNum];

            atomicAdd(&devTau[deviSig[pNum]], tempSW);
            
            ctfTemp = devctfP[itr]
                    * devctfP[itr]
                    * tempSW;

            addFTD(devDataT,
                   ctfTemp,
                   (RFLOAT)oldCor(0),
                   (RFLOAT)oldCor(1),
                   (RFLOAT)oldCor(2),
                   vdim);
        }
    }
}

__global__ void kernel_InsertF(Complex* devDataF,
                               Complex* devtranP,
                               RFLOAT* devctfP,
                               RFLOAT* devsigRcpP,
                               double* dev_mat,
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int batch,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    extern __shared__ double rotMat[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    
    Complex tran(0.0, 0.0);
    Mat33 mat;
    
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        if (insertIdx < dev_nc[iNum])
        {
            pNum = itr % npxl;
            cNum = iNum * mReco + insertIdx;
            
            for (int r = 0; r < 9; r++)
                rotMat[threadIdx.x * 9 + r] = dev_mat[cNum * 9 + r];
            
            mat.init(rotMat, threadIdx.x);
        
            Vec3 newCor((double)deviCol[pNum], 
                        (double)deviRow[pNum], 
                        0);
            Vec3 oldCor = mat * newCor;
            
            tran = devtranP[itr];
            tran *= (devctfP[itr]
                    * devsigRcpP[itr]
                    * dev_ws_data[smidx][iNum]);

            addFTC(devDataF,
                   tran,
                   (RFLOAT)oldCor(0),
                   (RFLOAT)oldCor(1),
                   (RFLOAT)oldCor(2),
                   vdim);
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(RFLOAT* devDataT,
                               RFLOAT* devctfP,
                               RFLOAT* devTau,
                               double* dev_mat,
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int* deviSig,
                               int batch,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    extern __shared__ double rotMat[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    RFLOAT ctfTemp;
    Mat33 mat;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        if (insertIdx < dev_nc[iNum])
        {
            pNum = itr % npxl;
            cNum = iNum * mReco + insertIdx;
            
            for (int r = 0; r < 9; r++)
                rotMat[threadIdx.x * 9 + r] = dev_mat[cNum * 9 + r];
            
            mat.init(rotMat, threadIdx.x);
        
            Vec3 newCor((double)deviCol[pNum], 
                        (double)deviRow[pNum], 
                        0);
            Vec3 oldCor = mat * newCor;
            
            atomicAdd(&devTau[deviSig[pNum]], dev_ws_data[smidx][iNum]);
            
            ctfTemp = devctfP[itr]
                    * devctfP[itr]
                    * dev_ws_data[smidx][iNum];

            addFTD(devDataT,
                   ctfTemp,
                   (RFLOAT)oldCor(0),
                   (RFLOAT)oldCor(1),
                   (RFLOAT)oldCor(2),
                   vdim);
        }
    }
}

__global__ void kernel_InsertF(Complex* devDataF,
                               Complex* devtranP,
                               RFLOAT* devctfP,
                               double* dev_mat,
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int batch,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    extern __shared__ double rotMat[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    
    Complex tran(0.0, 0.0);
    Mat33 mat;
    
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        if (insertIdx < dev_nc[iNum])
        {
            pNum = itr % npxl;
            cNum = iNum * mReco + insertIdx;
            
            for (int r = 0; r < 9; r++)
                rotMat[threadIdx.x * 9 + r] = dev_mat[cNum * 9 + r];
            
            mat.init(rotMat, threadIdx.x);
        
            Vec3 newCor((double)deviCol[pNum], 
                        (double)deviRow[pNum], 
                        0);
            Vec3 oldCor = mat * newCor;
            
            tran = devtranP[itr];
            tran *= (devctfP[itr]
                    * dev_ws_data[smidx][iNum]);

            addFTC(devDataF,
                   tran,
                   (RFLOAT)oldCor(0),
                   (RFLOAT)oldCor(1),
                   (RFLOAT)oldCor(2),
                   vdim);
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO3D(double* devO,
                                 int* devC,
                                 double* dev_mat,
                                 double* dev_nt,
                                 double* dev_offs,
                                 int* dev_nc,
                                 int insertIdx,
                                 int mReco,
                                 int batchSize)
{
    extern __shared__ double oCor[];

    double* resX = oCor;
    double* resY = &resX[blockDim.x];
    double* resZ = &resY[blockDim.x];
    int* resC = (int*)&resZ[blockDim.x];

    int ncIdx;
    int nrIdx;
    int tran0;
    int tran1;

    resX[threadIdx.x] = 0;
    resY[threadIdx.x] = 0;
    resZ[threadIdx.x] = 0;
    resC[threadIdx.x] = 0;

    __syncthreads();

    for (int itr = threadIdx.x; itr < batchSize; itr += blockDim.x)
    {
        if (insertIdx < dev_nc[itr])
        {
            ncIdx = itr * mReco + insertIdx;
            nrIdx = ncIdx * 9;
            tran0 = dev_nt[ncIdx * 2] - dev_offs[itr * 2];
            tran1 = dev_nt[ncIdx * 2 + 1] - dev_offs[itr * 2 + 1];

            resX[threadIdx.x] += -1 * (dev_mat[nrIdx] * tran0
                                    + dev_mat[nrIdx + 3] * tran1);

            resY[threadIdx.x] += -1 * (dev_mat[nrIdx + 1] * tran0
                                    + dev_mat[nrIdx + 4] * tran1);

            resZ[threadIdx.x] += -1 * (dev_mat[nrIdx + 2] * tran0
                                    + dev_mat[nrIdx + 5] * tran1);

            resC[threadIdx.x] += 1;
        }

    }
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            resX[threadIdx.x] += resX[threadIdx.x + i];
            resY[threadIdx.x] += resY[threadIdx.x + i];
            resZ[threadIdx.x] += resZ[threadIdx.x + i];
            resC[threadIdx.x] += resC[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&devO[0], resX[0]);
        atomicAdd(&devO[1], resY[0]);
        atomicAdd(&devO[2], resZ[0]);
        atomicAdd(&devC[0], resC[0]);
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO3D(double* devO,
                                 int* devC,
                                 double* dev_mat,
                                 double* dev_nt,
                                 int* dev_nc,
                                 int insertIdx,
                                 int mReco,
                                 int batchSize)
{
    extern __shared__ double oCor[];

    double* resX = oCor;
    double* resY = &resX[blockDim.x];
    double* resZ = &resY[blockDim.x];
    int* resC = (int*)&resZ[blockDim.x];

    int ncIdx;
    int nrIdx;
    int tran0;
    int tran1;

    resX[threadIdx.x] = 0;
    resY[threadIdx.x] = 0;
    resZ[threadIdx.x] = 0;
    resC[threadIdx.x] = 0;

    __syncthreads();

    for (int itr = threadIdx.x; itr < batchSize; itr += blockDim.x)
    {
        if (insertIdx < dev_nc[itr])
        {
            ncIdx = itr * mReco + insertIdx;
            nrIdx = ncIdx * 9;
            tran0 = dev_nt[ncIdx * 2];
            tran1 = dev_nt[ncIdx * 2 + 1];

            resX[threadIdx.x] += -1 * (dev_mat[nrIdx] * tran0
                                    + dev_mat[nrIdx + 3] * tran1);

            resY[threadIdx.x] += -1 * (dev_mat[nrIdx + 1] * tran0
                                    + dev_mat[nrIdx + 4] * tran1);

            resZ[threadIdx.x] += -1 * (dev_mat[nrIdx + 2] * tran0
                                    + dev_mat[nrIdx + 5] * tran1);

            resC[threadIdx.x] += 1;
        }

    }
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            resX[threadIdx.x] += resX[threadIdx.x + i];
            resY[threadIdx.x] += resY[threadIdx.x + i];
            resZ[threadIdx.x] += resZ[threadIdx.x + i];
            resC[threadIdx.x] += resC[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&devO[0], resX[0]);
        atomicAdd(&devO[1], resY[0]);
        atomicAdd(&devO[2], resZ[0]);
        atomicAdd(&devC[0], resC[0]);
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(RFLOAT* devDataT,
                               RFLOAT* devctfP,
                               RFLOAT* devsigRcpP,
                               RFLOAT* devTau,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int* deviSig,
                               int batch,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    extern __shared__ double rotMat[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;

    RFLOAT ctfTemp;
    RFLOAT tempSW = 0.0;
    Mat33 mat;
    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        pNum = itr % npxl;
        cNum = iNum * mReco + insertIdx;
        
        for (int r = 0; r < 9; r++)
            rotMat[threadIdx.x * 9 + r] = dev_mat[cNum * 9 + r];
        
        mat.init(rotMat, threadIdx.x);
        
        Vec3 newCor((double)deviCol[pNum], 
                    (double)deviRow[pNum], 
                    0);
        Vec3 oldCor = mat * newCor;

        tempSW = devsigRcpP[itr]
               * dev_ws_data[smidx][iNum];

        atomicAdd(&devTau[deviSig[pNum]], tempSW);
            
        ctfTemp = devctfP[itr]
                * devctfP[itr]
                * tempSW;

        addFTD(devDataT,
               ctfTemp,
               (RFLOAT)oldCor(0),
               (RFLOAT)oldCor(1),
               (RFLOAT)oldCor(2),
               vdim);
    }
}

__global__ void kernel_InsertF(Complex* devDataF,
                               Complex* devtranP,
                               RFLOAT* devctfP,
                               RFLOAT* devsigRcpP,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int batch,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    extern __shared__ double rotMat[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    
    Complex tran(0.0, 0.0);
    Mat33 mat;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        pNum = itr % npxl;
        cNum = iNum * mReco + insertIdx;
        
        for (int r = 0; r < 9; r++)
            rotMat[threadIdx.x * 9 + r] = dev_mat[cNum * 9 + r];
        
        mat.init(rotMat, threadIdx.x);
        
        Vec3 newCor((double)deviCol[pNum], 
                    (double)deviRow[pNum], 
                    0);
        Vec3 oldCor = mat * newCor;

        tran = devtranP[itr];
        tran *= (devctfP[itr]
                * devsigRcpP[itr]
                * dev_ws_data[smidx][iNum]);

        addFTC(devDataF,
               tran,
               (RFLOAT)oldCor(0),
               (RFLOAT)oldCor(1),
               (RFLOAT)oldCor(2),
               vdim);
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(RFLOAT* devDataT,
                               RFLOAT* devctfP,
                               RFLOAT* devTau,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int* deviSig,
                               int batch,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    extern __shared__ double rotMat[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;

    RFLOAT ctfTemp;
    Mat33 mat;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        pNum = itr % npxl;
        cNum = iNum * mReco + insertIdx;
        
        for (int r = 0; r < 9; r++)
            rotMat[threadIdx.x * 9 + r] = dev_mat[cNum * 9 + r];
        
        mat.init(rotMat, threadIdx.x);
        
        Vec3 newCor((double)deviCol[pNum], 
                    (double)deviRow[pNum], 
                    0);
        Vec3 oldCor = mat * newCor;

        atomicAdd(&devTau[deviSig[pNum]], dev_ws_data[smidx][iNum]);
        
        ctfTemp = devctfP[itr]
                * devctfP[itr]
                * dev_ws_data[smidx][iNum];

        addFTD(devDataT,
               ctfTemp,
               (RFLOAT)oldCor(0),
               (RFLOAT)oldCor(1),
               (RFLOAT)oldCor(2),
               vdim);
    }
}

__global__ void kernel_InsertF(Complex* devDataF,
                               Complex* devtranP,
                               RFLOAT* devctfP,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int batch,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    extern __shared__ double rotMat[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int iNum, pNum, cNum;
    
    Complex tran(0.0, 0.0);
    Mat33 mat;

    for (int itr = tid; itr < batch * npxl; itr += gridDim.x * blockDim.x)
    {
        iNum = itr / npxl;
        pNum = itr % npxl;
        cNum = iNum * mReco + insertIdx;
        
        for (int r = 0; r < 9; r++)
            rotMat[threadIdx.x * 9 + r] = dev_mat[cNum * 9 + r];
        
        mat.init(rotMat, threadIdx.x);
        
        Vec3 newCor((double)deviCol[pNum], 
                    (double)deviRow[pNum], 
                    0);
        Vec3 oldCor = mat * newCor;

        tran = devtranP[itr];
        tran *= (devctfP[itr]
                * dev_ws_data[smidx][iNum]);

        addFTC(devDataF,
               tran,
               (RFLOAT)oldCor(0),
               (RFLOAT)oldCor(1),
               (RFLOAT)oldCor(2),
               vdim);
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO3D(double* devO,
                                 int* devC,
                                 double* dev_mat,
                                 double* dev_nt,
                                 double* dev_offs,
                                 int insertIdx,
                                 int mReco,
                                 int batchSize)
{
    extern __shared__ double oResCor[];

    double* resX = oResCor;
    double* resY = &resX[blockDim.x];
    double* resZ = &resY[blockDim.x];
    int* resC = (int*)&resZ[blockDim.x];

    int ncIdx;
    int nrIdx;
    int tran0;
    int tran1;

    resX[threadIdx.x] = 0;
    resY[threadIdx.x] = 0;
    resZ[threadIdx.x] = 0;
    resC[threadIdx.x] = 0;

    __syncthreads();

    for (int itr = threadIdx.x; itr < batchSize; itr += blockDim.x)
    {
        ncIdx = itr * mReco + insertIdx;
        nrIdx = ncIdx * 9;
        tran0 = dev_nt[ncIdx * 2] - dev_offs[itr * 2];
        tran1 = dev_nt[ncIdx * 2 + 1] - dev_offs[itr * 2 + 1];

        resX[threadIdx.x] += -1 * (dev_mat[nrIdx] * tran0
                                + dev_mat[nrIdx + 3] * tran1);

        resY[threadIdx.x] += -1 * (dev_mat[nrIdx + 1] * tran0
                                + dev_mat[nrIdx + 4] * tran1);

        resZ[threadIdx.x] += -1 * (dev_mat[nrIdx + 2] * tran0
                                + dev_mat[nrIdx + 5] * tran1);

        resC[threadIdx.x] += 1;

    }
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            resX[threadIdx.x] += resX[threadIdx.x + i];
            resY[threadIdx.x] += resY[threadIdx.x + i];
            resZ[threadIdx.x] += resZ[threadIdx.x + i];
            resC[threadIdx.x] += resC[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&devO[0], resX[0]);
        atomicAdd(&devO[1], resY[0]);
        atomicAdd(&devO[2], resZ[0]);
        atomicAdd(&devC[0], resC[0]);
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO3D(double* devO,
                                 int* devC,
                                 double* dev_mat,
                                 double* dev_nt,
                                 int insertIdx,
                                 int mReco,
                                 int batchSize)
{
    extern __shared__ double oData[];

    double* resX = oData;
    double* resY = &resX[blockDim.x];
    double* resZ = &resY[blockDim.x];
    int* resC = (int*)&resZ[blockDim.x];

    int ncIdx;
    int nrIdx;
    int tran0;
    int tran1;

    resX[threadIdx.x] = 0;
    resY[threadIdx.x] = 0;
    resZ[threadIdx.x] = 0;
    resC[threadIdx.x] = 0;

    __syncthreads();

    for (int itr = threadIdx.x; itr < batchSize; itr += blockDim.x)
    {
        ncIdx = itr * mReco + insertIdx;
        nrIdx = ncIdx * 9;
        tran0 = dev_nt[ncIdx * 2];
        tran1 = dev_nt[ncIdx * 2 + 1];

        resX[threadIdx.x] += -1 * (dev_mat[nrIdx] * tran0
                                + dev_mat[nrIdx + 3] * tran1);

        resY[threadIdx.x] += -1 * (dev_mat[nrIdx + 1] * tran0
                                + dev_mat[nrIdx + 4] * tran1);

        resZ[threadIdx.x] += -1 * (dev_mat[nrIdx + 2] * tran0
                                + dev_mat[nrIdx + 5] * tran1);

        resC[threadIdx.x] += 1;

    }
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            resX[threadIdx.x] += resX[threadIdx.x + i];
            resY[threadIdx.x] += resY[threadIdx.x + i];
            resZ[threadIdx.x] += resZ[threadIdx.x + i];
            resC[threadIdx.x] += resC[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&devO[0], resX[0]);
        atomicAdd(&devO[1], resY[0]);
        atomicAdd(&devO[2], resZ[0]);
        atomicAdd(&devC[0], resC[0]);
    }
}

/**
 * @brief 
 *
 * @param 
 * @param 
 * @param 
 */
__global__ void kernel_SetSF(RFLOAT* devDataT,
                             RFLOAT* sf)
{
    sf[threadIdx.x] = 1.0 / devDataT[threadIdx.x];
}

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_SetSF2D(RFLOAT* devDataT,
                               RFLOAT* sf,
                               int dimSize)
{
    sf[threadIdx.x] = 1.0 / devDataT[threadIdx.x * dimSize];
}

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeTF2D(Complex *devDataF,
                                     RFLOAT *devDataT,
                                     RFLOAT *sf,
                                     int kIdx,
                                     int dimSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    while (index < dimSize)
    {
        devDataF[index] *= sf[kIdx];
        devDataT[index] *= sf[kIdx];

        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeTF(Complex* devDataF,
                                   RFLOAT* devDataT,
                                   RFLOAT* sf,
                                   int batch)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    while(index < batch)
    {
        devDataF[index] *= sf[0];
        devDataT[index] *= sf[0];
        
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeT(RFLOAT *devDataT,
                                  const int length,
                                  const int num,
                                  const RFLOAT sf)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.normalizeT(devDataT, length, num, sf);
}

/**
 * @brief Symmetrize T3D
 *
 * @param devDataT : the pointer of T3D
 * @param devSym : the pointer of Volume
 * @param devSymmat : the Symmetry Matrix
 * @param numSymMat : the size of the Symmetry Matrix
 * @param r : the range of T3D elements need to be symmetrized
 * @param interp : the way of interpolating
 * @param dim : the length of one side of T3D
 */
__global__ void kernel_SymmetrizeT(RFLOAT *devDataT,
                                   double *devSymmat,
                                   const int numSymMat,
                                   const int r,
                                   const int interp,
                                   const size_t num,
                                   const int dim,
                                   const size_t dimSize,
                                   cudaTextureObject_t texObject)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;
    Mat33 mat;

    while(index < dimSize)
    {
        i = (index + num) % (dim / 2 + 1);
        j = ((index + num) / (dim / 2 + 1)) % dim;
        k = ((index + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        for (int m = 0 ; m < numSymMat; m++)
        {
            mat.init(devSymmat, m);
            
            RFLOAT inc = 0.0;
            Vec3 newCor((double)i, (double)j, (double)k);
            Vec3 oldCor = mat * newCor;
            
#ifdef SINGLE_PRECISION
            if ((int)floorf(oldCor.squaredNorm3()) < r)
            {
                inc = getByInterpolationFT((RFLOAT)oldCor(0),
                                           (RFLOAT)oldCor(1),
                                           (RFLOAT)oldCor(2),
                                           interp,
                                           dim,
                                           texObject);
                
                devDataT[index] += inc;
            }
#else
            if ((int)floor(oldCor.squaredNorm3()) < r)
            {
                inc = getByInterpolationFT((RFLOAT)oldCor(0),
                                           (RFLOAT)oldCor(1),
                                           (RFLOAT)oldCor(2),
                                           interp,
                                           dim,
                                           texObject);
                
                devDataT[index] += inc;
            }
#endif    
        } 
#ifdef SINGLE_PRECISION
        devDataT[index] = fmaxf(devDataT[index], 1e-25);
#else
        devDataT[index] = fmax(devDataT[index], 1e-25);
#endif    
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief Normalize F: F = F * sf
 *
 * @param devDataF : the pointer of F3D
 * @param length : F3D's size
 * @param sf : the coefficient to Normalize F
 **/
__global__ void kernel_NormalizeF(Complex *devDataF,
                                  const int length,
                                  const int num,
                                  const RFLOAT sf)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.normalizeF(devDataF, length, num, sf);
}

/**
 * @brief Symmetrize F3D
 *
 * @param devDataF : the pointer of T3D
 * @param devSym : the pointer of Volume
 * @param devSymmat : the Symmetry Matrix
 * @param numSymMat : the size of the Symmetry Matrix
 * @param r : the range of T3D elements need to be symmetrized
 * @param interp : the way of interpolating
 * @param dim : the length of one side of F3D
 **/
__global__ void kernel_SymmetrizeF(Complex *devDataF,
                                   double *devSymmat,
                                   const int numSymMat,
                                   const int r,
                                   const int interp,
                                   const size_t num,
                                   const int dim,
                                   const size_t dimSize,
                                   cudaTextureObject_t texObject)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;
    Mat33 mat;

    while(index < dimSize)
    {
        i = (index + num) % (dim / 2 + 1);
        j = ((index + num) / (dim / 2 + 1)) % dim;
        k = ((index + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        for (int m = 0 ; m < numSymMat; m++)
        {
            mat.init(devSymmat, m);
            
            Complex inc(0.0, 0.0);
            
            Vec3 newCor((double)i, (double)j, (double)k);
            Vec3 oldCor = mat * newCor;
#ifdef SINGLE_PRECISION
            if ((int)floorf(oldCor.squaredNorm3()) < r)
            {
                inc = getByInterpolationFTC((RFLOAT)oldCor(0),
                                            (RFLOAT)oldCor(1),
                                            (RFLOAT)oldCor(2),
                                            interp,
                                            dim,
                                            texObject);
                
                devDataF[index] += inc;
            }
#else
            if ((int)floor(oldCor.squaredNorm3()) < r)
            {
                inc = getByInterpolationFTC((RFLOAT)oldCor(0),
                                            (RFLOAT)oldCor(1),
                                            (RFLOAT)oldCor(2),
                                            interp,
                                            dim,
                                            texObject);
                
                devDataF[index] += inc;
            }
#endif    
        }        
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ShellAverage2D(RFLOAT *devAvg2D,
                                      int *devCount2D,
                                      RFLOAT *devDataT,
                                      int dim,
                                      int r)
{
    extern __shared__ RFLOAT sum[];

    RFLOAT *sumAvg = sum;
    int *sumCount = (int*)&sumAvg[r];

    for (int itr = threadIdx.x; itr < r; itr += blockDim.x)
    {
        sumAvg[itr] = 0;
        sumCount[itr] = 0;
    }

    __syncthreads();

    int j = blockIdx.x;
    if(j >= dim / 2) j = j - dim;

    for (int itr = threadIdx.x; itr < r; itr += blockDim.x)
    {
        int quad = itr * itr + j * j;

        if(quad < r * r)
        {
#ifdef SINGLE_PRECISION
            int u = (int)rintf(sqrtf((float)quad));
#else
            int u = (int)rint(sqrt((double)quad));
#endif
            if (u < r)
            {
                atomicAdd(&sumAvg[u], devDataT[itr + blockIdx.x * blockDim.x]);
                atomicAdd(&sumCount[u], 1);
            }
        }
    }

    __syncthreads();

    for (int itr = threadIdx.x; itr < r; itr += blockDim.x)
    {
        //devAvg2D[itr + blockIdx.x * r] = sumAvg[itr];
        //devCount2D[itr + blockIdx.x * r] = sumCount[itr];
        devAvg2D[itr * gridDim.x + blockIdx.x] = sumAvg[itr];
        devCount2D[itr * gridDim.x + blockIdx.x] = sumCount[itr];
    }

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ShellAverage(RFLOAT *devAvg2D,
                                    int *devCount2D,
                                    RFLOAT *devDataT,
                                    int volumeShift,
                                    int smidxShift,
                                    int lineSize,
                                    int r,
                                    int dim,
                                    size_t dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j, k;

    extern __shared__ RFLOAT sum[];

    RFLOAT *sumAvg = sum;
    int *sumCount = (int*)&sumAvg[r];

    for (int itr = threadIdx.x; itr < r; itr+= blockDim.x)
    {
        sumAvg[itr] = 0;
        sumCount[itr] = 0;
    }

    __syncthreads();

    while(tid < dimSize)
    {
        i = (tid + volumeShift) % (dim / 2 + 1);
        j = ((tid + volumeShift) / (dim / 2 + 1)) % dim;
        k = ((tid + volumeShift) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if(quad < r * r)
        {
#ifdef SINGLE_PRECISION
            int u = (int)rintf(sqrtf((float)quad));
#else
            int u = (int)rint(sqrt((double)quad));
#endif
            if (u < r)
            {
                atomicAdd(&sumAvg[u], devDataT[tid]);
                atomicAdd(&sumCount[u], 1);
            }
        }
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

    for (int itr = threadIdx.x; itr < r; itr+= blockDim.x)
    {
        devAvg2D[itr * lineSize + smidxShift + blockIdx.x] += sumAvg[itr];
        devCount2D[itr * lineSize + smidxShift + blockIdx.x] += sumCount[itr];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateAvg(RFLOAT *devAvg2D,
                                    int *devCount2D,
                                    RFLOAT *devAvg,
                                    int *devCount,
                                    int dim,
                                    int r)
{
    for (int itr = threadIdx.x; itr < r; itr+= blockDim.x)
    {
        devAvg[threadIdx.x] = 0;
        devCount[threadIdx.x] = 0;
    }

    __syncthreads();

    for (int itr = threadIdx.x; itr < r; itr += blockDim.x)
    {
        for (int i = 0; i < dim; i++)
        {
            devAvg[itr] += devAvg2D[itr + i * r];
            devCount[itr] += devCount2D[itr + i * r];
        }

        devAvg[itr] /= devCount[itr];

        if(itr == r - 1){
            devAvg[r] = devAvg[r - 1];
            devAvg[r + 1] = devAvg[r - 1];
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateAvg(RFLOAT *devAvg2D,
                                    int *devCount2D,
                                    RFLOAT *devAvg,
                                    int *devCount,
                                    int lineSize)
{
    extern __shared__ RFLOAT cal[];

    RFLOAT *calAvg = cal;
    int *calCount = (int*)&calAvg[blockDim.x];

    calAvg[threadIdx.x] = 0.0;
    calCount[threadIdx.x] = 0;
    int lineShift = blockIdx.x * lineSize;
    for (int itr = threadIdx.x; itr < lineSize; itr += blockDim.x)
    {
        calAvg[threadIdx.x] += devAvg2D[lineShift + itr];
        calCount[threadIdx.x] += devCount2D[lineShift + itr];
    }
    
    __syncthreads();
    
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }

    while (j != 0)
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                calAvg[threadIdx.x] += calAvg[threadIdx.x + j];
                calCount[threadIdx.x] += calCount[threadIdx.x + j];
            }

        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                calAvg[threadIdx.x] += calAvg[threadIdx.x + j];
                calCount[threadIdx.x] += calCount[threadIdx.x + j];
            }

        }

        __syncthreads();

        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        j /= 2;
    }

    if (threadIdx.x == 0)
    {
        devAvg[blockIdx.x] = calAvg[threadIdx.x];
        devCount[blockIdx.x] = calCount[threadIdx.x];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_mergeAvgCount(RFLOAT *devAvg,
                                     int *devCount,
                                     int r)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < r)
        devAvg[tid] /= devCount[tid];
    
    if(tid == r - 1){
        devAvg[r] = devAvg[r - 1];
        devAvg[r + 1] = devAvg[r - 1];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateFSC2D(RFLOAT *devDataT,
                                      RFLOAT *devFSC,
                                      RFLOAT *devAvg,
                                      bool joinHalf,
                                      int fscMatsize,
                                      int wiener,
                                      int dim,
                                      int pf,
                                      int r)
{
    int j = blockIdx.x;
    int jDim = dim / 2 + 1;
    if(j >= dim / 2) j = j - dim;

    for (int itr = threadIdx.x; itr < jDim; itr += blockDim.x)
    {
        int quad = itr * itr + j * j;

        //if (itr + blockIdx.x * jDim == 59)
        //    printf("itr:%d, j:%d, quad:%d, wiener:%d, r:%d\n", itr, j, quad, wiener, r);
        if(quad >= wiener && quad < r)
        {
#ifdef SINGLE_PRECISION
            int u = (int)rintf(sqrtf((float)quad));
            float FSC = (u / pf >= fscMatsize)
                      ? 0
                      : devFSC[u / pf];

            FSC = fmaxf(1e-3, fminf(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrtf(2 * FSC / (1 + FSC));
#else
            if (joinHalf)
                FSC = sqrtf(2 * FSC / (1 + FSC));
#endif

#else
            int u = (int)rint(sqrt((double)quad));
            double FSC = (u / pf >= fscMatsize)
                       ? 0
                       : devFSC[u / pf];

            FSC = fmax(1e-3, fmin(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrt(2 * FSC / (1 + FSC));
#else
            if (joinHalf)
                FSC = sqrt(2 * FSC / (1 + FSC));
#endif
#endif
            devDataT[itr + blockIdx.x * jDim] += (1 - FSC) / FSC * devAvg[u];
        }
#ifdef SINGLE_PRECISION
        devDataT[itr + blockIdx.x * jDim] = fmaxf(devDataT[itr + blockIdx.x * jDim], 1e-25);
#else
        devDataT[itr + blockIdx.x * jDim] = fmax(devDataT[itr + blockIdx.x * jDim], 1e-25);
#endif
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateFSC2D(RFLOAT *devDataT,
                                      RFLOAT *devFSC,
                                      bool joinHalf,
                                      int fscMatsize,
                                      int wiener,
                                      int dim,
                                      int pf,
                                      int r)
{
    int j = blockIdx.x;
    int jDim = dim / 2 + 1;
    if(j >= dim / 2) j = j - dim;

    for (int itr = threadIdx.x; itr < jDim; itr += blockDim.x)
    {
        int quad = itr * itr + j * j;

        if(quad >= wiener && quad < r)
        {
#ifdef SINGLE_PRECISION
            int u = (int)rintf(sqrtf((float)quad));
            float FSC = (u / pf >= fscMatsize)
                      ? 0
                      : devFSC[u / pf];

            FSC = fmaxf(1e-3, fminf(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrtf(2 * FSC / (1 + FSC));
#else
            if (joinHalf)
                FSC = sqrtf(2 * FSC / (1 + FSC));
#endif
#else
            int u = (int)rint(sqrt((double)quad));
            double FSC = (u / pf >= fscMatsize)
                       ? 0
                       : devFSC[u / pf];

            FSC = fmax(1e-3, fmin(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrt(2 * FSC / (1 + FSC));
#else
            if (joinHalf)
                FSC = sqrt(2 * FSC / (1 + FSC));
#endif
#endif
            devDataT[itr + blockIdx.x * jDim] /= FSC;
        }
#ifdef SINGLE_PRECISION
        devDataT[itr + blockIdx.x * jDim] = fmaxf(devDataT[itr + blockIdx.x * jDim], 1e-25);
#else
        devDataT[itr + blockIdx.x * jDim] = fmax(devDataT[itr + blockIdx.x * jDim], 1e-25);
#endif
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateFSC(RFLOAT *devDataT,
                                    RFLOAT *devFSC,
                                    RFLOAT *devAvg,
                                    size_t num,
                                    bool joinHalf,
                                    int fscMatsize,
                                    int wiener,
                                    int r,
                                    int pf,
                                    int dim,
                                    size_t dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j, k;

    while(tid < dimSize)
    {
        i = (tid + num) % (dim / 2 + 1);
        j = ((tid + num) / (dim / 2 + 1)) % dim;
        k = ((tid + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if(quad >= wiener && quad < r)
        {
#ifdef SINGLE_PRECISION
            int u = (int)rintf(sqrtf((float)quad));
            float FSC = (u / pf >= fscMatsize)
                      ? 0
                      : devFSC[u / pf];

            FSC = fmaxf(1e-3, fminf(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrtf(2 * FSC / (1 + FSC));
#else
            if (joinHalf)
                FSC = sqrtf(2 * FSC / (1 + FSC));
#endif

#else
            int u = (int)rint(sqrt((double)quad));
            double FSC = (u / pf >= fscMatsize)
                       ? 0
                       : devFSC[u / pf];

            FSC = fmax(1e-3, fmin(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrt(2 * FSC / (1 + FSC));
#else
            if (joinHalf)
                FSC = sqrt(2 * FSC / (1 + FSC));
#endif

#endif
            devDataT[tid] += (1 - FSC) / FSC * devAvg[u];
        }
#ifdef SINGLE_PRECISION
        devDataT[tid] = fmaxf(devDataT[tid], 1e-25);
#else
        devDataT[tid] = fmax(devDataT[tid], 1e-25);
#endif
        tid += blockDim.x * gridDim.x;
    }

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateFSC(RFLOAT *devDataT,
                                    RFLOAT *devFSC,
                                    size_t num,
                                    bool joinHalf,
                                    int fscMatsize,
                                    int wiener,
                                    int r,
                                    int pf,
                                    int dim,
                                    size_t dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j, k;

    while(tid < dimSize)
    {
        i = (tid + num) % (dim / 2 + 1);
        j = ((tid + num) / (dim / 2 + 1)) % dim;
        k = ((tid + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if(quad >= wiener && quad < r)
        {
#ifdef SINGLE_PRECISION
            int u = (int)rintf(sqrtf((float)quad));
            float FSC = (u / pf >= fscMatsize)
                      ? 0
                      : devFSC[u / pf];

            FSC = fmaxf(1e-3, fminf(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrtf(2 * FSC / (1 + FSC));
#else
            if (joinHalf)
                FSC = sqrtf(2 * FSC / (1 + FSC));
#endif

#else
            int u = (int)rint(sqrt((double)quad));
            double FSC = (u / pf >= fscMatsize)
                       ? 0
                       : devFSC[u / pf];

            FSC = fmax(1e-3, fmin(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrt(2 * FSC / (1 + FSC));
#else
            if (joinHalf)
                FSC = sqrt(2 * FSC / (1 + FSC));
#endif

#endif
            devDataT[tid] /= FSC;
        }
#ifdef SINGLE_PRECISION
        devDataT[tid] = fmaxf(devDataT[tid], 1e-25);
#else
        devDataT[tid] = fmax(devDataT[tid], 1e-25);
#endif
        tid += blockDim.x * gridDim.x;
    }

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_WienerConst(RFLOAT *devDataT,
                                   int wiener,
                                   int r,
                                   int num,
                                   int dim,
                                   int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.wienerConst(devDataT,
                            wiener,
                            r,
                            num,
                            dim,
                            dimSize);

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateW2D(RFLOAT *devDataW,
                                    RFLOAT *devDataT,
                                    const int dim,
                                    const int r)
{
    int j = blockIdx.x;
    int jDim = dim / 2 + 1;
    if(j >= dim / 2) j = j - dim;

    for (int itr = threadIdx.x; itr < jDim; itr += jDim)
    {
        int quad = itr * itr + j * j;

        if (quad < r)
        {
#ifdef SINGLE_PRECISION
            devDataW[itr
                     + jDim
                     * blockIdx.x] = 1.0 / fmaxf(fabsf(devDataT[itr
                                                                + jDim
                                                                * blockIdx.x]), 1e-6);
#else
            devDataW[itr
                     + jDim
                     * blockIdx.x] = 1.0 / fmax(fabs(devDataT[itr
                                                              + jDim
                                                              * blockIdx.x]), 1e-6);
#endif
        }
        else
        {
            devDataW[itr + jDim * blockIdx.x] = 0.0;
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateW(RFLOAT *devDataW,
                                  RFLOAT *devDataT,
                                  const size_t shift,
                                  const int r,
                                  const int dim,
                                  const size_t length)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;

    while(tid < length)
    {
        i = (tid + shift) % (dim / 2 + 1);
        j = ((tid + shift) / (dim / 2 + 1)) % dim;
        k = ((tid + shift) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if (quad < r)
        {
#ifdef SINGLE_PRECISION
            devDataW[tid] = 1.0 / fmaxf(fabsf(devDataT[tid]), 1e-6);
#else
            devDataW[tid] = 1.0 / fmax(fabs(devDataT[tid]), 1e-6);
#endif
        }
        else
        {
            devDataW[tid] = 0.0;
        }
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InitialW2D(RFLOAT *devDataW,
                                  int initWR,
                                  int dim)
{
    int j = blockIdx.x;
    int jDim = dim / 2 + 1;
    if(j >= dim / 2) j = j - dim;

    for (int itr = threadIdx.x; itr < jDim; itr += blockDim.x)
    {
        int quad = itr * itr + j * j;

        if (quad < initWR)
        {
            devDataW[itr + jDim * blockIdx.x] = 1;
        }
        else
        {
            devDataW[itr + jDim * blockIdx.x] = 0;
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InitialW(RFLOAT *devDataW,
                                int initWR,
                                int dim,
                                size_t dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;

    while(tid < dimSize)
    {
        i = (tid) % (dim / 2 + 1);
        j = ((tid) / (dim / 2 + 1)) % dim;
        k = ((tid) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if (quad < initWR)
        {
            devDataW[tid] = 1;
        }
        else
        {
            devDataW[tid] = 0;
        }

        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InitialW(RFLOAT *devDataW,
                                int initWR,
                                int shift,
                                int dim,
                                size_t dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;

    while(tid < dimSize)
    {
        i = (tid + shift) % (dim / 2 + 1);
        j = ((tid + shift) / (dim / 2 + 1)) % dim;
        k = ((tid + shift) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if (quad < initWR)
        {
            devDataW[tid] = 1;
        }
        else
        {
            devDataW[tid] = 0;
        }

        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_DeterminingC2D(Complex *devDataC,
                                      RFLOAT *devDataT,
                                      RFLOAT *devDataW,
                                      int length)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < length)
    {
        devDataC[tid].set(devDataT[tid] * devDataW[tid], 0);
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_DeterminingC(RFLOAT *devDataC,
                                    RFLOAT *devDataT,
                                    RFLOAT *devDataW,
                                    const size_t length)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < length)
    {
        devDataC[tid] = devDataT[tid] * devDataW[tid];
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_convoluteC2D(RFLOAT* devRealC,
                                    TabFunction tabfunc,
                                    RFLOAT nf,
                                    int padSize,
                                    int dim,
                                    int dimSize)
{
    int j = blockIdx.x;
    int index, i;
    if(j >= dim / 2) j = j - dim;

    for (int itr = threadIdx.x; itr < dim; itr += blockDim.x)
    {
        i = itr;
        if(i >= dim / 2) i = i - dim;

        index = itr + dim * blockIdx.x;

        devRealC[index] = devRealC[index]
                        / dimSize
                        * tabfunc((RFLOAT)(i * i + j * j)
                                  / (padSize * padSize))
                        / nf;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ConvoluteC(RFLOAT *devDataC,
                                  TabFunction tabfunc,
                                  RFLOAT nf,
                                  int dim,
                                  size_t shift,
                                  int padSize,
                                  size_t batch)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;

    while(tid < batch)
    {
        i = (tid + shift) % dim;
        j = ((tid + shift) / dim) % dim;
        k = ((tid + shift) / dim) / dim;

        if(i >= dim / 2) i = i - dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        devDataC[tid] = devDataC[tid]
                      * tabfunc((RFLOAT)(i*i + j*j + k*k)
                                / (padSize * padSize))
                      / nf;

        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_convoluteC(RFLOAT *devDoubleC,
                                  TabFunction tabfunc,
                                  RFLOAT nf,
                                  int padSize,
                                  int dim,
                                  size_t dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;

    while(tid < dimSize)
    {
        i = tid % dim;
        j = (tid / dim) % dim;
        k = (tid / dim) / dim;

        if(i >= dim / 2) i = i - dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        devDoubleC[tid] = devDoubleC[tid]
                          / dimSize
                          * tabfunc((RFLOAT)(i*i + j*j + k*k) / (padSize * padSize))
                          / nf;
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_RecalculateW2D(RFLOAT *devDataW,
                                      Complex *devDataC,
                                      int initWR,
                                      int dim)
{
    int j = blockIdx.x;
    int jDim = dim / 2 + 1;
    if(j >= dim / 2) j = j - dim;

    RFLOAT mode = 0.0, u, x, y;

    for (int itr = threadIdx.x; itr < jDim; itr += blockDim.x)
    {
        int quad = itr * itr + j * j;
        int index = itr + jDim * blockIdx.x;

        if (quad < initWR)
        {
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[index].real());
            y = fabsf(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            devDataW[index] /= fmaxf(mode, 1e-6);
#else
            x = fabs(devDataC[index].real());
            y = fabs(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            devDataW[index] /= fmax(mode, 1e-6);
#endif
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_RecalculateW(Complex *devDataC,
                                    RFLOAT *devDataW,
                                    int initWR,
                                    size_t shift,
                                    int dim,
                                    size_t dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;

    RFLOAT mode = 0.0, u, x, y;
    while(tid < dimSize)
    {
        i = (tid + shift) % (dim / 2 + 1);
        j = ((tid + shift) / (dim / 2 + 1)) % dim;
        k = ((tid + shift) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if (quad < initWR)
        {
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[tid].real());
            y = fabsf(devDataC[tid].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            devDataW[tid + shift] /= fmaxf(mode, 1e-6);
#else
            x = fabs(devDataC[tid].real());
            y = fabs(devDataC[tid].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            devDataW[tid + shift] /= fmax(mode, 1e-6);
#endif
        }
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_RecalculateW(RFLOAT *devDataW,
                                    Complex *devDataC,
                                    int initWR,
                                    int dim,
                                    size_t dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k;

    RFLOAT mode = 0.0, u, x, y;
    while(tid < dimSize)
    {
        i = (tid) % (dim / 2 + 1);
        j = ((tid) / (dim / 2 + 1)) % dim;
        k = ((tid) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if (quad < initWR)
        {
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[tid].real());
            y = fabsf(devDataC[tid].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            devDataW[tid] /= fmaxf(mode, 1e-6);
#else
            x = fabs(devDataC[tid].real());
            y = fabs(devDataC[tid].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            devDataW[tid] /= fmax(mode, 1e-6);
#endif
        }
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCAVG2D(RFLOAT *diff,
                                   int *counter,
                                   Complex *devDataC,
                                   int r,
                                   int dim)
{
    extern __shared__ RFLOAT sumAvg[];

    RFLOAT *sumDiff = sumAvg;
    int *sumCount = (int*)&sumDiff[blockDim.x];

    sumDiff[threadIdx.x] = 0;
    sumCount[threadIdx.x] = 0;

    __syncthreads();

    RFLOAT mode = 0, u, x, y;
    bool flag = true;

    int j = blockIdx.x;
    int jDim = dim / 2 + 1;
    if(j >= dim / 2) j = j - dim;

    for (int itr = threadIdx.x; itr < jDim; itr += blockDim.x)
    {
        int index = itr + blockIdx.x * jDim;
        int quad = itr * itr + j * j;

        if(quad < r)
        {
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[index].real());
            y = fabsf(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            sumDiff[threadIdx.x] += fabsf(mode - 1);
#else
            x = fabs(devDataC[index].real());
            y = fabs(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            sumDiff[threadIdx.x] += fabs(mode - 1);
#endif
            sumCount[threadIdx.x] += 1;
        }
    }

    __syncthreads();

    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0)
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                sumDiff[threadIdx.x] += sumDiff[threadIdx.x + j];
                sumCount[threadIdx.x] += sumCount[threadIdx.x + j];
            }

        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                sumDiff[threadIdx.x] += sumDiff[threadIdx.x + j];
                sumCount[threadIdx.x] += sumCount[threadIdx.x + j];
            }

        }

        __syncthreads();

        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;

        j /= 2;
    }

    if (threadIdx.x == 0)
    {

        diff[blockIdx.x] = sumDiff[0];
        counter[blockIdx.x] = sumCount[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCAVG(RFLOAT *diff,
                                 int *counter,
                                 Complex *devDataC,
                                 int shift,
                                 int r,
                                 int dim,
                                 size_t dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ RFLOAT sumAvg3D[];

    RFLOAT *sumDiff = sumAvg3D;
    int *sumCount = (int*)&sumDiff[blockDim.x];

    RFLOAT mode = 0, u, x, y;
    int i, j, k;
    bool flag = true;

    sumDiff[threadIdx.x] = 0;
    sumCount[threadIdx.x] = 0;

    __syncthreads();

    while(tid < dimSize)
    {
        i = (tid + shift) % (dim / 2 + 1);
        j = ((tid + shift) / (dim / 2 + 1)) % dim;
        k = ((tid + shift) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if(quad < r)
        {
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[tid].real());
            y = fabsf(devDataC[tid].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            sumDiff[threadIdx.x] += fabsf(mode - 1);
#else
            x = fabs(devDataC[tid].real());
            y = fabs(devDataC[tid].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            sumDiff[threadIdx.x] += fabs(mode - 1);
#endif
            sumCount[threadIdx.x] += 1;
        }
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

    if (blockDim.x % 2 == 0)
    {
        i = blockDim.x / 2;
        flag = true;
    }
    else
    {
        i = blockDim.x / 2 + 1;
        flag = false;
    }
    while (i != 0)
    {
        if (flag)
        {
            if (threadIdx.x < i)
            {
                sumDiff[threadIdx.x] += sumDiff[threadIdx.x + i];
                sumCount[threadIdx.x] += sumCount[threadIdx.x + i];
            }

        }
        else
        {
            if (threadIdx.x < i - 1)
            {
                sumDiff[threadIdx.x] += sumDiff[threadIdx.x + i];
                sumCount[threadIdx.x] += sumCount[threadIdx.x + i];
            }

        }

        __syncthreads();

        if(i % 2 != 0 && i != 1)
        {
            i++;
            flag = false;
        }
        else
            flag = true;

        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        diff[blockIdx.x] += sumDiff[0];
        counter[blockIdx.x] += sumCount[0];
    }

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCMAX2D(RFLOAT *devMax,
                                   Complex *devDataC,
                                   int r,
                                   int dim)
{
    extern __shared__ RFLOAT sMax2D[];

    sMax2D[threadIdx.x] = 0;

    __syncthreads();

    RFLOAT temp = 0.0, mode = 0.0, u, x, y;
    bool flag = true;

    int j = blockIdx.x;
    int jDim = dim / 2 + 1;
    if(j >= dim / 2) j = j - dim;

    for (int itr = threadIdx.x; itr < jDim; itr += blockDim.x)
    {
        int index = itr + blockIdx.x * jDim;
        int quad = itr * itr + j * j;

        if(quad < r)
        {
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[index].real());
            y = fabsf(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            if (fabsf(mode - 1) >= temp)
                temp = fabsf(mode - 1);
#else
            x = fabs(devDataC[index].real());
            y = fabs(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            if (fabs(mode - 1) >= temp)
                temp = fabs(mode - 1);
#endif
        }
    }
    sMax2D[threadIdx.x] = temp;

    __syncthreads();

    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }

    while (j != 0)
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                if (sMax2D[threadIdx.x] < sMax2D[threadIdx.x + j])
                {
                    sMax2D[threadIdx.x] = sMax2D[threadIdx.x + j];
                }
            }

        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                if (sMax2D[threadIdx.x] < sMax2D[threadIdx.x + j])
                {
                    sMax2D[threadIdx.x] = sMax2D[threadIdx.x + j];
                }
            }

        }

        __syncthreads();

        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        j /= 2;
    }

    if (threadIdx.x == 0)
    {
        devMax[blockIdx.x] = sMax2D[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCMAX(RFLOAT *devMax,
                                 Complex *devDataC,
                                 int shift,
                                 int r,
                                 int dim,
                                 size_t dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ RFLOAT sMax[];

    int i, j, k;
    RFLOAT temp = 0.0, mode = 0.0, u, x, y;
    bool flag = true;

    sMax[threadIdx.x] = 0;

    __syncthreads();

    while(tid < dimSize)
    {
        i = (tid + shift) % (dim / 2 + 1);
        j = ((tid + shift) / (dim / 2 + 1)) % dim;
        k = ((tid + shift) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        int quad = i * i + j * j + k * k;

        if(quad < r)
        {
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[tid].real());
            y = fabsf(devDataC[tid].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            if (fabsf(mode - 1) >= temp)
                temp = fabsf(mode - 1);
#else
            x = fabs(devDataC[tid].real());
            y = fabs(devDataC[tid].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            if (fabs(mode - 1) >= temp)
                temp = fabs(mode - 1);
#endif
        }
        tid += blockDim.x * gridDim.x;
    }

    sMax[threadIdx.x] = temp;

    __syncthreads();

    if (blockDim.x % 2 == 0)
    {
        i = blockDim.x / 2;
        flag = true;
    }
    else
    {
        i = blockDim.x / 2 + 1;
        flag = false;
    }

    while (i != 0)
    {
        if (flag)
        {
            if (threadIdx.x < i)
            {
                if (sMax[threadIdx.x] < sMax[threadIdx.x + i])
                {
                    sMax[threadIdx.x] = sMax[threadIdx.x + i];
                }
            }

        }
        else
        {
            if (threadIdx.x < i - 1)
            {
                if (sMax[threadIdx.x] < sMax[threadIdx.x + i])
                {
                    sMax[threadIdx.x] = sMax[threadIdx.x + i];
                }
            }

        }

        __syncthreads();

        if(i % 2 != 0 && i != 1)
        {
            i++;
            flag = false;
        }
        else
            flag = true;
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        if (devMax[blockIdx.x] < sMax[0])
            devMax[blockIdx.x] = sMax[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_mergeCAvg(RFLOAT *devDiff,
                                 int *devCount,
                                 int totalNum)
{
    extern __shared__ RFLOAT mC[];
    RFLOAT *mCAvg = mC;
    int *mCCount = (int*)&mCAvg[blockDim.x];

    mCAvg[threadIdx.x] = 0.0;
    mCCount[threadIdx.x] = 0;
    for (int itr = threadIdx.x; itr < totalNum; itr += blockDim.x)
    {
        mCAvg[threadIdx.x] += devDiff[itr];
        mCCount[threadIdx.x] += devCount[itr];
    }
    
    __syncthreads();
    
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }

    while (j != 0)
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                mCAvg[threadIdx.x] += mCAvg[threadIdx.x + j];
                mCCount[threadIdx.x] += mCCount[threadIdx.x + j];
            }
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                mCAvg[threadIdx.x] += mCAvg[threadIdx.x + j];
                mCCount[threadIdx.x] += mCCount[threadIdx.x + j];
            }
        }

        __syncthreads();

        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        j /= 2;
    }

    if (threadIdx.x == 0)
    {
        devDiff[0] = mCAvg[0];
        devCount[0] = mCCount[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_mergeCMax(RFLOAT *devMax,
                                 int totalNum)
{
    extern __shared__ RFLOAT mC[];

    mC[threadIdx.x] = 0.0;
    for (int itr = threadIdx.x; itr < totalNum; itr += blockDim.x)
    {
        if (mC[threadIdx.x] < devMax[itr])
            mC[threadIdx.x] = devMax[itr];
    }
    
    __syncthreads();
    
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }

    while (j != 0)
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                if (mC[threadIdx.x] < mC[threadIdx.x + j])
                    mC[threadIdx.x] = mC[threadIdx.x + j];
            }
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                if (mC[threadIdx.x] < mC[threadIdx.x + j])
                    mC[threadIdx.x] = mC[threadIdx.x + j];
            }
        }

        __syncthreads();

        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        j /= 2;
    }

    if (threadIdx.x == 0)
    {
        devMax[0] = mC[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_NormalizeFW2D(Complex *devDst,
                                     Complex *devDataF,
                                     RFLOAT *devDataW,
                                     const int r,
                                     const int pdim,
                                     const int fdim)
{
    int pj;
    int dIdx;

    int j = blockIdx.x;
    int jDim = fdim / 2 + 1;

    if(j >= fdim / 2)
    {
        j = j - fdim;
        pj = j + pdim;
    }
    else
        pj = j;

    for (int itr = threadIdx.x; itr < jDim; itr += blockDim.x)
    {
        dIdx = itr + pj * (pdim / 2 + 1);

        int index = itr + jDim * blockIdx.x;
        int quad = itr * itr + j * j;

        if (quad < r)
        {
            devDst[dIdx] = devDataF[index]
                         * devDataW[index];
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_NormalizeFW(Complex *devDst,
                                   Complex *devDataF,
                                   RFLOAT *devDataW,
                                   const size_t length,
                                   const size_t shift,
                                   const int r,
                                   const int pdim,
                                   const int fdim)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k, pj, pk, quad;
    int dIdx;

    while(tid < length)
    {
        i = (tid + shift) % (fdim / 2 + 1);
        j = ((tid + shift) / (fdim / 2 + 1)) % fdim;
        k = ((tid + shift) / (fdim / 2 + 1)) / fdim;

        if(j >= fdim / 2)
        {
            j = j - fdim;
            pj = j + pdim;
        }
        else
            pj = j;

        if(k >= fdim / 2)
        {
            k = k - fdim;
            pk = k + pdim;
        }
        else
            pk = k;

        dIdx = i + pj * (pdim / 2 + 1)
                 + pk * (pdim / 2 + 1) * pdim;

        quad = i * i + j * j + k * k;

        if (quad < r)
        {
            devDst[dIdx] = devDataF[tid]
                         * devDataW[tid];
        }

        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
__global__ void kernel_NormalizeP2D(RFLOAT *devDstR,
                                    int dimSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    while(index < dimSize)
    {
        devDstR[index] /= dimSize;
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
__global__ void kernel_NormalizeP(RFLOAT *devDstR,
                                  size_t length,
                                  size_t shift,
                                  size_t dimSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    while (index < length)
    {
        devDstR[index + shift] /= dimSize;
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_LowpassF(Complex *devDataF,
                                RFLOAT thres,
                                RFLOAT ew,
                                const int num,
                                const int dim,
                                const int dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.lowpassF(devDataF,
                         thres,
                         ew,
                         num,
                         dim,
                         dimSize);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF2D(RFLOAT *devDstI,
                                  RFLOAT *devMkb,
                                  RFLOAT nf,
                                  const int dim)
{
    int j = blockIdx.x, i, index, mkbIndex;
    if(j >= dim / 2) j = dim - j;

    for (int itr = threadIdx.x; itr < dim; itr += blockDim.x)
    {
        i = itr;
        if(i >= dim / 2) i = dim - i;

        index = itr + dim * blockIdx.x;
        mkbIndex = j * (dim / 2 + 1) + i;

        devDstI[index] = devDstI[index]
                       / devMkb[mkbIndex]
                       * nf;
#ifdef RECONSTRUCTOR_REMOVE_NEG
            if (devDstI[index] < 0)
                devDstI[index] = 0;
#endif
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF(RFLOAT *devDst,
                                RFLOAT *devMkb,
                                RFLOAT nf,
                                const int dim,
                                const size_t dimSize,
                                const size_t shift)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k, mkbIndex;

    while(tid < dimSize)
    {
        i = (tid + shift) % dim;
        j = ((tid + shift) / dim) % dim;
        k = ((tid + shift) / dim) / dim;

        if(i >= dim / 2) i = dim - i;
        if(j >= dim / 2) j = dim - j;
        if(k >= dim / 2) k = dim - k;

        mkbIndex = k * (dim / 2 + 1)
                     * (dim / 2 + 1)
                 + j * (dim / 2 + 1)
                 + i;

        devDst[tid] = devDst[tid]
                    / devMkb[mkbIndex]
                    * nf;
#ifdef RECONSTRUCTOR_REMOVE_NEG
        if (devDst[tid] < 0)
            devDst[tid] = 0;
#endif
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF2D(RFLOAT *devDstI,
                                  RFLOAT *devTik,
                                  const int dim)
{
    int j = blockIdx.x, i, index, mkbIndex;
    if(j >= dim / 2) j = dim - j;

    for (int itr = threadIdx.x; itr < dim; itr += blockDim.x)
    {
        i = itr;
        if(i >= dim / 2) i = dim - i;

        index = itr + dim * blockIdx.x;
        mkbIndex = j * (dim / 2 + 1) + i;

        devDstI[index] = devDstI[index] / devTik[mkbIndex];
#ifdef RECONSTRUCTOR_REMOVE_NEG
            if (devDstI[index] < 0)
                devDstI[index] = 0;
#endif
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF(RFLOAT *devDst,
                                RFLOAT *devTik,
                                const int dim,
                                const size_t dimSize,
                                const size_t shift)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j, k, mkbIndex;

    while(tid < dimSize)
    {
        i = (tid + shift) % dim;
        j = ((tid + shift) / dim) % dim;
        k = ((tid + shift) / dim) / dim;

        if(i >= dim / 2) i = dim - i;
        if(j >= dim / 2) j = dim - j;
        if(k >= dim / 2) k = dim - k;

        mkbIndex = k * (dim / 2 + 1)
                     * (dim / 2 + 1)
                 + j * (dim / 2 + 1)
                 + i;

        devDst[tid] /= devTik[mkbIndex];

#ifdef RECONSTRUCTOR_REMOVE_NEG
        if (devDst[tid] < 0)
            devDst[tid] = 0;
#endif

        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_TranslateI2D(Complex* devSrc,
                                    RFLOAT ox,
                                    RFLOAT oy,
                                    int r,
                                    int dim,
                                    int dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j;
    RFLOAT rCol;
    RFLOAT rRow;
    RFLOAT phase;
    for (int itr = tid; itr < dimSize; itr += gridDim.x * blockDim.x)
    {
        i = itr % (dim / 2 + 1);
        j = itr / (dim / 2 + 1);
        if(j >= dim / 2) j = j - dim;

        rCol = ox / dim;
        rRow = oy / dim;

        Complex imgTemp(0.0, 0.0);
        int quad = i * i + j * j;
        if (quad < r * r)
        {
            phase = PI_2 * (i * rCol + j * rRow);
#ifdef SINGLE_PRECISION
            imgTemp.set(cosf(-phase), sinf(-phase));
#else
            imgTemp.set(cos(-phase), sin(-phase));
#endif
            devSrc[itr] *= imgTemp;
        }
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_TranslateI(Complex* devRef,
                                  RFLOAT ox,
                                  RFLOAT oy,
                                  RFLOAT oz,
                                  int r,
                                  size_t shift,
                                  int dim,
                                  size_t batch)
{
    int i, j, k, quad;
    RFLOAT phase;
    Complex imgTemp(0.0, 0.0);

    RFLOAT rCol = ox / dim;
    RFLOAT rRow = oy / dim;
    RFLOAT rSlc = oz / dim;
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    while(index < batch)
    {
        i = (index + shift) % (dim / 2 + 1);
        j = ((index + shift) / (dim / 2 + 1)) % dim;
        k = ((index + shift) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        quad = i * i + j * j + k * k;
        if (quad < r * r)
        {
            phase = PI_2 * (i * rCol + j * rRow + k * rSlc);
#ifdef SINGLE_PRECISION
            imgTemp.set(cosf(-phase), sinf(-phase));
#else
            imgTemp.set(cos(-phase), sin(-phase));
#endif
            devRef[index] *= imgTemp;
        }

        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_SoftMask(RFLOAT *devMask,
                                RFLOAT r,
                                RFLOAT ew,
                                int dim,
                                size_t imgSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j;
    RFLOAT u;

    while(index < imgSize)
    {
        i = index % dim;
        j = index / dim;
        if(i >= dim / 2) i = i - dim;
        if(j >= dim / 2) j = j - dim;

#ifdef SINGLE_PRECISION
        u = sqrtf(i * i + j * j);
#else
        u = sqrt(i * i + j * j);
#endif
        if (u > r + ew)
        {
            devMask[index] = 0;
        }
        else
        {
            if (u >= r)
            {
#ifdef SINGLE_PRECISION
               devMask[index] = 0.5 + 0.5 * cosf((u - r) / ew * PI);
#else
               devMask[index] = 0.5 + 0.5 * cos((u - r) / ew * PI);
#endif
            }
            else
               devMask[index] = 1;
        }

        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_MulMask(RFLOAT *dev_image,
                               RFLOAT *devMask,
                               int imgIdx,
                               int dim,
                               size_t imgSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    size_t shift = index + imgIdx * imgSize;

    while(index < imgSize)
    {
        dev_image[shift] = dev_image[shift]
                         / imgSize
                         * devMask[index];
        index += blockDim.x * gridDim.x;
        shift += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_CTF(Complex *devCtf,
                           CTFAttr *ctfData,
                           RFLOAT *dpara,
                           RFLOAT pixelSize,
                           int nRow,
                           int nCol,
                           size_t imgSize)
{
    size_t shift = blockIdx.x * imgSize;
    int i, j;
    RFLOAT lambda, w1, w2, K1, K2, u, angle, defocus, ki;

    Complex valueCtf(0.0, 0.0);

    for (int index = threadIdx.x; index < imgSize; index += blockDim.x) 
    {
        i = index % (nCol / 2 + 1);
        j = index / (nCol / 2 + 1);

        if (j >= nRow / 2) j = j - nRow;

#ifdef SINGLE_PRECISION
        lambda = 12.2643247 
               / sqrt(ctfData[blockIdx.x].voltage 
                      * (1 + ctfData[blockIdx.x].voltage * 0.978466e-6));

        w1 = sqrtf(1 - ctfData[blockIdx.x].amplitudeContrast 
                     * ctfData[blockIdx.x].amplitudeContrast);
        w2 = ctfData[blockIdx.x].amplitudeContrast;

        K1 = PI * lambda;
        K2 = DIVPI2 * ctfData[blockIdx.x].Cs 
           * lambda * lambda * lambda;

        u = sqrtf((i / (pixelSize * nCol)) 
                  * (i / (pixelSize * nCol)) 
                  + (j / (pixelSize * nRow)) 
                  * (j / (pixelSize * nRow)));

        angle = atan2f(j, i) - ctfData[blockIdx.x].defocusTheta;
        defocus = -(ctfData[blockIdx.x].defocusU 
                    + ctfData[blockIdx.x].defocusV 
                    + (ctfData[blockIdx.x].defocusU - ctfData[blockIdx.x].defocusV) 
                    * cosf(2 * angle)) * dpara[blockIdx.x] 
                / 2;
        ki = K1 * defocus * u * u 
           + K2 * u * u * u * u 
           - ctfData[blockIdx.x].phaseShift;

        valueCtf.set(-w1 * sinf(ki) + w2 * cosf(ki), 0.0);
        //if (blockIdx.x == 2 && index == 6398)
        //    printf("i:%d, j:%d, w1:%lf, w2:%lf, ki:%lf, res:%lf\n", i, j, w1, w2, ki, valueCtf.real()); 
#else
        lambda = 12.2643247 
               / sqrt(ctfData[blockIdx.x].voltage 
                      * (1 + ctfData[blockIdx.x].voltage * 0.978466e-6));

        w1 = sqrt(1 - ctfData[blockIdx.x].amplitudeContrast 
                    * ctfData[blockIdx.x].amplitudeContrast);
        w2 = ctfData[blockIdx.x].amplitudeContrast;

        K1 = PI * lambda;
        K2 = DIVPI2 * ctfData[blockIdx.x].Cs
           * lambda * lambda * lambda;

        u = sqrt((i / (pixelSize * nCol)) 
                 * (i / (pixelSize * nCol)) 
                 + (j / (pixelSize * nRow)) 
                 * (j / (pixelSize * nRow)));

        angle = atan2(j, i) - ctfData[blockIdx.x].defocusTheta;
        defocus = -(ctfData[blockIdx.x].defocusU 
                    + ctfData[blockIdx.x].defocusV 
                    + (ctfData[blockIdx.x].defocusU - ctfData[blockIdx.x].defocusV) 
                    * cos(2 * angle)) * dpara[blockIdx.x]
                / 2;
        ki = K1 * defocus * u * u 
           + K2 * u * u * u * u 
           - ctfData[blockIdx.x].phaseShift;

        valueCtf.set(-w1 * sin(ki) + w2 * cos(ki), 0.0);
#endif
        devCtf[shift + index] = valueCtf;
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Background(RFLOAT *devDst,
                                  RFLOAT *devSumG,
                                  RFLOAT *devSumWG,
                                  const int dim,
                                  RFLOAT r,
                                  RFLOAT edgeWidth,
                                  const int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ RFLOAT sBack[];

    RFLOAT *sumWeight = sBack;
    RFLOAT *sumS = (RFLOAT*)&sumWeight[dim];

    Constructor constructor;
    constructor.init(tid);

    constructor.background(devDst,
                           sumWeight,
                           sumS,
                           devSumG,
                           devSumWG,
                           r,
                           edgeWidth,
                           dim,
                           dimSize,
                           threadIdx.x,
                           blockIdx.x);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateBg(RFLOAT *devSumG,
                                   RFLOAT *devSumWG,
                                   RFLOAT *bg,
                                   int dim)
{
    int tid = threadIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.calculateBg(devSumG,
                            devSumWG,
                            bg,
                            dim);

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_SoftMaskD(RFLOAT *devDst,
                                 RFLOAT *bg,
                                 RFLOAT r,
                                 RFLOAT edgeWidth,
                                 const int dim,
                                 const int dimSize,
                                 const int shift)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.softMaskD(devDst,
                          bg,
                          r,
                          edgeWidth,
                          dim,
                          dimSize,
                          shift);
}

///////////////////////////////////////////////////////////////

} // end namespace cuthunder

///////////////////////////////////////////////////////////////
