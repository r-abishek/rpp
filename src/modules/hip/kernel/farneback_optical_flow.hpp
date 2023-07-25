#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "copy.hpp"
#include "resize.hpp"
#include "resize_scale_intensity.hpp"
#include "box_filter_unsaturated.hpp"
#include "gaussian_filter_unsaturated.hpp"

#include <string>
#include <iomanip>
#include <fstream>
using namespace std;

#define FARNEBACK_FRAME_WIDTH 960                       // Farneback algorithm frame width
#define FARNEBACK_FRAME_HEIGHT 540                      // Farneback algorithm frame height
#define FARNEBACK_OUTPUT_FRAME_SIZE 518400              // 960 * 540
#define FARNEBACK_FRAME_MIN_SIZE 32                     // set minimum frame size
#define BORDER_SIZE 5                                   // set border size

static const float borderVals[BORDER_SIZE + 1] = {0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f, 1.f};

template <Rpp32s polyExpNbhoodSize>
__global__ void farneback_polynomial_expansion_tensor(float *srcPtr,
                                                      int2 srcStridesNH,
                                                      float *dstPtr,
                                                      int3 dstStridesNCH,
                                                      float *g,
                                                      float *xg,
                                                      float *xxg,
                                                      double4 invG11033355_f4,
                                                      int tileSizeX,
                                                      RpptROIPtr roiTensorPtrSrc)
{
    const int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const int x = hipBlockIdx_x * (hipBlockDim_x - 2 * polyExpNbhoodSize) + hipThreadIdx_x - polyExpNbhoodSize;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    RpptROI roiSrc = roiTensorPtrSrc[id_z];

    if (y < roiSrc.xywhROI.roiHeight)
    {
        extern __shared__ float smem[];
        volatile float *row = smem + hipThreadIdx_x;
        int xWarped = fminf(fmaxf(x, 0), roiSrc.xywhROI.roiWidth - 1);

        row[0] = srcPtr[(y * srcStridesNH.y) + xWarped] * g[0];
        row[hipBlockDim_x] = 0.f;
        row[2 * hipBlockDim_x] = 0.f;

        for (int k = 1; k <= polyExpNbhoodSize; ++k)
        {
            float t0 = srcPtr[((int)fmaxf(y - k, 0) * srcStridesNH.y) + xWarped];
            float t1 = srcPtr[((int)fminf(y + k, roiSrc.xywhROI.roiHeight - 1) * srcStridesNH.y) + xWarped];

            row[0] += g[k] * (t0 + t1);
            row[hipBlockDim_x] += xg[k] * (t1 - t0);
            row[2 * hipBlockDim_x] += xxg[k] * (t0 + t1);
        }

        __syncthreads();

        if ((hipThreadIdx_x >= polyExpNbhoodSize) && (hipThreadIdx_x + polyExpNbhoodSize < hipBlockDim_x) && (x < roiSrc.xywhROI.roiWidth))
        {
            float b1 = g[0] * row[0];
            float b3 = g[0] * row[hipBlockDim_x];
            float b5 = g[0] * row[2 * hipBlockDim_x];
            float b2 = 0, b4 = 0, b6 = 0;

            for (int k = 1; k <= polyExpNbhoodSize; ++k)
            {
                b1 += (row[k] + row[-k]) * g[k];
                b4 += (row[k] + row[-k]) * xxg[k];
                b2 += (row[k] - row[-k]) * xg[k];
                b3 += (row[k + hipBlockDim_x] + row[-k + hipBlockDim_x]) * g[k];
                b6 += (row[k + hipBlockDim_x] - row[-k + hipBlockDim_x]) * xg[k];
                b5 += (row[k + 2 * hipBlockDim_x] + row[-k + 2*hipBlockDim_x]) * g[k];
            }

            dstPtr[(y * dstStridesNCH.z) + xWarped] = b3*invG11033355_f4.x;
            dstPtr[dstStridesNCH.x + (y * dstStridesNCH.z) + xWarped] = b2*invG11033355_f4.x;
            dstPtr[(2 * dstStridesNCH.x) + (y * dstStridesNCH.z) + xWarped] = b1*invG11033355_f4.y + b5*invG11033355_f4.z;
            dstPtr[(3 * dstStridesNCH.x) + (y * dstStridesNCH.z) + xWarped] = b1*invG11033355_f4.y + b4*invG11033355_f4.z;
            dstPtr[(4 * dstStridesNCH.x) + (y * dstStridesNCH.z) + xWarped] = b6*invG11033355_f4.w;
        }
    }
}

__global__ void farneback_matrices_update_tensor(float *mVecCurrCompX,
                                                 float *mVecCurrCompY,
                                                 float *polyResPrev,
                                                 float *polyResCurr,
                                                 float *polyMatrices,
                                                 uint3 stridesNCH,
                                                 float *border,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    RpptROI roiSrc = roiTensorPtrSrc[id_z];

    int mVecCurrCompXIdx = id_z * stridesNCH.x;
    int mVecCurrCompYIdx = id_z * stridesNCH.x;

    if (id_y < roiSrc.xywhROI.roiHeight && id_x < roiSrc.xywhROI.roiWidth)
    {
        float dx = mVecCurrCompX[mVecCurrCompXIdx + (id_y * stridesNCH.z) + id_x];
        float dy = mVecCurrCompY[mVecCurrCompYIdx + (id_y * stridesNCH.z) + id_x];
        float fx = id_x + dx;
        float fy = id_y + dy;

        int x1 = floorf(fx);
        int y1 = floorf(fy);
        fx -= x1;
        fy -= y1;

        float r2, r3, r4, r5, r6;

        if ((x1 >= 0) && (y1 >= 0) && (x1 < roiSrc.xywhROI.roiWidth - 1) && (y1 < roiSrc.xywhROI.roiHeight - 1))
        {
            float a00 = (1.0f - fx) * (1.0f - fy);
            float a01 = fx * (1.0f - fy);
            float a10 = (1.0f - fx) * fy;
            float a11 = fx * fy;

            r2 = a00 * polyResCurr[y1 * stridesNCH.z + x1] +
                    a01 * polyResCurr[y1 * stridesNCH.z + x1 + 1] +
                    a10 * polyResCurr[(y1 + 1) * stridesNCH.z + x1] +
                    a11 * polyResCurr[(y1 + 1) * stridesNCH.z + x1 + 1];

            r3 = a00 * polyResCurr[stridesNCH.x + (y1 * stridesNCH.z) + x1] +
                    a01 * polyResCurr[stridesNCH.x + (y1 * stridesNCH.z) + x1 + 1] +
                    a10 * polyResCurr[stridesNCH.x + ((y1 + 1) * stridesNCH.z) + x1] +
                    a11 * polyResCurr[stridesNCH.x + ((y1 + 1) * stridesNCH.z) + x1 + 1];

            r4 = a00 * polyResCurr[(2 * stridesNCH.x) + (y1 * stridesNCH.z) + x1] +
                    a01 * polyResCurr[(2 * stridesNCH.x) + (y1 * stridesNCH.z) + x1 + 1] +
                    a10 * polyResCurr[(2 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1] +
                    a11 * polyResCurr[(2 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1 + 1];

            r5 = a00 * polyResCurr[(3 * stridesNCH.x) + (y1 * stridesNCH.z) + x1] +
                    a01 * polyResCurr[(3 * stridesNCH.x) + (y1 * stridesNCH.z) + x1 + 1] +
                    a10 * polyResCurr[(3 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1] +
                    a11 * polyResCurr[(3 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1 + 1];

            r6 = a00 * polyResCurr[(4 * stridesNCH.x) + (y1 * stridesNCH.z) + x1] +
                    a01 * polyResCurr[(4 * stridesNCH.x) + (y1 * stridesNCH.z) + x1 + 1] +
                    a10 * polyResCurr[(4 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1] +
                    a11 * polyResCurr[(4 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1 + 1];

            r4 = (polyResPrev[(2 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] + r4) * 0.5f;
            r5 = (polyResPrev[(3 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] + r5) * 0.5f;
            r6 = (polyResPrev[(4 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] + r6) * 0.25f;
        }
        else
        {
            r2 = r3 = 0.f;

            r4 = (polyResPrev[(2 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x]);
            r5 = (polyResPrev[(3 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x]);
            r6 = (polyResPrev[(4 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x]) * 0.5f;
        }

        r2 = (polyResPrev[(id_y * stridesNCH.z) + id_x] - r2) * 0.5f;
        r3 = (polyResPrev[stridesNCH.x + (id_y * stridesNCH.z) + id_x] - r3) * 0.5f;

        r2 += r4*dy + r6*dx;
        r3 += r6*dy + r5*dx;

        float scale =
                border[::min(id_x, BORDER_SIZE)] *
                border[::min(id_y, BORDER_SIZE)] *
                border[::min(roiSrc.xywhROI.roiWidth - id_x - 1, BORDER_SIZE)] *
                border[::min(roiSrc.xywhROI.roiHeight - id_y - 1, BORDER_SIZE)];

        r2 *= scale; r3 *= scale; r4 *= scale;
        r5 *= scale; r6 *= scale;

        polyMatrices[(id_y * stridesNCH.z) + id_x] = r4*r4 + r6*r6;
        polyMatrices[stridesNCH.x + (id_y * stridesNCH.z) + id_x] = (r4 + r5)*r6;
        polyMatrices[(2 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] = r5*r5 + r6*r6;
        polyMatrices[(3 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] = r4*r2 + r6*r3;
        polyMatrices[(4 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] = r6*r2 + r5*r3;
    }
}

__global__ void farneback_motion_vectors_update_tensor(float *polyMatricesBlurred,
                                                       float *mVecCurrCompX,
                                                       float *mVecCurrCompY,
                                                       uint2 stridesNH,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    const int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const int z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    RpptROI roiSrc = roiTensorPtrSrc[z];

    if ((y < roiSrc.xywhROI.roiHeight) && (x < roiSrc.xywhROI.roiWidth))
    {
        float g11 = polyMatricesBlurred[y * stridesNH.y + x];
        float g12 = polyMatricesBlurred[stridesNH.x + (y * stridesNH.y) + x];
        float g22 = polyMatricesBlurred[(2 * stridesNH.x) + (y * stridesNH.y) + x];
        float h1 = polyMatricesBlurred[(3 * stridesNH.x) + (y * stridesNH.y) + x];
        float h2 = polyMatricesBlurred[(4 * stridesNH.x) + (y * stridesNH.y) + x];

        float detInv = 1.0f / (g11*g22 - g12*g12 + 0.001f);

        mVecCurrCompX[y * stridesNH.y + x] = (g11*h2 - g12*h1) * detInv;
        mVecCurrCompY[y * stridesNH.y + x] = (g22*h1 - g12*h2) * detInv;
    }
}

RppStatus hip_exec_farneback_polynomial_expansion_tensor(Rpp32f *pyramidLevelF32,
                                                         RpptDescPtr srcCompDescPtr,
                                                         Rpp32f *polyRes,
                                                         RpptDescPtr mVecCompDescPtr,
                                                         Rpp32s polyExpNbhoodSize,
                                                         Rpp32f *g,
                                                         Rpp32f *xg,
                                                         Rpp32f *xxg,
                                                         double4 invG11033355_f4,
                                                         RpptROIPtr roiTensorPtrSrc,
                                                         RpptRoiType roiType,
                                                         rpp::Handle& handle)
{
    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = roiTensorPtrSrc[0].xywhROI.roiWidth;
    int globalThreads_y = roiTensorPtrSrc[0].xywhROI.roiHeight;
    int globalThreads_z = handle.GetBatchSize();

    int tileSizeX = localThreads_x - 2 * polyExpNbhoodSize;
    int ldsSize = 3 * localThreads_x * sizeof(float);

    if (polyExpNbhoodSize == 5)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(farneback_polynomial_expansion_tensor<5>),
                           dim3(ceil((float)globalThreads_x/tileSizeX), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           ldsSize,
                           handle.GetStream(),
                           pyramidLevelF32,
                           make_int2(srcCompDescPtr->strides.nStride, srcCompDescPtr->strides.hStride),
                           polyRes,
                           make_int3(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.cStride, mVecCompDescPtr->strides.hStride),
                           g,
                           xg,
                           xxg,
                           invG11033355_f4,
                           tileSizeX,
                           roiTensorPtrSrc);
    }
    else if (polyExpNbhoodSize == 7)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(farneback_polynomial_expansion_tensor<7>),
                           dim3(ceil((float)globalThreads_x/tileSizeX), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           ldsSize,
                           handle.GetStream(),
                           pyramidLevelF32,
                           make_int2(srcCompDescPtr->strides.nStride, srcCompDescPtr->strides.hStride),
                           polyRes,
                           make_int3(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.cStride, mVecCompDescPtr->strides.hStride),
                           g,
                           xg,
                           xxg,
                           invG11033355_f4,
                           tileSizeX,
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}

RppStatus hip_exec_farneback_matrices_update_tensor(Rpp32f *mVecCurrCompX,
                                                    Rpp32f *mVecCurrCompY,
                                                    Rpp32f *polyResPrev,
                                                    Rpp32f *polyResCurr,
                                                    Rpp32f *polyMatrices,
                                                    RpptDescPtr mVecCompDescPtr,
                                                    Rpp32f *border,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    rpp::Handle& handle)
{
    int localThreads_x = 32;
    int localThreads_y = 8;
    int localThreads_z = 1;
    int globalThreads_x = roiTensorPtrSrc[0].xywhROI.roiWidth;
    int globalThreads_y = roiTensorPtrSrc[0].xywhROI.roiHeight;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(farneback_matrices_update_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       mVecCurrCompX,
                       mVecCurrCompY,
                       polyResPrev,
                       polyResCurr,
                       polyMatrices,
                       make_uint3(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.cStride, mVecCompDescPtr->strides.hStride),
                       border,
                       roiTensorPtrSrc);

    return RPP_SUCCESS;
}

RppStatus hip_exec_farneback_motion_vectors_update_tensor(Rpp32f *polyMatricesBlurred,
                                                          Rpp32f *mVecCurrCompX,
                                                          Rpp32f *mVecCurrCompY,
                                                          RpptDescPtr mVecCompDescPtr,
                                                          RpptROIPtr roiTensorPtrSrc,
                                                          RpptRoiType roiType,
                                                          rpp::Handle& handle)
{
    int localThreads_x = 32;
    int localThreads_y = 8;
    int localThreads_z = 1;
    int globalThreads_x = roiTensorPtrSrc[0].xywhROI.roiWidth;
    int globalThreads_y = roiTensorPtrSrc[0].xywhROI.roiHeight;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(farneback_motion_vectors_update_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       polyMatricesBlurred,
                       mVecCurrCompX,
                       mVecCurrCompY,
                       make_uint2(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.hStride),
                       roiTensorPtrSrc);

    return RPP_SUCCESS;
}

RppStatus hip_exec_farneback_optical_flow_tensor(Rpp8u *src1Ptr,
                                                 Rpp8u *src2Ptr,
                                                 RpptDescPtr srcCompDescPtr,
                                                 Rpp32f *mVecCompX,
                                                 Rpp32f *mVecCompY,
                                                 RpptDescPtr mVecCompDescPtr,
                                                 Rpp32f pyramidScale,
                                                 Rpp32s numPyramidLevels,
                                                 Rpp32s windowSize,
                                                 Rpp32s numIterations,
                                                 Rpp32s polyExpNbhoodSize,
                                                 Rpp32f polyExpStdDev,
                                                 rpp::Handle& handle)
{
    // Precompute inverse pyramid scale
    Rpp32f oneOverPyramidScale = 1.0f / pyramidScale;

    // Create internal srcComp and mVecComp descriptors for batch processing
    RpptDesc srcCompBatchDesc = *srcCompDescPtr;
    RpptDescPtr srcCompBatchDescPtr = &srcCompBatchDesc;
    srcCompBatchDescPtr->n = 2;
    RpptDesc mVecCompBatchDesc = *mVecCompDescPtr;
    RpptDescPtr mVecCompBatchDescPtr = &mVecCompBatchDesc;
    mVecCompBatchDescPtr->n = 2;
    RpptDesc mVecCompBatch5Desc = *mVecCompDescPtr;
    RpptDescPtr mVecCompBatch5DescPtr = &mVecCompBatch5Desc;
    mVecCompBatch5DescPtr->n = 5;

    // Use preallocated buffers for 4 960x540 frames (for previous/current motion vectors in x/y)
    Rpp32f *preallocMem, *src1F32, *src2F32, *src1F32Blurred, *src2F32Blurred, *pyramidLevelPrevF32, *pyramidLevelCurrF32, *polyResPrev, *polyResCurr, *polyMatrices, *polyMatricesBlurred;
    Rpp32f *mVecPrevCompX, *mVecPrevCompY, *mVecCurrCompX, *mVecCurrCompY;
    preallocMem = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    hipMemset(preallocMem, 0, FARNEBACK_OUTPUT_FRAME_SIZE * 30 * sizeof(Rpp32f));
    hipDeviceSynchronize();
    src1F32 = preallocMem + 64;                                                 // previous frame (after 64 byte offset)
    src2F32 = src1F32 + FARNEBACK_OUTPUT_FRAME_SIZE;                            // current frame
    src1F32Blurred = src2F32 + FARNEBACK_OUTPUT_FRAME_SIZE;                     // blurred previous frame
    src2F32Blurred = src1F32Blurred + FARNEBACK_OUTPUT_FRAME_SIZE;              // blurred current frame
    pyramidLevelPrevF32 = src2F32Blurred + FARNEBACK_OUTPUT_FRAME_SIZE;         // pyramid level previous frame
    pyramidLevelCurrF32 = pyramidLevelPrevF32 + FARNEBACK_OUTPUT_FRAME_SIZE;    // pyramid level current frame
    polyResPrev = pyramidLevelCurrF32 + FARNEBACK_OUTPUT_FRAME_SIZE;            // 5 polynomial results for previous frame
    polyResCurr = polyResPrev + (FARNEBACK_OUTPUT_FRAME_SIZE * 5);              // 5 polynomial results for current frame
    polyMatrices = polyResCurr + (FARNEBACK_OUTPUT_FRAME_SIZE * 5);             // 5 polynomial matrices frames
    polyMatricesBlurred = polyMatrices + (FARNEBACK_OUTPUT_FRAME_SIZE * 5);     // 5 blurred polynomial matrices frames
    mVecPrevCompX = polyMatricesBlurred + (FARNEBACK_OUTPUT_FRAME_SIZE * 5);    // motion vector previous X component
    mVecPrevCompY = mVecPrevCompX + FARNEBACK_OUTPUT_FRAME_SIZE;                // motion vector previous Y component
    mVecCurrCompX = mVecPrevCompY + FARNEBACK_OUTPUT_FRAME_SIZE;                // motion vector current X component
    mVecCurrCompY = mVecCurrCompX + FARNEBACK_OUTPUT_FRAME_SIZE;                // motion vector current Y component

    // Creating U8 pointer interpretations from F32 buffers
    Rpp8u *src1U8Blurred = (Rpp8u *)src1F32Blurred;
    Rpp8u *src2U8Blurred = (Rpp8u *)src2F32Blurred;
    Rpp8u *pyramidLevelPrevU8 = (Rpp8u *)pyramidLevelPrevF32;
    Rpp8u *pyramidLevelCurrU8 = (Rpp8u *)pyramidLevelCurrF32;

    Rpp64f scale = 1.0; // change to 32f?
    Rpp32s numPyramidLevelsCropped = 0;
    for (; numPyramidLevelsCropped < numPyramidLevels; numPyramidLevelsCropped++)
    {
        scale *= pyramidScale;
        if (srcCompDescPtr->w * scale < FARNEBACK_FRAME_MIN_SIZE || srcCompDescPtr->h * scale < FARNEBACK_FRAME_MIN_SIZE)
            break;
    }

    // Commenting out temporarily to check box filter with U8
    hip_exec_copy_tensor(src1Ptr,
                         srcCompDescPtr,
                         src1F32,
                         mVecCompDescPtr,
                         handle);
    hip_exec_copy_tensor(src2Ptr,
                         srcCompDescPtr,
                         src2F32,
                         mVecCompDescPtr,
                         handle);
    
    // hip_exec_copy_tensor(src1Ptr,
    //                      srcCompBatchDescPtr,
    //                      src1F32,
    //                      mVecCompBatchDescPtr,
    //                      handle);
    hipDeviceSynchronize();

    Rpp32s bufIncrement = polyExpNbhoodSize * 2 + 1;
    std::vector<Rpp32f> xSquareBuf(bufIncrement);
    Rpp32f *xSquare = &xSquareBuf[0] + polyExpNbhoodSize;

    Rpp32f *buf;
    hipHostMalloc(&buf, (polyExpNbhoodSize * 6 + 3) * sizeof(Rpp32f));  // pinned memory sustitute for std::vector<Rpp32f> buf(polyExpNbhoodSize * 6 + 3););
    Rpp32f *g = &buf[0] + polyExpNbhoodSize;
    Rpp32f *xg = g + bufIncrement;
    Rpp32f *xxg = xg + bufIncrement;

    if (polyExpStdDev < RPP_MACHEPS)
        polyExpStdDev = polyExpNbhoodSize * 0.3f;

    Rpp32f s = 0; // changed to 32f
    Rpp32f oneOverTwoStddevSquare = 1.0f / (2.0f * polyExpStdDev * polyExpStdDev); // changed to 32f

    for (int x = -polyExpNbhoodSize; x <= polyExpNbhoodSize; x++)
    {
        xSquare[x] = x * x;     // percompute xSquare for multiple future uses
        g[x] = (Rpp32f)std::exp(-xSquare[x] * oneOverTwoStddevSquare);
        s += g[x];
    }
    s = 1.0f / s;
    for (int x = -polyExpNbhoodSize; x <= polyExpNbhoodSize; x++)
    {
        g[x] *= s;
        xg[x] = x * g[x];
        xxg[x] = xSquare[x] * g[x];
    }

    float4 gaussian00113355_f4 = (float4) 0.0f;
    for (int y = -polyExpNbhoodSize; y <= polyExpNbhoodSize; y++)
    {
        for (int x = -polyExpNbhoodSize; x <= polyExpNbhoodSize; x++)
        {
            float4 multiplier_f4 = make_float4(1, xSquare[x], xSquare[x] * xSquare[x], xSquare[x] * xSquare[y]);
            float4 gygx_f4 = (float4) (g[y] * g[x]);
            gaussian00113355_f4 += (gygx_f4 * multiplier_f4);   // float4 increment to process [   (g[y] * g[x])   |   (g[y] * g[x] * x * x)   |   (g[y] * g[x] * x * x * x * x)   |   (g[y] * g[x] * x * x * y * y)   ]
        }
    }

    Rpp32f yw = gaussian00113355_f4.y * gaussian00113355_f4.w;
    Rpp32f yz = gaussian00113355_f4.y * gaussian00113355_f4.z;
    Rpp32f xyzw = gaussian00113355_f4.x * yz * gaussian00113355_f4.w;
    Rpp32f y2 = gaussian00113355_f4.y * gaussian00113355_f4.y;
    Rpp32f y3w = y2 * yw;
    Rpp32f xyw3 = gaussian00113355_f4.x * yw * gaussian00113355_f4.w * gaussian00113355_f4.w;
    Rpp32f y2zw = yz * yw;
    Rpp32f oneOverDetG = -1.0f / ((xyzw * yw) - (xyw3 * gaussian00113355_f4.y) - (xyzw * yz) + (y3w * yz));

    // Gauss-Jordan simplified calculations for 4 out of 36 elements in 6 x 6 matrix inverse
    double4 invG11033355_f4; // initialization for 4 elements from the invG matrix - (1,1), (0,3), (3,3), (5,5)
    invG11033355_f4.x = (((y3w + y3w - xyzw) * gaussian00113355_f4.w) - xyw3 + (xyzw * gaussian00113355_f4.z) - (y2zw * gaussian00113355_f4.y)) * oneOverDetG;
    invG11033355_f4.y = ((y3w * gaussian00113355_f4.w) - (y2zw * gaussian00113355_f4.y)) * oneOverDetG;
    invG11033355_f4.z = (xyzw - y3w) * gaussian00113355_f4.y * oneOverDetG;
    invG11033355_f4.w = (((y3w + y3w - xyzw) * gaussian00113355_f4.y) + (((yz * yz) - (yw * yw)) * gaussian00113355_f4.x) - (y2 * y2 * gaussian00113355_f4.z)) * oneOverDetG;

    // Pinned memory allocations
    Rpp32f *stdDevPtrForGaussian, *border;
    hipHostMalloc(&stdDevPtrForGaussian, mVecCompDescPtr->n * sizeof(Rpp32f));
    hipHostMalloc(&border, (BORDER_SIZE + 1) * sizeof(Rpp32f));
    *(d_float6_s *)border = *(d_float6_s *)borderVals;
    RpptImagePatch *pyramidImgPatchPtr;
    hipHostMalloc(&pyramidImgPatchPtr, sizeof(RpptImagePatch));
    RpptROI *roiTensorPtrSrc, *roiTensorPtrPyramid;
    hipHostMalloc(&roiTensorPtrSrc, mVecCompDescPtr->n * 2 * sizeof(RpptROI));
    hipHostMalloc(&roiTensorPtrPyramid, mVecCompDescPtr->n * 5 * sizeof(RpptROI));
    for (int roiIdx = 0; roiIdx < 2; roiIdx++)
        roiTensorPtrSrc[roiIdx] = {0, 0, mVecCompDescPtr->w, mVecCompDescPtr->h};

    for (int k = numPyramidLevelsCropped; k >= 0; k--)
    {
        // hipStreamSynchronize(streams[0]);
        hipDeviceSynchronize();

        scale = 1.0;
        for (int i = 0; i < k; i++)
            scale *= pyramidScale;

        Rpp64f sigma = (1.0 / scale - 1) * 0.5; // could use Rpp32f
        Rpp32s smoothSize = ((Rpp32s)roundf(sigma * 5)) | 1;
        smoothSize = fminf(fmaxf(smoothSize, 3), 9); // cap needed for RPP compatibility on filter sizes 3,5,7,9

        pyramidImgPatchPtr->width = std::roundf(srcCompDescPtr->w * scale);
        pyramidImgPatchPtr->height = std::roundf(srcCompDescPtr->h * scale);

        if (k == numPyramidLevelsCropped)
        {
            hipMemset(mVecCurrCompX, 0, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f));
            hipMemset(mVecCurrCompY, 0, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f));
        }
        else
        {
            if (k == 0)
            {
                hipMemcpy(mVecCurrCompX, mVecCompX, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
                hipMemcpy(mVecCurrCompY, mVecCompY, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
            }

            // Batched fused resize_scale_intensity for previous and current frames
            RppStatus rsiReturn = hip_exec_resize_scale_intensity_tensor(mVecPrevCompX, mVecCompBatchDescPtr, mVecCurrCompX, mVecCompBatchDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, oneOverPyramidScale, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hipDeviceSynchronize();
            if (rsiReturn != RppStatus::RPP_SUCCESS)
                return RPP_ERROR;
        }

        // Gaussian filter for previous and current frames for the calculated smoothSize and sigma
        if (sigma == 0)
            sigma = 1;
        for (int batchCount = 0; batchCount < mVecCompDescPtr->n; batchCount++)
            stdDevPtrForGaussian[batchCount] = sigma;
        hip_exec_gaussian_filter_f32_tensor(src1F32, mVecCompDescPtr, src1F32Blurred, mVecCompDescPtr, smoothSize, stdDevPtrForGaussian, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hip_exec_gaussian_filter_f32_tensor(src2F32, mVecCompDescPtr, src2F32Blurred, mVecCompDescPtr, smoothSize, stdDevPtrForGaussian, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hipDeviceSynchronize();

        // Resize previous and current frames to pyramidLevel
        hip_exec_resize_tensor(src1F32Blurred, mVecCompDescPtr, pyramidLevelPrevF32, mVecCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hip_exec_resize_tensor(src2F32Blurred, mVecCompDescPtr, pyramidLevelCurrF32, mVecCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hipDeviceSynchronize();

        // Set new roi to be the pyramidLevel size
        for (int roiIdx = 0; roiIdx < 5; roiIdx++)
            roiTensorPtrPyramid[roiIdx] = {0, 0, pyramidImgPatchPtr->width, pyramidImgPatchPtr->height};
        
        // Run farneback polynomial expansion for previous and current frame to get 5 polyResPrev matrices and 5 polyResCurr matrices
        hip_exec_farneback_polynomial_expansion_tensor(pyramidLevelPrevF32, mVecCompDescPtr, polyResPrev, mVecCompDescPtr, polyExpNbhoodSize, g, xg, xxg, invG11033355_f4, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
        hip_exec_farneback_polynomial_expansion_tensor(pyramidLevelCurrF32, mVecCompDescPtr, polyResCurr, mVecCompDescPtr, polyExpNbhoodSize, g, xg, xxg, invG11033355_f4, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
        
        // Run farneback matrices update to get the 5 updated polyMatrices
        hip_exec_farneback_matrices_update_tensor(mVecCurrCompX, mVecCurrCompY, polyResPrev, polyResCurr, polyMatrices, mVecCompDescPtr, border, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);

        for (int i = 0; i < numIterations; i++)
        {
            hipDeviceSynchronize();

            // Run batched box filtering with user provided windowSize for all 5 float polyMatrices
            hip_exec_box_filter_f32_tensor(polyMatrices, mVecCompBatch5DescPtr, polyMatricesBlurred, mVecCompBatch5DescPtr, windowSize, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hipDeviceSynchronize();
            
            // Optimal combined pointer swap for 5 polyMatrices
            Rpp32f *temp;
            temp = polyMatrices;
            polyMatrices = polyMatricesBlurred;
            polyMatricesBlurred = temp;

            // Update Farneback Motion Vectors
            hip_exec_farneback_motion_vectors_update_tensor(polyMatrices, mVecCurrCompX, mVecCurrCompY, mVecCompDescPtr, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hipDeviceSynchronize();

            if (i < numIterations - 1)
            {
                // Run farneback matrices update to get 5 updated polyMatrices
                hip_exec_farneback_matrices_update_tensor(mVecCurrCompX, mVecCurrCompY, polyResPrev, polyResCurr, polyMatrices, mVecCompDescPtr, border, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
                hipDeviceSynchronize();
            }
        }

        // Optimal pointer swap for current motion vectors in X and Y
        Rpp32f *temp;
        temp = mVecPrevCompX;
        mVecPrevCompX = mVecCurrCompX;
        mVecCurrCompX = temp;
        temp = mVecPrevCompY;
        mVecPrevCompY = mVecCurrCompY;
        mVecCurrCompY = temp;
    }

    hipDeviceSynchronize();
    hipMemcpy(mVecCompX, mVecPrevCompX, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
    hipMemcpy(mVecCompY, mVecPrevCompY, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);

    hipHostFree(&buf);
    hipHostFree(&stdDevPtrForGaussian);
    hipHostFree(&border);
    hipHostFree(&pyramidImgPatchPtr);
    hipHostFree(&roiTensorPtrSrc);
    hipHostFree(&roiTensorPtrPyramid);

    return RPP_SUCCESS;
}
