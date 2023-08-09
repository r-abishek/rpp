#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void image_min_max_grid_result_tensor(float *srcPtr,
                                                 int2 blocksAndBufferSizePerImage_i2,
                                                 float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialMinMaxLDS[1024];                        // 4096 floats of src reduced to 1024 in a 512 x 1 thread block
    float2 *partialMinMaxLDS_f2 = (float2 *)partialMinMaxLDS;       // float2 pointer to beginning of buffer in LDS

    uint srcIdx = (id_z * blocksAndBufferSizePerImage_i2.x);
    float2 *srcPtr_f2 = (float2 *)srcPtr;
    float2 srcRef_f2 = srcPtr_f2[srcIdx];
    partialMinMaxLDS_f2[hipThreadIdx_x] = srcRef_f2;                // vectorized float2 initialization of LDS to srcRef_f2 using all 512 x 1 threads

    if (id_x >= blocksAndBufferSizePerImage_i2.y)
    {
        return;
    }

    int xAlignedLength = blocksAndBufferSizePerImage_i2.x & ~3;     // alignedLength for vectorized global loads
    int xDiff = blocksAndBufferSizePerImage_i2.x - xAlignedLength;  // difference between bufferLength and alignedLength
    srcIdx += (srcIdx + id_x);

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local memory
    if (id_x + 8 > blocksAndBufferSizePerImage_i2.y)
        for(int i = xDiff; i < 4; i++)
            src_f8.f2[i] = srcRef_f2;                               // local memory reset of invalid values (from the vectorized global load) to srcRef_f2
    partialMinMaxLDS_f2[hipThreadIdx_x].x = fminf(src_f8.f1[0], fminf(src_f8.f1[2], fminf(src_f8.f1[4], src_f8.f1[6])));    // perform small work of min/max a d_float8 vector and store in LDS
    partialMinMaxLDS_f2[hipThreadIdx_x].y = fmaxf(src_f8.f1[1], fmaxf(src_f8.f1[3], fmaxf(src_f8.f1[5], src_f8.f1[7])));    // perform small work of min/max a d_float8 vector and store in LDS
    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 1024 floats on 512 threads per block in x dimension
    for (int threadMax = 256; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            rpp_hip_math_minmax2(partialMinMaxLDS_f2[hipThreadIdx_x], partialMinMaxLDS_f2[hipThreadIdx_x + threadMax], partialMinMaxLDS_f2[hipThreadIdx_x]);
        __syncthreads();
    }

    // Final vectorized float2 store of min and max to dst
    if (hipThreadIdx_x == 0)
    {
        float2 *dstPtr_f2 = (float2 *)dstPtr;
        dstPtr_f2[hipBlockIdx_z] = partialMinMaxLDS_f2[0];
    }
}

template <typename T, typename U>
__global__ void image_min_max_pln1_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          U *imageMinMaxArr,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialMinMaxLDS[32][32];                                  // 32 rows of src, 256 reduced cols of src in a 16 x 16 thread block (each producing float2 outputs for min and max)
    float *partialMinMaxLDSRowPtr = &partialMinMaxLDS[hipThreadIdx_y][0];       // float pointer to beginning of each row in LDS
    float2 *partialMinMaxLDSRowPtr_f2 = (float2 *)partialMinMaxLDSRowPtr;       // float2 pointer to beginning of each row in LDS

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRef = srcPtr[srcIdx];
    partialMinMaxLDSRowPtr_f2[hipThreadIdx_x] = (float2) srcRef;                // vectorized float2 initialization of LDS to srcRef using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        float2 *imageMinMaxArr_f2 = (float2 *)imageMinMaxArr;
        imageMinMaxArr_f2[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = (float2) srcRef;
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;        // difference between roiWidth and alignedLength
    srcIdx += (((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x));

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);               // load 8 pixels to local memory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = srcRef;                                              // local memory reset of invalid values (from the vectorized global load) to srcRef
    rpp_hip_math_minmax8(src_f8, partialMinMaxLDSRowPtr_f2[hipThreadIdx_x]);    // perform small work of reducing vector d_float8 to float2 min-max and store in LDS
    __syncthreads();                                                            // syncthreads after LDS load

    // Vectorized reduction of 16 float2s (32 floats) on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            rpp_hip_math_minmax2(partialMinMaxLDSRowPtr_f2[hipThreadIdx_x], partialMinMaxLDSRowPtr_f2[hipThreadIdx_x + threadMax], partialMinMaxLDSRowPtr_f2[hipThreadIdx_x]);
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Vectorized reduction of 32 float2s (64 floats) on 32 threads per block in y dimension
        for (int threadMax = 16, increment = 256; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                rpp_hip_math_minmax2(partialMinMaxLDSRowPtr_f2[0], partialMinMaxLDSRowPtr_f2[increment], partialMinMaxLDSRowPtr_f2[0]);
            __syncthreads();
        }

        // Final vectorized float2 store of min and max to dst
        if (hipThreadIdx_y == 0)
        {
            float2 *imageMinMaxArr_f2 = (float2 *)imageMinMaxArr;
            imageMinMaxArr_f2[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialMinMaxLDSRowPtr_f2[0];
        }
    }
}

template <typename T, typename U>
RppStatus hip_exec_image_min_max_tensor(T *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        U *imageMinMaxArr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y * 2;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int gridDim_x = (int) ceil((float)globalThreads_x/localThreads_x);
    int gridDim_y = (int) ceil((float)globalThreads_y/localThreads_y);
    int gridDim_z = (int) ceil((float)globalThreads_z/localThreads_z);
    int numOfBlocksPerImage = gridDim_x * gridDim_y;

    float *imagePartialMinMaxArr;
    imagePartialMinMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        int xBufferSizePerImage = numOfBlocksPerImage * 2;
        hipMemset(imagePartialMinMaxArr, 0, xBufferSizePerImage * gridDim_z * sizeof(float));
        hipLaunchKernelGGL(image_min_max_pln1_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialMinMaxArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(image_min_max_grid_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(512, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialMinMaxArr,
                           make_int2(numOfBlocksPerImage, xBufferSizePerImage),
                           imageMinMaxArr);
    }

    return RPP_SUCCESS;
}