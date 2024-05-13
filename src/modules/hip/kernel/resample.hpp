#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - resample kernel --------------------
// Single channel resample support
__global__ void resample_1channel_hip_tensor(float *srcPtr,
                                             float *dstPtr,
                                             int srcLength,
                                             int outEnd,
                                             double scale,
                                             RpptResamplingWindow &window,
                                             int block)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    extern __shared__ float windowCoeffs_smem[];
    float *windowCoeffsSh = windowCoeffs_smem;
    for (int k = hipThreadIdx_x; k < window.lookupSize; k += block)
        windowCoeffsSh[k] = window.lookup[k];
    __syncthreads();
    window.lookup = windowCoeffsSh;

    int outBlock = id_x * block;
    if (outBlock >= outEnd)
        return;

    int blockEnd = std::min(outBlock + block, outEnd);
    double inBlockRaw = outBlock * scale;
    int inBlockRounded = static_cast<int>(inBlockRaw);
    float inPos = inBlockRaw - inBlockRounded;
    float fscale = scale;

    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
    {
        int loc0, loc1;
        window.input_range(inPos, &loc0, &loc1);

        if (loc0 + inBlockRounded < 0)
            loc0 = -inBlockRounded;
        if (loc1 + inBlockRounded > srcLength)
            loc1 = srcLength - inBlockRounded;
        int locInWindow = loc0;
        float locBegin = locInWindow - inPos;
        float accum = 0.0f;

        for (; locInWindow < loc1; locInWindow++, locBegin++)
        {
            float w = window(locBegin);
            accum += srcPtr[inBlockRounded + locInWindow] * w;
        }
        dstPtr[outPos] = accum;
    }
}

// Generic n channel resample support
__global__ void resample_nchannel_hip_tensor(float *srcPtr,
                                             float *dstPtr,
                                             int2 srcDims,
                                             int outEnd,
                                             double scale,
                                             RpptResamplingWindow &window,
                                             int block)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    int outBlock = id_x * block;
    if (outBlock >= outEnd)
        return;

    int blockEnd = std::min(outBlock + block, outEnd);
    double inBlockRaw = outBlock * scale;
    int inBlockRounded = static_cast<int>(inBlockRaw);
    float inPos = inBlockRaw - inBlockRounded;
    float fscale = scale;
    float tempBuf[RPPT_MAX_CHANNELS] = {0.0f}; // Considering max channels as 3

    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
    {
        int loc0, loc1;
        window.input_range(inPos, &loc0, &loc1);

        if (loc0 + inBlockRounded < 0)
            loc0 = -inBlockRounded;
        if (loc1 + inBlockRounded > srcDims.x)
            loc1 = srcDims.x - inBlockRounded;
        int locInWindow = loc0;
        float locBegin = locInWindow - inPos;
        int2 ofs_i2 = make_int2(loc0, loc1);
        ofs_i2 *= (int2)srcDims.y;
        int idx = inBlockRounded * srcDims.y;

        for (int inOfs = ofs_i2.x; inOfs < ofs_i2.y; inOfs += srcDims.y, locBegin++)
        {
            float w = window(locBegin);
            for (int c = 0; c < srcDims.y; c++)
                tempBuf[c] += srcPtr[idx + inOfs + c] * w;
        }
        int dstLoc = outPos * srcDims.y;
        for (int c = 0; c < srcDims.y; c++)
            dstPtr[dstLoc + c] = tempBuf[c];
    }
}

// -------------------- Set 1 - resample kernels executor --------------------

RppStatus hip_exec_resample_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *inRateTensor,
                                   Rpp32f *outRateTensor,
                                   Rpp32s *srcDimsTensor,
                                   RpptResamplingWindow &window,
                                   rpp::Handle& handle)
{
    int batchSize = handle.GetBatchSize();

    for(int i = 0; i < batchSize; i++)
    {
        float inRate = inRateTensor[i];
        float outRate = outRateTensor[i];
        int srcLength = srcDimsTensor[i * 2];
        int numChannels = srcDimsTensor[i * 2 + 1];
        if (inRate == outRate) // No need of Resampling, do a direct memcpy
            hipMemcpy(dstPtr, srcPtr, srcLength * numChannels * sizeof(float), hipMemcpyDeviceToDevice);
        else
        {
            int outEnd = std::ceil(srcLength * outRate / inRate);
            double scale = static_cast<double>(inRate) / outRate;
            int block = 256; // 1 << 8
            int length = (outEnd / block) + 1;
            int globalThreads_x = length;
            int globalThreads_y = 1;
            int globalThreads_z = 1;
            size_t sharedMemSize = (window.lookupSize + (RPPT_MAX_CHANNELS + 1) * block) * sizeof(float);

            if (numChannels == 1)
            {
                hipLaunchKernelGGL(resample_1channel_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/256), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                                   dim3(256, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                                   sharedMemSize,
                                   handle.GetStream(),
                                   srcPtr + i * srcDescPtr->strides.nStride,
                                   dstPtr + i * dstDescPtr->strides.nStride,
                                   srcLength,
                                   outEnd,
                                   scale,
                                   window,
                                   block);
            }
            else
            {
                hipLaunchKernelGGL(resample_nchannel_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/256), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                                   dim3(256, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                                   0,
                                   handle.GetStream(),
                                   srcPtr + i * srcDescPtr->strides.nStride,
                                   dstPtr + i * dstDescPtr->strides.nStride,
                                   make_int2(srcLength, numChannels),
                                   outEnd,
                                   scale,
                                   window,
                                   block);
            }
        }

    }

    return RPP_SUCCESS;
}