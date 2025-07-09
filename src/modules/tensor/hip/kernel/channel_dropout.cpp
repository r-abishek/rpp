/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_tensor_executors.hpp"
#include <random>

// -------------------- Set 0 - Dropout main kernels --------------------
template <typename T>
__device__ __forceinline__ T channel_dropout_zero();

template <>
__device__ __forceinline__ Rpp8u channel_dropout_zero<Rpp8u>() { return 0; }

template <>
__device__ __forceinline__ Rpp8s channel_dropout_zero<Rpp8s>() { return static_cast<Rpp8s>(-128); }

template <>
__device__ __forceinline__ Rpp32f channel_dropout_zero<Rpp32f>() { return 0.0f; }

template <>
__device__ __forceinline__ half channel_dropout_zero<half>() { return __float2half(0.0f); }

// PKD3 kernel
template <typename T>
__global__ void channel_dropout_pkd_hip_tensor(T *srcPtr,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               const uint8_t *channelMask,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    int maskBase = id_z * 3;

    for (int c = 0; c < 3; c++)
    {
        if (channelMask[maskBase + c])
            dstPtr[dstIdx + c] = srcPtr[dstIdx + c];
        else
            dstPtr[dstIdx + c] = channel_dropout_zero<T>();
    }
}

// PLN kernel
template <typename T>
__global__ void channel_dropout_pln_hip_tensor(T *srcPtr,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               const uint8_t *channelMask,
                                               int channels,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    for (int c = 0; c < channels; c++)
    {
        uint dstIdx = (id_z * dstStridesNCH.x) + (c * dstStridesNCH.y) + (id_y * dstStridesNCH.z) + id_x;
        int maskIdx = id_z * channels + c;

        T val = srcPtr[dstIdx];
        dstPtr[dstIdx] = channelMask[maskIdx] ? val : channel_dropout_zero<T>();
    }
}

// PLN3 kernel
template <typename T>
__global__ void channel_dropout_pln3_hip_tensor(T *srcPtr,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
                                                const uint8_t *channelMask,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint baseIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    int maskBase = id_z * 3;

    // Channel 0
    dstPtr[baseIdx] = channelMask[maskBase + 0] ? srcPtr[baseIdx] : channel_dropout_zero<T>();

    // Channel 1
    uint ch1Idx = baseIdx + dstStridesNCH.y;
    dstPtr[ch1Idx] = channelMask[maskBase + 1] ? srcPtr[ch1Idx] : channel_dropout_zero<T>();

    // Channel 2
    uint ch2Idx = ch1Idx + dstStridesNCH.y;
    dstPtr[ch2Idx] = channelMask[maskBase + 2] ? srcPtr[ch2Idx] : channel_dropout_zero<T>();
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_channel_dropout_tensor(T *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          T *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32f *dropProb,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle &handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    // Generate channel mask on host
    std::mt19937 gen(std::random_device{}());
    std::vector<uint8_t> channelMaskHost(globalThreads_z * srcDescPtr->c);
    for (int b = 0; b < globalThreads_z; b++)
    {
        std::bernoulli_distribution keepDist(1.0f - dropProb[b]);
        bool anyKept = false;
        int base = b * srcDescPtr->c;
        for (int c = 0; c < srcDescPtr->c; c++)
        {
            channelMaskHost[base + c] = keepDist(gen);
            anyKept |= channelMaskHost[base + c];
        }
        // Ensure at least one channel is kept
        if (!anyKept)
            channelMaskHost[base + (gen() % srcDescPtr->c)] = 1;
    }

    uint8_t *d_channelMask = nullptr;
    hipMalloc(&d_channelMask, channelMaskHost.size() * sizeof(uint8_t));
    hipMemcpy(d_channelMask, channelMaskHost.data(), channelMaskHost.size() * sizeof(uint8_t), hipMemcpyHostToDevice);

    // PKD3 -> PKD3 (NHWC -> NHWC)
    if (srcDescPtr->layout == RpptLayout::NHWC && dstDescPtr->layout == RpptLayout::NHWC && srcDescPtr->c == 3)
    {
        hipMemcpyAsync(dstPtr, srcPtr, srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(channel_dropout_pkd_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0, handle.GetStream(),
                           dstPtr, dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           d_channelMask, roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipFree(d_channelMask);
        return RPP_SUCCESS;
    }
    // PLN3 -> PLN3 (NCHW -> NCHW)
    else if (srcDescPtr->layout == RpptLayout::NCHW && dstDescPtr->layout == RpptLayout::NCHW && srcDescPtr->c == 3)
    {
        hipMemcpyAsync(dstPtr, srcPtr, srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(channel_dropout_pln3_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0, handle.GetStream(),
                           dstPtr, dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           d_channelMask, roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipFree(d_channelMask);
        return RPP_SUCCESS;
    }
    // PLN1 -> PLN1 (NCHW -> NCHW, c==1)
    else if (srcDescPtr->layout == RpptLayout::NCHW && dstDescPtr->layout == RpptLayout::NCHW && srcDescPtr->c == 1)
    {
        hipMemcpyAsync(dstPtr, srcPtr, srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(channel_dropout_pln_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0, 
                           handle.GetStream(),
                           dstPtr, dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           d_channelMask, dstDescPtr->c, roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipFree(d_channelMask);
        return RPP_SUCCESS;
    }
    // PKD3 -> PLN3 (NHWC -> NCHW)
    else if (srcDescPtr->layout == RpptLayout::NHWC && dstDescPtr->layout == RpptLayout::NCHW && srcDescPtr->c == 3)
    {
        hipLaunchKernelGGL(convert_pkd3_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0, 
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(channel_dropout_pln3_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), globalThreads_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0, handle.GetStream(),
                           dstPtr, dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           d_channelMask, roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipFree(d_channelMask);
        return RPP_SUCCESS;
    }
    // PLN3 -> PKD3 (NCHW -> NHWC)
    else if (srcDescPtr->layout == RpptLayout::NCHW && dstDescPtr->layout == RpptLayout::NHWC && srcDescPtr->c == 3)
    {
        hipLaunchKernelGGL(convert_pln3_pkd3_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0, 
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(channel_dropout_pkd_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0, 
                           handle.GetStream(),
                           dstPtr, dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           d_channelMask, roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipFree(d_channelMask);
        return RPP_SUCCESS;
    }
    else
    {
        hipFree(d_channelMask);
        return RPP_ERROR;
    }
}

template RppStatus hip_exec_channel_dropout_tensor<Rpp8u>(Rpp8u*,
                                                          RpptDescPtr,
                                                          Rpp8u*,
                                                          RpptDescPtr,
                                                          Rpp32f*,
                                                          RpptROIPtr,
                                                          RpptRoiType,
                                                          rpp::Handle&);
template RppStatus hip_exec_channel_dropout_tensor<Rpp8s>(Rpp8s*,
                                                          RpptDescPtr,
                                                          Rpp8s*,
                                                          RpptDescPtr,
                                                          Rpp32f*,
                                                          RpptROIPtr,
                                                          RpptRoiType,
                                                          rpp::Handle&);
template RppStatus hip_exec_channel_dropout_tensor<Rpp32f>(Rpp32f*,
                                                           RpptDescPtr,
                                                           Rpp32f*,
                                                           RpptDescPtr,
                                                           Rpp32f*,
                                                           RpptROIPtr,
                                                           RpptRoiType,
                                                           rpp::Handle&);
template RppStatus hip_exec_channel_dropout_tensor<half>(half*,
                                                         RpptDescPtr,
                                                         half*,
                                                         RpptDescPtr,
                                                         Rpp32f*,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);