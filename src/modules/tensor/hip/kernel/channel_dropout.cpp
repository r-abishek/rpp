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

// -------------------- Set 0 - dropout main kernels --------------------
template <typename T>
__global__ void channel_dropout_pkd_hip_tensor(T *dstPtr,
                                               const T *srcPtr,
                                               uint2 stridesNH,
                                               bool *channelMaskTensor,
                                               int channels)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= stridesNH.x / stridesNH.y || id_y >= stridesNH.y || id_z >= gridDim.z)
        return;

    uint srcIdx = id_z * stridesNH.x + id_y * stridesNH.y + id_x * channels;
    uint maskOffset = id_z * channels;

    for (int c = 0; c < channels; c++)
    {
        if (channelMaskTensor[maskOffset + c])
            dstPtr[srcIdx + c] = srcPtr[srcIdx + c];
        else
            dstPtr[srcIdx + c] = static_cast<T>(0);
    }
}

template <typename T>
__global__ void channel_dropout_pln_hip_tensor(T *dstPtr,
                                               const T *srcPtr,
                                               uint3 stridesNCH,
                                               bool *channelMaskTensor,
                                               int channels)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= stridesNCH.y || id_y >= stridesNCH.z || id_z >= gridDim.z)
        return;

    for (int c = 0; c < channels; c++)
    {
        uint maskOffset = id_z * channels + c;
        uint srcIdx = id_z * stridesNCH.x + c * stridesNCH.y + id_y * stridesNCH.z + id_x;

        if (channelMaskTensor[maskOffset])
            dstPtr[srcIdx] = srcPtr[srcIdx];
        else
            dstPtr[srcIdx] = static_cast<T>(0);
    }
}

template <typename T>
__global__ void channel_dropout_pkd3_to_pln3_hip_tensor(T *dstPtr,
                                                        const T *srcPtr,
                                                        uint3 dstStrides,
                                                        bool *channelMaskTensor)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int width = dstStrides.y / 3;  // Assumes 3 channels

    if (x >= width || y >= dstStrides.z || z >= gridDim.z)
        return;

    int srcIdx = z * dstStrides.x + y * dstStrides.y + x * 3;
    int maskOffset = z * 3;

    for (int c = 0; c < 3; c++)
    {
        int dstIdx = z * dstStrides.x + c * dstStrides.y + y * dstStrides.z + x;
        dstPtr[dstIdx] = channelMaskTensor[maskOffset + c] ? srcPtr[srcIdx + c] : static_cast<T>(0);
    }
}

template <typename T>
__global__ void channel_dropout_pln3_to_pkd3_hip_tensor(T *dstPtr,
                                                        const T *srcPtr,
                                                        uint3 srcStrides,
                                                        bool *channelMaskTensor)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int width = srcStrides.y / 3;

    if (x >= width || y >= srcStrides.z || z >= gridDim.z)
        return;

    int dstIdx = z * srcStrides.x + y * srcStrides.y + x * 3;
    int maskOffset = z * 3;

    for (int c = 0; c < 3; c++)
    {
        int srcIdx = z * srcStrides.x + c * srcStrides.y + y * srcStrides.z + x;
        dstPtr[dstIdx + c] = channelMaskTensor[maskOffset + c] ? srcPtr[srcIdx] : static_cast<T>(0);
    }
}

template <typename T>
__global__ void channel_dropout_pln1_hip_tensor(T *dstPtr,
                                                const T *srcPtr,
                                                uint3 strides,
                                                bool *channelMaskTensor)
{
    int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (x >= strides.y || y >= strides.z || z >= gridDim.z)
        return;

    int maskOffset = z;
    int idx = z * strides.x + y * strides.z + x;
    dstPtr[idx] = channelMaskTensor[maskOffset] ? srcPtr[idx] : static_cast<T>(0);
}

// -------------------- Set 1 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_channel_dropout_tensor(T *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          T *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          bool *channelMaskTensor,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle &handle)
{
    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int channels = dstDescPtr->c;

    if (srcDescPtr->layout == RpptLayout::NHWC && dstDescPtr->layout == RpptLayout::NHWC)
    {
        hipLaunchKernelGGL(channel_dropout_pkd_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           srcPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           channelMaskTensor,
                           channels);
    }
    else if (srcDescPtr->layout == RpptLayout::NCHW && dstDescPtr->layout == RpptLayout::NCHW)
    {
        hipLaunchKernelGGL(channel_dropout_pln_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           srcPtr,
                           make_uint3(dstDescPtr->strides.nStride,
                                      dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           channelMaskTensor,
                           channels);
    }
    else if (srcDescPtr->layout == RpptLayout::NHWC && dstDescPtr->layout == RpptLayout::NCHW)
    {
        hipLaunchKernelGGL(channel_dropout_pkd3_to_pln3_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           srcPtr,
                           make_uint3(dstDescPtr->strides.nStride,
                                      dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           channelMaskTensor);
    }
    else if (srcDescPtr->layout == RpptLayout::NCHW && dstDescPtr->layout == RpptLayout::NHWC)
    {
        hipLaunchKernelGGL(channel_dropout_pln3_to_pkd3_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride,
                                      srcDescPtr->strides.cStride,
                                      srcDescPtr->strides.hStride),
                           channelMaskTensor);
    }
    else if (srcDescPtr->layout == RpptLayout::NCHW && dstDescPtr->layout == RpptLayout::NCHW && srcDescPtr->c == 1)
    {
        hipLaunchKernelGGL(channel_dropout_pln1_hip_tensor<T>,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           srcPtr,
                           make_uint3(dstDescPtr->strides.nStride,
                                      dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           channelMaskTensor);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_channel_dropout_tensor<Rpp8u>(Rpp8u*,
                                                          RpptDescPtr,
                                                          Rpp8u*,
                                                          RpptDescPtr,
                                                          bool*,
                                                          RpptROIPtr,
                                                          RpptRoiType,
                                                          rpp::Handle&);

template RppStatus hip_exec_channel_dropout_tensor<half>(half*,
                                                         RpptDescPtr,
                                                         half*,
                                                         RpptDescPtr,
                                                         bool*,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_channel_dropout_tensor<Rpp32f>(Rpp32f*,
                                                           RpptDescPtr,
                                                           Rpp32f*,
                                                           RpptDescPtr,
                                                           bool*,
                                                           RpptROIPtr,
                                                           RpptRoiType,
                                                           rpp::Handle&);

template RppStatus hip_exec_channel_dropout_tensor<Rpp8s>(Rpp8s*,
                                                          RpptDescPtr,
                                                          Rpp8s*,
                                                          RpptDescPtr,
                                                          bool*,
                                                          RpptROIPtr,
                                                          RpptRoiType,
                                                          rpp::Handle&);
