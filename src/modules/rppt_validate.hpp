/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

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

#ifndef RPPT_VALIDATE_OPERATIONS_FUNCTIONS
#define RPPT_VALIDATE_OPERATIONS_FUNCTIONS

#include <iostream>
#include <stdlib.h>

#include "rpp.h"
#include "rppdefs.h"
#include "rpp/handle.hpp"

#ifdef OCL_COMPILE
#include <CL/cl.h>
#elif defined (HIP_COMPILE)
#include <hip/hip_runtime_api.h>
#endif

inline RppLayoutParams get_layout_params(RpptLayout layout, Rpp32u channels)
{
    RppLayoutParams layoutParams;
    if(layout == RpptLayout::NCHW || layout == RpptLayout::NCDHW)
    {
        if (channels == 1) // PLN1
        {
            layoutParams.channelParam = 1;
            layoutParams.bufferMultiplier = 1;
        }
        else if (channels == 3) // PLN3
        {
            layoutParams.channelParam = 3;
            layoutParams.bufferMultiplier = 1;
        }
    }
    else if(layout == RpptLayout::NHWC || layout == RpptLayout::NDHWC)
    {
        //PKD
        layoutParams.channelParam = 1;
        layoutParams.bufferMultiplier = channels;
    }
    return layoutParams;
}

inline int check_roi_out_of_bounds(RpptROIPtr roiPtrImage, RpptDescPtr srcDescPtr, RpptRoiType type)
{
    int x, y, w, h;
    if (type == RpptRoiType::XYWH)
    {
        x = ((0 <= roiPtrImage->xywhROI.xy.x) && (roiPtrImage->xywhROI.xy.x < srcDescPtr->w)) ? roiPtrImage->xywhROI.xy.x : -1;
        y = ((0 <= roiPtrImage->xywhROI.xy.y) && (roiPtrImage->xywhROI.xy.y < srcDescPtr->h)) ? roiPtrImage->xywhROI.xy.y : -1;
        w = ((roiPtrImage->xywhROI.roiWidth) <= srcDescPtr->w) ? roiPtrImage->xywhROI.roiWidth : -1;
        h = ((roiPtrImage->xywhROI.roiHeight) <= srcDescPtr->h) ? roiPtrImage->xywhROI.roiHeight : -1;
    }
    else if (type == RpptRoiType::LTRB)
    {
        x = ((0 <= roiPtrImage->ltrbROI.lt.x) && (roiPtrImage->ltrbROI.lt.x < srcDescPtr->w)) ? roiPtrImage->ltrbROI.lt.x : -1;
        y = ((0 <= roiPtrImage->ltrbROI.lt.y) && (roiPtrImage->ltrbROI.lt.y < srcDescPtr->h)) ? roiPtrImage->ltrbROI.lt.y : -1;
        w = ((0 <= roiPtrImage->ltrbROI.rb.x) && (roiPtrImage->ltrbROI.rb.x < srcDescPtr->w)) ? roiPtrImage->ltrbROI.rb.x - roiPtrImage->ltrbROI.lt.x + 1 : -1;
        h = ((0 <= roiPtrImage->ltrbROI.rb.y) && (roiPtrImage->ltrbROI.rb.y < srcDescPtr->h)) ? roiPtrImage->ltrbROI.rb.y - roiPtrImage->ltrbROI.lt.y + 1 : -1;
    }
    if ((x < 0) || (y < 0) || (w < 0) || (h < 0))
        return -1;
    return 0;
}

#ifdef GPU_SUPPORT

inline void copy_param_float(float *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.floatArr[paramIndex].floatmem, handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem, sizeof(Rpp32f) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.floatArr[paramIndex].floatmem, CL_FALSE, 0, sizeof(Rpp32f) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_float3(float *param, rpp::Handle& handle, Rpp32u paramIndex)
{
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.float3Arr[paramIndex].floatmem, param, sizeof(Rpp32f) * handle.GetBatchSize() * 3, hipMemcpyHostToDevice);
#endif
}

inline void copy_param_uint(uint *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.uintArr[paramIndex].uintmem, handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.uintArr[paramIndex].uintmem, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_int(int *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.intArr[paramIndex].intmem, handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem, sizeof(Rpp32s) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.intArr[paramIndex].intmem, CL_FALSE, 0, sizeof(Rpp32s) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_RpptRGB(RpptRGB *param, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.rgbArr.rgbmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.rgbArr.rgbmem, handle.GetInitHandle()->mem.mcpu.rgbArr.rgbmem, sizeof(RpptRGB) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#endif // backend
}

#endif // GPU_SUPPORT

#endif // RPPT_VALIDATE_OPERATIONS_FUNCTIONS
