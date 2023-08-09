/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_statistical_operations.h"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    #include "hip/hip_tensor_statistical_operations.hpp"
#endif // HIP_COMPILE

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** cartesian_to_polar ********************/

RppStatus rppt_cartesian_to_polar_gpu(RppPtr_t srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      RppPtr_t dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      RpptAngleType angleType,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcGenericDescPtr->dataType != RpptDataType::F32)   // src tensor data type is F32 for cartesian coordinates
        return RPP_ERROR_INVALID_SRC_DATA_TYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32)   // dst tensor data type is F32 for polar coordinates
        return RPP_ERROR_INVALID_DST_DATA_TYPE;

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;
    rpp_tensor_generic_to_image_desc(srcGenericDescPtr, srcDescPtr);
    rpp_tensor_generic_to_image_desc(dstGenericDescPtr, dstDescPtr);

    if (srcDescPtr->c != 2)
        return RPP_ERROR_INVALID_SRC_CHANNELS;  // src tensor channels is 2 for (x, y) coordinates
    if (dstDescPtr->c != 2)
        return RPP_ERROR_INVALID_DST_CHANNELS;  // dst tensor channels is 2 for (magnitude, angle) coordinates

    hip_exec_cartesian_to_polar_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       angleType,
                                       roiTensorPtrSrc,
                                       roiType,
                                       rpp::deref(rppHandle));

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** image_sum ********************/

RppStatus rppt_image_sum_gpu(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t imageSumArr,
                             Rpp32u imageSumArrLength,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c == 1)
    {
        if (imageSumArrLength < srcDescPtr->n)      // sum of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (imageSumArrLength < srcDescPtr->n * 4)  // sum of each channel, and total sum of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_image_sum_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp32f*>(imageSumArr),
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        hip_exec_image_sum_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  static_cast<Rpp32f*>(imageSumArr),
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_image_sum_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  static_cast<Rpp32f*>(imageSumArr),
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        hip_exec_image_sum_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp32f*>(imageSumArr),
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** image_min_max ********************/

RppStatus rppt_image_min_max_gpu(RppPtr_t srcPtr,
                                 RpptGenericDescPtr srcGenericDescPtr,
                                 RppPtr_t imageMinMaxArr,
                                 Rpp32u imageMinMaxArrLength,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    RpptDesc srcDesc;
    RpptDescPtr srcDescPtr;
    srcDescPtr = &srcDesc;
    rpp_tensor_generic_to_image_desc(srcGenericDescPtr, srcDescPtr);

    if (srcDescPtr->c == 1)
    {
        if (imageMinMaxArrLength < srcDescPtr->n * 2)   // min and max of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (imageMinMaxArrLength < srcDescPtr->n * 8)   // min and max of each channel, and overall min and max of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_image_min_max_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp32f*>(imageMinMaxArr),
                                      roiTensorPtrSrc,
                                      roiType,
                                      rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        hip_exec_image_min_max_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      static_cast<Rpp32f*>(imageMinMaxArr),
                                      roiTensorPtrSrc,
                                      roiType,
                                      rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_image_min_max_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      static_cast<Rpp32f*>(imageMinMaxArr),
                                      roiTensorPtrSrc,
                                      roiType,
                                      rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        hip_exec_image_min_max_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp32f*>(imageMinMaxArr),
                                      roiTensorPtrSrc,
                                      roiType,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** normalize_minmax ********************/

RppStatus rppt_normalize_minmax_gpu(RppPtr_t srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32f *imageMinMaxArr,
                                    Rpp32u imageMinMaxArrLength,
                                    Rpp32f newMin,
                                    Rpp32f newMax,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;
    rpp_tensor_generic_to_image_desc(srcGenericDescPtr, srcDescPtr);
    rpp_tensor_generic_to_image_desc(dstGenericDescPtr, dstDescPtr);

    if (srcDescPtr->c == 1)
    {
        if (imageMinMaxArrLength < srcDescPtr->n * 2)   // min and max of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
        for (int i = 0; i < imageMinMaxArrLength; i += 2)
            if (imageMinMaxArr[i] == imageMinMaxArr[i + 1])
                return RPP_ERROR_INVALID_ARGUMENTS;
    }
    else if (srcDescPtr->c == 3)
    {
        if (imageMinMaxArrLength < srcDescPtr->n * 8)   // min and max of each channel, and overall min and max of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_normalize_minmax_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         static_cast<Rpp32f*>(imageMinMaxArr),
                                         newMin,
                                         newMax,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_normalize_minmax_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         static_cast<Rpp32f*>(imageMinMaxArr),
                                         newMin,
                                         newMax,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_normalize_minmax_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         static_cast<Rpp32f*>(imageMinMaxArr),
                                         newMin,
                                         newMax,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_normalize_minmax_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         static_cast<Rpp32f*>(imageMinMaxArr),
                                         newMin,
                                         newMax,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
