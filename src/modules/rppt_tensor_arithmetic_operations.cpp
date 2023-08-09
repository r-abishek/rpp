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
#include "rppt_tensor_arithmetic_operations.h"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    #include "hip/hip_tensor_arithmetic_operations.hpp"
#endif // HIP_COMPILE

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** multiply_scalar ********************/

RppStatus rppt_multiply_scalar_gpu(RppPtr_t srcPtr,
                                   RpptGenericDescPtr srcGenericDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32f mulParam,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcGenericDescPtr->dataType != RpptDataType::F32)   // F32 data type check for float scalar multiply
        return RPP_ERROR_INVALID_SRC_DATA_TYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32)   // F32 data type check for float scalar multiply
        return RPP_ERROR_INVALID_DST_DATA_TYPE;

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    rpp_tensor_generic_to_image_desc(srcGenericDescPtr, srcDescPtr);
    rpp_tensor_generic_to_image_desc(dstGenericDescPtr, dstDescPtr);

    hip_exec_multiply_scalar_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    mulParam,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
