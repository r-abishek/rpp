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
#include "rppt_tensor_vision_operations.h"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    #include "hip/hip_tensor_vision_operations.hpp"
#endif // HIP_COMPILE

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** farneback_optical_flow ********************/

RppStatus rppt_farneback_optical_flow_gpu(RppPtr_t src1Ptr,
                                          RppPtr_t src2Ptr,
                                          RpptDescPtr srcCompDescPtr,
                                          RppPtr_t mVecCompX,
                                          RppPtr_t mVecCompY,
                                          RpptDescPtr mVecCompDescPtr,
                                          Rpp32f pyramidScale,
                                          Rpp32s numPyramidLevels,
                                          Rpp32s windowSize,
                                          Rpp32s numIterations,
                                          Rpp32s polyExpNbhoodSize,
                                          Rpp32f polyExpStdDev,
                                          rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcCompDescPtr->layout != RpptLayout::NCHW)             return RPP_ERROR_INVALID_SRC_LAYOUT;    // src1Ptr and src2Ptr must be NCHW
    if (mVecCompDescPtr->layout != RpptLayout::NCHW)            return RPP_ERROR_INVALID_DST_LAYOUT;    // mVecCompX and mVecCompY must be NCHW
    if (srcCompDescPtr->dataType != RpptDataType::U8)           return RPP_ERROR_INVALID_SRC_DATA_TYPE; // src1Ptr and src2Ptr must be U8
    if (mVecCompDescPtr->dataType != RpptDataType::F32)         return RPP_ERROR_INVALID_DST_DATA_TYPE; // mVecCompX and mVecCompY must be F32
    if (srcCompDescPtr->c != 1)                                 return RPP_ERROR_INVALID_SRC_CHANNELS;  // src1Ptr and src2Ptr must be single channel
    if (mVecCompDescPtr->c != 1)                                return RPP_ERROR_INVALID_DST_CHANNELS;  // mVecCompX and mVecCompY must be single channel
    if ((srcCompDescPtr->w) != (mVecCompDescPtr->w))            return RPP_ERROR_MISMATCH_SRC_AND_DST_WIDTHS;   // src and dst widths must match
    if ((srcCompDescPtr->h) != (mVecCompDescPtr->h))            return RPP_ERROR_MISMATCH_SRC_AND_DST_HEIGHTS;  // src and dst heights must match
    if ((polyExpNbhoodSize != 5) && (polyExpNbhoodSize != 7))   return RPP_ERROR_INVALID_ARGUMENTS;     //  polyExpNbhoodSize must be 5 or 7

    return hip_exec_farneback_optical_flow_tensor(static_cast<Rpp8u*>(src1Ptr) + srcCompDescPtr->offsetInBytes,
                                                  static_cast<Rpp8u*>(src2Ptr) + srcCompDescPtr->offsetInBytes,
                                                  srcCompDescPtr,
                                                  (Rpp32f*) (static_cast<Rpp8u*>(mVecCompX) + mVecCompDescPtr->offsetInBytes),
                                                  (Rpp32f*) (static_cast<Rpp8u*>(mVecCompY) + mVecCompDescPtr->offsetInBytes),
                                                  mVecCompDescPtr,
                                                  pyramidScale,
                                                  numPyramidLevels,
                                                  windowSize,
                                                  numIterations,
                                                  polyExpNbhoodSize,
                                                  polyExpStdDev,
                                                  rpp::deref(rppHandle));

#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
