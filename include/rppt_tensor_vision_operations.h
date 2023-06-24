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

#ifndef RPPT_TENSOR_VISION_OPERATIONS_H
#define RPPT_TENSOR_VISION_OPERATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** farneback_optical_flow ********************/

// Farneback optical flow motion vector generation
// Farnebäck, G. Two-frame motion estimation based on polynomial expansion. Proc. Scand. Conf. Image Anal. 3, 363–370 (2003).

// *param[in] src1Ptr source tensor memory for frame1
// *param[in] src2Ptr source tensor memory for frame2 (single image buffer optimization assumption - src2Ptr = src1Ptr + srcCompDescPtr->strides.nStride)
// *param[in] srcCompDescPtr source component tensor descriptor for src1Ptr and src2Ptr (must be of layout NCHW, with srcCompDescPtr->c = 1 - same descriptor for src1Ptr and src2Ptr)
// *param[out] mVecCompX motion vector x component tensor memory
// *param[out] mVecCompY motion vector y component tensor memory
// *param[in] mVecCompDescPtr motion vector component tensor descriptor for mVecCompX and mVecCompY (must be of layout NCHW, type F32 with mVecCompDescPtr->c = 1 - same descriptor for mVecCompX and mVecCompY)
// *param[in] pyramidScale An Rpp32f value specifying the image scale for building image pyramids. (pyramidScale < 1, and typical value is 0.5)
// *param[in] numPyramidLevels An Rpp32s value specifying the number of pyramid levels for building image pyramids. (if numPyramidLevels = 1, only the initial image is used, if numPyramidLevels = 5, then 4 additional pyramid levels are created)
// *param[in] windowSize An Rpp32s value specifying the window size. (Larger the window size, faster the motion vector generation, and more blurry the motion field - windowSize = 3/5/7/9)
// *param[in] numIterations An Rpp32s value specifying the number of iterations at each pyramid level
// *param[in] polyExpNbhoodSize An Rpp32s value specifying the pixel neighborhood size used to find the polynomial expansion between pixels (typical values are polyExpNbhoodSize = 5 or 7)
// *param[in] polyExpStdDev An Rpp32f value specifying standard deviation of the gaussian used in the polynomial expansion (typical values are polyExpStdDev = 1.1 or 1.5 for polyExpNbhoodSize = 5 or 7)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

#ifdef GPU_SUPPORT
RppStatus rppt_farneback_optical_flow_gpu(RppPtr_t src1Ptr, RppPtr_t src2Ptr, RppPtr_t dstPtrIntermU8, RppPtr_t dstPtrIntermF32, RppPtr_t dh_cudaResizdStrided, RpptDescPtr srcCompDescPtr, RppPtr_t mVecCompX, RppPtr_t mVecCompY, RpptDescPtr mVecCompDescPtr, Rpp32f pyramidScale, Rpp32s numPyramidLevels, Rpp32s windowSize, Rpp32s numIterations, Rpp32s polyExpNbhoodSize, Rpp32f polyExpStdDev, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_VISION_OPERATIONS_H
