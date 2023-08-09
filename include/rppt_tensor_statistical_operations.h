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

#ifndef RPPT_TENSOR_STATISTICAL_OPERATIONS_H
#define RPPT_TENSOR_STATISTICAL_OPERATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** cartesian_to_polar ********************/

// Cartesian to polar operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcGenericDescPtr source tensor descriptor (F32 NHWC/NCHW tensor with channels dimension = 2 for x,y coordinates)
// *param[out] dstPtr destination tensor memory
// *param[in] dstGenericDescPtr destination tensor descriptor (F32 NHWC/NCHW tensor with channels dimension = 2 for magnitude,angle coordinates)
// *param[in] angleType type of angle output (A single RpptAngleType enum specifying RADIANS or DEGREES for all polar coordinate ouputs)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

#ifdef GPU_SUPPORT
RppStatus rppt_cartesian_to_polar_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, RpptAngleType angleType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** image_sum ********************/

// Image sum finder operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] imageSumArr destination array of minimum length (srcPtr->n * srcPtr->c)
// *param[in] imageSumArrLength length of provided destination array (minimum length = srcPtr->n * srcPtr->c)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

#ifdef GPU_SUPPORT
RppStatus rppt_image_sum_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageSumArr, Rpp32u imageSumArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** image_min_max ********************/

// Image min_max finder operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcGenericDescPtr source tensor descriptor
// *param[out] imageMinMaxArr destination minmax array (length >= srcPtr->n * srcPtr->c * 2)
// *param[in] imageMinMaxArrLength length of provided destination array (minimum length = srcPtr->n * srcPtr->c * 2)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

#ifdef GPU_SUPPORT
RppStatus rppt_image_min_max_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t imageMinMaxArr, Rpp32u imageMinMaxArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** normalize_minmax ********************/

// Minmax normalize operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcGenericDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstGenericDescPtr destination tensor descriptor
// *param[in] imageMinMaxArr minmax array containing min, max of each image in batch (min != max for any single image, and length >= srcPtr->n * srcPtr->c * 2)
// *param[in] imageMinMaxArrLength length of provided destination array (minimum length = srcPtr->n * srcPtr->c * 2)
// *param[in] newMin new Rpp32f minimum value to use for normalize operation
// *param[in] newMax new Rpp32f maximum value to use for normalize operation
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

#ifdef GPU_SUPPORT
RppStatus rppt_normalize_minmax_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *imageMinMaxArr, Rpp32u imageMinMaxArrLength, Rpp32f newMin, Rpp32f newMax, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_STATISTICAL_OPERATIONS_H
