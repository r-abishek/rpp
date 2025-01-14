#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

/* BitwiseOR is logical operation only on U8/I8 types.
   For a Rpp32f precision image (pixel values from 0-1), the BitwiseOR is applied on a 0-255
   range-translated approximation, of the original 0-1 decimal-range image.
   The bitwise operation is applied to the char representation of the raw floating-point data in memory */

template <typename T>
RppStatus hip_exec_bitwise_or_tensor(T *srcPtr1,
                                     T *srcPtr2,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle);
