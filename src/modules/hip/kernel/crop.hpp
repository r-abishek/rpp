#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
RppStatus hip_exec_crop_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle);