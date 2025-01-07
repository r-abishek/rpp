#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
RppStatus hip_exec_crop_and_patch_tensor(T *srcPtr1,
                                         T *srcPtr2,
                                         RpptDescPtr srcDescPtr,
                                         T *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptROIPtr cropTensorPtr,
                                         RpptROIPtr patchTensorPtr,
                                         RpptRoiType roiType,
                                         rpp::Handle& handle);
