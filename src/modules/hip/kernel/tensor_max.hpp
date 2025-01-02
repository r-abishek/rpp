#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T, typename U>
RppStatus hip_exec_tensor_max(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *maxArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle);