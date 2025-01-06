#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rpp_cpu_common.hpp"

template <typename T>
RppStatus hip_exec_color_twist_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle);