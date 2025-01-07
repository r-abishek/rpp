#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T1, typename T2>
RppStatus hip_exec_lut_tensor(T1 *srcPtr,
                              RpptDescPtr srcDescPtr,
                              T2 *dstPtr,
                              RpptDescPtr dstDescPtr,
                              T2 *lutPtr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle);