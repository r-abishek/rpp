#include <random>
#include <hip/hip_runtime.h>

#include "rpp_hip_common.hpp"
#include "spatter_mask.hpp"

template <typename T>
RppStatus hip_exec_spatter_tensor(T *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  T *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptRGB spatterColor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  rpp::Handle& handle);