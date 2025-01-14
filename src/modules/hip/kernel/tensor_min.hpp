#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 2 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_min(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *minArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle &handle);
