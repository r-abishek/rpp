#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 1 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_mean(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *tensorMeanArr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle);
