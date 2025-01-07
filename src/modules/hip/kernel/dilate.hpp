#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_dilate_tensor(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u kernelSize,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle);