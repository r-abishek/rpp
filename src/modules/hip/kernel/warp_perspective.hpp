#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_warp_perspective_tensor(T *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           T *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *perspectiveTensor,
                                           RpptInterpolationType interpolationType,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rpp::Handle& handle);