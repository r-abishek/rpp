#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_resize_crop_mirror_tensor(T *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             T *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptImagePatchPtr dstImgSizes,
                                             RpptInterpolationType interpolationType,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             rpp::Handle& handle);


