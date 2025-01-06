#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T, typename U>
RppStatus hip_exec_resize_mirror_normalize_tensor(T *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  U *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  RpptImagePatchPtr dstImgSizes,
                                                  RpptInterpolationType interpolationType,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  rpp::Handle& handle);
