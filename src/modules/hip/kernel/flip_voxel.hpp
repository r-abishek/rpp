#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
RppStatus hip_exec_flip_voxel_tensor(T *srcPtr,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     T *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc,
                                     Rpp32u *horizontalTensor,
                                     Rpp32u *verticalTensor,
                                     Rpp32u *depthTensor,
                                     RpptRoi3DType roiType,
                                     rpp::Handle& handle);