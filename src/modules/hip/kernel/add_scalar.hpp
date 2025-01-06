#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

RppStatus hip_exec_add_scalar_tensor(Rpp32f *srcPtr,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc,
                                     Rpp32f *addTensor,
                                     rpp::Handle& handle);
