#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

RppStatus hip_exec_multiply_scalar_tensor(Rpp32f *srcPtr,
                                          RpptGenericDescPtr srcGenericDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptGenericDescPtr dstGenericDescPtr,
                                          RpptROI3DPtr roiGenericPtrSrc,
                                          Rpp32f *mulTensor,
                                          rpp::Handle& handle);
