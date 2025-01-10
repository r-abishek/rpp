#include <hip/hip_runtime.h>
#include <omp.h>
#include "rpp_hip_common.hpp"

RppStatus hip_exec_fused_multiply_add_scalar_tensor(Rpp32f *srcPtr,
                                       RpptGenericDescPtr srcGenericDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptGenericDescPtr dstGenericDescPtr,
                                       RpptROI3DPtr roiGenericPtrSrc,
                                       Rpp32f *mulTensor,
                                       Rpp32f *addTensor,
                                       rpp::Handle& handle);