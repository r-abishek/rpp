#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T, typename U>
RppStatus hip_exec_log_generic_tensor(T *srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      U *dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      uint *roiTensor,
                                      rpp::Handle& handle);
