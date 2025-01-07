#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
RppStatus hip_exec_copy_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               rpp::Handle& handle);