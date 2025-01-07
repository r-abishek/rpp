#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
RppStatus hip_exec_salt_and_pepper_noise_tensor(T *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                T *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                RpptXorwowState *xorwowInitialStatePtr,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                rpp::Handle& handle);
