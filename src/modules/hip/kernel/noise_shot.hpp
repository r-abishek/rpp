#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rng_seed_stream.hpp"

template <typename T>
RppStatus hip_exec_shot_noise_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle);