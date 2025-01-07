#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rng_seed_stream.hpp"

template <typename T>
RppStatus hip_exec_gaussian_noise_voxel_tensor(T *srcPtr,
                                               RpptGenericDescPtr srcGenericDescPtr,
                                               T *dstPtr,
                                               RpptGenericDescPtr dstGenericDescPtr,
                                               RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                               Rpp32f *meanTensor,
                                               Rpp32f *stdDevTensor,
                                               RpptROI3DPtr roiGenericPtrSrc,
                                               rpp::Handle& handle);