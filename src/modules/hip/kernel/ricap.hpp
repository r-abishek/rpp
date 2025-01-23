#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
RppStatus hip_exec_ricap_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32u *permutationTensor,
                                RpptROIPtr roiPtrInputCropRegion,
                                RpptRoiType roiType,
                                rpp::Handle& handle);


