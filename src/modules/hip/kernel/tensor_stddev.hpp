#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

template <typename T>
RppStatus hip_exec_tensor_stddev(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp32f *imageStddevArr,
                                 Rpp32f *meanTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle);