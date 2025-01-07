#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
RppStatus hip_exec_fog_tensor(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              T *dstPtr,
                              RpptDescPtr dstDescPtr,
                              Rpp32f *d_fogAlphaMaskPtr,
                              Rpp32f *d_fogIntensityMaskPtr,
                              Rpp32f *intensityFactor,
                              Rpp32f *greyFactor,
                              Rpp32u *maskLocOffsetX,
                              Rpp32u *maskLocOffsetY,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle);