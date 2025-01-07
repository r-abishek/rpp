#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

RppStatus hip_exec_down_mixing_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcDimsTensor,
                                      bool normalizeWeights,
                                      rpp::Handle& handle);