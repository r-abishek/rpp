#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T, typename U>
RppStatus hip_exec_erase_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptRoiLtrb *anchorBoxInfoTensor,
                                U *colorsTensor,
                                Rpp32u *numBoxesTensor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle);
