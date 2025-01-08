#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 2 - Kernel Executors --------------------

RppStatus hip_exec_lens_correction_tensor(RpptDescPtr dstDescPtr,
                                          Rpp32f *rowRemapTable,
                                          Rpp32f *colRemapTable,
                                          RpptDescPtr remapTableDescPtr,
                                          Rpp32f *cameraMatrix,
                                          Rpp32f *distanceCoeffs,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle& handle);