#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void normalize_minmax_hip_compute(d_float8 *val_f8, float2 *oldMinOldRange_f2, float2 *newMinNewRange_f2)
{
    float4 oldMin_f4 = (float4) oldMinOldRange_f2->x;
    val_f8->f4[0] -= oldMin_f4;
    val_f8->f4[1] -= oldMin_f4;
    rpp_hip_math_fmaf8_const(val_f8, val_f8, newMinNewRange_f2->y / oldMinOldRange_f2->y, newMinNewRange_f2->x);
}

template <typename T>
__global__ void normalize_minmax_pln1_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint2 dstStridesNH,
                                            float *imageMinMaxArr,
                                            float2 newMinNewRange_f2,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    float2 *imageMinMaxArr_f2 = (float2 *)imageMinMaxArr;
    float2 oldMinOldRange_f2 = imageMinMaxArr_f2[id_z];
    oldMinOldRange_f2.y -= oldMinOldRange_f2.x;

    d_float8 val_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
    normalize_minmax_hip_compute(&val_f8, &oldMinOldRange_f2, &newMinNewRange_f2);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
}

template <typename T>
RppStatus hip_exec_normalize_minmax_tensor(T *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           T *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *imageMinMaxArr,
                                           Rpp32f newMin,
                                           Rpp32f newMax,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->c == 1) && (dstDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(normalize_minmax_pln1_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           imageMinMaxArr,
                           make_float2(newMin, newMax - newMin),
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}
