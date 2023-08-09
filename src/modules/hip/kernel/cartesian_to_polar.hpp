#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void cartesian_to_polar_radians_hip_compute(d_float16 *src_f16, d_float16 *dst_f16)
{
    dst_f16->f4[0] = (src_f16->f4[0] * src_f16->f4[0]) + (src_f16->f4[2] * src_f16->f4[2]);
    dst_f16->f4[1] = (src_f16->f4[1] * src_f16->f4[1]) + (src_f16->f4[3] * src_f16->f4[3]);
    rpp_hip_math_sqrt8(&(dst_f16->f8[0]), &(dst_f16->f8[0]));
    rpp_hip_math_atan2_8(&(src_f16->f8[1]), &(src_f16->f8[0]), &(dst_f16->f8[1]));
}

__device__ void cartesian_to_polar_radians_degrees_conversion_hip_compute(d_float16 *dst_f16)
{
    rpp_hip_math_multiply8_const(&(dst_f16->f8[1]), &(dst_f16->f8[1]), (float4)ONE_EIGHTY_OVER_PI);
    rpp_hip_math_add8_const(&(dst_f16->f8[1]), &(dst_f16->f8[1]), (float4)360);
    rpp_hip_math_fmod8_const(&(dst_f16->f8[1]), &(dst_f16->f8[1]), 360.0f);
}

template <typename T>
__global__ void cartesian_to_polar_pkd_tensor(T *srcPtr,
                                              uint2 srcStridesNH,
                                              T *dstPtr,
                                              uint2 dstStridesNH,
                                              RpptAngleType angleType,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 2);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 2;

    d_float16 src_f16, dst_f16;

    rpp_hip_load16_pkd2_and_unpack_to_float16_pln2(srcPtr + srcIdx, &src_f16);
    cartesian_to_polar_radians_hip_compute(&src_f16, &dst_f16);
    if (angleType == RpptAngleType::DEGREES)
        cartesian_to_polar_radians_degrees_conversion_hip_compute(&dst_f16);
    rpp_hip_pack_float16_pln2_and_store16_pkd2(dstPtr + dstIdx, &dst_f16);
}

template <typename T>
__global__ void cartesian_to_polar_pln_tensor(T *srcPtr,
                                              uint3 srcStridesNCH,
                                              T *dstPtr,
                                              uint3 dstStridesNCH,
                                              RpptAngleType angleType,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float16 src_f16, dst_f16;

    rpp_hip_load16_pln2_and_unpack_to_float16_pln2(srcPtr + srcIdx, srcStridesNCH.y, &src_f16);
    cartesian_to_polar_radians_hip_compute(&src_f16, &dst_f16);
    if (angleType == RpptAngleType::DEGREES)
        cartesian_to_polar_radians_degrees_conversion_hip_compute(&dst_f16);
    rpp_hip_pack_float16_pln2_and_store16_pln2(dstPtr + dstIdx, dstStridesNCH.y, &dst_f16);
}

template <typename T>
__global__ void cartesian_to_polar_pkd2_pln2_tensor(T *srcPtr,
                                                    uint2 srcStridesNH,
                                                    T *dstPtr,
                                                    uint3 dstStridesNCH,
                                                    RpptAngleType angleType,
                                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 2);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float16 src_f16, dst_f16;

    rpp_hip_load16_pkd2_and_unpack_to_float16_pln2(srcPtr + srcIdx, &src_f16);
    cartesian_to_polar_radians_hip_compute(&src_f16, &dst_f16);
    if (angleType == RpptAngleType::DEGREES)
        cartesian_to_polar_radians_degrees_conversion_hip_compute(&dst_f16);
    rpp_hip_pack_float16_pln2_and_store16_pln2(dstPtr + dstIdx, dstStridesNCH.y, &dst_f16);
}

template <typename T>
__global__ void cartesian_to_polar_pln2_pkd2_tensor(T *srcPtr,
                                                    uint3 srcStridesNCH,
                                                    T *dstPtr,
                                                    uint2 dstStridesNH,
                                                    RpptAngleType angleType,
                                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 2;

    d_float16 src_f16, dst_f16;

    rpp_hip_load16_pln2_and_unpack_to_float16_pln2(srcPtr + srcIdx, srcStridesNCH.y, &src_f16);
    cartesian_to_polar_radians_hip_compute(&src_f16, &dst_f16);
    if (angleType == RpptAngleType::DEGREES)
        cartesian_to_polar_radians_degrees_conversion_hip_compute(&dst_f16);
    rpp_hip_pack_float16_pln2_and_store16_pkd2(dstPtr + dstIdx, &dst_f16);
}

template <typename T>
RppStatus hip_exec_cartesian_to_polar_tensor(T *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             T *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptAngleType angleType,
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

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 2 + 7) >> 3;
        hipLaunchKernelGGL(cartesian_to_polar_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           angleType,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(cartesian_to_polar_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           angleType,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(cartesian_to_polar_pkd2_pln2_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           angleType,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
        hipLaunchKernelGGL(cartesian_to_polar_pln2_pkd2_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           angleType,
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}
