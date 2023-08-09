#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- resize_scale_intensity device helper --------------------

__device__ void resize_scale_intensity_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)(srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) / dstDimsWH->x;
    float hRatio = (float)(srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) / dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    locDst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [0-3]
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [4-7]
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [0-3]
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [4-7]
}

// -------------------- resize_scale_intensity device kernel --------------------

template <typename T, typename U>   // resize_bilinear_pln_tensor alone supporting bit-depth change (U8->F32) for optical flow
__global__ void resize_scale_intensity_bilinear_pln_tensor(T *srcPtr,
                                                           uint3 srcStridesNCH,
                                                           U *dstPtr,
                                                           uint3 dstStridesNCH,
                                                           int channelsDst,
                                                           RpptImagePatchPtr dstImgSize,
                                                           float intensityScaleFactor,
                                                           RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float4 intensityScaleFactor_f4 = (float4)intensityScaleFactor;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_scale_intensity_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
    dst_f8.f4[0] *= intensityScaleFactor_f4;
    dst_f8.f4[1] *= intensityScaleFactor_f4;
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        dst_f8.f4[0] *= intensityScaleFactor_f4;
        dst_f8.f4[1] *= intensityScaleFactor_f4;
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        dst_f8.f4[0] *= intensityScaleFactor_f4;
        dst_f8.f4[1] *= intensityScaleFactor_f4;
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

// -------------------- resize_scale_intensity kernel executor --------------------

template <typename T, typename U>
RppStatus hip_exec_resize_scale_intensity_tensor(T *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 U *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSize,
                                                 RpptInterpolationType interpolationType,
                                                 Rpp32f intensityScaleFactor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);
    
    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->layout != RpptLayout::NCHW)
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if (dstDescPtr->layout != RpptLayout::NCHW)
        return RPP_ERROR_INVALID_DST_LAYOUT;

    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    hipLaunchKernelGGL(resize_scale_intensity_bilinear_pln_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                       dstDescPtr->c,
                       dstImgSize,
                       intensityScaleFactor,
                       roiTensorPtrSrc);

    return RPP_SUCCESS;
}
