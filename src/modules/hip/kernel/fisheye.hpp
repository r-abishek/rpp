#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ __constant__ float4 TWO_F4 = static_cast<float4>(2.0);
__device__ __constant__ float4 ONE_F4 = static_cast<float4>(1.0);

// -------------------- Set 0 - fisheye kernel device helpers --------------------

__device__ void fisheye_srcloc_hip_compute(int index, int2 *widthHeight_i2, int4 *srcRoiPtr_i4, d_float8 *normX_f8,
                                           d_float8 *normY_f8, d_float8 *dist_f8, d_float16 *locSrc_f16)
{
    float dist = dist_f8->f1[index];
    if ((dist >= 0.0) && (dist <= 1.0))
    {
        float distNew = sqrtf(1.0 - dist * dist);
        distNew = (dist + (1.0 - distNew)) * 0.5f;
        if (distNew <= 1.0)
        {
            float theta = atan2f(normY_f8->f1[index], normX_f8->f1[index]);
            float newX = distNew * cosf(theta);
            float newY = distNew * sinf(theta);
            locSrc_f16->f8[0].f1[index] = (((newX + 1) * widthHeight_i2->x) * 0.5f) + static_cast<float>(srcRoiPtr_i4->x);
            locSrc_f16->f8[1].f1[index] = (((newY + 1) * widthHeight_i2->y) * 0.5f) + static_cast<float>(srcRoiPtr_i4->y);
        }
    }
}

__device__ void fisheye_roi_and_srclocs_hip_compute(int2 *idxy_i2, int2 *widthHeight_i2, int4 *srcRoiPtr_i4, d_float16 *locSrc_f16)
{
    d_float8 normY_f8, normX_f8, dist_f8, increment_f8;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);

    // compute the normalized x and y coordinates
    normY_f8.f4[0] = static_cast<float4>(((static_cast<float>(2 * idxy_i2->y) / widthHeight_i2->y)) - 1);
    normY_f8.f4[1] = normY_f8.f4[0];    
    normX_f8.f4[0] = (TWO_F4 * (static_cast<float4>(idxy_i2->x) + increment_f8.f4[0]) / static_cast<float4>(widthHeight_i2->x)) - ONE_F4;
    normX_f8.f4[1] = (TWO_F4 * (static_cast<float4>(idxy_i2->x) + increment_f8.f4[1]) / static_cast<float4>(widthHeight_i2->x)) - ONE_F4;
    
    // compute the euclidean distance using the normalized x and y coordinates
    dist_f8.f4[0] = ((normX_f8.f4[0] * normX_f8.f4[0]) + (normY_f8.f4[0] * normY_f8.f4[0]));
    dist_f8.f4[1] = ((normX_f8.f4[1] * normX_f8.f4[1]) + (normY_f8.f4[1] * normY_f8.f4[1]));
    rpp_hip_math_sqrt8(&dist_f8, &dist_f8);

    // compute src locations for the given dst locations
    fisheye_srcloc_hip_compute(0, widthHeight_i2, srcRoiPtr_i4, &normX_f8, &normY_f8, &dist_f8, locSrc_f16);
    fisheye_srcloc_hip_compute(1, widthHeight_i2, srcRoiPtr_i4, &normX_f8, &normY_f8, &dist_f8, locSrc_f16);
    fisheye_srcloc_hip_compute(2, widthHeight_i2, srcRoiPtr_i4, &normX_f8, &normY_f8, &dist_f8, locSrc_f16);
    fisheye_srcloc_hip_compute(3, widthHeight_i2, srcRoiPtr_i4, &normX_f8, &normY_f8, &dist_f8, locSrc_f16);
    fisheye_srcloc_hip_compute(4, widthHeight_i2, srcRoiPtr_i4, &normX_f8, &normY_f8, &dist_f8, locSrc_f16);
    fisheye_srcloc_hip_compute(5, widthHeight_i2, srcRoiPtr_i4, &normX_f8, &normY_f8, &dist_f8, locSrc_f16);
    fisheye_srcloc_hip_compute(6, widthHeight_i2, srcRoiPtr_i4, &normX_f8, &normY_f8, &dist_f8, locSrc_f16);
    fisheye_srcloc_hip_compute(7, widthHeight_i2, srcRoiPtr_i4, &normX_f8, &normY_f8, &dist_f8, locSrc_f16);
}

// -------------------- Set 1 - fisheye kernels --------------------

template <typename T>
__global__ void fisheye_pkd_hip_tensor(T *srcPtr,
                                       uint2 srcStridesNH,
                                       T *dstPtr,
                                       uint2 dstStridesNH,
                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int width = (srcRoi_i4.z - srcRoi_i4.x) + 1;
    int height =  (srcRoi_i4.w - srcRoi_i4.y) + 1;

    if ((id_y >= height) || (id_x >= width))
        return;

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    int2 idxy_i2 = make_int2(id_x, id_y);
    int2 widthHeight_i2 = make_int2(width, height);
   
    // initialize the src location values with invalid values
    d_float16 locSrc_f16;
    locSrc_f16.f8[0].f4[0] = static_cast<float4>(-1);
    locSrc_f16.f8[0].f4[1] = static_cast<float4>(-1);
    locSrc_f16.f8[1].f4[0] = static_cast<float4>(-1);
    locSrc_f16.f8[1].f4[1] = static_cast<float4>(-1);
    
    // compute the src location values for the given dst locations
    fisheye_roi_and_srclocs_hip_compute(&idxy_i2, &widthHeight_i2, &srcRoi_i4, &locSrc_f16);
    
    d_float24 pix_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &pix_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void fisheye_pln_hip_tensor(T *srcPtr,
                                       uint3 srcStridesNCH,
                                       T *dstPtr,
                                       uint3 dstStridesNCH,
                                       int channelsDst,
                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int width = (srcRoi_i4.z - srcRoi_i4.x) + 1;
    int height =  (srcRoi_i4.w - srcRoi_i4.y) + 1;

    if ((id_y >= height) || (id_x >= width))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int2 idxy_i2 = make_int2(id_x, id_y);
    int2 widthHeight_i2 = make_int2(width, height);
   
    // initialize the src location values with invalid values
    d_float16 locSrc_f16;
    locSrc_f16.f8[0].f4[0] = static_cast<float4>(-1);
    locSrc_f16.f8[0].f4[1] = static_cast<float4>(-1);
    locSrc_f16.f8[1].f4[0] = static_cast<float4>(-1);
    locSrc_f16.f8[1].f4[1] = static_cast<float4>(-1);
    
    // compute the src location values for the given dst locations
    fisheye_roi_and_srclocs_hip_compute(&idxy_i2, &widthHeight_i2, &srcRoi_i4, &locSrc_f16);

    d_float8 pix_f8;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &pix_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);    
    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
    }
}

template <typename T>
__global__ void fisheye_pkd3_pln3_hip_tensor(T *srcPtr,
                                             uint2 srcStridesNH,
                                             T *dstPtr,
                                             uint3 dstStridesNCH,
                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int width = (srcRoi_i4.z - srcRoi_i4.x) + 1;
    int height =  (srcRoi_i4.w - srcRoi_i4.y) + 1;

    if ((id_y >= height) || (id_x >= width))
        return;

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int2 idxy_i2 = make_int2(id_x, id_y);
    int2 widthHeight_i2 = make_int2(width, height);
   
    // initialize the src location values with invalid values
    d_float16 locSrc_f16;
    locSrc_f16.f8[0].f4[0] = static_cast<float4>(-1);
    locSrc_f16.f8[0].f4[1] = static_cast<float4>(-1);
    locSrc_f16.f8[1].f4[0] = static_cast<float4>(-1);
    locSrc_f16.f8[1].f4[1] = static_cast<float4>(-1);
    
    // compute the src location values for the given dst locations
    fisheye_roi_and_srclocs_hip_compute(&idxy_i2, &widthHeight_i2, &srcRoi_i4, &locSrc_f16);

    d_float24 pix_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &pix_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);    
}

template <typename T>
__global__ void fisheye_pln3_pkd3_hip_tensor(T *srcPtr,
                                             uint3 srcStridesNCH,
                                             T *dstPtr,
                                             uint2 dstStridesNH,
                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int width = (srcRoi_i4.z - srcRoi_i4.x) + 1;
    int height =  (srcRoi_i4.w - srcRoi_i4.y) + 1;

    if ((id_y >= height) || (id_x >= width))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    int2 idxy_i2 = make_int2(id_x, id_y);
    int2 widthHeight_i2 = make_int2(width, height);
   
    // initialize the src location values with invalid values
    d_float16 locSrc_f16;
    locSrc_f16.f8[0].f4[0] = static_cast<float4>(-1);
    locSrc_f16.f8[0].f4[1] = static_cast<float4>(-1);
    locSrc_f16.f8[1].f4[0] = static_cast<float4>(-1);
    locSrc_f16.f8[1].f4[1] = static_cast<float4>(-1);
    
    // compute the src location values for the given dst locations
    fisheye_roi_and_srclocs_hip_compute(&idxy_i2, &widthHeight_i2, &srcRoi_i4, &locSrc_f16);

    d_float24 pix_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

// -------------------- Set 2 - fisheye kernels executor --------------------

template <typename T>
RppStatus hip_exec_fisheye_tensor(T *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  T *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    Rpp32s globalThreads_x = (dstDescPtr->w + 7) >> 3;
    Rpp32s globalThreads_y = dstDescPtr->h;
    Rpp32s globalThreads_z = dstDescPtr->n;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(fisheye_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(fisheye_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(fisheye_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(fisheye_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}