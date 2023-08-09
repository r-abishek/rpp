#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rpp_cpu_common.hpp"

__device__ void hsv_to_rgb_1RGB_hip_compute(float *pixelHR, float *pixelSG, float *pixelVB)
{
    float hue = *pixelHR * 6.0f;
    float sat = *pixelSG;
    float v = *pixelVB;
    float3 rgb_f3;

    // HSV to RGB
    int hueIntegerPart = (int) hue;
    float hueFractionPart = hue - hueIntegerPart;
    float vsat = v * sat;
    float vsatf = vsat * hueFractionPart;
    float p = v - vsat;
    float q = v - vsatf;
    float t = v - vsat + vsatf;
    switch (hueIntegerPart)
    {
        case 0: rgb_f3.x = v; rgb_f3.y = t; rgb_f3.z = p; break;
        case 1: rgb_f3.x = q; rgb_f3.y = v; rgb_f3.z = p; break;
        case 2: rgb_f3.x = p; rgb_f3.y = v; rgb_f3.z = t; break;
        case 3: rgb_f3.x = p; rgb_f3.y = q; rgb_f3.z = v; break;
        case 4: rgb_f3.x = t; rgb_f3.y = p; rgb_f3.z = v; break;
        case 5: rgb_f3.x = v; rgb_f3.y = p; rgb_f3.z = q; break;
    }
    rgb_f3 *= (float3) 255.0f;
    *pixelHR = rgb_f3.x;
    *pixelSG = rgb_f3.y;
    *pixelVB = rgb_f3.z;
}

__device__ void hsv_to_bgr_1BGR_hip_compute(float *pixelHB, float *pixelSG, float *pixelVR)
{
    float hue = *pixelHB * 6.0f;
    float sat = *pixelSG;
    float v = *pixelVR;
    float3 bgr_f3;

    // HSV to BGR
    int hueIntegerPart = (int) hue;
    float hueFractionPart = hue - hueIntegerPart;
    float vsat = v * sat;
    float vsatf = vsat * hueFractionPart;
    float p = v - vsat;
    float q = v - vsatf;
    float t = v - vsat + vsatf;
    switch (hueIntegerPart)
    {
        case 0: bgr_f3.z = v; bgr_f3.y = t; bgr_f3.x = p; break;
        case 1: bgr_f3.z = q; bgr_f3.y = v; bgr_f3.x = p; break;
        case 2: bgr_f3.z = p; bgr_f3.y = v; bgr_f3.x = t; break;
        case 3: bgr_f3.z = p; bgr_f3.y = q; bgr_f3.x = v; break;
        case 4: bgr_f3.z = t; bgr_f3.y = p; bgr_f3.x = v; break;
        case 5: bgr_f3.z = v; bgr_f3.y = p; bgr_f3.x = q; break;
    }
    bgr_f3 *= (float3) 255.0f;
    *pixelHB = bgr_f3.x;
    *pixelSG = bgr_f3.y;
    *pixelVR = bgr_f3.z;
}

__device__ void hsv_to_rgb_8RGB_hip_compute(d_float24 *pix_f24)
{
    hsv_to_rgb_1RGB_hip_compute(&(pix_f24->f1[ 0]), &(pix_f24->f1[ 8]), &(pix_f24->f1[16]));
    hsv_to_rgb_1RGB_hip_compute(&(pix_f24->f1[ 1]), &(pix_f24->f1[ 9]), &(pix_f24->f1[17]));
    hsv_to_rgb_1RGB_hip_compute(&(pix_f24->f1[ 2]), &(pix_f24->f1[10]), &(pix_f24->f1[18]));
    hsv_to_rgb_1RGB_hip_compute(&(pix_f24->f1[ 3]), &(pix_f24->f1[11]), &(pix_f24->f1[19]));
    hsv_to_rgb_1RGB_hip_compute(&(pix_f24->f1[ 4]), &(pix_f24->f1[12]), &(pix_f24->f1[20]));
    hsv_to_rgb_1RGB_hip_compute(&(pix_f24->f1[ 5]), &(pix_f24->f1[13]), &(pix_f24->f1[21]));
    hsv_to_rgb_1RGB_hip_compute(&(pix_f24->f1[ 6]), &(pix_f24->f1[14]), &(pix_f24->f1[22]));
    hsv_to_rgb_1RGB_hip_compute(&(pix_f24->f1[ 7]), &(pix_f24->f1[15]), &(pix_f24->f1[23]));
}

__device__ void hsv_to_bgr_8BGR_hip_compute(d_float24 *pix_f24)
{
    hsv_to_bgr_1BGR_hip_compute(&(pix_f24->f1[ 0]), &(pix_f24->f1[ 8]), &(pix_f24->f1[16]));
    hsv_to_bgr_1BGR_hip_compute(&(pix_f24->f1[ 1]), &(pix_f24->f1[ 9]), &(pix_f24->f1[17]));
    hsv_to_bgr_1BGR_hip_compute(&(pix_f24->f1[ 2]), &(pix_f24->f1[10]), &(pix_f24->f1[18]));
    hsv_to_bgr_1BGR_hip_compute(&(pix_f24->f1[ 3]), &(pix_f24->f1[11]), &(pix_f24->f1[19]));
    hsv_to_bgr_1BGR_hip_compute(&(pix_f24->f1[ 4]), &(pix_f24->f1[12]), &(pix_f24->f1[20]));
    hsv_to_bgr_1BGR_hip_compute(&(pix_f24->f1[ 5]), &(pix_f24->f1[13]), &(pix_f24->f1[21]));
    hsv_to_bgr_1BGR_hip_compute(&(pix_f24->f1[ 6]), &(pix_f24->f1[14]), &(pix_f24->f1[22]));
    hsv_to_bgr_1BGR_hip_compute(&(pix_f24->f1[ 7]), &(pix_f24->f1[15]), &(pix_f24->f1[23]));
}

template <typename T, typename U>
__global__ void hsv_to_rgb_pkd3_tensor(T *srcPtr,
                                       uint2 srcStridesNH,
                                       U *dstPtr,
                                       uint2 dstStridesNH,
                                       uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    hsv_to_rgb_8RGB_hip_compute(&pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T, typename U>
__global__ void hsv_to_rgb_pln3_tensor(T *srcPtr,
                                       uint3 srcStridesNCH,
                                       U *dstPtr,
                                       uint3 dstStridesNCH,
                                       uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + id_x;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    hsv_to_rgb_8RGB_hip_compute(&pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T, typename U>
__global__ void hsv_to_rgb_pkd3_pln3_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            U *dstPtr,
                                            uint3 dstStridesNCH,
                                            uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    hsv_to_rgb_8RGB_hip_compute(&pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T, typename U>
__global__ void hsv_to_rgb_pln3_pkd3_tensor(T *srcPtr,
                                            uint3 srcStridesNCH,
                                            U *dstPtr,
                                            uint2 dstStridesNH,
                                            uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    hsv_to_rgb_8RGB_hip_compute(&pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T, typename U>
__global__ void hsv_to_bgr_pkd3_tensor(T *srcPtr,
                                       uint2 srcStridesNH,
                                       U *dstPtr,
                                       uint2 dstStridesNH,
                                       uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    hsv_to_bgr_8BGR_hip_compute(&pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T, typename U>
__global__ void hsv_to_bgr_pln3_tensor(T *srcPtr,
                                       uint3 srcStridesNCH,
                                       U *dstPtr,
                                       uint3 dstStridesNCH,
                                       uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + id_x;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    hsv_to_bgr_8BGR_hip_compute(&pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T, typename U>
__global__ void hsv_to_bgr_pkd3_pln3_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            U *dstPtr,
                                            uint3 dstStridesNCH,
                                            uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    hsv_to_bgr_8BGR_hip_compute(&pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T, typename U>
__global__ void hsv_to_bgr_pln3_pkd3_tensor(T *srcPtr,
                                            uint3 srcStridesNCH,
                                            U *dstPtr,
                                            uint2 dstStridesNH,
                                            uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    hsv_to_bgr_8BGR_hip_compute(&pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T, typename U>
RppStatus hip_exec_hsv_to_rgbbgr_tensor(T *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        U *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptSubpixelLayout dstSubpixelLayout,
                                        rpp::Handle& handle)
{
    if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        int localThreads_x = LOCAL_THREADS_X;
        int localThreads_y = LOCAL_THREADS_Y;
        int localThreads_z = LOCAL_THREADS_Z;
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if (dstSubpixelLayout == RpptSubpixelLayout::RGBtype)
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
                hipLaunchKernelGGL(hsv_to_rgb_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   make_uint2(srcDescPtr->w, srcDescPtr->h));
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(hsv_to_rgb_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   make_uint2(srcDescPtr->w, srcDescPtr->h));
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(hsv_to_rgb_pkd3_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   make_uint2(srcDescPtr->w, srcDescPtr->h));
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
                hipLaunchKernelGGL(hsv_to_rgb_pln3_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   make_uint2(srcDescPtr->w, srcDescPtr->h));
            }
        }
        else if (dstSubpixelLayout == RpptSubpixelLayout::BGRtype)
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
                hipLaunchKernelGGL(hsv_to_bgr_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   make_uint2(srcDescPtr->w, srcDescPtr->h));
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(hsv_to_bgr_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   make_uint2(srcDescPtr->w, srcDescPtr->h));
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(hsv_to_bgr_pkd3_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   make_uint2(srcDescPtr->w, srcDescPtr->h));
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
                hipLaunchKernelGGL(hsv_to_bgr_pln3_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   make_uint2(srcDescPtr->w, srcDescPtr->h));
            }
        }
    }

    return RPP_SUCCESS;
}
