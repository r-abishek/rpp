#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rpp_cpu_common.hpp"

__device__ __forceinline__ void snow_1RGB_hip_compute(float *pixelR, float *pixelG, float *pixelB, float *brightnessCoefficient)
{
    // RGB to HSV

    float hue, sat, l, add;
    float rf, gf, bf, cmax, cmin, delta;
    rf = *pixelR;
    gf = *pixelG;
    bf = *pixelB;
    cmax = fmaxf(fmaxf(rf, gf), bf);
    cmin = fminf(fminf(rf, gf), bf);
    delta = cmax - cmin;
    hue = 0.0f;
    sat = 0.0f;
    add = 0.0f;
    l = (cmax + cmin) / 2.0f;
    if(cmax == cmin)
    {
        hue = 0.0f;
        sat = 0.0f;
    }
    else
    {
        //saturation calculation
        if(l <= 0.5)
        {
            sat = delta / (cmax + cmin);
        }
        else
        {
            sat = delta / (2.0f - cmax - cmin);
        }
        //hue calculation
        if(cmax == rf)
        {
            hue = (gf - bf) / delta + (g < b ? 6 : 0);
        }
        else if(cmax == gf)
        {
            hue = 2.0f + (bf - rf) / delta;
        }
        else
        {
            hue = 4.0f + (rf - gf) / delta;
        }
    }

    // Modify L 

    l = l * 2.5f;

    // HSV to RGB with brightness/contrast adjustment

    if(sat == 0.0f)
    {
        *pixelR = l;
        *pixelG = l;
        *pixelB = l;
    }
    else
    {
        float p, q;
        q = l < 0.5f ? l * (1.0f + sat) : l + sat - l * sat;
        p = 2.0f * l - q;
        snow_1hueToRGB_hip_compute(&p, &q, &(hue + 1/3), pixelR);
        snow_1hueToRGB_hip_compute(&p, &q, &hue, pixelG);
        snow_1hueToRGB_hip_compute(&p, &q, &(hue - 1/3), pixelB);
    }
}

__device__ __forceinline__ void snow_1hueToRGB_hip_compute(float *p, float *q, float *t, float *r)
{
    if(t < 0.0f) t += 1.0f;
    if(t > 1.0f) t -= 1.0f;
    if(t < 1.0f / 6.0f)
    {
        *r = *p + (*q - *p) * 6.0f * t;
    }
    else if(t < 1.0f / 2.0f)
    {
        *r = *q;
    }
    else if(t < 2.0f / 3.0f)
    {
        *r = *p + (*q - *p) * (2.0f / 3.0f - t) * 6.0f;
    }
    else
    {
        *r = *p;
    }

}

__device__ __forceinline__ void snow_8RGB_hip_compute(d_float24 *pix_f24, float *brightnessCoefficient)
{
    snow_1RGB_hip_compute(&(pix_f24->f1[ 0]), &(pix_f24->f1[ 8]), &(pix_f24->f1[16]), brightnessCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 1]), &(pix_f24->f1[ 9]), &(pix_f24->f1[17]), brightnessCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 2]), &(pix_f24->f1[10]), &(pix_f24->f1[18]), brightnessCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 3]), &(pix_f24->f1[11]), &(pix_f24->f1[19]), brightnessCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 4]), &(pix_f24->f1[12]), &(pix_f24->f1[20]), brightnessCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 5]), &(pix_f24->f1[13]), &(pix_f24->f1[21]), brightnessCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 6]), &(pix_f24->f1[14]), &(pix_f24->f1[22]), brightnessCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 7]), &(pix_f24->f1[15]), &(pix_f24->f1[23]), brightnessCoefficient);
}

__device__ __forceinline__ void snow_hip_compute(uchar *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient)
{
    float4 normalizer_f4 = static_cast<float4>(ONE_OVER_255);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient);
    rpp_hip_pixel_check_0to255(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(float *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient)
{
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(half *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient)
{
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(schar *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient)
{
    float4 i8Offset_f4 = static_cast<float4>(128.0f);
    float4 normalizer_f4 = static_cast<float4>(ONE_OVER_255);
    rpp_hip_math_add24_const(pix_f24, pix_f24, i8Offset_f4);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient);
    rpp_hip_pixel_check_0to255(pix_f24);
    rpp_hip_math_subtract24_const(pix_f24, pix_f24, i8Offset_f4);
}

template <typename T>
__global__ void snow_pkd_hip_tensor(T *srcPtr,
                                    uint2 srcStridesNH,
                                    T *dstPtr,
                                    uint2 dstStridesNH,
                                    float *brightnessCoefficient,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float brightnessCoefficient = 3.5f;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, &brightnessCoefficient);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void snow_pln_hip_tensor(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    T *dstPtr,
                                    uint3 dstStridesNCH,
                                    float *brightnessCoefficient,
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

    float4 colorTwistParams_f4 = make_float4(brightnessTensor[id_z], contrastTensor[id_z], hueTensor[id_z], saturationTensor[id_z]);

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, &colorTwistParams_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void snow_pkd3_pln3_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          float *brightnessCoefficient,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float4 colorTwistParams_f4 = make_float4(brightnessTensor[id_z], contrastTensor[id_z], hueTensor[id_z], saturationTensor[id_z]);

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, &colorTwistParams_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void snow_pln3_pkd3_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float *brightnessCoefficient,
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
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float4 colorTwistParams_f4 = make_float4(brightnessTensor[id_z], contrastTensor[id_z], hueTensor[id_z], saturationTensor[id_z]);

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, &colorTwistParams_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_snow_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rppt32f *brightnessCoefficient,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
            hipLaunchKernelGGL(snow_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               brightnessCoefficient,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(snow_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               brightnessCoefficient,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(snow_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               brightnessCoefficient,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(snow_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               brightnessCoefficient,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}