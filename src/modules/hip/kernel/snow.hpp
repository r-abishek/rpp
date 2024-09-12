#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rpp_cpu_common.hpp"

__device__ __forceinline__ void snow_1hueToRGB_hip_compute(float *p, float *q, float t, float *r)
{
    if(t < 0.0f) t += 1.0f;
    if(t > 1.0f) t -= 1.0f;
    if(t < ONE_OVER_6)
    {
        *r = fmaf(6.0f * t, (*q - *p), *p);
        // *r = *p + (*q - *p) * 6.0f * t;
    }
    else if(t < 0.5f)
    {
        *r = *q;
    }
    else if(t < TWO_OVER_3)
    {
        *r = fmaf((TWO_OVER_3 - t) * 6.0f, (*q - *p), *p);
        // *r = *p + (*q - *p) * (TWO_OVER_3 - t) * 6.0f;
    }
    else
    {
        *r = *p;
    }
}
__device__ __forceinline__ void snow_1RGB_hip_compute(float *pixelR, float *pixelG, float *pixelB, float *brightnessCoefficient, float *snowCoefficient)
{
    // RGB to HSL
    float hue, sat, l;
    float rf, gf, bf, cmax, cmin, delta;
    float lower_threshold = 0.0f;
    float upper_threshold = 0.39215686f;
    float brightnessFactor = 2.5f;

    rf = *pixelR;
    gf = *pixelG;
    bf = *pixelB;
    cmax = fmaxf(fmaxf(rf, gf), bf);
    cmin = fminf(fminf(rf, gf), bf);
    delta = cmax - cmin;
    hue = 0.0f;
    sat = 0.0f;
    l = (cmax + cmin) * 0.5f;
    if(delta != 0.0f)
    {
        //saturation calculation
        float mul = (l < 0.5) ? l : (1.0f - l);
        sat = delta / (mul * 2);
        // if(l <= 0.5)
        // {
        //     sat = delta / (cmax + cmin);
        // }
        // else
        // {
        //     sat = delta / (2.0f - cmax - cmin);
        // }
        
        // hue calculation
        if(cmax == rf)
        {
            hue = (gf - bf) / delta + (gf < bf ? 6 : 0);
        }
        else if(cmax == gf)
        {
            hue = 2.0f + (bf - rf) / delta;
        }
        else
        {
            hue = 4.0f + (rf - gf) / delta;
        }
        // switch(cmax) {
        //     case rf :{
        //         hue = (gf - bf) / delta + (gf < bf ? 6 : 0);
        //         break;
        //     }   
        //     case gf :{
        //         hue = 2.0f + (bf - rf) / delta;
        //         break;
        //     }
        //     case bf :{
        //         hue = 4.0f + (rf - gf) / delta;
        //         break;
        //     }
                    
        // }
        hue = hue * ONE_OVER_6;

    }

    //Lighter the darken images
    if(l >= lower_threshold && l <= upper_threshold)
    {
        l = l * fmaf((brightnessFactor - 1.0f), (1.0f - (l - lower_threshold) / (upper_threshold - lower_threshold)), 1.0f);
    }

    // Modify L 
    if(l <= *snowCoefficient && !((hue>=0.5 && hue <= 0.56) && (sat >= 0.196) && (l >= 0.196)))
    {
        l = l * (*brightnessCoefficient);
    }

    // HSL to RGB with brightness/contrast adjustment

    // if(sat == 0.0f)
    // {
    //     *pixelR = l;
    //     *pixelG = l;
    //     *pixelB = l;
    // }
    // else
    // {
    //     float p, q;
    //     q = l < 0.5f ? l * (1.0f + sat) : l + sat - l * sat;
    //     p = 2.0f * l - q;
    //     snow_1hueToRGB_hip_compute(&p, &q, (hue + ONE_OVER_3), pixelR);
    //     snow_1hueToRGB_hip_compute(&p, &q, hue, pixelG);
    //     snow_1hueToRGB_hip_compute(&p, &q, (hue - ONE_OVER_3), pixelB);
    // }
    float4 xt_f4 = make_float4(
        6.0f * (hue - TWO_OVER_3),
        0.0f,
        6.0f *(1.0 - hue),
        0.0f
    );
    if(hue < TWO_OVER_3)
    {
        xt_f4.x = 0.0f;
        xt_f4.y = 6.0f * (TWO_OVER_3 - hue);
        xt_f4.z = 6.0f * (hue - ONE_OVER_3);
    }
    if(hue < ONE_OVER_3)
    {
        xt_f4.x = 6.0f * (ONE_OVER_3 - hue);
        xt_f4.y = 6.0f * hue;
        xt_f4.z = 0.0f;
    }
    xt_f4.x = fminf(xt_f4.x, 1.0f);
    xt_f4.y = fminf(xt_f4.y, 1.0f);
    xt_f4.z = fminf(xt_f4.z, 1.0f);

    float sat2 = 2.0f * sat;
    float satinv = 1.0f - sat;
    float luminv = 1.0f - l;
    float lum2m1 = (2.0f * l) -1.0f;
    float4 ct_f4 = (sat2 * xt_f4) + satinv;

    float4 rgb_f4;
    if(l >= 0.5f)
    {
        rgb_f4 = (luminv * ct_f4) + lum2m1;
    }
    else
    {
        rgb_f4 = l * ct_f4;
    }
    *pixelR = rgb_f4.x;
    *pixelG = rgb_f4.y;
    *pixelB = rgb_f4.z;
}


__device__ __forceinline__ void snow_8RGB_hip_compute(d_float24 *pix_f24, float *brightnessCoefficient, float *snowCoefficient)
{
    snow_1RGB_hip_compute(&(pix_f24->f1[ 0]), &(pix_f24->f1[ 8]), &(pix_f24->f1[16]), brightnessCoefficient, snowCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 1]), &(pix_f24->f1[ 9]), &(pix_f24->f1[17]), brightnessCoefficient, snowCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 2]), &(pix_f24->f1[10]), &(pix_f24->f1[18]), brightnessCoefficient, snowCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 3]), &(pix_f24->f1[11]), &(pix_f24->f1[19]), brightnessCoefficient, snowCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 4]), &(pix_f24->f1[12]), &(pix_f24->f1[20]), brightnessCoefficient, snowCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 5]), &(pix_f24->f1[13]), &(pix_f24->f1[21]), brightnessCoefficient, snowCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 6]), &(pix_f24->f1[14]), &(pix_f24->f1[22]), brightnessCoefficient, snowCoefficient);
    snow_1RGB_hip_compute(&(pix_f24->f1[ 7]), &(pix_f24->f1[15]), &(pix_f24->f1[23]), brightnessCoefficient, snowCoefficient);
}

__device__ __forceinline__ void snow_hip_compute(uchar *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient, float *snowCoefficient)
{
    float4 normalizer_f4 = static_cast<float4>(ONE_OVER_255);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient, snowCoefficient);
    normalizer_f4 = static_cast<float4>(255.0f);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    rpp_hip_pixel_check_0to255(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(float *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient, float *snowCoefficient)
{
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient, snowCoefficient);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(half *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient, float *snowCoefficient)
{
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient, snowCoefficient);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(schar *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient, float *snowCoefficient)
{
    float4 i8Offset_f4 = static_cast<float4>(128.0f);
    float4 normalizer_f4 = static_cast<float4>(ONE_OVER_255);
    rpp_hip_math_add24_const(pix_f24, pix_f24, i8Offset_f4);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient, snowCoefficient);
    normalizer_f4 = static_cast<float4>(255.0f);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    rpp_hip_pixel_check_0to255(pix_f24);
    rpp_hip_math_subtract24_const(pix_f24, pix_f24, i8Offset_f4);
}

// template <typename T>
// __global__ void snow_pkd_hip_tensor(T *srcPtr,
//                                     uint2 srcStridesNH,
//                                     T *dstPtr,
//                                     uint2 dstStridesNH,
//                                     float *brightnessCoefficient,
//                                     float *snowCoefficient,
//                                     RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

//     d_float24 pix_f24;

//     rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
//     snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowCoefficient);
//     rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
// }

template <typename T>
__global__ void snow_pkd_hip_tensor(T *srcPtr,
                                    uint2 srcStridesNH,
                                    T *dstPtr,
                                    uint2 dstStridesNH,
                                    float *brightnessCoefficient,
                                    float *snowCoefficient,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    
    float r = static_cast<float>(*(srcPtr + srcIdx));
    float g = static_cast<float>(*(srcPtr + srcIdx + 1));
    float b = static_cast<float>(*(srcPtr + srcIdx + 2));
    r *= ONE_OVER_255;
    g *= ONE_OVER_255;
    b *= ONE_OVER_255;
    // printf("SRc : %d  Dst : %d\n ",(int)*(srcPtr + srcIdx),(int)*(dstPtr + dstIdx));
    snow_1RGB_hip_compute(&r, &b, &g, brightnessCoefficient, snowCoefficient);
    r *= 255; 
    g *= 255; 
    b *= 255; 
    rpp_hip_pixel_check_and_store(r, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(g, &dstPtr[dstIdx + 1]);
    rpp_hip_pixel_check_and_store(b, &dstPtr[dstIdx + 2]);

    // printf("SRc1 : %d  Dst1 : %d\n ",(int)*(srcPtr + srcIdx),(int)*(dstPtr + dstIdx));

    // d_float24 pix_f24;

    // rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    // snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowCoefficient);
    // rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

// template <typename T>
// __global__ void snow_pkd_hip_tensor(T *srcPtr,
//                                     uint2 srcStridesNH,
//                                     T *dstPtr,
//                                     uint2 dstStridesNH,
//                                     float *brightnessCoefficient,
//                                     float *snowCoefficient,
//                                     RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

//     d_float24 pix_f24;

//     rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
//     snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowCoefficient);
//     rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
// }

template <typename T>
__global__ void snow_pln_hip_tensor(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    T *dstPtr,
                                    uint3 dstStridesNCH,
                                    float *brightnessCoefficient,
                                    float *snowCoefficient,
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

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowCoefficient);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void snow_pkd3_pln3_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          float *brightnessCoefficient,
                                          float *snowCoefficient,
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

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowCoefficient);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void snow_pln3_pkd3_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float *brightnessCoefficient,
                                          float *snowCoefficient,
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

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowCoefficient);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_snow_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *brightnessCoefficient,
                               Rpp32f *snowCoefficient,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        int globalThreads_x = (dstDescPtr->w);
        // int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();
        *snowCoefficient = *snowCoefficient * (255.0f / 2.0f);
        *snowCoefficient = *snowCoefficient + (255.0f / 3.0f);

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            // globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
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
                               snowCoefficient,
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
                               snowCoefficient,
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
                               snowCoefficient,
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
                               snowCoefficient,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}