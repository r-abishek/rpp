#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - gaussian_filter device helpers --------------------

__device__ void gaussian_filter_f32_3x3_row_hip_compute(d_float10 *srcPtr_f10, d_float8 *dst_f8, float *filter)
{
    d_float10 src_f10 = *srcPtr_f10;
    // float src_f1;
    // uint3 src_ui3;
    // src_ui3 = *(uint3 *)srcPtr;
    // src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f10.f1[0], filter[0], dst_f8->f1[0]);
    // src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f10.f1[1], filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f10.f1[1], filter[0], dst_f8->f1[1]);
    // src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f10.f1[2], filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f10.f1[2], filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f10.f1[2], filter[0], dst_f8->f1[2]);
    // src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[1] = fmaf(src_f10.f1[3], filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f10.f1[3], filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f10.f1[3], filter[0], dst_f8->f1[3]);
    // src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f10.f1[4], filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f10.f1[4], filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f10.f1[4], filter[0], dst_f8->f1[4]);
    // src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f10.f1[5], filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f10.f1[5], filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f10.f1[5], filter[0], dst_f8->f1[5]);
    // src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[4] = fmaf(src_f10.f1[6], filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f10.f1[6], filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f10.f1[6], filter[0], dst_f8->f1[6]);
    // src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[5] = fmaf(src_f10.f1[7], filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f10.f1[7], filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f10.f1[7], filter[0], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f10.f1[8], filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f10.f1[8], filter[1], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f10.f1[9], filter[2], dst_f8->f1[7]);
}

__device__ void gaussian_filter_f32_5x5_row_hip_compute(d_float12 *srcPtr_f12, d_float8 *dst_f8, float *filter)
{
    d_float12 src_f12 = *srcPtr_f12;
    // float src_f1;
    // uint3 src_ui3;
    // src_ui3 = *(uint3 *)srcPtr;
    // src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f12.f1[0], filter[0], dst_f8->f1[0]);
    // src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f12.f1[1], filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f12.f1[1], filter[0], dst_f8->f1[1]);
    // src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f12.f1[2], filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f12.f1[2], filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f12.f1[2], filter[0], dst_f8->f1[2]);
    // src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f12.f1[3], filter[3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f12.f1[3], filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f12.f1[3], filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f12.f1[3], filter[0], dst_f8->f1[3]);
    // src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[0] = fmaf(src_f12.f1[4], filter[4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f12.f1[4], filter[3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f12.f1[4], filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f12.f1[4], filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f12.f1[4], filter[0], dst_f8->f1[4]);
    // src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[1] = fmaf(src_f12.f1[5], filter[4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f12.f1[5], filter[3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f12.f1[5], filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f12.f1[5], filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f12.f1[5], filter[0], dst_f8->f1[5]);
    // src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f12.f1[6], filter[4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f12.f1[6], filter[3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f12.f1[6], filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f12.f1[6], filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f12.f1[6], filter[0], dst_f8->f1[6]);
    // src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f12.f1[7], filter[4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f12.f1[7], filter[3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f12.f1[7], filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f12.f1[7], filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f12.f1[7], filter[0], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[4] = fmaf(src_f12.f1[8], filter[4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f12.f1[8], filter[3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f12.f1[8], filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f12.f1[8], filter[1], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[5] = fmaf(src_f12.f1[9], filter[4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f12.f1[9], filter[3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f12.f1[9], filter[2], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack2(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f12.f1[10], filter[4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f12.f1[10], filter[3], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack3(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f12.f1[11], filter[4], dst_f8->f1[7]);
}

__device__ void gaussian_filter_f32_7x7_row_hip_compute(d_float14 *srcPtr_f14, d_float8 *dst_f8, float *filter)
{
    d_float14 src_f14 = *srcPtr_f14;
    // float src_f1;
    // uint4 src_ui4 = *(uint4 *)srcPtr;
    // src_f1 = rpp_hip_unpack0(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f14.f1[0], filter[0], dst_f8->f1[0]);
    // src_f1 = rpp_hip_unpack1(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f14.f1[1], filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[1], filter[0], dst_f8->f1[1]);
    // src_f1 = rpp_hip_unpack2(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f14.f1[2], filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[2], filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[2], filter[0], dst_f8->f1[2]);
    // src_f1 = rpp_hip_unpack3(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f14.f1[3], filter[3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[3], filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[3], filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[3], filter[0], dst_f8->f1[3]);
    // src_f1 = rpp_hip_unpack0(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f14.f1[4], filter[4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[4], filter[3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[4], filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[4], filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[4], filter[0], dst_f8->f1[4]);
    // src_f1 = rpp_hip_unpack1(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f14.f1[5], filter[5], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[5], filter[4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[5], filter[3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[5], filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[5], filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[5], filter[0], dst_f8->f1[5]);
    // src_f1 = rpp_hip_unpack2(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f14.f1[6], filter[6], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[6], filter[5], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[6], filter[4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[6], filter[3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[6], filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[6], filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[6], filter[0], dst_f8->f1[6]);
    // src_f1 = rpp_hip_unpack3(src_ui4.y);
    dst_f8->f1[1] = fmaf(src_f14.f1[7], filter[6], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[7], filter[5], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[7], filter[4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[7], filter[3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[7], filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[7], filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[7], filter[0], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack0(src_ui4.z);
    dst_f8->f1[2] = fmaf(src_f14.f1[8], filter[6], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[8], filter[5], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[8], filter[4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[8], filter[3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[8], filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[8], filter[1], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack1(src_ui4.z);
    dst_f8->f1[3] = fmaf(src_f14.f1[9], filter[6], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[9], filter[5], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[9], filter[4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[9], filter[3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[9], filter[2], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack2(src_ui4.z);
    dst_f8->f1[4] = fmaf(src_f14.f1[10], filter[6], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[10], filter[5], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[10], filter[4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[10], filter[3], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack3(src_ui4.z);
    dst_f8->f1[5] = fmaf(src_f14.f1[11], filter[6], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[11], filter[5], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[11], filter[4], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack0(src_ui4.w);
    dst_f8->f1[6] = fmaf(src_f14.f1[12], filter[6], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[12], filter[5], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack1(src_ui4.w);
    dst_f8->f1[7] = fmaf(src_f14.f1[13], filter[6], dst_f8->f1[7]);
}

__device__ void gaussian_filter_f32_9x9_row_hip_compute(d_float16 *srcPtr_f16, d_float8 *dst_f8, float *filter)
{
    d_float16 src_f16 = *srcPtr_f16;
    // float src_f1;
    // uint4 src_ui4 = *(uint4 *)srcPtr;
    // src_f1 = rpp_hip_unpack0(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f16.f1[0], filter[0], dst_f8->f1[0]);
    // src_f1 = rpp_hip_unpack1(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f16.f1[1], filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[1], filter[0], dst_f8->f1[1]);
    // src_f1 = rpp_hip_unpack2(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f16.f1[2], filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[2], filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[2], filter[0], dst_f8->f1[2]);
    // src_f1 = rpp_hip_unpack3(src_ui4.x);
    dst_f8->f1[0] = fmaf(src_f16.f1[3], filter[3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[3], filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[3], filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[3], filter[0], dst_f8->f1[3]);
    // src_f1 = rpp_hip_unpack0(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f16.f1[4], filter[4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[4], filter[3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[4], filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[4], filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[4], filter[0], dst_f8->f1[4]);
    // src_f1 = rpp_hip_unpack1(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f16.f1[5], filter[5], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[5], filter[4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[5], filter[3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[5], filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[5], filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[5], filter[0], dst_f8->f1[5]);
    // src_f1 = rpp_hip_unpack2(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f16.f1[6], filter[6], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[6], filter[5], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[6], filter[4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[6], filter[3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[6], filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[6], filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[6], filter[0], dst_f8->f1[6]);
    // src_f1 = rpp_hip_unpack3(src_ui4.y);
    dst_f8->f1[0] = fmaf(src_f16.f1[7], filter[7], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[7], filter[6], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[7], filter[5], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[7], filter[4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[7], filter[3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[7], filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[7], filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[7], filter[0], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack0(src_ui4.z);
    dst_f8->f1[0] = fmaf(src_f16.f1[8], filter[8], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[8], filter[7], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[8], filter[6], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[8], filter[5], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[8], filter[4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[8], filter[3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[8], filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[8], filter[1], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack1(src_ui4.z);
    dst_f8->f1[1] = fmaf(src_f16.f1[9], filter[8], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[9], filter[7], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[9], filter[6], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[9], filter[5], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[9], filter[4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[9], filter[3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[9], filter[2], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack2(src_ui4.z);
    dst_f8->f1[2] = fmaf(src_f16.f1[10], filter[8], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[10], filter[7], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[10], filter[6], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[10], filter[5], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[10], filter[4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[10], filter[3], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack3(src_ui4.z);
    dst_f8->f1[3] = fmaf(src_f16.f1[11], filter[8], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[11], filter[7], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[11], filter[6], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[11], filter[5], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[11], filter[4], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack0(src_ui4.w);
    dst_f8->f1[4] = fmaf(src_f16.f1[12], filter[8], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[12], filter[7], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[12], filter[6], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[12], filter[5], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack1(src_ui4.w);
    dst_f8->f1[5] = fmaf(src_f16.f1[13], filter[8], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[13], filter[7], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[13], filter[6], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack2(src_ui4.w);
    dst_f8->f1[6] = fmaf(src_f16.f1[14], filter[8], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[14], filter[7], dst_f8->f1[7]);
    // src_f1 = rpp_hip_unpack3(src_ui4.w);
    dst_f8->f1[7] = fmaf(src_f16.f1[15], filter[8], dst_f8->f1[7]);
}

// -------------------- Set 1 - PLN1->PLN1 for F32 without saturation check --------------------

// kernelSize = 3
template <typename T>
__global__ void gaussian_filter_3x3_pln_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float9 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ float src_lds[16][128];

    // OLD correct code
    // int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    // d_float9 filter_f9 = filterTensor[id_z];
    // float *filter_row1 = &filter_f9.f1[0];
    // float *filter_row2 = &filter_f9.f1[3];
    // float *filter_row3 = &filter_f9.f1[6];
    // sum_f8.f4[0] = (float4) 0;
    // sum_f8.f4[1] = (float4) 0;
    // if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    //     (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    // else
    //     *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;

    // NEW experimental code
    int srcIdx = (id_z * srcStridesNCH.x) +
                 (std::min(std::max(id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y, 0), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
                //  (std::min(std::max(id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x, 0), roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float9 filter_f9 = filterTensor[id_z];
    float *filter_row1 = &filter_f9.f1[0];
    float *filter_row2 = &filter_f9.f1[3];
    float *filter_row3 = &filter_f9.f1[6];
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    float *srcPos_lds = &src_lds[hipThreadIdx_y][hipThreadIdx_x8];
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, (d_float8 *)srcPos_lds);
    if (id_x_i < 0)
        for(int i = 0; i < -id_x_i; i++)
            srcPos_lds[i] = srcPos_lds[-id_x_i];
    int diff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x_i;
    if ((diff >= 0) && (diff < 8))
        for(int i = diff; i < 8; i++)
            srcPos_lds[i] = srcPos_lds[diff - 1];


    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_f32_3x3_row_hip_compute((d_float10 *)&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        gaussian_filter_f32_3x3_row_hip_compute((d_float10 *)&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        gaussian_filter_f32_3x3_row_hip_compute((d_float10 *)&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

// kernelSize = 5
template <typename T>
__global__ void gaussian_filter_5x5_pln_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float25 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ float src_lds[16][128];

    // OLD correct code
    // int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    // d_float25 filter_f25 = filterTensor[id_z];
    // float *filter_row1 = &filter_f25.f1[0];
    // float *filter_row2 = &filter_f25.f1[5];
    // float *filter_row3 = &filter_f25.f1[10];
    // float *filter_row4 = &filter_f25.f1[15];
    // float *filter_row5 = &filter_f25.f1[20];
    // sum_f8.f4[0] = (float4) 0;
    // sum_f8.f4[1] = (float4) 0;
    // if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    //     (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    // else
    //     *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;

    // NEW experimental code
    int srcIdx = (id_z * srcStridesNCH.x) +
                 (std::min(std::max(id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y, 0), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
                //  (std::min(std::max(id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x, 0), roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float25 filter_f25 = filterTensor[id_z];
    float *filter_row1 = &filter_f25.f1[0];
    float *filter_row2 = &filter_f25.f1[5];
    float *filter_row3 = &filter_f25.f1[10];
    float *filter_row4 = &filter_f25.f1[15];
    float *filter_row5 = &filter_f25.f1[20];
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    float *srcPos_lds = &src_lds[hipThreadIdx_y][hipThreadIdx_x8];
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, (d_float8 *)srcPos_lds);
    if (id_x_i < 0)
        for(int i = 0; i < -id_x_i; i++)
            srcPos_lds[i] = srcPos_lds[-id_x_i];
    int diff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x_i;
    if ((diff >= 0) && (diff < 8))
        for(int i = diff; i < 8; i++)
            srcPos_lds[i] = srcPos_lds[diff - 1];

    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        gaussian_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        gaussian_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        gaussian_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        gaussian_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

// kernelSize = 7
template <typename T>
__global__ void gaussian_filter_7x7_pln_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float49 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ float src_lds[16][128];

    // OLD correct code
    // int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    // d_float49 filter_f49 = filterTensor[id_z];
    // float *filter_row1 = &filter_f49.f1[0];
    // float *filter_row2 = &filter_f49.f1[7];
    // float *filter_row3 = &filter_f49.f1[14];
    // float *filter_row4 = &filter_f49.f1[21];
    // float *filter_row5 = &filter_f49.f1[28];
    // float *filter_row6 = &filter_f49.f1[35];
    // float *filter_row7 = &filter_f49.f1[42];
    // sum_f8.f4[0] = (float4) 0;
    // sum_f8.f4[1] = (float4) 0;
    // if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    //     (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    // else
    //     *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;

    // NEW experimental code
    int srcIdx = (id_z * srcStridesNCH.x) +
                 (std::min(std::max(id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y, 0), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
                //  (std::min(std::max(id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x, 0), roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float49 filter_f49 = filterTensor[id_z];
    float *filter_row1 = &filter_f49.f1[0];
    float *filter_row2 = &filter_f49.f1[7];
    float *filter_row3 = &filter_f49.f1[14];
    float *filter_row4 = &filter_f49.f1[21];
    float *filter_row5 = &filter_f49.f1[28];
    float *filter_row6 = &filter_f49.f1[35];
    float *filter_row7 = &filter_f49.f1[42];
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    float *srcPos_lds = &src_lds[hipThreadIdx_y][hipThreadIdx_x8];
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, (d_float8 *)srcPos_lds);
    if (id_x_i < 0)
        for(int i = 0; i < -id_x_i; i++)
            srcPos_lds[i] = srcPos_lds[-id_x_i];
    int diff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x_i;
    if ((diff >= 0) && (diff < 8))
        for(int i = diff; i < 8; i++)
            srcPos_lds[i] = srcPos_lds[diff - 1];

    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        gaussian_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        gaussian_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        gaussian_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        gaussian_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        gaussian_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
        gaussian_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

// kernelSize = 9
template <typename T>
__global__ void gaussian_filter_9x9_pln_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               d_float81 *filterTensor)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ float src_lds[16][128];

    // OLD correct code
    // int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    // d_float81 filter_f81 = filterTensor[id_z];
    // float *filter_row1 = &filter_f81.f1[0];
    // float *filter_row2 = &filter_f81.f1[9];
    // float *filter_row3 = &filter_f81.f1[18];
    // float *filter_row4 = &filter_f81.f1[27];
    // float *filter_row5 = &filter_f81.f1[36];
    // float *filter_row6 = &filter_f81.f1[45];
    // float *filter_row7 = &filter_f81.f1[54];
    // float *filter_row8 = &filter_f81.f1[63];
    // float *filter_row9 = &filter_f81.f1[72];
    // sum_f8.f4[0] = (float4) 0;
    // sum_f8.f4[1] = (float4) 0;
    // if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    //     (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_lds[hipThreadIdx_y][hipThreadIdx_x8]);
    // else
    //     *(uint2 *)&src_lds[hipThreadIdx_y][hipThreadIdx_x8] = (uint2)0;

    // NEW experimental code
    int srcIdx = (id_z * srcStridesNCH.x) +
                 (std::min(std::max(id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y, 0), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
                //  (std::min(std::max(id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x, 0), roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    d_float81 filter_f81 = filterTensor[id_z];
    float *filter_row1 = &filter_f81.f1[0];
    float *filter_row2 = &filter_f81.f1[9];
    float *filter_row3 = &filter_f81.f1[18];
    float *filter_row4 = &filter_f81.f1[27];
    float *filter_row5 = &filter_f81.f1[36];
    float *filter_row6 = &filter_f81.f1[45];
    float *filter_row7 = &filter_f81.f1[54];
    float *filter_row8 = &filter_f81.f1[63];
    float *filter_row9 = &filter_f81.f1[72];
    sum_f8.f4[0] = (float4) 0;
    sum_f8.f4[1] = (float4) 0;
    float *srcPos_lds = &src_lds[hipThreadIdx_y][hipThreadIdx_x8];
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, (d_float8 *)srcPos_lds);
    if (id_x_i < 0)
        for(int i = 0; i < -id_x_i; i++)
            srcPos_lds[i] = srcPos_lds[-id_x_i];
    int diff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x_i;
    if ((diff >= 0) && (diff < 8))
        for(int i = diff; i < 8; i++)
            srcPos_lds[i] = srcPos_lds[diff - 1];

    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8, filter_row6);
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8, filter_row7);
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8, filter_row8);
        gaussian_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8, filter_row9);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

// -------------------- Set 2 - Gaussian kernel creators --------------------

__device__ float gaussian(int iSquare, int j, float mulFactor)
{
    float expFactor = - (iSquare + (j * j)) * mulFactor;
    expFactor = expf(expFactor);
    return expFactor;
}

template <typename T>
__global__ void create_gaussian_kernel(T *filterTensor,
                                       float *stdDevTensor,
                                       int kernelSize,
                                       int filterStride,
                                       int batchSize)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    if(id_x >= batchSize)
        return;

    T filter_temp;
    T *filter = &filterTensor[id_x];
    float stdDev = stdDevTensor[id_x];
    int cnt = 0;
    float mulFactor = 1 / (2 * stdDev * stdDev);
    float kernelSum = 0.0f;
    int startIdx = -(kernelSize / 2);
    int endIdx = -startIdx;
    for(int i = startIdx; i <= endIdx; i++)
    {
        int iSquare = i * i;
        for(int j = startIdx; j <= endIdx; j++)
        {
            filter_temp.f1[cnt] = gaussian(iSquare, j, mulFactor);
            kernelSum += filter_temp.f1[cnt];
            cnt++;
        }
    }
    kernelSum = (1.0f / kernelSum);

    // Multiply kernel values by (1 / kernelSum)
    cnt = 0;
    for(int i = startIdx; i <= endIdx; i++)
    {
        for(int j = startIdx; j <= endIdx; j++)
        {
            filter_temp.f1[cnt] *= kernelSum;
            cnt++;
        }
    }

    *(T *)filter = filter_temp;
}

// -------------------- Set 3 - Kernel Executors --------------------

static RppStatus hip_exec_create_gaussian_kernel(Rpp32f *filterTensor,
                                                 Rpp32s kernelSize,
                                                 Rpp32f *stdDevTensor,
                                                 rpp::Handle &handle)
{
    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = handle.GetBatchSize();
    int globalThreads_y = 1;
    int globalThreads_z = 1;
    int numValues = kernelSize * kernelSize;

    if (kernelSize == 3)
        hipLaunchKernelGGL(create_gaussian_kernel,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           (d_float9 *)filterTensor,
                           stdDevTensor,
                           kernelSize,
                           numValues,
                           handle.GetBatchSize());
    else if (kernelSize == 5)
        hipLaunchKernelGGL(create_gaussian_kernel,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           (d_float25 *)filterTensor,
                           stdDevTensor,
                           kernelSize,
                           numValues,
                           handle.GetBatchSize());
    else if (kernelSize == 7)
        hipLaunchKernelGGL(create_gaussian_kernel,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           (d_float49 *)filterTensor,
                           stdDevTensor,
                           kernelSize,
                           numValues,
                           handle.GetBatchSize());
    else if (kernelSize == 9)
        hipLaunchKernelGGL(create_gaussian_kernel,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           (d_float81 *)filterTensor,
                           stdDevTensor,
                           kernelSize,
                           numValues,
                           handle.GetBatchSize());

    return RPP_SUCCESS;
}

template <typename T>
RppStatus hip_exec_gaussian_filter_f32_tensor(T *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              T *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32u kernelSize,
                                              Rpp32f *stdDevTensor,
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

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    // Create a filter of size (kernel size x kernel size)
    float *filterTensor = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    hip_exec_create_gaussian_kernel((float *)filterTensor,
                                    kernelSize,
                                    stdDevTensor,
                                    handle);

    float kernelHost[81];
    hipMemcpy(kernelHost, filterTensor, kernelSize * kernelSize * sizeof(Rpp32f), hipMemcpyDeviceToHost);
    // std::cerr << "\n\nPRINTING GAUSSIAN:\n";
    // for (int i = 0; i < kernelSize; i++)
    // {
    //     for (int j = 0; j < kernelSize; j++)
    //     {
    //         std::cerr << kernelHost[i * kernelSize + j] << ", ";
    //     }
    //     std::cerr << "\n";
    // }

    if ((srcDescPtr->c == 1) && (dstDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(gaussian_filter_3x3_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               (d_float9 *)filterTensor);
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(gaussian_filter_5x5_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               (d_float25 *)filterTensor);
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(gaussian_filter_7x7_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               (d_float49 *)filterTensor);
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(gaussian_filter_9x9_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc,
                               (d_float81 *)filterTensor);
        }
    }

    return RPP_SUCCESS;
}
