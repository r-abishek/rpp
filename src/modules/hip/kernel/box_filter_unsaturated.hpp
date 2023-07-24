#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - box_filter device helpers --------------------

__device__ void box_filter_f32_3x3_row_hip_compute(d_float10 *srcPtr_f10, d_float8 *dst_f8)
{
    d_float10 src_f10 = *srcPtr_f10;
    dst_f8->f1[0] = fmaf(src_f10.f1[0], 0.1111111f, dst_f8->f1[0]);
    dst_f8->f1[0] = fmaf(src_f10.f1[1], 0.1111111f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f10.f1[1], 0.1111111f, dst_f8->f1[1]);
    dst_f8->f1[0] = fmaf(src_f10.f1[2], 0.1111111f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f10.f1[2], 0.1111111f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f10.f1[2], 0.1111111f, dst_f8->f1[2]);
    dst_f8->f1[1] = fmaf(src_f10.f1[3], 0.1111111f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f10.f1[3], 0.1111111f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f10.f1[3], 0.1111111f, dst_f8->f1[3]);
    dst_f8->f1[2] = fmaf(src_f10.f1[4], 0.1111111f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f10.f1[4], 0.1111111f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f10.f1[4], 0.1111111f, dst_f8->f1[4]);
    dst_f8->f1[3] = fmaf(src_f10.f1[5], 0.1111111f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f10.f1[5], 0.1111111f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f10.f1[5], 0.1111111f, dst_f8->f1[5]);
    dst_f8->f1[4] = fmaf(src_f10.f1[6], 0.1111111f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f10.f1[6], 0.1111111f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f10.f1[6], 0.1111111f, dst_f8->f1[6]);
    dst_f8->f1[5] = fmaf(src_f10.f1[7], 0.1111111f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f10.f1[7], 0.1111111f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f10.f1[7], 0.1111111f, dst_f8->f1[7]);
    dst_f8->f1[6] = fmaf(src_f10.f1[8], 0.1111111f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f10.f1[8], 0.1111111f, dst_f8->f1[7]);
    dst_f8->f1[7] = fmaf(src_f10.f1[9], 0.1111111f, dst_f8->f1[7]);
}

__device__ void box_filter_f32_5x5_row_hip_compute(d_float12 *srcPtr_f12, d_float8 *dst_f8)
{
    d_float12 src_f12 = *srcPtr_f12;
    dst_f8->f1[0] = fmaf(src_f12.f1[0], 0.04f, dst_f8->f1[0]);
    dst_f8->f1[0] = fmaf(src_f12.f1[1], 0.04f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f12.f1[1], 0.04f, dst_f8->f1[1]);
    dst_f8->f1[0] = fmaf(src_f12.f1[2], 0.04f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f12.f1[2], 0.04f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f12.f1[2], 0.04f, dst_f8->f1[2]);
    dst_f8->f1[0] = fmaf(src_f12.f1[3], 0.04f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f12.f1[3], 0.04f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f12.f1[3], 0.04f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f12.f1[3], 0.04f, dst_f8->f1[3]);
    dst_f8->f1[0] = fmaf(src_f12.f1[4], 0.04f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f12.f1[4], 0.04f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f12.f1[4], 0.04f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f12.f1[4], 0.04f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f12.f1[4], 0.04f, dst_f8->f1[4]);
    dst_f8->f1[1] = fmaf(src_f12.f1[5], 0.04f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f12.f1[5], 0.04f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f12.f1[5], 0.04f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f12.f1[5], 0.04f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f12.f1[5], 0.04f, dst_f8->f1[5]);
    dst_f8->f1[2] = fmaf(src_f12.f1[6], 0.04f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f12.f1[6], 0.04f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f12.f1[6], 0.04f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f12.f1[6], 0.04f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f12.f1[6], 0.04f, dst_f8->f1[6]);
    dst_f8->f1[3] = fmaf(src_f12.f1[7], 0.04f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f12.f1[7], 0.04f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f12.f1[7], 0.04f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f12.f1[7], 0.04f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f12.f1[7], 0.04f, dst_f8->f1[7]);
    dst_f8->f1[4] = fmaf(src_f12.f1[8], 0.04f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f12.f1[8], 0.04f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f12.f1[8], 0.04f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f12.f1[8], 0.04f, dst_f8->f1[7]);
    dst_f8->f1[5] = fmaf(src_f12.f1[9], 0.04f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f12.f1[9], 0.04f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f12.f1[9], 0.04f, dst_f8->f1[7]);
    dst_f8->f1[6] = fmaf(src_f12.f1[10], 0.04f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f12.f1[10], 0.04f, dst_f8->f1[7]);
    dst_f8->f1[7] = fmaf(src_f12.f1[11], 0.04f, dst_f8->f1[7]);
}

__device__ void box_filter_f32_7x7_row_hip_compute(d_float14 *srcPtr_f14, d_float8 *dst_f8)
{
    d_float14 src_f14 = *srcPtr_f14;
    dst_f8->f1[0] = fmaf(src_f14.f1[0], 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[0] = fmaf(src_f14.f1[1], 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[1], 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[0] = fmaf(src_f14.f1[2], 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[2], 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[2], 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[0] = fmaf(src_f14.f1[3], 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[3], 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[3], 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[3], 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[0] = fmaf(src_f14.f1[4], 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[4], 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[4], 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[4], 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[4], 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[0] = fmaf(src_f14.f1[5], 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[5], 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[5], 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[5], 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[5], 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[5], 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[0] = fmaf(src_f14.f1[6], 0.02040816f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f14.f1[6], 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[6], 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[6], 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[6], 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[6], 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[6], 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[1] = fmaf(src_f14.f1[7], 0.02040816f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f14.f1[7], 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[7], 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[7], 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[7], 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[7], 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[7], 0.02040816f, dst_f8->f1[7]);
    dst_f8->f1[2] = fmaf(src_f14.f1[8], 0.02040816f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f14.f1[8], 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[8], 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[8], 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[8], 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[8], 0.02040816f, dst_f8->f1[7]);
    dst_f8->f1[3] = fmaf(src_f14.f1[9], 0.02040816f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f14.f1[9], 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[9], 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[9], 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[9], 0.02040816f, dst_f8->f1[7]);
    dst_f8->f1[4] = fmaf(src_f14.f1[10], 0.02040816f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f14.f1[10], 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[10], 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[10], 0.02040816f, dst_f8->f1[7]);
    dst_f8->f1[5] = fmaf(src_f14.f1[11], 0.02040816f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f14.f1[11], 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[11], 0.02040816f, dst_f8->f1[7]);
    dst_f8->f1[6] = fmaf(src_f14.f1[12], 0.02040816f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f14.f1[12], 0.02040816f, dst_f8->f1[7]);
    dst_f8->f1[7] = fmaf(src_f14.f1[13], 0.02040816f, dst_f8->f1[7]);
}

__device__ void box_filter_f32_9x9_row_hip_compute(d_float16 *srcPtr_f16, d_float8 *dst_f8)
{
    d_float16 src_f16 = *srcPtr_f16;
    dst_f8->f1[0] = fmaf(src_f16.f1[0], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[0] = fmaf(src_f16.f1[1], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[1], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[0] = fmaf(src_f16.f1[2], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[2], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[2], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[0] = fmaf(src_f16.f1[3], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[3], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[3], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[3], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[0] = fmaf(src_f16.f1[4], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[4], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[4], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[4], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[4], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[0] = fmaf(src_f16.f1[5], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[5], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[5], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[5], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[5], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[5], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[0] = fmaf(src_f16.f1[6], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[6], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[6], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[6], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[6], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[6], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[6], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[0] = fmaf(src_f16.f1[7], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[7], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[7], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[7], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[7], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[7], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[7], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[7], 0.01234568f, dst_f8->f1[7]);
    dst_f8->f1[0] = fmaf(src_f16.f1[8], 0.01234568f, dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f16.f1[8], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[8], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[8], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[8], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[8], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[8], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[8], 0.01234568f, dst_f8->f1[7]);
    dst_f8->f1[1] = fmaf(src_f16.f1[9], 0.01234568f, dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f16.f1[9], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[9], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[9], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[9], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[9], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[9], 0.01234568f, dst_f8->f1[7]);
    dst_f8->f1[2] = fmaf(src_f16.f1[10], 0.01234568f, dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f16.f1[10], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[10], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[10], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[10], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[10], 0.01234568f, dst_f8->f1[7]);
    dst_f8->f1[3] = fmaf(src_f16.f1[11], 0.01234568f, dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f16.f1[11], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[11], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[11], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[11], 0.01234568f, dst_f8->f1[7]);
    dst_f8->f1[4] = fmaf(src_f16.f1[12], 0.01234568f, dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f16.f1[12], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[12], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[12], 0.01234568f, dst_f8->f1[7]);
    dst_f8->f1[5] = fmaf(src_f16.f1[13], 0.01234568f, dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f16.f1[13], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[13], 0.01234568f, dst_f8->f1[7]);
    dst_f8->f1[6] = fmaf(src_f16.f1[14], 0.01234568f, dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f16.f1[14], 0.01234568f, dst_f8->f1[7]);
    dst_f8->f1[7] = fmaf(src_f16.f1[15], 0.01234568f, dst_f8->f1[7]);
}

// -------------------- Set 1 - PLN1->PLN1 for F32 without saturation check --------------------

// kernelSize = 3
__global__ void box_filter_f32_3x3_pln_tensor(float *srcPtr,
                                              uint3 srcStridesNCH,
                                              float *dstPtr,
                                              uint3 dstStridesNCH,
                                              uint padLength,
                                              uint2 tileSize,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ float src_lds[16][128];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 (std::min(std::max(id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y, 0), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
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
        box_filter_f32_3x3_row_hip_compute((d_float10 *)&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_3x3_row_hip_compute((d_float10 *)&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_3x3_row_hip_compute((d_float10 *)&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

// kernelSize = 5
__global__ void box_filter_f32_5x5_pln_tensor(float *srcPtr,
                                              uint3 srcStridesNCH,
                                              float *dstPtr,
                                              uint3 dstStridesNCH,
                                              uint padLength,
                                              uint2 tileSize,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ float src_lds[16][128];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 (std::min(std::max(id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y, 0), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
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
        box_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_5x5_row_hip_compute((d_float12 *)&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

// kernelSize = 7
__global__ void box_filter_f32_7x7_pln_tensor(float *srcPtr,
                                              uint3 srcStridesNCH,
                                              float *dstPtr,
                                              uint3 dstStridesNCH,
                                              uint padLength,
                                              uint2 tileSize,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ float src_lds[16][128];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 (std::min(std::max(id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y, 0), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
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
        box_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_7x7_row_hip_compute((d_float14 *)&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

// kernelSize = 9
__global__ void box_filter_f32_9x9_pln_tensor(float *srcPtr,
                                              uint3 srcStridesNCH,
                                              float *dstPtr,
                                              uint3 dstStridesNCH,
                                              uint padLength,
                                              uint2 tileSize,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ float src_lds[16][128];

    int srcIdx = (id_z * srcStridesNCH.x) +
                 (std::min(std::max(id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y, 0), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1) * srcStridesNCH.z) +
                 (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
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
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 5][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 6][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 7][hipThreadIdx_x8], &sum_f8);
        box_filter_f32_9x9_row_hip_compute((d_float16 *)&src_lds[hipThreadIdx_y + 8][hipThreadIdx_x8], &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

// -------------------- Set 2 - Kernel Executors --------------------

RppStatus hip_exec_box_filter_f32_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32u kernelSize,
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

    if ((srcDescPtr->c == 1) && (dstDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(box_filter_f32_3x3_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(box_filter_f32_5x5_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(box_filter_f32_7x7_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(box_filter_f32_9x9_pln_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
