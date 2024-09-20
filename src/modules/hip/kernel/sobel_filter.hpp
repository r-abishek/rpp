#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ __constant__ float sobel3x3XHip[9] = {-1, 0, 1,
                                              -2, 0, 2,
                                              -1, 0, 1};
__device__ __constant__ float sobel3x3YHip[9] = {-1, -2, -1,
                                                  0, 0, 0,
                                                  1, 2, 1};
__device__ __constant__ float sobel5x5XHip[25] = {-1,  -2,   0,   2,   1,
                                               -4,  -8,   0,   8,   4,
                                               -6, -12,   0,  12,   6,
                                               -4,  -8,   0,   8,   4,
                                               -1,  -2,   0,   2,   1};
__device__ __constant__ float sobel5x5YHip[25] = {-1,  -4,  -6,  -4,  -1,
                                                  -2,  -8, -12,  -8,  -2,
                                                  0,   0,   0,   0,   0,
                                                2,   8,  12,   8,   2,
                                                1,   4,   6,   4,   1};

// -------------------- sobel_filter device helpers --------------------

__device__ __forceinline__ void sobel_filter_bidirection_hip_compute(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8)
{
    rpp_hip_math_multiply8(src1_f8, src1_f8, src1_f8);
    rpp_hip_math_multiply8(src2_f8, src2_f8, src2_f8);
    rpp_hip_math_add8(src1_f8, src2_f8, dst_f8);
    rpp_hip_math_sqrt8(dst_f8, dst_f8);
}

__device__ __forceinline__ void sobel_filter_3x3_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, float *filter)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(reinterpret_cast<uint3 *>(srcPtr));
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[0], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[0], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[0], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[1] = fmaf(src_f1, filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[0], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f1, filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[0], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f1, filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[0], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[4] = fmaf(src_f1, filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[0], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[5] = fmaf(src_f1, filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[0], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f1, filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f1, filter[2], dst_f8->f1[7]);
}

__device__ __forceinline__ void sobel_filter_3x3_bidirectional_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8x, d_float8 *dst_f8y, float *filterX, float *filterY)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(reinterpret_cast<uint3 *>(srcPtr));
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8x->f1[0] = fmaf(src_f1, filterX[0], dst_f8x->f1[0]);
    dst_f8y->f1[0] = fmaf(src_f1, filterY[0], dst_f8y->f1[0]);

    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8x->f1[0] = fmaf(src_f1, filterX[1], dst_f8x->f1[0]);
    dst_f8y->f1[0] = fmaf(src_f1, filterY[1], dst_f8y->f1[0]);
    dst_f8x->f1[1] = fmaf(src_f1, filterX[0], dst_f8x->f1[1]);
    dst_f8y->f1[1] = fmaf(src_f1, filterY[0], dst_f8y->f1[1]);

    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8x->f1[0] = fmaf(src_f1, filterX[2], dst_f8x->f1[0]);
    dst_f8y->f1[0] = fmaf(src_f1, filterY[2], dst_f8y->f1[0]);
    dst_f8x->f1[1] = fmaf(src_f1, filterX[1], dst_f8x->f1[1]);
    dst_f8y->f1[1] = fmaf(src_f1, filterY[1], dst_f8y->f1[1]);
    dst_f8x->f1[2] = fmaf(src_f1, filterX[0], dst_f8x->f1[2]);
    dst_f8y->f1[2] = fmaf(src_f1, filterY[0], dst_f8y->f1[2]);

    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8x->f1[1] = fmaf(src_f1, filterX[2], dst_f8x->f1[1]);
    dst_f8y->f1[1] = fmaf(src_f1, filterY[2], dst_f8y->f1[1]);
    dst_f8x->f1[2] = fmaf(src_f1, filterX[1], dst_f8x->f1[2]);
    dst_f8y->f1[2] = fmaf(src_f1, filterY[1], dst_f8y->f1[2]);
    dst_f8x->f1[3] = fmaf(src_f1, filterX[0], dst_f8x->f1[3]);
    dst_f8y->f1[3] = fmaf(src_f1, filterY[0], dst_f8y->f1[3]);

    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8x->f1[2] = fmaf(src_f1, filterX[2], dst_f8x->f1[2]);
    dst_f8y->f1[2] = fmaf(src_f1, filterY[2], dst_f8y->f1[2]);
    dst_f8x->f1[3] = fmaf(src_f1, filterX[1], dst_f8x->f1[3]);
    dst_f8y->f1[3] = fmaf(src_f1, filterY[1], dst_f8y->f1[3]);
    dst_f8x->f1[4] = fmaf(src_f1, filterX[0], dst_f8x->f1[4]);
    dst_f8y->f1[4] = fmaf(src_f1, filterY[0], dst_f8y->f1[4]);

    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8x->f1[3] = fmaf(src_f1, filterX[2], dst_f8x->f1[3]);
    dst_f8y->f1[3] = fmaf(src_f1, filterY[2], dst_f8y->f1[3]);
    dst_f8x->f1[4] = fmaf(src_f1, filterX[1], dst_f8x->f1[4]);
    dst_f8y->f1[4] = fmaf(src_f1, filterY[1], dst_f8y->f1[4]);
    dst_f8x->f1[5] = fmaf(src_f1, filterX[0], dst_f8x->f1[5]);
    dst_f8y->f1[5] = fmaf(src_f1, filterY[0], dst_f8y->f1[5]);

    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8x->f1[4] = fmaf(src_f1, filterX[2], dst_f8x->f1[4]);
    dst_f8y->f1[4] = fmaf(src_f1, filterY[2], dst_f8y->f1[4]);
    dst_f8x->f1[5] = fmaf(src_f1, filterX[1], dst_f8x->f1[5]);
    dst_f8y->f1[5] = fmaf(src_f1, filterY[1], dst_f8y->f1[5]);
    dst_f8x->f1[6] = fmaf(src_f1, filterX[0], dst_f8x->f1[6]);
    dst_f8y->f1[6] = fmaf(src_f1, filterY[0], dst_f8y->f1[6]);

    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8x->f1[5] = fmaf(src_f1, filterX[2], dst_f8x->f1[5]);
    dst_f8y->f1[5] = fmaf(src_f1, filterY[2], dst_f8y->f1[5]);
    dst_f8x->f1[6] = fmaf(src_f1, filterX[1], dst_f8x->f1[6]);
    dst_f8y->f1[6] = fmaf(src_f1, filterY[1], dst_f8y->f1[6]);
    dst_f8x->f1[7] = fmaf(src_f1, filterX[0], dst_f8x->f1[7]);
    dst_f8y->f1[7] = fmaf(src_f1, filterY[0], dst_f8y->f1[7]);

    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8x->f1[6] = fmaf(src_f1, filterX[2], dst_f8x->f1[6]);
    dst_f8y->f1[6] = fmaf(src_f1, filterY[2], dst_f8y->f1[6]);
    dst_f8x->f1[7] = fmaf(src_f1, filterX[1], dst_f8x->f1[7]);
    dst_f8y->f1[7] = fmaf(src_f1, filterY[1], dst_f8y->f1[7]);

    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8x->f1[7] = fmaf(src_f1, filterX[2], dst_f8x->f1[7]);
    dst_f8y->f1[7] = fmaf(src_f1, filterY[2], dst_f8y->f1[7]);
}

__device__ void sobel_filter_5x5_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8, float *filter)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(reinterpret_cast<uint3 *>(srcPtr));
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[0], dst_f8->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[1], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[0], dst_f8->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[2], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[1], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[0], dst_f8->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8->f1[0] = fmaf(src_f1, filter[3], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[2], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[1], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[0], dst_f8->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8->f1[0] = fmaf(src_f1, filter[4], dst_f8->f1[0]);
    dst_f8->f1[1] = fmaf(src_f1, filter[3], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[2], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[1], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[0], dst_f8->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8->f1[1] = fmaf(src_f1, filter[4], dst_f8->f1[1]);
    dst_f8->f1[2] = fmaf(src_f1, filter[3], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[2], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[1], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[0], dst_f8->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8->f1[2] = fmaf(src_f1, filter[4], dst_f8->f1[2]);
    dst_f8->f1[3] = fmaf(src_f1, filter[3], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[2], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[1], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[0], dst_f8->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8->f1[3] = fmaf(src_f1, filter[4], dst_f8->f1[3]);
    dst_f8->f1[4] = fmaf(src_f1, filter[3], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[2], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[1], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[0], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8->f1[4] = fmaf(src_f1, filter[4], dst_f8->f1[4]);
    dst_f8->f1[5] = fmaf(src_f1, filter[3], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[2], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[1], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8->f1[5] = fmaf(src_f1, filter[4], dst_f8->f1[5]);
    dst_f8->f1[6] = fmaf(src_f1, filter[3], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[2], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui3.z);
    dst_f8->f1[6] = fmaf(src_f1, filter[4], dst_f8->f1[6]);
    dst_f8->f1[7] = fmaf(src_f1, filter[3], dst_f8->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui3.z);
    dst_f8->f1[7] = fmaf(src_f1, filter[4], dst_f8->f1[7]);
}

__device__ void sobel_filter_5x5_bidirectional_row_hip_compute(uchar *srcPtr, d_float8 *dst_f8x, d_float8 *dst_f8y, float *filterX, float *filterY)
{
    float src_f1;
    uint3 src_ui3;
    src_ui3 = *(reinterpret_cast<uint3 *>(srcPtr));
    src_f1 = rpp_hip_unpack0(src_ui3.x);
    dst_f8x->f1[0] = fmaf(src_f1, filterX[0], dst_f8x->f1[0]);
    dst_f8y->f1[0] = fmaf(src_f1, filterY[0], dst_f8y->f1[0]);
    src_f1 = rpp_hip_unpack1(src_ui3.x);
    dst_f8x->f1[0] = fmaf(src_f1, filterX[1], dst_f8x->f1[0]);
    dst_f8x->f1[1] = fmaf(src_f1, filterX[0], dst_f8x->f1[1]);
    dst_f8y->f1[0] = fmaf(src_f1, filterY[1], dst_f8y->f1[0]);
    dst_f8y->f1[1] = fmaf(src_f1, filterX[0], dst_f8y->f1[1]);
    src_f1 = rpp_hip_unpack2(src_ui3.x);
    dst_f8x->f1[0] = fmaf(src_f1, filterX[2], dst_f8x->f1[0]);
    dst_f8x->f1[1] = fmaf(src_f1, filterX[1], dst_f8x->f1[1]);
    dst_f8x->f1[2] = fmaf(src_f1, filterX[0], dst_f8x->f1[2]);
    dst_f8y->f1[0] = fmaf(src_f1, filterY[2], dst_f8y->f1[0]);
    dst_f8y->f1[1] = fmaf(src_f1, filterY[1], dst_f8y->f1[1]);
    dst_f8y->f1[2] = fmaf(src_f1, filterY[0], dst_f8y->f1[2]);
    src_f1 = rpp_hip_unpack3(src_ui3.x);
    dst_f8x->f1[0] = fmaf(src_f1, filterX[3], dst_f8x->f1[0]);
    dst_f8x->f1[1] = fmaf(src_f1, filterX[2], dst_f8x->f1[1]);
    dst_f8x->f1[2] = fmaf(src_f1, filterX[1], dst_f8x->f1[2]);
    dst_f8x->f1[3] = fmaf(src_f1, filterX[0], dst_f8x->f1[3]);
    dst_f8y->f1[0] = fmaf(src_f1, filterY[3], dst_f8y->f1[0]);
    dst_f8y->f1[1] = fmaf(src_f1, filterY[2], dst_f8y->f1[1]);
    dst_f8y->f1[2] = fmaf(src_f1, filterY[1], dst_f8y->f1[2]);
    dst_f8y->f1[3] = fmaf(src_f1, filterY[0], dst_f8y->f1[3]);
    src_f1 = rpp_hip_unpack0(src_ui3.y);
    dst_f8x->f1[0] = fmaf(src_f1, filterX[4], dst_f8x->f1[0]);
    dst_f8x->f1[1] = fmaf(src_f1, filterX[3], dst_f8x->f1[1]);
    dst_f8x->f1[2] = fmaf(src_f1, filterX[2], dst_f8x->f1[2]);
    dst_f8x->f1[3] = fmaf(src_f1, filterX[1], dst_f8x->f1[3]);
    dst_f8x->f1[4] = fmaf(src_f1, filterX[0], dst_f8x->f1[4]);
    dst_f8y->f1[0] = fmaf(src_f1, filterY[4], dst_f8y->f1[0]);
    dst_f8y->f1[1] = fmaf(src_f1, filterY[3], dst_f8y->f1[1]);
    dst_f8y->f1[2] = fmaf(src_f1, filterY[2], dst_f8y->f1[2]);
    dst_f8y->f1[3] = fmaf(src_f1, filterY[1], dst_f8y->f1[3]);
    dst_f8y->f1[4] = fmaf(src_f1, filterY[0], dst_f8y->f1[4]);
    src_f1 = rpp_hip_unpack1(src_ui3.y);
    dst_f8x->f1[1] = fmaf(src_f1, filterX[4], dst_f8x->f1[1]);
    dst_f8x->f1[2] = fmaf(src_f1, filterX[3], dst_f8x->f1[2]);
    dst_f8x->f1[3] = fmaf(src_f1, filterX[2], dst_f8x->f1[3]);
    dst_f8x->f1[4] = fmaf(src_f1, filterX[1], dst_f8x->f1[4]);
    dst_f8x->f1[5] = fmaf(src_f1, filterX[0], dst_f8x->f1[5]);
    dst_f8y->f1[1] = fmaf(src_f1, filterY[4], dst_f8y->f1[1]);
    dst_f8y->f1[2] = fmaf(src_f1, filterY[3], dst_f8y->f1[2]);
    dst_f8y->f1[3] = fmaf(src_f1, filterY[2], dst_f8y->f1[3]);
    dst_f8y->f1[4] = fmaf(src_f1, filterY[1], dst_f8y->f1[4]);
    dst_f8y->f1[5] = fmaf(src_f1, filterY[0], dst_f8y->f1[5]);
    src_f1 = rpp_hip_unpack2(src_ui3.y);
    dst_f8x->f1[2] = fmaf(src_f1, filterX[4], dst_f8x->f1[2]);
    dst_f8x->f1[3] = fmaf(src_f1, filterX[3], dst_f8x->f1[3]);
    dst_f8x->f1[4] = fmaf(src_f1, filterX[2], dst_f8x->f1[4]);
    dst_f8x->f1[5] = fmaf(src_f1, filterX[1], dst_f8x->f1[5]);
    dst_f8x->f1[6] = fmaf(src_f1, filterX[0], dst_f8x->f1[6]);
    dst_f8y->f1[2] = fmaf(src_f1, filterY[4], dst_f8y->f1[2]);
    dst_f8y->f1[3] = fmaf(src_f1, filterY[3], dst_f8y->f1[3]);
    dst_f8y->f1[4] = fmaf(src_f1, filterY[2], dst_f8y->f1[4]);
    dst_f8y->f1[5] = fmaf(src_f1, filterY[1], dst_f8y->f1[5]);
    dst_f8y->f1[6] = fmaf(src_f1, filterY[0], dst_f8y->f1[6]);
    src_f1 = rpp_hip_unpack3(src_ui3.y);
    dst_f8x->f1[3] = fmaf(src_f1, filterX[4], dst_f8x->f1[3]);
    dst_f8x->f1[4] = fmaf(src_f1, filterX[3], dst_f8x->f1[4]);
    dst_f8x->f1[5] = fmaf(src_f1, filterX[2], dst_f8x->f1[5]);
    dst_f8x->f1[6] = fmaf(src_f1, filterX[1], dst_f8x->f1[6]);
    dst_f8x->f1[7] = fmaf(src_f1, filterX[0], dst_f8x->f1[7]);
    dst_f8y->f1[3] = fmaf(src_f1, filterY[4], dst_f8y->f1[3]);
    dst_f8y->f1[4] = fmaf(src_f1, filterY[3], dst_f8y->f1[4]);
    dst_f8y->f1[5] = fmaf(src_f1, filterY[2], dst_f8y->f1[5]);
    dst_f8y->f1[6] = fmaf(src_f1, filterY[1], dst_f8y->f1[6]);
    dst_f8y->f1[7] = fmaf(src_f1, filterY[0], dst_f8y->f1[7]);
    src_f1 = rpp_hip_unpack0(src_ui3.z);
    dst_f8x->f1[4] = fmaf(src_f1, filterX[4], dst_f8x->f1[4]);
    dst_f8x->f1[5] = fmaf(src_f1, filterX[3], dst_f8x->f1[5]);
    dst_f8x->f1[6] = fmaf(src_f1, filterX[2], dst_f8x->f1[6]);
    dst_f8x->f1[7] = fmaf(src_f1, filterX[1], dst_f8x->f1[7]);
    dst_f8y->f1[4] = fmaf(src_f1, filterY[4], dst_f8y->f1[4]);
    dst_f8y->f1[5] = fmaf(src_f1, filterY[3], dst_f8y->f1[5]);
    dst_f8y->f1[6] = fmaf(src_f1, filterY[2], dst_f8y->f1[6]);
    dst_f8y->f1[7] = fmaf(src_f1, filterY[1], dst_f8y->f1[7]);
    src_f1 = rpp_hip_unpack1(src_ui3.z);
    dst_f8x->f1[5] = fmaf(src_f1, filterX[4], dst_f8x->f1[5]);
    dst_f8x->f1[6] = fmaf(src_f1, filterX[3], dst_f8x->f1[6]);
    dst_f8x->f1[7] = fmaf(src_f1, filterX[2], dst_f8x->f1[7]);
    dst_f8y->f1[5] = fmaf(src_f1, filterY[4], dst_f8y->f1[5]);
    dst_f8y->f1[6] = fmaf(src_f1, filterY[3], dst_f8y->f1[6]);
    dst_f8y->f1[7] = fmaf(src_f1, filterY[2], dst_f8y->f1[7]);
    src_f1 = rpp_hip_unpack2(src_ui3.z);
    dst_f8x->f1[6] = fmaf(src_f1, filterX[4], dst_f8x->f1[6]);
    dst_f8x->f1[7] = fmaf(src_f1, filterX[3], dst_f8x->f1[7]);
    dst_f8y->f1[6] = fmaf(src_f1, filterY[4], dst_f8y->f1[6]);
    dst_f8y->f1[7] = fmaf(src_f1, filterY[3], dst_f8y->f1[7]);
    src_f1 = rpp_hip_unpack3(src_ui3.z);
    dst_f8x->f1[7] = fmaf(src_f1, filterX[4], dst_f8x->f1[7]);
    dst_f8y->f1[7] = fmaf(src_f1, filterY[4], dst_f8y->f1[7]);
}

template <typename T>
__global__ void sobel_filter_3x3_pln_bidirection_tensor(T *srcPtr,
                                                        uint3 srcStridesNCH,
                                                        T *dstPtr,
                                                        uint3 dstStridesNCH,
                                                        int channelsDst,
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

    d_float8 sum_f8x, sum_f8y, sum_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filterRowX1 = &sobel3x3XHip[0];
    float *filterRowX2 = &filterRowX1[3];
    float *filterRowX3 = &filterRowX1[6];
    float *filterRowY1 = &sobel3x3YHip[0];
    float *filterRowY2 = &filterRowY1[3];
    float *filterRowY3 = &filterRowY1[6];
    sum_f8x.f4[0] = static_cast<float4>(0);
    sum_f8x.f4[1] = static_cast<float4>(0);
    sum_f8y.f4[0] = static_cast<float4>(0);
    sum_f8y.f4[1] = static_cast<float4>(0);
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else if(id_y_i < 0)
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx + srcStridesNCH.z, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    if(id_x_i < 0)
        src_smem[hipThreadIdx_y][0] = src_smem[hipThreadIdx_y][1];
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        sobel_filter_3x3_bidirectional_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8x, &sum_f8y, filterRowX1, filterRowY1);
        sobel_filter_3x3_bidirectional_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8x, &sum_f8y, filterRowX2, filterRowY2);
        sobel_filter_3x3_bidirectional_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8x, &sum_f8y, filterRowX3, filterRowY3);
        rpp_hip_pixel_check_0to255(&sum_f8x);
        rpp_hip_pixel_check_0to255(&sum_f8y);
        rpp_hip_adjust_range(dstPtr, &sum_f8x);
        rpp_hip_adjust_range(dstPtr, &sum_f8y);
        sobel_filter_bidirection_hip_compute(&sum_f8x, &sum_f8y, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_5x5_pln_bidirection_tensor(T *srcPtr,
                                                        uint3 srcStridesNCH,
                                                        T *dstPtr,
                                                        uint3 dstStridesNCH,
                                                        int channelsDst,
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

    d_float8 sum_f8x, sum_f8y, sum_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filterRowX1 = &sobel5x5XHip[0];
    float *filterRowX2 = &filterRowX1[5];
    float *filterRowX3 = &filterRowX1[10];
    float *filterRowX4 = &filterRowX1[15];
    float *filterRowX5 = &filterRowX1[20];
    float *filterRowY1 = &sobel5x5YHip[0];
    float *filterRowY2 = &filterRowY1[5];
    float *filterRowY3 = &filterRowY1[10];
    float *filterRowY4 = &filterRowY1[15];
    float *filterRowY5 = &filterRowY1[20];
    sum_f8x.f4[0] = static_cast<float4>(0);
    sum_f8x.f4[1] = static_cast<float4>(0);
    sum_f8y.f4[0] = static_cast<float4>(0);
    sum_f8y.f4[1] = static_cast<float4>(0);
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else if(id_y_i < 0)
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx + 2 * srcStridesNCH.z, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    if(id_x_i < 0)
    {
        src_smem[hipThreadIdx_y][0] = src_smem[hipThreadIdx_y][2];
        src_smem[hipThreadIdx_y][1] = src_smem[hipThreadIdx_y][2];
    }

    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        sobel_filter_5x5_bidirectional_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8x, &sum_f8y, filterRowX1, filterRowY1);
        sobel_filter_5x5_bidirectional_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8x, &sum_f8y, filterRowX2, filterRowY2);
        sobel_filter_5x5_bidirectional_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8x, &sum_f8y, filterRowX3, filterRowY3);
        sobel_filter_5x5_bidirectional_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8x, &sum_f8y, filterRowX4, filterRowY4);
        sobel_filter_5x5_bidirectional_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8x, &sum_f8y, filterRowX5, filterRowY5);
        rpp_hip_pixel_check_0to255(&sum_f8x);
        rpp_hip_pixel_check_0to255(&sum_f8y);
        rpp_hip_adjust_range(dstPtr, &sum_f8x);
        rpp_hip_adjust_range(dstPtr, &sum_f8y);
        sobel_filter_bidirection_hip_compute(&sum_f8x, &sum_f8y, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_3x3_pln_unidirection_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               int sobelType)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    d_float8 sum_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filter_row1 = ((!sobelType) ? sobel3x3XHip : sobel3x3YHip);
    float *filter_row2 = &filter_row1[3];
    float *filter_row3 = &filter_row1[6];
    sum_f8.f4[0] = static_cast<float4>(0);
    sum_f8.f4[1] = static_cast<float4>(0);
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else if(id_y_i < 0)
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx + srcStridesNCH.z, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    if(id_x_i < 0)
        src_smem[hipThreadIdx_y][0] = src_smem[hipThreadIdx_y][1];
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        sobel_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        sobel_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        sobel_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        rpp_hip_pixel_check_0to255(&sum_f8);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
__global__ void sobel_filter_5x5_pln_unidirection_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint padLength,
                                               uint2 tileSize,
                                               RpptROIPtr roiTensorPtrSrc,
                                               int sobelType)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    float *filter_row1 = (!sobelType) ? sobel5x5XHip : sobel5x5YHip;
    float *filter_row2 = &filter_row1[5];
    float *filter_row3 = &filter_row1[10];
    float *filter_row4 = &filter_row1[15];
    float *filter_row5 = &filter_row1[20];
    sum_f8.f4[0] = static_cast<float4>(0);
    sum_f8.f4[1] = static_cast<float4>(0);
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    else if(id_y_i < 0)
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx + 2 * srcStridesNCH.z, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    if(id_x_i < 0)
    {
        src_smem[hipThreadIdx_y][0] = src_smem[hipThreadIdx_y][2];
        src_smem[hipThreadIdx_y][1] = src_smem[hipThreadIdx_y][2];
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        sobel_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &sum_f8, filter_row1);
        sobel_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 1][hipThreadIdx_x8], &sum_f8, filter_row2);
        sobel_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 2][hipThreadIdx_x8], &sum_f8, filter_row3);
        sobel_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 3][hipThreadIdx_x8], &sum_f8, filter_row4);
        sobel_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y + 4][hipThreadIdx_x8], &sum_f8, filter_row5);
        rpp_hip_pixel_check_0to255(&sum_f8);
        rpp_hip_adjust_range(dstPtr, &sum_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &sum_f8);
    }
}

template <typename T>
RppStatus hip_exec_sobel_filter_tensor(T *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          T *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32u sobelType,
                                          Rpp32u kernelSize,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (SMEM_LENGTH_X - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;
    bool combined = (sobelType == 2);

    if (kernelSize == 3)
    {
        if(combined)
        {
            hipLaunchKernelGGL(sobel_filter_3x3_pln_bidirection_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else
        {
            hipLaunchKernelGGL(sobel_filter_3x3_pln_unidirection_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
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
                               sobelType);
        }
    }
    else if (kernelSize == 5)
    {
        if(combined)
        {
            hipLaunchKernelGGL(sobel_filter_5x5_pln_bidirection_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else
        {
            hipLaunchKernelGGL(sobel_filter_5x5_pln_unidirection_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
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
                               sobelType);
        }
    }

    return RPP_SUCCESS;
}