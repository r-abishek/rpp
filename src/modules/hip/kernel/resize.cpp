#include <hip/hip_runtime.h>
#include <half.hpp>
#include "rpp_hip_host_decls.hpp"

using half_float::half;

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
#define RANGE_CHECK(value, lower, upper) ((value > upper) ? upper : (value < lower) ? lower : value)

__device__ __forceinline__ void CalculateCubicCoefficients(float* coeffs, float x)
{
    float A = -0.5f;
    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.0f - coeffs[0] - coeffs[1] - coeffs[2];
}

extern "C" __global__ void resize_pln(unsigned char *srcPtr,
                                      unsigned char *dstPtr,
                                      const unsigned int source_height,
                                      const unsigned int source_width,
                                      const unsigned int dest_height,
                                      const unsigned int dest_width,
                                      const unsigned int channel)
{
    int A, B, C, D, x, y, index, pixVal;
    float x_ratio = ((float)(source_width - 1 )) / dest_width ;
    float y_ratio = ((float)(source_height - 1 )) / dest_height;
    float x_diff, y_diff, ya, yb;

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel)
    {
        return;
    }

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    unsigned int pixId;

    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;

    A = srcPtr[x + y * source_width + id_z * source_height * source_width];
    B = srcPtr[x + 1  + y * source_width + id_z * source_height * source_width];
    C = srcPtr[x + (y + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x+1) + (y+1) * source_width + id_z * source_height * source_width];

    pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                   B * (x_diff) * (1 - y_diff) +
                   C * (y_diff) * (1 - x_diff) +
                   D * (x_diff * y_diff));

    dstPtr[pixId] = saturate_8u(pixVal);
}

extern "C" __global__ void resize_pkd(unsigned char *srcPtr,
                                      unsigned char *dstPtr,
                                      const unsigned int source_height,
                                      const unsigned int source_width,
                                      const unsigned int dest_height,
                                      const unsigned int dest_width,
                                      const unsigned int channel)
{
    int A, B, C, D, x, y, index, pixVal;
    float x_ratio = ((float)(source_width -1 )) / dest_width;
    float y_ratio = ((float)(source_height -1 )) / dest_height;
    float x_diff, y_diff, ya, yb;

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel)
    {
        return;
    }

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    unsigned int pixId;
    pixId = id_x * channel + id_y * dest_width * channel + id_z;

    A = srcPtr[x * channel + y * source_width * channel + id_z];
    B = srcPtr[(x + 1) * channel + y * source_width * channel + id_z];
    C = srcPtr[x * channel + (y + 1) * source_width * channel + id_z];
    D = srcPtr[(x + 1) * channel + (y + 1) * source_width * channel + id_z];

    pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                   B * (x_diff) * (1 - y_diff) +
                   C * (y_diff) * (1 - x_diff) +
                   D * (x_diff * y_diff));

    dstPtr[pixId] = saturate_8u(pixVal);
}

extern "C" __global__ void resize_crop_pln(unsigned char *srcPtr,
                                           unsigned char *dstPtr,
                                           const unsigned int source_height,
                                           const unsigned int source_width,
                                           const unsigned int dest_height,
                                           const unsigned int dest_width,
                                           const unsigned int x1,
                                           const unsigned int y1,
                                           const unsigned int x2,
                                           const unsigned int y2,
                                           const unsigned int padding,
                                           const unsigned int type,
                                           const unsigned int channel)
{
    int A, B, C, D, x, y, index, pixVal;
    float x_ratio = ((float)(x2 - x1 )) / dest_width;
    float y_ratio = ((float)(y2 - y1 )) / dest_height;
    float x_diff, y_diff, ya, yb;
    A = B = C = D = 0;

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel)
    {
        return;
    }

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    unsigned int pixId;

    if(type == 0)
    {
        pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    }
    else
    {
        pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height + ((dest_width + padding * 2) * padding) + (id_y * padding * 2) + (padding);
    }

    A = srcPtr[(x + x1) + (y + y1) * source_width + id_z * source_height * source_width];
    B = srcPtr[(x + x1 + 1)  + (y + y1) * source_width + id_z * source_height * source_width];
    C = srcPtr[(x + x1) + (y + y1 + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x + x1 + 1) + (y + y1 + 1) * source_width + id_z * source_height * source_width];

    pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                   B * (x_diff) * (1 - y_diff) +
                   C * (y_diff) * (1 - x_diff) +
                   D * (x_diff * y_diff));

    dstPtr[pixId] = saturate_8u(pixVal);
}

extern "C" __global__ void resize_crop_pkd(unsigned char *srcPtr,
                                           unsigned char *dstPtr,
                                           const unsigned int source_height,
                                           const unsigned int source_width,
                                           const unsigned int dest_height,
                                           const unsigned int dest_width,
                                           const unsigned int x1,
                                           const unsigned int y1,
                                           const unsigned int x2,
                                           const unsigned int y2,
                                           const unsigned int padding,
                                           const unsigned int type,
                                           const unsigned int channel)
{
    int A, B, C, D, x, y, index, pixVal;
    float x_ratio = ((float)(x2 - x1 )) / dest_width;
    float y_ratio = ((float)(y2 - y1 )) / dest_height;
    float x_diff, y_diff, ya, yb;
    A = B = C = D = 0;

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel)
    {
        return;
    }

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    unsigned int pixId;

    if(type == 0)
    {
        pixId = id_x * channel + id_y * dest_width * channel + id_z;
    }
    else
    {
        pixId = id_x * channel + id_y * dest_width * channel + id_z + ((dest_width + padding * 2) * channel * padding) + (id_y * padding * 2 * channel) + (padding * channel);
    }

    A = srcPtr[(x + x1) * channel + (y + y1) * source_width * channel + id_z];
    B = srcPtr[(x + x1 + 1) * channel + (y + y1) * source_width * channel + id_z];
    C = srcPtr[(x + x1) * channel + (y + y1 + 1) * source_width * channel + id_z];
    D = srcPtr[(x + x1 + 1) * channel + (y + y1 + 1) * source_width * channel + id_z];

    pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                   B * (x_diff) * (1 - y_diff) +
                   C * (y_diff) * (1 - x_diff) +
                   D * (x_diff * y_diff));

    dstPtr[pixId] = saturate_8u(pixVal);
}

extern "C" __global__ void resize_batch(unsigned char *srcPtr,
                                        unsigned char *dstPtr,
                                        unsigned int *source_height,
                                        unsigned int *source_width,
                                        unsigned int *dest_height,
                                        unsigned int *dest_width,
                                        unsigned int *max_source_width,
                                        unsigned int *max_dest_width,
                                        unsigned long *source_batch_index,
                                        unsigned long *dest_batch_index,
                                        const unsigned int channel,
                                        unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                        unsigned int *dest_inc,
                                        const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }
    
    float x_ratio = ((float)(source_width[id_z] -1 )) / dest_width[id_z];
    float y_ratio = ((float)(source_height[id_z] -1 )) / dest_height[id_z];
    int x = (int)(x_ratio * id_x);
    int y = (int)(y_ratio * id_y);
    float x_diff = (x_ratio * id_x) - x;
    float y_diff = (y_ratio * id_y) - y;
    unsigned long dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * plnpkdindex;

    for(int indextmp = 0; indextmp < channel; indextmp++)
    {
        int A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * plnpkdindex + indextmp * source_inc[id_z]];
        int B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * plnpkdindex + indextmp * source_inc[id_z]];
        int C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * plnpkdindex + indextmp * source_inc[id_z]];
        int D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * plnpkdindex + indextmp * source_inc[id_z]];

        int pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                           B * (x_diff) * (1 - y_diff) +
                           C * (y_diff) * (1 - x_diff) +
                           D * (x_diff * y_diff));

        dstPtr[dst_pixIdx] = saturate_8u(pixVal);
        dst_pixIdx += dest_inc[id_z];
    }
}

extern "C" __global__ void resize_crop_batch(unsigned char *srcPtr,
                                             unsigned char *dstPtr,
                                             unsigned int *source_height,
                                             unsigned int *source_width,
                                             unsigned int *dest_height,
                                             unsigned int *dest_width,
                                             unsigned int *max_source_width,
                                             unsigned int *max_dest_width,
                                             unsigned int *xroi_begin,
                                             unsigned int *xroi_end,
                                             unsigned int *yroi_begin,
                                             unsigned int *yroi_end,
                                             unsigned long long *source_batch_index,
                                             unsigned long long *dest_batch_index,
                                             const unsigned int channel,
                                             unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                             unsigned int *dest_inc,
                                             const unsigned int padding,
                                             const unsigned int type,
                                             const int in_plnpkdind, // use 1 pln 3 for pkd
                                             const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    int x = (int)(x_ratio * id_x);
    int y = (int)(y_ratio * id_y);
    float x_diff = (x_ratio * id_x) - x;
    float y_diff = (y_ratio * id_y) - y;
    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;
    unsigned long dst_pixIdx = 0;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            int A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                               B * (x_diff) * (1 - y_diff) +
                               C * (y_diff) * (1 - x_diff) +
                               D * (x_diff * y_diff));
            dstPtr[dst_pixIdx] = saturate_8u(pixVal);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void resize_nn_crop_batch(unsigned char *srcPtr,
                                             unsigned char *dstPtr,
                                             unsigned int *source_height,
                                             unsigned int *source_width,
                                             unsigned int *dest_height,
                                             unsigned int *dest_width,
                                             unsigned int *max_source_width,
                                             unsigned int *max_dest_width,
                                             unsigned int *xroi_begin,
                                             unsigned int *xroi_end,
                                             unsigned int *yroi_begin,
                                             unsigned int *yroi_end,
                                             unsigned long long *source_batch_index,
                                             unsigned long long *dest_batch_index,
                                             const unsigned int channel,
                                             unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                             unsigned int *dest_inc,
                                             const unsigned int padding,
                                             const unsigned int type,
                                             const int in_plnpkdind, // use 1 pln 3 for pkd
                                             const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z])) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z])) / dest_height[id_z];
    int x = (int)(xroi_begin[id_z] + (x_ratio * (id_x + 0.5f)));
    int y = (int)(yroi_begin[id_z] + (y_ratio * (id_y + 0.5f)));
    unsigned long dst_pixIdx, src_pixIdx;

    if (x < source_width[id_z] && y < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        src_pixIdx = source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = saturate_8u(srcPtr[src_pixIdx + indextmp * source_inc[id_z]]);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}
extern "C" __global__ void resize_cubic_crop_batch(unsigned char *srcPtr,
                                             unsigned char *dstPtr,
                                             unsigned int *source_height,
                                             unsigned int *source_width,
                                             unsigned int *dest_height,
                                             unsigned int *dest_width,
                                             unsigned int *max_source_width,
                                             unsigned int *max_dest_width,
                                             unsigned int *xroi_begin,
                                             unsigned int *xroi_end,
                                             unsigned int *yroi_begin,
                                             unsigned int *yroi_end,
                                             unsigned long long *source_batch_index,
                                             unsigned long long *dest_batch_index,
                                             const unsigned int channel,
                                             unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                             unsigned int *dest_inc,
                                             const unsigned int padding,
                                             const unsigned int type,
                                             const int in_plnpkdind, // use 1 pln 3 for pkd
                                             const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }
    int kernel_size = 4;
    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z])) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z])) / dest_height[id_z];
    float xf = (x_ratio * (id_x + 0.5f) - 0.5f);
    float yf = (y_ratio * (id_y + 0.5f) - 0.5f);
    int x = (int)xf;
    int y = (int)yf;
    float x_diff = xf - x;
    float y_diff = yf - y;
    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;
    unsigned long dst_pixIdx = 0;
    float A, B, C, D, coeffs_x[4], coeffs_y[4];
    int width_limit = source_width[id_z] - 1;
    int height_limit = source_height[id_z] - 1;
    if (x < source_width[id_z] && y < source_height[id_z])
    {
        CalculateCubicCoefficients(coeffs_x, x_diff);
        CalculateCubicCoefficients(coeffs_y, y_diff);
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            A = B = C = D = 0;
            for (int k = 0; k < kernel_size; k++)
            {
                int xIdx = RANGE_CHECK(x + 1 + k - (kernel_size / 2), 0, width_limit);
                A += srcPtr[source_batch_index[id_z] + (xIdx + (RANGE_CHECK((y - 1), 0, height_limit) * max_source_width[id_z])) * in_plnpkdind + indextmp * source_inc[id_z]] * coeffs_x[k];
                B += srcPtr[source_batch_index[id_z] + (xIdx + (RANGE_CHECK(y, 0, height_limit)  * max_source_width[id_z])) * in_plnpkdind + indextmp * source_inc[id_z]] * coeffs_x[k];
                C += srcPtr[source_batch_index[id_z] + (xIdx + (RANGE_CHECK((y + 1), 0, height_limit) * max_source_width[id_z])) * in_plnpkdind + indextmp * source_inc[id_z]] * coeffs_x[k];
                D += srcPtr[source_batch_index[id_z] + (xIdx + (RANGE_CHECK((y + 2), 0, height_limit) * max_source_width[id_z])) * in_plnpkdind + indextmp * source_inc[id_z]] * coeffs_x[k];
            }            
            dstPtr[dst_pixIdx] = saturate_8u(A * coeffs_y[0] + B * coeffs_y[1] + C * coeffs_y[2] + D * coeffs_y[3]);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}


extern "C" __global__ void resize_crop_batch_int8(signed char *srcPtr,
                                                  signed char *dstPtr,
                                                  unsigned int *source_height,
                                                  unsigned int *source_width,
                                                  unsigned int *dest_height,
                                                  unsigned int *dest_width,
                                                  unsigned int *max_source_width,
                                                  unsigned int *max_dest_width,
                                                  unsigned int *xroi_begin,
                                                  unsigned int *xroi_end,
                                                  unsigned int *yroi_begin,
                                                  unsigned int *yroi_end,
                                                  unsigned long long *source_batch_index,
                                                  unsigned long long *dest_batch_index,
                                                  const unsigned int channel,
                                                  unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                  unsigned int *dest_inc,
                                                  const unsigned int padding,
                                                  const unsigned int type,
                                                  const int in_plnpkdind, // use 1 pln 3 for pkd
                                                  const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    int x = (int)(x_ratio * id_x);
    int y = (int)(y_ratio * id_y);
    float x_diff = (x_ratio * id_x) - x;
    float y_diff = (y_ratio * id_y) - y;
    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;
    unsigned long dst_pixIdx = 0;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            int A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            signed char pixVal = (signed char)(A * (1 - x_diff) * (1 - y_diff) +
                                               B * (x_diff) * (1 - y_diff) +
                                               C * (y_diff) * (1 - x_diff) +
                                               D * (x_diff * y_diff));
            dstPtr[dst_pixIdx] = pixVal;
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void resize_nn_crop_batch_int8(signed char *srcPtr,
                                                     signed char *dstPtr,
                                                     unsigned int *source_height,
                                                     unsigned int *source_width,
                                                     unsigned int *dest_height,
                                                     unsigned int *dest_width,
                                                     unsigned int *max_source_width,
                                                     unsigned int *max_dest_width,
                                                     unsigned int *xroi_begin,
                                                     unsigned int *xroi_end,
                                                     unsigned int *yroi_begin,
                                                     unsigned int *yroi_end,
                                                     unsigned long long *source_batch_index,
                                                     unsigned long long *dest_batch_index,
                                                     const unsigned int channel,
                                                     unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                     unsigned int *dest_inc,
                                                     const unsigned int padding,
                                                     const unsigned int type,
                                                     const int in_plnpkdind, // use 1 pln 3 for pkd
                                                     const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z])) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z])) / dest_height[id_z];
    int x = (int)(xroi_begin[id_z] + (x_ratio * (id_x + 0.5f)));
    int y = (int)(yroi_begin[id_z] + (y_ratio * (id_y + 0.5f)));
    unsigned long dst_pixIdx, src_pixIdx;

    if (x < source_width[id_z] && y < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        src_pixIdx = source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = (signed char)(srcPtr[src_pixIdx + indextmp * source_inc[id_z]]);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}


// extern "C" __global__ void resize_crop_batch_fp16(
//     half *srcPtr, half *dstPtr,
//     unsigned int *source_height, unsigned int *source_width,
//     unsigned int *dest_height, unsigned int *dest_width,
//     unsigned int *max_source_width,
//     unsigned int *max_dest_width, int *xroi_begin,
//     int *xroi_end, int *yroi_begin, int *yroi_end,
//     unsigned long *source_batch_index,
//     unsigned long *dest_batch_index, const unsigned int channel,
//     unsigned int
//         *source_inc, // use width * height for pln and 1 for pkd
//     unsigned int *dest_inc, const unsigned int padding,
//     const unsigned int type,
//     const int in_plnpkdind, const int out_plnpkdind  // use 1 pln 3 for pkd
// ) {
//   int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//   int x, y;
//   float x_ratio =
//       ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
//   float y_ratio =
//       ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
//   float x_diff, y_diff, ya, yb;

//   unsigned long dst_pixIdx = 0;

//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   x = (int)(x_ratio * id_x);
//   y = (int)(y_ratio * id_y);

//   x_diff = (x_ratio * id_x) - x;
//   y_diff = (y_ratio * id_y) - y;

//   x = xroi_begin[id_z] + x;
//   y = yroi_begin[id_z] + y;

//   if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       float A = srcPtr[source_batch_index[id_z] +
//                  (x + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float B = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float C = srcPtr[source_batch_index[id_z] +
//                  (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float D = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];

//       float pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
//                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
//       dstPtr[dst_pixIdx] = (half)pixVal;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   } else {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = 0;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void resize_crop_batch_fp32(float *srcPtr,
                                                  float *dstPtr,
                                                  unsigned int *source_height,
                                                  unsigned int *source_width,
                                                  unsigned int *dest_height,
                                                  unsigned int *dest_width,
                                                  unsigned int *max_source_width,
                                                  unsigned int *max_dest_width,
                                                  unsigned int *xroi_begin,
                                                  unsigned int *xroi_end,
                                                  unsigned int *yroi_begin,
                                                  unsigned int *yroi_end,
                                                  unsigned long long *source_batch_index,
                                                  unsigned long long *dest_batch_index,
                                                  const unsigned int channel,
                                                  unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                  unsigned int *dest_inc,
                                                  const unsigned int padding,
                                                  const unsigned int type,
                                                  const int in_plnpkdind, // use 1 pln 3 for pkd
                                                  const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    int x = (int)(x_ratio * id_x);
    int y = (int)(y_ratio * id_y);
    float x_diff = (x_ratio * id_x) - x;
    float y_diff = (y_ratio * id_y) - y;
    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;
    unsigned long dst_pixIdx = 0;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            float A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float pixVal = A * (1 - x_diff) * (1 - y_diff) +
                        B * (x_diff) * (1 - y_diff) +
                        C * (y_diff) * (1 - x_diff) +
                        D * (x_diff * y_diff);
            dstPtr[dst_pixIdx] = pixVal;
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void resize_nn_crop_batch_fp32(float *srcPtr,
                                                    float *dstPtr,
                                                    unsigned int *source_height,
                                                    unsigned int *source_width,
                                                    unsigned int *dest_height,
                                                    unsigned int *dest_width,
                                                    unsigned int *max_source_width,
                                                    unsigned int *max_dest_width,
                                                    unsigned int *xroi_begin,
                                                    unsigned int *xroi_end,
                                                    unsigned int *yroi_begin,
                                                    unsigned int *yroi_end,
                                                    unsigned long long *source_batch_index,
                                                    unsigned long long *dest_batch_index,
                                                    const unsigned int channel,
                                                    unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                    unsigned int *dest_inc,
                                                    const unsigned int padding,
                                                    const unsigned int type,
                                                    const int in_plnpkdind, // use 1 pln 3 for pkd
                                                    const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z])) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z])) / dest_height[id_z];
    int x = (int)(xroi_begin[id_z] + (x_ratio * (id_x + 0.5f)));
    int y = (int)(yroi_begin[id_z] + (y_ratio * (id_y + 0.5f)));
    unsigned long dst_pixIdx, src_pixIdx;

    if (x < source_width[id_z] && y < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        src_pixIdx = source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = srcPtr[src_pixIdx + indextmp * source_inc[id_z]];
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void resize_crop_batch_u8_fp32(unsigned char *srcPtr,
                                                     float *dstPtr,
                                                     unsigned int *source_height,
                                                     unsigned int *source_width,
                                                     unsigned int *dest_height,
                                                     unsigned int *dest_width,
                                                     unsigned int *max_source_width,
                                                     unsigned int *max_dest_width,
                                                     unsigned int *xroi_begin,
                                                     unsigned int *xroi_end,
                                                     unsigned int *yroi_begin,
                                                     unsigned int *yroi_end,
                                                     unsigned long long *source_batch_index,
                                                     unsigned long long *dest_batch_index,
                                                     const unsigned int channel,
                                                     unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                     unsigned int *dest_inc,
                                                     const unsigned int padding,
                                                     const unsigned int type,
                                                     const int in_plnpkdind, // use 1 pln 3 for pkd
                                                     const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    int x = (int)(x_ratio * id_x);
    int y = (int)(y_ratio * id_y);
    float x_diff = (x_ratio * id_x) - x;
    float y_diff = (y_ratio * id_y) - y;
    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;
    unsigned long dst_pixIdx = 0;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            float A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float pixVal = A * (1 - x_diff) * (1 - y_diff) +
                        B * (x_diff) * (1 - y_diff) +
                        C * (y_diff) * (1 - x_diff) +
                        D * (x_diff * y_diff);
            dstPtr[dst_pixIdx] = pixVal * 0.00392157f;
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void resize_nn_crop_batch_u8_fp32(unsigned char *srcPtr,
                                                        float *dstPtr,
                                                        unsigned int *source_height,
                                                        unsigned int *source_width,
                                                        unsigned int *dest_height,
                                                        unsigned int *dest_width,
                                                        unsigned int *max_source_width,
                                                        unsigned int *max_dest_width,
                                                        unsigned int *xroi_begin,
                                                        unsigned int *xroi_end,
                                                        unsigned int *yroi_begin,
                                                        unsigned int *yroi_end,
                                                        unsigned long long *source_batch_index,
                                                        unsigned long long *dest_batch_index,
                                                        const unsigned int channel,
                                                        unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                        unsigned int *dest_inc,
                                                        const unsigned int padding,
                                                        const unsigned int type,
                                                        const int in_plnpkdind, // use 1 pln 3 for pkd
                                                        const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z])) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z])) / dest_height[id_z];
    int x = (int)(xroi_begin[id_z] + (x_ratio * (id_x + 0.5f)));
    int y = (int)(yroi_begin[id_z] + (y_ratio * (id_y + 0.5f)));
    unsigned long dst_pixIdx, src_pixIdx;

    if (x < source_width[id_z] && y < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        src_pixIdx = source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = srcPtr[src_pixIdx + indextmp * source_inc[id_z]] * 0.00392157f;
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

// extern "C" __global__ void resize_crop_batch_u8_fp16(
//     unsigned char *srcPtr, half *dstPtr,
//     unsigned int *source_height, unsigned int *source_width,
//     unsigned int *dest_height, unsigned int *dest_width,
//     unsigned int *max_source_width,
//     unsigned int *max_dest_width, int *xroi_begin,
//     int *xroi_end, int *yroi_begin, int *yroi_end,
//     unsigned long *source_batch_index,
//     unsigned long *dest_batch_index, const unsigned int channel,
//     unsigned int *source_inc, unsigned int *dest_inc, // use width * height for pln and 1 for pkd
//     const unsigned int padding, const unsigned int type,
//     const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
// ) {
//   int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//   int x, y, index;
//   float x_ratio =
//       ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
//   float y_ratio =
//       ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
//   float x_diff, y_diff, ya, yb;

//   unsigned long dst_pixIdx = 0;

//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   x = (int)(x_ratio * id_x);
//   y = (int)(y_ratio * id_y);

//   x_diff = (x_ratio * id_x) - x;
//   y_diff = (y_ratio * id_y) - y;

//   x = xroi_begin[id_z] + x;
//   y = yroi_begin[id_z] + y;

//   if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       half A = srcPtr[source_batch_index[id_z] +
//                  (x + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       half B = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       half C = srcPtr[source_batch_index[id_z] +
//                  (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       half D = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];

//       half pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
//                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
//       dstPtr[dst_pixIdx] = (half)(pixVal/255.0);
//       dst_pixIdx += dest_inc[id_z];
//     }
//   } else {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = 0;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void resize_crop_batch_u8_int8(unsigned char *srcPtr,
                                                     signed char *dstPtr,
                                                     unsigned int *source_height,
                                                     unsigned int *source_width,
                                                     unsigned int *dest_height,
                                                     unsigned int *dest_width,
                                                     unsigned int *max_source_width,
                                                     unsigned int *max_dest_width,
                                                     unsigned int *xroi_begin,
                                                     unsigned int *xroi_end,
                                                     unsigned int *yroi_begin,
                                                     unsigned int *yroi_end,
                                                     unsigned long long *source_batch_index,
                                                     unsigned long long *dest_batch_index,
                                                     const unsigned int channel,
                                                     unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                     unsigned int *dest_inc,
                                                     const unsigned int padding,
                                                     const unsigned int type,
                                                     const int in_plnpkdind, // use 1 pln 3 for pkd
                                                     const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    int x = (int)(x_ratio * id_x);
    int y = (int)(y_ratio * id_y);
    float x_diff = (x_ratio * id_x) - x;
    float y_diff = (y_ratio * id_y) - y;
    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;
    unsigned long dst_pixIdx = 0;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            float A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float pixVal = A * (1 - x_diff) * (1 - y_diff) +
                           B * (x_diff) * (1 - y_diff) +
                           C * (y_diff) * (1 - x_diff) +
                           D * (x_diff * y_diff);
            dstPtr[dst_pixIdx] = (signed char)(pixVal - 128);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = -128;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void resize_nn_crop_batch_u8_int8(unsigned char *srcPtr,
                                                        signed char *dstPtr,
                                                        unsigned int *source_height,
                                                        unsigned int *source_width,
                                                        unsigned int *dest_height,
                                                        unsigned int *dest_width,
                                                        unsigned int *max_source_width,
                                                        unsigned int *max_dest_width,
                                                        unsigned int *xroi_begin,
                                                        unsigned int *xroi_end,
                                                        unsigned int *yroi_begin,
                                                        unsigned int *yroi_end,
                                                        unsigned long long *source_batch_index,
                                                        unsigned long long *dest_batch_index,
                                                        const unsigned int channel,
                                                        unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                        unsigned int *dest_inc,
                                                        const unsigned int padding,
                                                        const unsigned int type,
                                                        const int in_plnpkdind, // use 1 pln 3 for pkd
                                                        const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z])) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z])) / dest_height[id_z];
    int x = (int)(xroi_begin[id_z] + (x_ratio * (id_x + 0.5f)));
    int y = (int)(yroi_begin[id_z] + (y_ratio * (id_y + 0.5f)));
    unsigned long dst_pixIdx, src_pixIdx;

    if (x < source_width[id_z] && y < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        src_pixIdx = source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = (signed char)(srcPtr[src_pixIdx + indextmp * source_inc[id_z]] - 128);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = -128;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void resize_crop_mirror_batch(unsigned char *srcPtr,
                                                    unsigned char *dstPtr,
                                                    unsigned int *source_height,
                                                    unsigned int *source_width,
                                                    unsigned int *dest_height,
                                                    unsigned int *dest_width,
                                                    unsigned int *max_source_width,
                                                    unsigned int *max_dest_width,
                                                    unsigned int *xroi_begin,
                                                    unsigned int *xroi_end,
                                                    unsigned int *yroi_begin,
                                                    unsigned int *yroi_end,
                                                    unsigned int *mirror,
                                                    unsigned long long *source_batch_index,
                                                    unsigned long long *dest_batch_index,
                                                    const unsigned int channel,
                                                    unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                    unsigned int *dest_inc,
                                                    const int in_plnpkdind, // use 1 pln 3 for pkd
                                                    const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int x, y;
    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    float x_diff, y_diff, ya, yb;

    unsigned long dst_pixIdx = 0;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            int A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];

            int pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                               B * (x_diff) * (1 - y_diff) +
                               C * (y_diff) * (1 - x_diff) +
                               D * (x_diff * y_diff));

            dstPtr[dst_pixIdx] = saturate_8u(pixVal);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void resize_crop_mirror_batch_int8(signed char *srcPtr,
                                                         signed char *dstPtr,
                                                         unsigned int *source_height,
                                                         unsigned int *source_width,
                                                         unsigned int *dest_height,
                                                         unsigned int *dest_width,
                                                         unsigned int *max_source_width,
                                                         unsigned int *max_dest_width,
                                                         unsigned int *xroi_begin,
                                                         unsigned int *xroi_end,
                                                         unsigned int *yroi_begin,
                                                         unsigned int *yroi_end,
                                                         unsigned int *mirror,
                                                         unsigned long long *source_batch_index,
                                                         unsigned long long *dest_batch_index,
                                                         const unsigned int channel,
                                                         unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                         unsigned int *dest_inc,
                                                         const int in_plnpkdind, // use 1 pln 3 for pkd
                                                         const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int x, y;
    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    float x_diff, y_diff, ya, yb;

    unsigned long dst_pixIdx = 0;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) * out_plnpkdind;

        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            int A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];

            int pixVal = (signed char)(A * (1 - x_diff) * (1 - y_diff) +
                                B * (x_diff) * (1 - y_diff) +
                                C * (y_diff) * (1 - x_diff) +
                                D * (x_diff * y_diff));

            dstPtr[dst_pixIdx] = pixVal;
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = -128;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

// extern "C" __global__ void resize_crop_mirror_batch_fp16(
//     half *srcPtr, half *dstPtr,
//     unsigned int *source_height, unsigned int *source_width,
//     unsigned int *dest_height, unsigned int *dest_width,
//     unsigned int *max_source_width,
//     unsigned int *max_dest_width, int *xroi_begin,
//     int *xroi_end, int *yroi_begin, int *yroi_end,
//     int *mirror, unsigned long *source_batch_index,
//     unsigned long *dest_batch_index, const unsigned int channel,
//     unsigned int
//         *source_inc, // use width * height for pln and 1 for pkd
//     unsigned int *dest_inc,
//     const int in_plnpkdind, const int out_plnpkdind  // use 1 pln 3 for pkd
// ) {
//   int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//   int x, y, index;
//   float x_ratio =
//       ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
//   float y_ratio =
//       ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
//   float x_diff, y_diff, ya, yb;

//   unsigned long dst_pixIdx = 0;

//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   x = (int)(x_ratio * id_x);
//   y = (int)(y_ratio * id_y);

//   x_diff = (x_ratio * id_x) - x;
//   y_diff = (y_ratio * id_y) - y;

//   x = xroi_begin[id_z] + x;
//   y = yroi_begin[id_z] + y;

//   if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
//                      out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       float A = srcPtr[source_batch_index[id_z] +
//                  (x + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float B = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float C = srcPtr[source_batch_index[id_z] +
//                  (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float D = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];

//       float pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
//                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
//       dstPtr[dst_pixIdx] = (half)pixVal;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   } else {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = 0;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void resize_crop_mirror_batch_fp32(float *srcPtr,
                                                         float *dstPtr,
                                                         unsigned int *source_height,
                                                         unsigned int *source_width,
                                                         unsigned int *dest_height,
                                                         unsigned int *dest_width,
                                                         unsigned int *max_source_width,
                                                         unsigned int *max_dest_width,
                                                         unsigned int *xroi_begin,
                                                         unsigned int *xroi_end,
                                                         unsigned int *yroi_begin,
                                                         unsigned int *yroi_end,
                                                         unsigned int *mirror,
                                                         unsigned long long *source_batch_index,
                                                         unsigned long long *dest_batch_index,
                                                         const unsigned int channel,
                                                         unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                         unsigned int *dest_inc,
                                                         const int in_plnpkdind, // use 1 pln 3 for pkd
                                                         const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int x, y, index;
    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    float x_diff, y_diff, ya, yb;

    unsigned long dst_pixIdx = 0;

    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    {
        return;
    }

    x = (int)(x_ratio * id_x);
    y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) * out_plnpkdind;

        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            float A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            float D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];

            float pixVal = A * (1 - x_diff) * (1 - y_diff) +
                           B * (x_diff) * (1 - y_diff) +
                           C * (y_diff) * (1 - x_diff) +
                           D * (x_diff * y_diff);

            dstPtr[dst_pixIdx] = pixVal;
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (int indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

extern "C" __global__ void random_crop_letterbox_batch(unsigned char *srcPtr,
                                                       unsigned char *dstPtr,
                                                       unsigned int *source_height,
                                                       unsigned int *source_width,
                                                       unsigned int *dest_height,
                                                       unsigned int *dest_width,
                                                       unsigned int *max_source_width,
                                                       unsigned int *max_dest_width,
                                                       unsigned int *xroi_begin,
                                                       unsigned int *xroi_end,
                                                       unsigned int *yroi_begin,
                                                       unsigned int *yroi_end,
                                                       unsigned long long *source_batch_index,
                                                       unsigned long long *dest_batch_index,
                                                       const unsigned int channel,
                                                       unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                                       unsigned int *dest_inc,
                                                       unsigned int padding,
                                                       const unsigned int type,
                                                       const int in_plnpkdind, // use 1 pln 3 for pkd
                                                       const int out_plnpkdind)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float x_ratio = ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
    float y_ratio = ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
    float x_diff, y_diff, ya, yb;

    int indextmp = 0;
    unsigned long dst_pixIdx = 0;
    unsigned int minVal = ((dest_height[id_z] < dest_width[id_z]) ? dest_height[id_z] : dest_width[id_z]);
    padding = (5 * minVal / 100);

    if (id_x >= dest_width[id_z] - padding || id_y >= dest_height[id_z] - padding || id_x < padding || id_y < padding)
    {
        return;
    }

    int x = (int)(x_ratio * id_x);
    int y = (int)(y_ratio * id_y);

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y;

    x = xroi_begin[id_z] + x;
    y = yroi_begin[id_z] + y;

    if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z])
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            int A = srcPtr[source_batch_index[id_z] + (x + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int B = srcPtr[source_batch_index[id_z] + ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int C = srcPtr[source_batch_index[id_z] + (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];
            int D = srcPtr[source_batch_index[id_z] + ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind + indextmp * source_inc[id_z]];

            int pixVal = (int)(A * (1 - x_diff) * (1 - y_diff) +
                               B * (x_diff) * (1 - y_diff) +
                               C * (y_diff) * (1 - x_diff) +
                               D * (x_diff * y_diff));

            dstPtr[dst_pixIdx] = saturate_8u(pixVal);
            dst_pixIdx += dest_inc[id_z];
        }
    }
    else
    {
        dst_pixIdx = dest_batch_index[id_z] + (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
        for (indextmp = 0; indextmp < channel; indextmp++)
        {
            dstPtr[dst_pixIdx] = 0;
            dst_pixIdx += dest_inc[id_z];
        }
    }
}

RppStatus hip_exec_resize_crop_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiResizeInterpType interpType)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *x, *roiWidth, *y, *roiHeight;
    if (type == 0)
    {
        x = handle.GetInitHandle()->mem.mgpu.roiPoints.x;
        roiWidth = handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth;
        y = handle.GetInitHandle()->mem.mgpu.roiPoints.y;
        roiHeight = handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight;
    }
    else if (type == 1)
    {
        x = handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem;
        roiWidth = handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem;
        y = handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem;
        roiHeight = handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem;
    }

    if(interpType == RppiResizeInterpType::NEAREST_NEIGHBOR)
    {
        hipLaunchKernelGGL(resize_nn_crop_batch,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }
    else if(interpType == RppiResizeInterpType::CUBIC)
    {
        hipLaunchKernelGGL(resize_cubic_crop_batch,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);  
    }
    else
    {
        hipLaunchKernelGGL(resize_crop_batch,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_batch_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiResizeInterpType interpType)
{
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = max_width;
    // int globalThreads_y = max_height;
    // int globalThreads_z = handle.GetBatchSize();

    // Rpp32u *x, *roiWidth, *y, *roiHeight;
    // if (type == 0)
    // {
    //     x = handle.GetInitHandle()->mem.mgpu.roiPoints.x;
    //     roiWidth = handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth;
    //     y = handle.GetInitHandle()->mem.mgpu.roiPoints.y;
    //     roiHeight = handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight;
    // }
    // else if (type == 1)
    // {
    //     x = handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem;
    //     roiWidth = handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem;
    //     y = handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem;
    //     roiHeight = handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem;
    // }

    // hipLaunchKernelGGL(resize_crop_batch_u8_fp16,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    handle.GetStream(),
    //                    srcPtr,
    //                    dstPtr,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
    //                    x,
    //                    roiWidth,
    //                    y,
    //                    roiHeight,
    //                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
    //                    handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
    //                    tensor_info._in_channels,
    //                    handle.GetInitHandle()->mem.mgpu.inc,
    //                    handle.GetInitHandle()->mem.mgpu.dstInc,
    //                    padding,
    //                    type,
    //                    in_plnpkdind,
    //                    out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_batch_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiResizeInterpType interpType)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *x, *roiWidth, *y, *roiHeight;
    if (type == 0)
    {
        x = handle.GetInitHandle()->mem.mgpu.roiPoints.x;
        roiWidth = handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth;
        y = handle.GetInitHandle()->mem.mgpu.roiPoints.y;
        roiHeight = handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight;
    }
    else if (type == 1)
    {
        x = handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem;
        roiWidth = handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem;
        y = handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem;
        roiHeight = handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem;
    }
    if(interpType == RppiResizeInterpType::NEAREST_NEIGHBOR)
    {
        hipLaunchKernelGGL(resize_nn_crop_batch_u8_fp32,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }
    else
    {
        hipLaunchKernelGGL(resize_crop_batch_u8_fp32,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_batch_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiResizeInterpType interpType)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *x, *roiWidth, *y, *roiHeight;
    if (type == 0)
    {
        x = handle.GetInitHandle()->mem.mgpu.roiPoints.x;
        roiWidth = handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth;
        y = handle.GetInitHandle()->mem.mgpu.roiPoints.y;
        roiHeight = handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight;
    }
    else if (type == 1)
    {
        x = handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem;
        roiWidth = handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem;
        y = handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem;
        roiHeight = handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem;
    }
    if(interpType == RppiResizeInterpType::NEAREST_NEIGHBOR)
    {
        hipLaunchKernelGGL(resize_nn_crop_batch_u8_int8,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }
    else
    {
        hipLaunchKernelGGL(resize_crop_batch_u8_int8,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiResizeInterpType interpType)
{
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = max_width;
    // int globalThreads_y = max_height;
    // int globalThreads_z = handle.GetBatchSize();

    // Rpp32u *x, *roiWidth, *y, *roiHeight;
    // if (type == 0)
    // {
    //     x = handle.GetInitHandle()->mem.mgpu.roiPoints.x;
    //     roiWidth = handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth;
    //     y = handle.GetInitHandle()->mem.mgpu.roiPoints.y;
    //     roiHeight = handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight;
    // }
    // else if (type == 1)
    // {
    //     x = handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem;
    //     roiWidth = handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem;
    //     y = handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem;
    //     roiHeight = handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem;
    // }

    // hipLaunchKernelGGL(resize_crop_batch_fp16,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    handle.GetStream(),
    //                    srcPtr,
    //                    dstPtr,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
    //                    x,
    //                    roiWidth,
    //                    y,
    //                    roiHeight,
    //                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
    //                    handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
    //                    tensor_info._in_channels,
    //                    handle.GetInitHandle()->mem.mgpu.inc,
    //                    handle.GetInitHandle()->mem.mgpu.dstInc,
    //                    padding,
    //                    type,
    //                    in_plnpkdind,
    //                    out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiResizeInterpType interpType)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *x, *roiWidth, *y, *roiHeight;
    if (type == 0)
    {
        x = handle.GetInitHandle()->mem.mgpu.roiPoints.x;
        roiWidth = handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth;
        y = handle.GetInitHandle()->mem.mgpu.roiPoints.y;
        roiHeight = handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight;
    }
    else if (type == 1)
    {
        x = handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem;
        roiWidth = handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem;
        y = handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem;
        roiHeight = handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem;
    }
    if(interpType == RppiResizeInterpType::NEAREST_NEIGHBOR)
    {
        hipLaunchKernelGGL(resize_nn_crop_batch_fp32,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }
    else
    {
        hipLaunchKernelGGL(resize_crop_batch_fp32,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiResizeInterpType interpType)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *x, *roiWidth, *y, *roiHeight;
    if (type == 0)
    {
        x = handle.GetInitHandle()->mem.mgpu.roiPoints.x;
        roiWidth = handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth;
        y = handle.GetInitHandle()->mem.mgpu.roiPoints.y;
        roiHeight = handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight;
    }
    else if (type == 1)
    {
        x = handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem;
        roiWidth = handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem;
        y = handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem;
        roiHeight = handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem;
    }
    if(interpType == RppiResizeInterpType::NEAREST_NEIGHBOR)
    {
        hipLaunchKernelGGL(resize_nn_crop_batch_int8,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }
    else
    {
        hipLaunchKernelGGL(resize_crop_batch_int8,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        dstPtr,
                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                        x,
                        roiWidth,
                        y,
                        roiHeight,
                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                        tensor_info._in_channels,
                        handle.GetInitHandle()->mem.mgpu.inc,
                        handle.GetInitHandle()->mem.mgpu.dstInc,
                        padding,
                        type,
                        in_plnpkdind,
                        out_plnpkdind);
    }

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_mirror_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(resize_crop_mirror_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                       tensor_info._in_channels,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_mirror_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = max_width;
    // int globalThreads_y = max_height;
    // int globalThreads_z = handle.GetBatchSize();

    // hipLaunchKernelGGL(resize_crop_mirror_batch_fp16,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    handle.GetStream(),
    //                    srcPtr,
    //                    dstPtr,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.height,
    //                    handle.GetInitHandle()->mem.mgpu.dstSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
    //                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
    //                    handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
    //                    tensor_info._in_channels,
    //                    handle.GetInitHandle()->mem.mgpu.inc,
    //                    handle.GetInitHandle()->mem.mgpu.dstInc,
    //                    in_plnpkdind,
    //                    out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_mirror_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(resize_crop_mirror_batch_fp32,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                       tensor_info._in_channels,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_resize_crop_mirror_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(resize_crop_mirror_batch_int8,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                       tensor_info._in_channels,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       in_plnpkdind,
                       out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus hip_exec_random_crop_letterbox_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32u padding, Rpp32u type, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(random_crop_letterbox_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.dstSize.height,
                       handle.GetInitHandle()->mem.mgpu.dstSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       handle.GetInitHandle()->mem.mgpu.dstInc,
                       padding,
                       type,
                       plnpkdind,
                       plnpkdind);

    return RPP_SUCCESS;
}