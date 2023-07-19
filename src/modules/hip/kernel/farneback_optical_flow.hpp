#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "copy.hpp"
#include "resize.hpp"
#include "resize_scale_intensity.hpp"
#include "box_filter_unsaturated.hpp"
#include "gaussian_filter_unsaturated.hpp"
#include "box_filter.hpp"

#include <string>
#include <iomanip>
#include <fstream>
using namespace std;


#define FARNEBACK_FRAME_WIDTH 960                       // Farneback algorithm frame width
#define FARNEBACK_FRAME_HEIGHT 540                      // Farneback algorithm frame height
#define FARNEBACK_OUTPUT_FRAME_SIZE 518400             // 960 * 540
#define FARNEBACK_OUTPUT_MVEC_SIZE 1036800             // 960 * 540 * 2
#define FARNEBACK_FRAME_MIN_SIZE 32                     // set minimum frame size
#define BORDER_SIZE 5                                   // set border size
// #define MAX_MVEC_WIDTH 1920                             // define max motion vector width
// #define MAX_MVEC_HEIGHT 1080                             // define max motion vector height

static const float borderVals[BORDER_SIZE + 1] = {0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f, 1.f};

template <Rpp32s polyExpNbhoodSize>
__global__ void farneback_polynomial_expansion_tensor(float *srcPtr,
                                                      int2 srcStridesNH,
                                                      float *dstPtr,
                                                      int3 dstStridesNCH,
                                                      float *g,
                                                      float *xg,
                                                      float *xxg,
                                                      double4 invG11033355_f4,
                                                      int tileSizeX,
                                                      RpptROIPtr roiTensorPtrSrc)
{
    // Original style
    const int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const int x = hipBlockIdx_x * (hipBlockDim_x - 2 * polyExpNbhoodSize) + hipThreadIdx_x - polyExpNbhoodSize;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // if(
    //     (hipBlockIdx_x == 0) &&
    //     (hipBlockIdx_y == 0) &&
    //     (hipBlockIdx_z == 0) &&
    //     (hipThreadIdx_x == 0) &&
    //     (hipThreadIdx_y == 0) &&
    //     (hipThreadIdx_z == 0)
    // )
    // {
    //     printf("\nInside farneback_polynomial_expansion_tensor:");
    //     printf("\nig11, ig03, ig33, ig55 = %f, %f, %f, %f", (float)invG11033355_f4.x, (float)invG11033355_f4.y, (float)invG11033355_f4.z, (float)invG11033355_f4.w);
    //     for (int i = 0; i < (polyExpNbhoodSize + 1); i++)
    //         printf("\ng[%d], xg[%d], xxg[%d] = %f, %f, %f", i, i, i, g[i], xg[i], xxg[i]);
    // }

    RpptROI roiSrc = roiTensorPtrSrc[id_z];

    if (y < roiSrc.xywhROI.roiHeight)
    {
        extern __shared__ float smem[];
        volatile float *row = smem + hipThreadIdx_x;
        int xWarped = fminf(fmaxf(x, 0), roiSrc.xywhROI.roiWidth - 1);

        row[0] = srcPtr[(y * srcStridesNH.y) + xWarped] * g[0];
        row[hipBlockDim_x] = 0.f;
        row[2 * hipBlockDim_x] = 0.f;

        for (int k = 1; k <= polyExpNbhoodSize; ++k)
        {
            float t0 = srcPtr[((int)fmaxf(y - k, 0) * srcStridesNH.y) + xWarped];
            float t1 = srcPtr[((int)fminf(y + k, roiSrc.xywhROI.roiHeight - 1) * srcStridesNH.y) + xWarped];

            row[0] += g[k] * (t0 + t1);
            row[hipBlockDim_x] += xg[k] * (t1 - t0);
            row[2 * hipBlockDim_x] += xxg[k] * (t0 + t1);
        }

        __syncthreads();

        if ((hipThreadIdx_x >= polyExpNbhoodSize) && (hipThreadIdx_x + polyExpNbhoodSize < hipBlockDim_x) && (x < roiSrc.xywhROI.roiWidth))
        {
            float b1 = g[0] * row[0];
            float b3 = g[0] * row[hipBlockDim_x];
            float b5 = g[0] * row[2 * hipBlockDim_x];
            float b2 = 0, b4 = 0, b6 = 0;

            for (int k = 1; k <= polyExpNbhoodSize; ++k)
            {
                b1 += (row[k] + row[-k]) * g[k];
                b4 += (row[k] + row[-k]) * xxg[k];
                b2 += (row[k] - row[-k]) * xg[k];
                b3 += (row[k + hipBlockDim_x] + row[-k + hipBlockDim_x]) * g[k];
                b6 += (row[k + hipBlockDim_x] - row[-k + hipBlockDim_x]) * xg[k];
                b5 += (row[k + 2 * hipBlockDim_x] + row[-k + 2*hipBlockDim_x]) * g[k];
            }

            dstPtr[(y * dstStridesNCH.z) + xWarped] = b3*invG11033355_f4.x;
            dstPtr[dstStridesNCH.x + (y * dstStridesNCH.z) + xWarped] = b2*invG11033355_f4.x;
            dstPtr[(2 * dstStridesNCH.x) + (y * dstStridesNCH.z) + xWarped] = b1*invG11033355_f4.y + b5*invG11033355_f4.z;
            dstPtr[(3 * dstStridesNCH.x) + (y * dstStridesNCH.z) + xWarped] = b1*invG11033355_f4.y + b4*invG11033355_f4.z;
            dstPtr[(4 * dstStridesNCH.x) + (y * dstStridesNCH.z) + xWarped] = b6*invG11033355_f4.w;
        }
    }











    // Modified Style
    // int id_x = hipBlockIdx_x * tileSizeX + hipThreadIdx_x - polyExpNbhoodSize;
    // int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // // if ((id_y == 0) && (id_z == 0))
    // // {
    // //     printf("\n id_x = %d", id_x);
    // // }

    // RpptROI roiSrc = roiTensorPtrSrc[id_z];

    // int srcIdx = id_z * srcStridesNH.x;          // + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    // int dstIdx = (id_z * dstStridesNCH.x * 5);   // + (id_y * dstStridesNH.y) + id_x;

    // // if ((id_x == 0) && (id_y == 0) && (id_z == 0)) {
    // //     printf("\nid_x, id_y, id_z = %d, %d, %d", id_x, id_y, id_z);
    // //     printf("\nsrcIdx, dstIdx = %d, %d", srcIdx, dstIdx);

    // if (id_y < roiSrc.xywhROI.roiHeight)
    // {
    //     extern __shared__ float lds[];
    //     volatile float *row = lds + hipThreadIdx_x;
    //     int xWarped = ::min(::max(id_x, 0), roiSrc.xywhROI.roiWidth - 1);

    //     // printf("\nroiSrc = %d, %d, %d, %d", roiSrc.xywhROI.xy.x, roiSrc.xywhROI.xy.y, roiSrc.xywhROI.roiWidth, roiSrc.xywhROI.roiHeight);
    //     // printf("\nxWarped = %d", xWarped);

    //     row[0] = srcPtr[srcIdx + (id_y * srcStridesNH.y) + xWarped] * g[0];                                                 // src(id_y, xWarped) * c_g[0];
    //     // printf("\nsrcPtr[0],  g[0] = %f, %f", srcPtr[0],  g[0]);
    //     row[hipBlockDim_x] = 0.f;
    //     row[2 * hipBlockDim_x] = 0.f;

    //     for (int k = 1; k <= polyExpNbhoodSize; ++k)
    //     {
    //         float t0 = srcPtr[srcIdx + (::max(id_y - k, 0) * srcStridesNH.y) + xWarped];                             // src(::max(id_y - k, 0), xWarped);
    //         float t1 = srcPtr[srcIdx + (::min(id_y + k, roiSrc.xywhROI.roiHeight - 1) * srcStridesNH.y) + xWarped];  // src(::min(id_y + k, roiSrc.xywhROI.roiHeight - 1), xWarped);

    //         // if ((id_x == -5) && (id_y == 0) && (id_z == 0))
    //         // {
    //         //     printf("\n\n t0, t1 = %f, %f", t0, t1);
    //         // }

    //         row[0] += g[k] * (t0 + t1);
    //         row[hipBlockDim_x] += xg[k] * (t1 - t0);
    //         row[2 * hipBlockDim_x] += xxg[k] * (t0 + t1);
    //     }

    //     __syncthreads();

    //     if (hipThreadIdx_x >= polyExpNbhoodSize && hipThreadIdx_x + polyExpNbhoodSize < hipBlockDim_x && id_x < roiSrc.xywhROI.roiWidth)
    //     {
    //         float b1 = g[0] * row[0];
    //         float b3 = g[0] * row[hipBlockDim_x];
    //         float b5 = g[0] * row[2 * hipBlockDim_x];
    //         float b2 = 0, b4 = 0, b6 = 0;

    //         for (int k = 1; k <= polyExpNbhoodSize; ++k)
    //         {
    //             b1 += (row[k] + row[-k]) * g[k];
    //             b4 += (row[k] + row[-k]) * xxg[k];
    //             b2 += (row[k] - row[-k]) * xg[k];
    //             b3 += (row[k + hipBlockDim_x] + row[-k + hipBlockDim_x]) * g[k];
    //             b6 += (row[k + hipBlockDim_x] - row[-k + hipBlockDim_x]) * xg[k];
    //             b5 += (row[k + 2 * hipBlockDim_x] + row[-k + 2 * hipBlockDim_x]) * g[k];
    //         }

    //         // dst(id_y, xWarped) = b3*c_ig11;
    //         // dst(roiSrc.xywhROI.roiHeight + id_y, xWarped) = b2*c_ig11;
    //         // dst(2*roiSrc.xywhROI.roiHeight + id_y, xWarped) = b1*c_ig03 + b5*c_ig33;
    //         // dst(3*roiSrc.xywhROI.roiHeight + id_y, xWarped) = b1*c_ig03 + b4*c_ig33;
    //         // dst(4*roiSrc.xywhROI.roiHeight + id_y, xWarped) = b6*c_ig55;

    //         dstIdx += ((id_y * dstStridesNCH.z) + xWarped);
    //         dstPtr[dstIdx] = b3 * invG11033355_f4.x;                            // polyRes[0]
    //         dstIdx += dstStridesNCH.x;
    //         dstPtr[dstIdx] = b2 * invG11033355_f4.x;                            // polyRes[1]
    //         dstIdx += dstStridesNCH.x;
    //         dstPtr[dstIdx] = b1 * invG11033355_f4.y + b5 * invG11033355_f4.z;   // polyRes[2]
    //         dstIdx += dstStridesNCH.x;
    //         dstPtr[dstIdx] = b1 * invG11033355_f4.y + b4 * invG11033355_f4.z;   // polyRes[3]
    //         dstIdx += dstStridesNCH.x;
    //         dstPtr[dstIdx] = b6 * invG11033355_f4.w;                            // polyRes[4]

    //         // printf("\nig11, ig03, ig33, ig55 = %f, %f, %f, %f", (float)invG11033355_f4.x, (float)invG11033355_f4.y, (float)invG11033355_f4.z, (float)invG11033355_f4.w);
    //         // printf("\nResult = %f, %f, %f, %f, %f", b3 * invG11033355_f4.x, b2 * invG11033355_f4.x, (b1 * invG11033355_f4.y + b5 * invG11033355_f4.z), (b1 * invG11033355_f4.y + b4 * invG11033355_f4.z), b6 * invG11033355_f4.w);

    //         // if ((id_x == -5) && (id_y == 0) && (id_z == 0))
    //         // {
    //         //     int x = 0;
    //         //     printf("\n");
    //         //     for (; x < 256; x++)
    //         //         printf(" %f", row[x]);
    //         //     printf("\n");
    //         //     for (; x < 512; x++)
    //         //         printf(" %f", row[x]);
    //         //     printf("\n");
    //         //     for (; x < 768; x++)
    //         //         printf(" %f", row[x]);
    //         //     printf("\n\n b1, b2, b3, b4, b5, b6 = %f, %f, %f, %f, %f, %f", b1, b2, b3, b4, b5, b6);
    //         //     printf("\n\n invG11033355_f4.x, invG11033355_f4.y, invG11033355_f4.z, invG11033355_f4.w = %f, %f, %f, %f", (float)invG11033355_f4.x, (float)invG11033355_f4.y, (float)invG11033355_f4.z, (float)invG11033355_f4.w);
    //         // }
    //     }
    // }
    // }
}

__global__ void farneback_matrices_update_tensor(float *mVecCurrCompX,
                                                 float *mVecCurrCompY,
                                                 float *polyResPrev,
                                                 float *polyResCurr,
                                                 float *polyMatrices,
                                                 uint3 stridesNCH,
                                                 float *border,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(
        (hipBlockIdx_x == 0) &&
        (hipBlockIdx_y == 0) &&
        (hipBlockIdx_z == 0) &&
        (hipThreadIdx_x == 0) &&
        (hipThreadIdx_y == 0) &&
        (hipThreadIdx_z == 0)
    )
    {
        printf("\nInside farneback_matrices_update_tensor:");
        printf("\nborder[0], border[1], border[2], border[3], border[4], border[5] = %f, %f, %f, %f, %f, %f", border[0], border[1], border[2], border[3], border[4], border[5]);
    }

    RpptROI roiSrc = roiTensorPtrSrc[id_z];

    int mVecCurrCompXIdx = id_z * stridesNCH.x;
    int mVecCurrCompYIdx = id_z * stridesNCH.x;

    if (id_y < roiSrc.xywhROI.roiHeight && id_x < roiSrc.xywhROI.roiWidth)
    {
        float dx = mVecCurrCompX[mVecCurrCompXIdx + (id_y * stridesNCH.z) + id_x];     // flowx(id_y, id_x);
        float dy = mVecCurrCompY[mVecCurrCompYIdx + (id_y * stridesNCH.z) + id_x];     // flowy(id_y, id_x);
        float fx = id_x + dx;
        float fy = id_y + dy;

        int x1 = floorf(fx);
        int y1 = floorf(fy);
        fx -= x1;
        fy -= y1;





        // Original style
        float r2, r3, r4, r5, r6;

        if ((x1 >= 0) && (y1 >= 0) && (x1 < roiSrc.xywhROI.roiWidth - 1) && (y1 < roiSrc.xywhROI.roiHeight - 1))
        {
            float a00 = (1.0f - fx) * (1.0f - fy);
            float a01 = fx * (1.0f - fy);
            float a10 = (1.0f - fx) * fy;
            float a11 = fx * fy;

            r2 = a00 * polyResCurr[y1 * stridesNCH.z + x1] +
                    a01 * polyResCurr[y1 * stridesNCH.z + x1 + 1] +
                    a10 * polyResCurr[(y1 + 1) * stridesNCH.z + x1] +
                    a11 * polyResCurr[(y1 + 1) * stridesNCH.z + x1 + 1];

            r3 = a00 * polyResCurr[stridesNCH.x + (y1 * stridesNCH.z) + x1] +
                    a01 * polyResCurr[stridesNCH.x + (y1 * stridesNCH.z) + x1 + 1] +
                    a10 * polyResCurr[stridesNCH.x + ((y1 + 1) * stridesNCH.z) + x1] +
                    a11 * polyResCurr[stridesNCH.x + ((y1 + 1) * stridesNCH.z) + x1 + 1];

            r4 = a00 * polyResCurr[(2 * stridesNCH.x) + (y1 * stridesNCH.z) + x1] +
                    a01 * polyResCurr[(2 * stridesNCH.x) + (y1 * stridesNCH.z) + x1 + 1] +
                    a10 * polyResCurr[(2 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1] +
                    a11 * polyResCurr[(2 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1 + 1];

            r5 = a00 * polyResCurr[(3 * stridesNCH.x) + (y1 * stridesNCH.z) + x1] +
                    a01 * polyResCurr[(3 * stridesNCH.x) + (y1 * stridesNCH.z) + x1 + 1] +
                    a10 * polyResCurr[(3 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1] +
                    a11 * polyResCurr[(3 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1 + 1];

            r6 = a00 * polyResCurr[(4 * stridesNCH.x) + (y1 * stridesNCH.z) + x1] +
                    a01 * polyResCurr[(4 * stridesNCH.x) + (y1 * stridesNCH.z) + x1 + 1] +
                    a10 * polyResCurr[(4 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1] +
                    a11 * polyResCurr[(4 * stridesNCH.x) + ((y1 + 1) * stridesNCH.z) + x1 + 1];

            r4 = (polyResPrev[(2 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] + r4) * 0.5f;
            r5 = (polyResPrev[(3 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] + r5) * 0.5f;
            r6 = (polyResPrev[(4 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] + r6) * 0.25f;
        }
        else
        {
            r2 = r3 = 0.f;

            r4 = (polyResPrev[(2 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x]);
            r5 = (polyResPrev[(3 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x]);
            r6 = (polyResPrev[(4 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x]) * 0.5f;
        }

        r2 = (polyResPrev[(id_y * stridesNCH.z) + id_x] - r2) * 0.5f;
        r3 = (polyResPrev[stridesNCH.x + (id_y * stridesNCH.z) + id_x] - r3) * 0.5f;

        r2 += r4*dy + r6*dx;
        r3 += r6*dy + r5*dx;

        float scale =
                border[::min(id_x, BORDER_SIZE)] *
                border[::min(id_y, BORDER_SIZE)] *
                border[::min(roiSrc.xywhROI.roiWidth - id_x - 1, BORDER_SIZE)] *
                border[::min(roiSrc.xywhROI.roiHeight - id_y - 1, BORDER_SIZE)];

        r2 *= scale; r3 *= scale; r4 *= scale;
        r5 *= scale; r6 *= scale;

        polyMatrices[(id_y * stridesNCH.z) + id_x] = r4*r4 + r6*r6;
        polyMatrices[stridesNCH.x + (id_y * stridesNCH.z) + id_x] = (r4 + r5)*r6;
        polyMatrices[(2 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] = r5*r5 + r6*r6;
        polyMatrices[(3 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] = r4*r2 + r6*r3;
        polyMatrices[(4 * stridesNCH.x) + (id_y * stridesNCH.z) + id_x] = r6*r2 + r5*r3;





















        // Modified style
        /*
        float r[5];

        if (x1 >= 0 && y1 >= 0 && x1 < roiSrc.xywhROI.roiWidth - 1 && y1 < roiSrc.xywhROI.roiHeight - 1)
        {
            float a00 = (1.f - fx) * (1.f - fy);
            float a01 = fx * (1.f - fy);
            float a10 = (1.f - fx) * fy;
            float a11 = fx * fy;

            int polyResCurrRow0Idx = y1 * stridesNCH.z + x1;
            int polyResCurrRow1Idx = polyResCurrRow0Idx + stridesNCH.z;

            for(int i = 0; i < 5; i++)
            {
                r[i] = a00 * polyResCurr[polyResCurrRow0Idx] +        // R1(y1, x1) +
                       a01 * polyResCurr[polyResCurrRow0Idx + 1] +    // R1(y1, x1 + 1) +
                       a10 * polyResCurr[polyResCurrRow1Idx] +        // R1(y1 + 1, x1) +
                       a11 * polyResCurr[polyResCurrRow1Idx + 1];     // R1(y1 + 1, x1 + 1);

                polyResCurrRow0Idx += stridesNCH.x;
                polyResCurrRow1Idx += stridesNCH.x;
            }

            int polyResPrevIdx = id_y * stridesNCH.z + id_x;

            r[0] = (polyResPrev[polyResPrevIdx] - r[0]) * 0.5f;
            polyResPrevIdx += stridesNCH.x;
            r[1] = (polyResPrev[polyResPrevIdx] - r[1]) * 0.5f;
            polyResPrevIdx += stridesNCH.x;
            r[2] = (polyResPrev[polyResPrevIdx] + r[2]) * 0.5f;
            polyResPrevIdx += stridesNCH.x;
            r[3] = (polyResPrev[polyResPrevIdx] + r[3]) * 0.5f;
            polyResPrevIdx += stridesNCH.x;
            r[4] = (polyResPrev[polyResPrevIdx] + r[4]) * 0.25f;
        }
        else
        {
            int polyResPrevIdx = id_y * stridesNCH.z + id_x;

            r[0] = polyResPrev[polyResPrevIdx] * 0.5f;
            polyResPrevIdx += stridesNCH.x;
            r[1] = polyResPrev[polyResPrevIdx] * 0.5f;
            polyResPrevIdx += stridesNCH.x;
            r[2] = (polyResPrev[polyResPrevIdx] + r[2]) * 0.5f;
            polyResPrevIdx += stridesNCH.x;
            r[3] = (polyResPrev[polyResPrevIdx] + r[3]) * 0.5f;
            polyResPrevIdx += stridesNCH.x;
            r[4] = (polyResPrev[polyResPrevIdx] + r[4]) * 0.5f;

            // r[0] = r[1] = r[2] = r[3] = r[4] = 5.0f;

            // if(id_x == 0 && id_y == 0 && id_z == 0)
            // printf("\n\n r[0], r[1], r[2], r[3], r[4] = %f, %f, %f, %f, %f", r[0], r[1], r[2], r[3], r[4]);
        }

        r[0] += r[2] * dy + r[4] * dx;
        r[1] += r[4] * dy + r[3] * dx;

        float scale = border[::min(id_x, BORDER_SIZE)] *
                      border[::min(id_y, BORDER_SIZE)] *
                      border[::min(roiSrc.xywhROI.roiWidth - id_x - 1, BORDER_SIZE)] *
                      border[::min(roiSrc.xywhROI.roiHeight - id_y - 1, BORDER_SIZE)];

        r[0] *= scale;
        r[1] *= scale;
        r[2] *= scale;
        r[3] *= scale;
        r[4] *= scale;

        int polyMatricesIdx = id_y * stridesNCH.z + id_x;
        polyMatrices[polyMatricesIdx] = r[2] * r[2] + r[4] * r[4];
        polyMatricesIdx += stridesNCH.x;
        polyMatrices[polyMatricesIdx] = (r[2] + r[3]) * r[4];
        polyMatricesIdx += stridesNCH.x;
        polyMatrices[polyMatricesIdx] = r[3] * r[3] + r[4] * r[4];
        polyMatricesIdx += stridesNCH.x;
        polyMatrices[polyMatricesIdx] = r[2] * r[0] + r[4] * r[1];
        polyMatricesIdx += stridesNCH.x;
        polyMatrices[polyMatricesIdx] = r[4] * r[0] + r[3] * r[1];
        */
    }
}

__global__ void farneback_motion_vectors_update_tensor(float *polyMatricesBlurred,
                                                       float *mVecCurrCompX,
                                                       float *mVecCurrCompY,
                                                       uint2 stridesNH,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    // Original Version
    const int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const int z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    RpptROI roiSrc = roiTensorPtrSrc[z];

    if ((y < roiSrc.xywhROI.roiHeight) && (x < roiSrc.xywhROI.roiWidth))
    {
        float g11 = polyMatricesBlurred[y * stridesNH.y + x];
        float g12 = polyMatricesBlurred[stridesNH.x + (y * stridesNH.y) + x];
        float g22 = polyMatricesBlurred[(2 * stridesNH.x) + (y * stridesNH.y) + x];
        float h1 = polyMatricesBlurred[(3 * stridesNH.x) + (y * stridesNH.y) + x];
        float h2 = polyMatricesBlurred[(4 * stridesNH.x) + (y * stridesNH.y) + x];

        float detInv = 1.0f / (g11*g22 - g12*g12 + 0.001f);

        mVecCurrCompX[y * stridesNH.y + x] = (g11*h2 - g12*h1) * detInv;
        mVecCurrCompY[y * stridesNH.y + x] = (g22*h1 - g12*h2) * detInv;
    }






    // Modified Version
    /*
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    RpptROI roiSrc = roiTensorPtrSrc[id_z];

    if (id_y < roiSrc.xywhROI.roiHeight && id_x < roiSrc.xywhROI.roiWidth)
    {
        int polyMatricesIdx = id_y * stridesNH.y + id_x;
        int mVecCurrIdx = polyMatricesIdx;

        float g11 = polyMatricesBlurred[polyMatricesIdx];   // M(id_y, id_x);
        polyMatricesIdx += stridesNH.x;
        float g12 = polyMatricesBlurred[polyMatricesIdx];   // M(height + id_y, id_x);
        polyMatricesIdx += stridesNH.x;
        float g22 = polyMatricesBlurred[polyMatricesIdx];   // M(2*height + id_y, id_x);
        polyMatricesIdx += stridesNH.x;
        float h1 = polyMatricesBlurred[polyMatricesIdx];    // M(3*height + id_y, id_x);
        polyMatricesIdx += stridesNH.x;
        float h2 = polyMatricesBlurred[polyMatricesIdx];    // M(4*height + id_y, id_x);

        float detInv = 1.f / (g11 * g22 - g12 * g12 + 1e-3f);

        mVecCurrCompX[mVecCurrIdx] = (g11 * h2 - g12 * h1) * detInv;
        mVecCurrCompY[mVecCurrIdx] = (g22 * h1 - g12 * h2) * detInv;
    }
    */
}

__global__ void saturate_left_tensor(float *mVecComp,
                                     uint2 stridesNH)
{
    const int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const int y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const int z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((y >= 540) || (x >= 960))
    {
        return;
        // mVecComp[(y * stridesNH.y) + x] = 0;
    }
    int srcIdx = (y * stridesNH.y) + x;
    mVecComp[srcIdx] = fminf(fmaxf(mVecComp[srcIdx], -24.0f), 101.0f);
}


// either g12 is higher, or g11/g22 are lower or

RppStatus hip_exec_farneback_polynomial_expansion_tensor(Rpp32f *pyramidLevelF32,
                                                         RpptDescPtr srcCompDescPtr,
                                                         Rpp32f *polyRes,
                                                         RpptDescPtr mVecCompDescPtr,
                                                         Rpp32s polyExpNbhoodSize,
                                                         Rpp32f *g,
                                                         Rpp32f *xg,
                                                         Rpp32f *xxg,
                                                         double4 invG11033355_f4,
                                                         RpptROIPtr roiTensorPtrSrc,
                                                         RpptRoiType roiType,
                                                         rpp::Handle& handle)
{
    // if (roiType == RpptRoiType::LTRB)
    //     hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = roiTensorPtrSrc[0].xywhROI.roiWidth; // can be changed to roi width and height?
    int globalThreads_y = roiTensorPtrSrc[0].xywhROI.roiHeight; // can be changed to roi width and height?
    int globalThreads_z = handle.GetBatchSize();

    int tileSizeX = localThreads_x - 2 * polyExpNbhoodSize;
    int ldsSize = 3 * localThreads_x * sizeof(float);

    std::cerr << "\nEntering farneback_polynomial_expansion";
    std::cerr << "\nfarneback_polynomial_expansion src nh strides -> " << srcCompDescPtr->strides.nStride << ", " << srcCompDescPtr->strides.hStride;
    std::cerr << "\nfarneback_polynomial_expansion dst nch strides -> " << mVecCompDescPtr->strides.nStride << ", " << mVecCompDescPtr->strides.cStride << ", " << mVecCompDescPtr->strides.hStride;
    std::cerr << "\nfarneback_polynomial_expansion roiTensorPtrSrc[0] w/h -> " << roiTensorPtrSrc[0].xywhROI.roiWidth << ", " << roiTensorPtrSrc[0].xywhROI.roiHeight;
    std::cerr << "\nfarneback_polynomial_expansion tileSizeX -> " << tileSizeX;
    std::cerr << "\nfarneback_polynomial_expansion ldsSize -> " << ldsSize;
    std::cerr << "\nfarneback_polynomial_expansion localThreads_x, localThreads_y, localThreads_z -> " << localThreads_x << ", " << localThreads_y << ", " << localThreads_z;
    std::cerr << "\nfarneback_polynomial_expansion globalThreads_x, globalThreads_y, globalThreads_z -> " << globalThreads_x << ", " << globalThreads_y << ", " << globalThreads_z;


    if (polyExpNbhoodSize == 5)
    {
        // std::cerr << "\nEntering farneback_polynomial_expansion_tensor<5>";
        hipLaunchKernelGGL(HIP_KERNEL_NAME(farneback_polynomial_expansion_tensor<5>),
                           dim3(ceil((float)globalThreads_x/tileSizeX), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           ldsSize,
                           handle.GetStream(),
                           pyramidLevelF32,
                           make_int2(srcCompDescPtr->strides.nStride, srcCompDescPtr->strides.hStride),
                           polyRes,
                           make_int3(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.cStride, mVecCompDescPtr->strides.hStride),
                           g,
                           xg,
                           xxg,
                           invG11033355_f4,
                           tileSizeX,
                           roiTensorPtrSrc);
    }
    else if (polyExpNbhoodSize == 7)
    {
        // std::cerr << "\nEntering farneback_polynomial_expansion_tensor<7>";
        hipLaunchKernelGGL(HIP_KERNEL_NAME(farneback_polynomial_expansion_tensor<7>),
                           dim3(ceil((float)globalThreads_x/tileSizeX), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           ldsSize,
                           handle.GetStream(),
                           pyramidLevelF32,
                           make_int2(srcCompDescPtr->strides.nStride, srcCompDescPtr->strides.hStride),
                           polyRes,
                           make_int3(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.cStride, mVecCompDescPtr->strides.hStride),
                           g,
                           xg,
                           xxg,
                           invG11033355_f4,
                           tileSizeX,
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}

RppStatus hip_exec_farneback_matrices_update_tensor(Rpp32f *mVecCurrCompX,
                                                    Rpp32f *mVecCurrCompY,
                                                    Rpp32f *polyResPrev,
                                                    Rpp32f *polyResCurr,
                                                    Rpp32f *polyMatrices,
                                                    RpptDescPtr mVecCompDescPtr,
                                                    Rpp32f *border,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    rpp::Handle& handle)
{
    // if (roiType == RpptRoiType::LTRB)
    //     hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = 32;
    int localThreads_y = 8;
    int localThreads_z = 1;
    int globalThreads_x = roiTensorPtrSrc[0].xywhROI.roiWidth;  // mVecCompDescPtr->w; // can be changed to roi width and height?
    int globalThreads_y = roiTensorPtrSrc[0].xywhROI.roiHeight; // mVecCompDescPtr->h; // can be changed to roi width and height?
    // int globalThreads_x = mVecCompDescPtr->w; // can be changed to roi width and height?
    // int globalThreads_y = mVecCompDescPtr->h; // can be changed to roi width and height?
    int globalThreads_z = handle.GetBatchSize();

    std::cerr << "\nEntering farneback_matrices_update_tensor";
    std::cerr << "\nfarneback_matrices_update_tensor mVecCompDescPtr nch strides -> " << mVecCompDescPtr->strides.nStride << ", " << mVecCompDescPtr->strides.cStride << ", " << mVecCompDescPtr->strides.hStride;
    std::cerr << "\nfarneback_matrices_update_tensor roiTensorPtrSrc[0] w/h -> " << roiTensorPtrSrc[0].xywhROI.roiWidth << ", " << roiTensorPtrSrc[0].xywhROI.roiHeight;
    std::cerr << "\nfarneback_matrices_update_tensor localThreads_x, localThreads_y, localThreads_z -> " << localThreads_x << ", " << localThreads_y << ", " << localThreads_z;
    std::cerr << "\nfarneback_matrices_update_tensor globalThreads_x, globalThreads_y, globalThreads_z -> " << globalThreads_x << ", " << globalThreads_y << ", " << globalThreads_z;

    hipLaunchKernelGGL(farneback_matrices_update_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       mVecCurrCompX,
                       mVecCurrCompY,
                       polyResPrev,
                       polyResCurr,
                       polyMatrices,
                       make_uint3(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.cStride, mVecCompDescPtr->strides.hStride),
                       border,
                    //    g,
                    //    xg,
                    //    xxg,
                    //    invG11033355_f4,
                       roiTensorPtrSrc);



    return RPP_SUCCESS;
}

RppStatus hip_exec_farneback_motion_vectors_update_tensor(Rpp32f *polyMatricesBlurred,
                                                          Rpp32f *mVecCurrCompX,
                                                          Rpp32f *mVecCurrCompY,
                                                          RpptDescPtr mVecCompDescPtr,
                                                          RpptROIPtr roiTensorPtrSrc,
                                                          RpptRoiType roiType,
                                                          rpp::Handle& handle)
{
    // if (roiType == RpptRoiType::LTRB)
    //     hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = 32;
    int localThreads_y = 8;
    int localThreads_z = 1;
    int globalThreads_x = roiTensorPtrSrc[0].xywhROI.roiWidth;  // mVecCompDescPtr->w; // can be changed to roi width and height?
    int globalThreads_y = roiTensorPtrSrc[0].xywhROI.roiHeight; // mVecCompDescPtr->h; // can be changed to roi width and height?
    // int globalThreads_x = mVecCompDescPtr->w; // can be changed to roi width and height?
    // int globalThreads_y = mVecCompDescPtr->h; // can be changed to roi width and height?
    int globalThreads_z = handle.GetBatchSize();

    std::cerr << "\nEntering farneback_motion_vectors_update_tensor";
    std::cerr << "\nfarneback_motion_vectors_update_tensor mVecCompDescPtr nh strides -> " << mVecCompDescPtr->strides.nStride << ", " << mVecCompDescPtr->strides.hStride;
    std::cerr << "\nfarneback_motion_vectors_update_tensor roiTensorPtrSrc[0] w/h -> " << roiTensorPtrSrc[0].xywhROI.roiWidth << ", " << roiTensorPtrSrc[0].xywhROI.roiHeight;
    std::cerr << "\nfarneback_motion_vectors_update_tensor localThreads_x, localThreads_y, localThreads_z -> " << localThreads_x << ", " << localThreads_y << ", " << localThreads_z;
    std::cerr << "\nfarneback_motion_vectors_update_tensor globalThreads_x, globalThreads_y, globalThreads_z -> " << globalThreads_x << ", " << globalThreads_y << ", " << globalThreads_z;
    hipLaunchKernelGGL(farneback_motion_vectors_update_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       polyMatricesBlurred,
                       mVecCurrCompX,
                       mVecCurrCompY,
                       make_uint2(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.hStride),
                       roiTensorPtrSrc);

    return RPP_SUCCESS;
}

RppStatus hip_exec_saturate_left_tensor(Rpp32f *mVecCompX,
                                        RpptDescPtr mVecCompDescPtr,
                                        rpp::Handle& handle)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = 960;
    int globalThreads_y = 540;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(saturate_left_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       mVecCompX,
                       make_uint2(mVecCompDescPtr->strides.nStride, mVecCompDescPtr->strides.hStride));

    return RPP_SUCCESS;
}

RppStatus hip_exec_farneback_optical_flow_tensor(Rpp8u *src1Ptr,
                                                 Rpp8u *src2Ptr,
                                                 Rpp8u *dstPtrIntermU8, // to be removed
                                                 Rpp32f *dstPtrIntermF32, // to be removed
                                                 Rpp32f *dh_cudaResizdStrided, // to be removed
                                                 RpptDescPtr srcCompDescPtr,
                                                 Rpp32f *mVecCompX,
                                                 Rpp32f *mVecCompY,
                                                 RpptDescPtr mVecCompDescPtr,
                                                 Rpp32f pyramidScale,
                                                 Rpp32s numPyramidLevels,
                                                 Rpp32s windowSize,
                                                 Rpp32s numIterations,
                                                 Rpp32s polyExpNbhoodSize,
                                                 Rpp32f polyExpStdDev,
                                                 rpp::Handle& handle)
{
    Rpp32f oneOverPyramidScale = 1.0f / pyramidScale;
    // std::cerr << "\noneOverPyramidScale = " << oneOverPyramidScale;
    Rpp64f pyramidScaleDouble = (Rpp64f) pyramidScale;

    // reinterpret src and dst with modified tensor descriptors for batched U8-F32 conversion
    // RpptDesc srcDesc, mVecDesc;
    // RpptDescPtr srcDescPtr = &srcDesc;
    // RpptDescPtr mVecDescPtr = &mVecDesc;
    // srcDesc = *srcCompDescPtr;
    // srcDescPtr->n = 2;
    // mVecDesc = *mVecCompDescPtr;
    // mVecDescPtr->n = 2;

    // NOT REQD - initialize default hip kernel launch params
    // int localThreads_x = LOCAL_THREADS_X;
    // int localThreads_y = LOCAL_THREADS_Y;
    // int localThreads_z = LOCAL_THREADS_Z;
    // int globalThreads_x = (mVecDescPtr->strides.hStride + 7) >> 3;
    // int globalThreads_y = mVecDescPtr->h;
    // int globalThreads_z = mVecDescPtr->n;

    // NOT REQD - Use batching instead of streams
    // initialize multiple streams with first stream same as rpp handle stream
    // hipStream_t streams[5];
    // streams[0] = handle.GetStream();
    // hipStreamCreate(&streams[1]);
    // hipStreamCreate(&streams[2]);
    // hipStreamCreate(&streams[3]);
    // hipStreamCreate(&streams[4]);

    // Use preallocated buffers for 4 960x540 frames (for previous/current motion vectors in x/y)
    // Rpp32s srcCompBufSize = MAX_MVEC_WIDTH * MAX_MVEC_HEIGHT;
    Rpp32f *preallocMem, *srcF32, *srcF32Blurred, *pyramidLevel, *polyRes, *polyMatrices, *polyMatricesBlurred;
    Rpp32f *mVecPrevCompX, *mVecPrevCompY, *mVecCurrCompX, *mVecCurrCompY;
    preallocMem = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    hipMemset(preallocMem, 0, FARNEBACK_OUTPUT_FRAME_SIZE * 30 * sizeof(Rpp32f));
    hipDeviceSynchronize();
    srcF32 = preallocMem + 64;                                              // 2 frames - previous and current (after 64 byte offset)
    srcF32Blurred = srcF32 + FARNEBACK_OUTPUT_MVEC_SIZE;                    // 2 blurred frames - previous and current
    pyramidLevel = srcF32Blurred + FARNEBACK_OUTPUT_MVEC_SIZE;              // 2 pyramid frames - previous and current
    polyRes = pyramidLevel + FARNEBACK_OUTPUT_MVEC_SIZE;                    // 10 polyRes frames - 5 based on previous frame and 5 based on current frame
    polyMatrices = polyRes + (FARNEBACK_OUTPUT_FRAME_SIZE * 10);            // 5 polyMatrices frames
    polyMatricesBlurred = polyMatrices + (FARNEBACK_OUTPUT_FRAME_SIZE * 5); // 5 blurred polyMatrices frames
    mVecPrevCompX = polyMatricesBlurred + (FARNEBACK_OUTPUT_FRAME_SIZE * 5);
    mVecPrevCompY = mVecPrevCompX + FARNEBACK_OUTPUT_FRAME_SIZE;
    mVecCurrCompX = mVecPrevCompY + FARNEBACK_OUTPUT_FRAME_SIZE;
    mVecCurrCompY = mVecCurrCompX + FARNEBACK_OUTPUT_FRAME_SIZE;

    // Rpp32f *src1F32, *src2F32, *src1F32Blurred, *src2F32Blurred;
    Rpp32f *src1F32 = srcF32;
    Rpp32f *src2F32 = src1F32 + FARNEBACK_OUTPUT_FRAME_SIZE;
    Rpp32f *src1F32Blurred = srcF32Blurred;
    Rpp32f *src2F32Blurred = src1F32Blurred + FARNEBACK_OUTPUT_FRAME_SIZE;

    Rpp32f *pyramidLevelPrevF32 = pyramidLevel;
    Rpp32f *pyramidLevelCurrF32 = pyramidLevelPrevF32 + FARNEBACK_OUTPUT_FRAME_SIZE;
    Rpp32f *polyResPrev = polyRes;
    Rpp32f *polyResCurr = polyResPrev + (FARNEBACK_OUTPUT_FRAME_SIZE * 5);

    // Creating U8 pointer interpretations from F32 buffers
    Rpp8u *src1U8Blurred = (Rpp8u *)src1F32Blurred;
    Rpp8u *src2U8Blurred = (Rpp8u *)src2F32Blurred;
    Rpp8u *pyramidLevelPrevU8 = (Rpp8u *)pyramidLevelPrevF32;
    Rpp8u *pyramidLevelCurrU8 = (Rpp8u *)pyramidLevelCurrF32;

    // NOT REQD NOW - Set pre-allocated device buffer for pyramid image patch
    // RpptImagePatch *fbackImgPatchPtr = (RpptImagePatch *)handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem;

    Rpp64f scale = 1.0; // changed to 32f
    Rpp32s numPyramidLevelsCropped = 0;
    for (; numPyramidLevelsCropped < numPyramidLevels; numPyramidLevelsCropped++)
    {
        scale *= pyramidScale;
        if (srcCompDescPtr->w * scale < FARNEBACK_FRAME_MIN_SIZE || srcCompDescPtr->h * scale < FARNEBACK_FRAME_MIN_SIZE) // TODO: scale multiplier is loop independent
            break;
    }
    // std::cerr << "\nscale = " << scale;


    // Commenting out temporarily to check box filter with U8
    hip_exec_copy_tensor(src1Ptr,
                         srcCompDescPtr,
                         src1F32,
                         mVecCompDescPtr,
                         handle);
    hip_exec_copy_tensor(src2Ptr,
                         srcCompDescPtr,
                         src2F32,
                         mVecCompDescPtr,
                         handle);
    hipDeviceSynchronize();

    // Batched buffer copy with fused bit depth converison (U8 to F32) for 2 frames
    // hip_exec_copy_tensor(src1Ptr + srcDescPtr->offsetInBytes,
    //                      srcDescPtr,
    //                      mVecPrevCompX,
    //                      mVecDescPtr,
    //                      handle);

    // -------------------- stage output dump check --------------------
    // std::cerr << "\nAbout to copy d to d...";
    // hipMemcpy(dstPtrIntermF32, src1F32, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
    // hipDeviceSynchronize();
    // std::cerr << "\nCopy d to d complete...";
    // return RPP_SUCCESS;

    // hipDeviceSynchronize();
    // Rpp32f h_src1F32[FARNEBACK_OUTPUT_FRAME_SIZE], h_src2F32[FARNEBACK_OUTPUT_FRAME_SIZE];
    // hipMemcpy(h_src1F32, src1F32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
    // hipMemcpy(h_src2F32, src2F32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
    // rpp_tensor_write_to_file("src1F32", h_src1F32, mVecCompDescPtr);
    // rpp_tensor_write_to_file("src2F32", h_src2F32, mVecCompDescPtr);

    // Call kernel instead?
    // hipLaunchKernelGGL(copy_pln1_pln1_tensor,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    streams[0],
    //                    src1Ptr,
    //                    make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
    //                    mVecPrevCompX,
    //                    make_uint2(mVecDescPtr->strides.nStride, mVecDescPtr->strides.hStride),
    //                    make_uint2(srcDescPtr->w, srcDescPtr->h));

    Rpp32s bufIncrement = polyExpNbhoodSize * 2 + 1;
    std::vector<Rpp32f> xSquareBuf(bufIncrement);
    Rpp32f *xSquare = &xSquareBuf[0] + polyExpNbhoodSize;

    Rpp32f *buf;
    hipHostMalloc(&buf, (polyExpNbhoodSize * 6 + 3) * sizeof(Rpp32f));  // pinned memory sustitute for std::vector<Rpp32f> buf(polyExpNbhoodSize * 6 + 3););
    Rpp32f *g = &buf[0] + polyExpNbhoodSize;
    Rpp32f *xg = g + bufIncrement;
    Rpp32f *xxg = xg + bufIncrement;

    if (polyExpStdDev < RPP_MACHEPS)
        polyExpStdDev = polyExpNbhoodSize * 0.3f;
    // std::cout << "\npolyExpStdDev = " << polyExpStdDev;

    Rpp32f s = 0; // changed to 32f
    Rpp32f oneOverTwoStddevSquare = 1.0f / (2.0f * polyExpStdDev * polyExpStdDev); // changed to 32f
    // std::cout << "\noneOverTwoStddevSquare = " << oneOverTwoStddevSquare;

    for (int x = -polyExpNbhoodSize; x <= polyExpNbhoodSize; x++)
    {
        xSquare[x] = x * x;     // percompute xSquare for multiple future uses
        g[x] = (Rpp32f)std::exp(-xSquare[x] * oneOverTwoStddevSquare);
        s += g[x];
        // std::cout << "\n\n g[x] = " << g[x];
    }
    // std::cout << "\ns before inversion = " << s;
    s = 1.0f / s;
    // std::cout << "\ns after inversion = " << s;
    for (int x = -polyExpNbhoodSize; x <= polyExpNbhoodSize; x++)
    {
        g[x] *= s;
        xg[x] = x * g[x];
        xxg[x] = xSquare[x] * g[x];
        // std::cout << "\n\n g, xg, xxg = " << g[x] << ", " << xg[x] << ", " << xxg[x];
    }

    // TEMP printing g, xg, xxg
    // for (int x = 0; x < 8; x++)
    // {
    //     std::cout << "\ng, xg, xxg = " << g[x] << ", " << xg[x] << ", " << xxg[x];
    // }

    float4 gaussian00113355_f4 = (float4) 0.0f;
    for (int y = -polyExpNbhoodSize; y <= polyExpNbhoodSize; y++)
    {
        for (int x = -polyExpNbhoodSize; x <= polyExpNbhoodSize; x++)
        {
            float4 multiplier_f4 = make_float4(1, xSquare[x], xSquare[x] * xSquare[x], xSquare[x] * xSquare[y]);
            float4 gygx_f4 = (float4) (g[y] * g[x]);
            gaussian00113355_f4 += (gygx_f4 * multiplier_f4);   // float4 increment to process [   (g[y] * g[x])   |   (g[y] * g[x] * x * x)   |   (g[y] * g[x] * x * x * x * x)   |   (g[y] * g[x] * x * x * y * y)   ]
        }
    }

    // std::cout << "\ngaussian00113355_f4.x = " << gaussian00113355_f4.x;
    // std::cout << "\ngaussian00113355_f4.y = " << gaussian00113355_f4.y;
    // std::cout << "\ngaussian00113355_f4.z = " << gaussian00113355_f4.z;
    // std::cout << "\ngaussian00113355_f4.w = " << gaussian00113355_f4.w;

    Rpp32f yw = gaussian00113355_f4.y * gaussian00113355_f4.w;
    Rpp32f yz = gaussian00113355_f4.y * gaussian00113355_f4.z;
    Rpp32f xyzw = gaussian00113355_f4.x * yz * gaussian00113355_f4.w;
    Rpp32f y2 = gaussian00113355_f4.y * gaussian00113355_f4.y;
    Rpp32f y3w = y2 * yw;
    Rpp32f xyw3 = gaussian00113355_f4.x * yw * gaussian00113355_f4.w * gaussian00113355_f4.w;
    Rpp32f y2zw = yz * yw;
    Rpp32f oneOverDetG = -1.0f / ((xyzw * yw) - (xyw3 * gaussian00113355_f4.y) - (xyzw * yz) + (y3w * yz));

    // Gauss-Jordan simplified calculations for 4 out of 36 elements in 6 x 6 matrix inverse
    double4 invG11033355_f4; // initialization for 4 elements from the invG matrix - (1,1), (0,3), (3,3), (5,5)
    invG11033355_f4.x = (((y3w + y3w - xyzw) * gaussian00113355_f4.w) - xyw3 + (xyzw * gaussian00113355_f4.z) - (y2zw * gaussian00113355_f4.y)) * oneOverDetG;
    invG11033355_f4.y = ((y3w * gaussian00113355_f4.w) - (y2zw * gaussian00113355_f4.y)) * oneOverDetG;
    invG11033355_f4.z = (xyzw - y3w) * gaussian00113355_f4.y * oneOverDetG;
    invG11033355_f4.w = (((y3w + y3w - xyzw) * gaussian00113355_f4.y) + (((yz * yz) - (yw * yw)) * gaussian00113355_f4.x) - (y2 * y2 * gaussian00113355_f4.z)) * oneOverDetG;

    // std::cout << "\nig11 (invG11033355_f4.x) = " << invG11033355_f4.x;
    // std::cout << "\nig03 (invG11033355_f4.y) = " << invG11033355_f4.y;
    // std::cout << "\nig33 (invG11033355_f4.z) = " << invG11033355_f4.z;
    // std::cout << "\nig55 (invG11033355_f4.w) = " << invG11033355_f4.w;

    // Pinned memory allocations
    Rpp32f *stdDevPtrForGaussian, *border;
    hipHostMalloc(&stdDevPtrForGaussian, mVecCompDescPtr->n * sizeof(Rpp32f));
    hipHostMalloc(&border, (BORDER_SIZE + 1) * sizeof(Rpp32f));
    *(d_float6_s *)border = *(d_float6_s *)borderVals;
    RpptImagePatch *pyramidImgPatchPtr;
    hipHostMalloc(&pyramidImgPatchPtr, sizeof(RpptImagePatch));
    RpptROI *roiTensorPtrSrc, *roiTensorPtrPyramid;
    hipHostMalloc(&roiTensorPtrSrc, mVecCompDescPtr->n * sizeof(RpptROI));
    hipHostMalloc(&roiTensorPtrPyramid, mVecCompDescPtr->n * sizeof(RpptROI));

    *roiTensorPtrSrc = {0, 0, mVecCompDescPtr->w, mVecCompDescPtr->h};

    // scale optimization trial
    // Rpp32f scalesBuf[numPyramidLevelsCropped + 1] = {1};
    // Rpp32f *scales = &scalesBuf[1];

    for (int k = numPyramidLevelsCropped; k >= 0; k--) // original version
    // for (int k = 1; k >= 0; k -= 1) // moddified version
    {
        std::cerr << "\n\n\n\n --------->>>> Loop calculations for iteration - " << k;

        // hipStreamSynchronize(streams[0]);
        hipDeviceSynchronize();

        scale = 1.0;
        for (int i = 0; i < k; i++)
            scale *= pyramidScale;
        // scale optimization trial
        // if (k = numPyramidLevelsCropped)
        //     for (int i = 0; i < k; i++)
        //         scales[i] = (scales[i - 1] * pyramidScale);

        Rpp64f sigma = (1.0 / scale - 1) * 0.5;   // CUDA has double ????????? >>>>>>>>>>
        Rpp32s smoothSize = ((Rpp32s)roundf(sigma * 5)) | 1;
        smoothSize = fminf(fmaxf(smoothSize, 3), 9); // cap needed since RPP supports only filter sizes 3,5,7,9
        // std::cout << std::setprecision(12) << "\nsigma = " << sigma;
        // std::cout << "\nsmoothSize = " << smoothSize;

        pyramidImgPatchPtr->width = std::roundf(srcCompDescPtr->w * scale);
        pyramidImgPatchPtr->height = std::roundf(srcCompDescPtr->h * scale);

        // std::cout << "\nnumPyramidLevelsCropped = " << numPyramidLevelsCropped;
        // std::cout << std::setprecision(12) << "\nscale = " << scale;
        // std::cout << "\nsrcCompDescPtr->w = " << srcCompDescPtr->w;
        // std::cout << "\nsrcCompDescPtr->h = " << srcCompDescPtr->h;
        // std::cout << "\npyramidImgPatchPtr->width = " << pyramidImgPatchPtr->width;
        // std::cout << "\npyramidImgPatchPtr->height = " << pyramidImgPatchPtr->height;

        // Checking previous and current flows from iteration 2
        // -------------------- stage output dump check --------------------
        // if (k == numPyramidLevelsCropped - 1)
        // {
        //     hipDeviceSynchronize();
        //     std::cerr << "\nAbout to copy d to d...";
        //     hipMemcpy(dstPtrIntermF32, mVecPrevCompX, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipDeviceSynchronize();
        //     hipMemcpy(dstPtrIntermF32 + (FARNEBACK_OUTPUT_FRAME_SIZE * 2), mVecCurrCompX, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipDeviceSynchronize();
        //     std::cerr << "\nCopy d to d complete...";
        //     return RPP_SUCCESS;
        // }

        // RpptDesc mVecCompPydDesc;
        // RpptDescPtr mVecCompPydDescPtr = &mVecCompPydDesc;
        // mVecCompPydDesc = *mVecCompDescPtr;
        if (k == numPyramidLevelsCropped)
        {
            hipMemset(mVecCurrCompX, 0, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f));
            hipMemset(mVecCurrCompY, 0, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f));
            // mVecCompPydDescPtr->w = pyramidImgPatchPtr->width;
            // mVecCompPydDescPtr->h = pyramidImgPatchPtr->height;
            // mVecCompPydDescPtr->strides.hStride = pyramidImgPatchPtr->width;
            // mVecCompPydDescPtr->strides.cStride = pyramidImgPatchPtr->width * pyramidImgPatchPtr->height;
            // mVecCompPydDescPtr->strides.nStride = mVecCompPydDescPtr->strides.cStride;
        }
        else
        {
            if (k == 0)
            {
                hipMemcpy(mVecCurrCompX, mVecCompX, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
                hipMemcpy(mVecCurrCompY, mVecCompY, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);

                // -------------------- stage output dump check --------------------
                // hipDeviceSynchronize();
                // std::cerr << "\nAbout to copy d to d...";
                // hipMemcpy(dstPtrIntermF32, mVecCurrCompX, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
                // hipMemcpy(dstPtrIntermF32 + FARNEBACK_OUTPUT_FRAME_SIZE, mVecCurrCompY, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
                // hipDeviceSynchronize();
                // std::cerr << "\nCopy d to d complete...";
                // return RPP_SUCCESS;
            }
            hip_exec_resize_scale_intensity_tensor(mVecPrevCompX, mVecCompDescPtr, mVecCurrCompX, mVecCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, oneOverPyramidScale, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hip_exec_resize_scale_intensity_tensor(mVecPrevCompY, mVecCompDescPtr, mVecCurrCompY, mVecCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, oneOverPyramidScale, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hipDeviceSynchronize();

            // Checking previous and current flows from iteration 2
            // -------------------- stage output dump check --------------------
            // if (k == 0)
            // {
            //     hipDeviceSynchronize();
            //     std::cerr << "\nAbout to copy d to d...";
            //     hipMemcpy(dstPtrIntermF32, mVecPrevCompX, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
            //     hipDeviceSynchronize();
            //     hipMemcpy(dstPtrIntermF32 + (FARNEBACK_OUTPUT_FRAME_SIZE * 2), mVecCurrCompX, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
            //     hipDeviceSynchronize();
            //     std::cerr << "\nCopy d to d complete...";
            //     return RPP_SUCCESS;
            // }
        }

        // if (k == numPyramidLevelsCropped)    // same as // if (!prevFlowX.data)
        // {
            // Do nothing since memset already done to 0

            // if (flags_ & OPTFLOW_USE_INITIAL_FLOW)
            // {
            //     cuda::resize(flowx0, curFlowX, Size(width, height), 0, 0, INTER_LINEAR, streams[0]);
            //     cuda::resize(flowy0, curFlowY, Size(width, height), 0, 0, INTER_LINEAR, streams[1]);
            //     curFlowX.convertTo(curFlowX, curFlowX.depth(), scale, streams[0]);
            //     curFlowY.convertTo(curFlowY, curFlowY.depth(), scale, streams[1]);
            // }
            // else
            // {
            //     curFlowX.setTo(0, streams[0]);
            //     curFlowY.setTo(0, streams[1]);
            // }

            // Stream experiments in separate PR
            // hipMemsetAsync(mVecCurrCompX, 0, srcCompBufSize * 4, streams[0]);
            // hipMemsetAsync(mVecCurrCompY, 0, srcCompBufSize * 4, streams[1]);

            // hipMemset(mVecCurrCompX, 0, FARNEBACK_OUTPUT_MVEC_SIZE * 4); // setting both mVecCurrCompX and mVecCurrCompY to 0
        // }
        /*
        if (k < numPyramidLevelsCropped) // else
        {
            // on second loop itereation onward
            // cuda::resize(prevFlowX, curFlowX, Size(width, height), 0, 0, INTER_LINEAR, streams[0]);
            // cuda::resize(prevFlowY, curFlowY, Size(width, height), 0, 0, INTER_LINEAR, streams[1]);
            // curFlowX.convertTo(curFlowX, curFlowX.depth(), 1./pyrScale_, streams[0]);
            // curFlowY.convertTo(curFlowY, curFlowY.depth(), 1./pyrScale_, streams[1]);

            hip_exec_resize_scale_intensity_tensor(mVecPrevCompX, mVecCompDescPtr, mVecCurrCompX, mVecCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, oneOverPyramidScale, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hip_exec_resize_scale_intensity_tensor(mVecPrevCompY, mVecCompDescPtr, mVecCurrCompY, mVecCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, oneOverPyramidScale, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);

            // -------------------- stage output dump check --------------------
            hipDeviceSynchronize();
            // std::cerr << "\nAbout to copy d to d...";
            // hipMemcpy(dstPtrIntermF32, mVecCurrCompX, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
            // hipDeviceSynchronize();
            // std::cerr << "\nCopy d to d complete...";
            // return RPP_SUCCESS;
        }*/




        // On need basis
        // GpuMat M = allocMatFromBuf(5*height, width, CV_32F, M_);
        // GpuMat bufM = allocMatFromBuf(5*height, width, CV_32F, bufM_);
        // GpuMat R[2] =
        // {
        //     allocMatFromBuf(5*height, width, CV_32F, R_[0]),
        //     allocMatFromBuf(5*height, width, CV_32F, R_[1])
        // };
        // GpuMat blurredFrame[2] =
        // {
        //     allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[0]),
        //     allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[1])
        // };
        // GpuMat pyrLevel[2] =
        // {
        //     allocMatFromBuf(height, width, CV_32F, pyrLevel_[0]),
        //     allocMatFromBuf(height, width, CV_32F, pyrLevel_[1])
        // };

        // RPP cannot use gaussian on HIP yet
        // Mat g = getGaussianKernel(smoothSize, sigma, CV_32F);
        // device::optflow_farneback::setGaussianBlurKernel(g.ptr<float>(smoothSize/2), smoothSize/2);

        // for (int i = 0; i < 2; i++)
        // {
        //     // device::optflow_farneback::gaussianBlurGpu(frames_[i], smoothSize/2, blurredFrame[i], BORDER_REFLECT101, StreamAccessor::getStream(streams[i]));
        //     // cuda::resize(blurredFrame[i], pyrLevel[i], Size(width, height), 0.0, 0.0, INTER_LINEAR, streams[i]);
        //     // device::optflow_farneback::polynomialExpansionGpu(pyrLevel[i], polyN_, R[i], StreamAccessor::getStream(streams[i]));

        //     // hip_exec_box_filter_tensor(src1Ptr,
        //     //                            mVecCompDescPtr,
        //     //                            src1U8Blurred,
        //     //                            mVecCompDescPtr,
        //     //                            smoothSize,
        //     //                            roiTensorPtrSrc,
        //     //                            RpptRoiType::XYWH,
        //     //                            handle);

        //     // -------------------- stage output dump check --------------------
        //     // hipDeviceSynchronize();
        //     // std::cerr << "\n\nAbout to copy d to d...";
        //     // hipMemcpy(dstPtrIntermU8, src1U8Blurred, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        //     // hipMemcpy(h_src2F32, src2F32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        //     // rpp_tensor_write_to_file("src1U8Blurred", h_src1U8Blurred, mVecCompDescPtr);
        //     // rpp_tensor_write_to_images("src1U8Blurred", h_src1U8Blurred, dstDescPtrRGB, fbackImgPatchPtr);
        //     // rpp_tensor_write_to_file("src2F32", h_src2F32, mVecCompDescPtr);
        //     // break;
        //     // exit(0);
        // }

        // Fixing ROI for RPP batch processing
        // *roiTensorPtrSrc = {0, 0, mVecCompDescPtr->w, mVecCompDescPtr->h};

        // std::cerr << "\nAbout to run 2 box filters...";
        // hip_exec_box_filter_tensor(src1Ptr, srcCompDescPtr, src1U8Blurred, srcCompDescPtr, smoothSize, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        // hip_exec_box_filter_tensor(src2Ptr, srcCompDescPtr, src2U8Blurred, srcCompDescPtr, smoothSize, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        // hip_exec_box_filter_f32_tensor(src1F32, mVecCompDescPtr, src1F32Blurred, mVecCompDescPtr, smoothSize, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        // hip_exec_box_filter_f32_tensor(src2F32, mVecCompDescPtr, src2F32Blurred, mVecCompDescPtr, smoothSize, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        // hipMemcpy(src1U8Blurred, src1Ptr, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        // hipMemcpy(src2U8Blurred, src2Ptr, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp8u), hipMemcpyDeviceToDevice);

        //
        // hipMemcpy(src1F32Blurred, src1F32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        // hipMemcpy(src2F32Blurred, src2F32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);

        // Mods to run Gaussian Filter instead
        if (sigma == 0)
            sigma = 1;
        for (int batchCount = 0; batchCount < mVecCompDescPtr->n; batchCount++)
            stdDevPtrForGaussian[batchCount] = sigma;
        hip_exec_gaussian_filter_f32_tensor(src1F32, mVecCompDescPtr, src1F32Blurred, mVecCompDescPtr, smoothSize, stdDevPtrForGaussian, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hip_exec_gaussian_filter_f32_tensor(src2F32, mVecCompDescPtr, src2F32Blurred, mVecCompDescPtr, smoothSize, stdDevPtrForGaussian, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hipDeviceSynchronize();
        std::cerr << "\n\nGAUSSIAN FILTER PARAMS smoothSize and sigma -> " << smoothSize << ", " << stdDevPtrForGaussian[0] << "\n\n";


        // -------------------- stage output dump check --------------------
        // if (k == 5)
        // {
        //     hipDeviceSynchronize();
        //     std::cerr << "\n\nAbout to copy d to d...";
        //     hipMemcpy(dstPtrIntermF32, src1F32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipMemcpy(dstPtrIntermF32 + FARNEBACK_OUTPUT_FRAME_SIZE, src2F32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipMemcpy(dstPtrIntermF32 + (2 * FARNEBACK_OUTPUT_FRAME_SIZE), src1F32Blurred, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipMemcpy(dstPtrIntermF32 + (3 * FARNEBACK_OUTPUT_FRAME_SIZE), src2F32Blurred, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipDeviceSynchronize();
        //     return RPP_SUCCESS;
        // }

        // std::cerr << "\nAbout to run 2 resizes...";
        // hip_exec_resize_tensor(src1U8Blurred, srcCompDescPtr, pyramidLevelPrevU8, srcCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        // hip_exec_resize_tensor(src2U8Blurred, srcCompDescPtr, pyramidLevelCurrU8, srcCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hip_exec_resize_tensor(src1F32Blurred, mVecCompDescPtr, pyramidLevelPrevF32, mVecCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hip_exec_resize_tensor(src2F32Blurred, mVecCompDescPtr, pyramidLevelCurrF32, mVecCompDescPtr, pyramidImgPatchPtr, RpptInterpolationType::BILINEAR, roiTensorPtrSrc, RpptRoiType::XYWH, handle);
        hipDeviceSynchronize();
        // -------------------- stage output dump check --------------------
        // if (k == 0)
        // {
        //     hipDeviceSynchronize();
        //     std::cerr << "\nAbout to copy d to d...";
        //     // hipMemcpy(dstPtrIntermU8, pyramidLevelPrevU8, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        //     hipMemcpy(dstPtrIntermF32, pyramidLevelPrevF32, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     return RPP_SUCCESS;
        // }

        // if(k == 1)
        //     return RPP_SUCCESS;

        *roiTensorPtrPyramid = {0, 0, pyramidImgPatchPtr->width, pyramidImgPatchPtr->height};
        // std::cerr << "\nAbout to run 2 farneback_polynomial_expansion_tensor...";
        // // verification process
        // Rpp32f *pyramidLevelPrevVerify;
        // hipHostMalloc(&pyramidLevelPrevVerify, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f));
        // pyramidLevelPrevVerify[0] = 75.000007600000004;
        // pyramidLevelPrevVerify[1] = 74.999748199999999;
        // pyramidLevelPrevVerify[2] = 74.999748199999999;
        // pyramidLevelPrevVerify[3] = 74.782745399999996;
        // pyramidLevelPrevVerify[4] = 76.164245600000001;
        // hip_exec_farneback_polynomial_expansion_tensor(pyramidLevelPrevVerify, mVecCompDescPtr, polyResPrev, mVecCompDescPtr, polyExpNbhoodSize, g, xg, xxg, invG11033355_f4, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
        // Rpp32f *polyResPrevVerify;
        // hipMemcpy(dstPtrIntermU8, src2U8Blurred, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp8u), hipMemcpyDeviceToDevice);

        // Override with pre-stored cudaResizdStrided buffer for 1st iteration - if needed for testing
        // hipMemcpy(pyramidLevelPrevF32, dh_cudaResizdStrided, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        // hipMemcpy(pyramidLevelCurrF32, dh_cudaResizdStrided + FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);

        // hip_exec_farneback_polynomial_expansion_tensor(pyramidLevelPrevU8, srcCompDescPtr, polyResPrev, mVecCompDescPtr, polyExpNbhoodSize, g, xg, xxg, invG11033355_f4, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
        // hip_exec_farneback_polynomial_expansion_tensor(pyramidLevelCurrU8, srcCompDescPtr, polyResCurr, mVecCompDescPtr, polyExpNbhoodSize, g, xg, xxg, invG11033355_f4, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
        hip_exec_farneback_polynomial_expansion_tensor(pyramidLevelPrevF32, mVecCompDescPtr, polyResPrev, mVecCompDescPtr, polyExpNbhoodSize, g, xg, xxg, invG11033355_f4, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
        // return RPP_SUCCESS;
        hip_exec_farneback_polynomial_expansion_tensor(pyramidLevelCurrF32, mVecCompDescPtr, polyResCurr, mVecCompDescPtr, polyExpNbhoodSize, g, xg, xxg, invG11033355_f4, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
        // -------------------- stage output dump check --------------------
        // if (k == 0)
        // {
        //     hipDeviceSynchronize();
        //     std::cerr << "\nAbout to copy d to d...";
        //     hipMemcpy(dstPtrIntermF32, polyResPrev, FARNEBACK_OUTPUT_FRAME_SIZE * 5 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipDeviceSynchronize();
        //     std::cerr << "\nCopy d to d complete...";
        //     return RPP_SUCCESS;
        // }

        // streams[1].waitForCompletion();
        // device::optflow_farneback::updateMatricesGpu(curFlowX, curFlowY, R[0], R[1], M, StreamAccessor::getStream(streams[0]));

        // std::cerr << "\nAbout to run farneback_matrices_update_tensor...";
        hip_exec_farneback_matrices_update_tensor(mVecCurrCompX, mVecCurrCompY, polyResPrev, polyResCurr, polyMatrices, mVecCompDescPtr, border, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);

        // -------------------- stage output dump check --------------------
        // if (k == 0)
        // {
        //     hipDeviceSynchronize();
        //     std::cerr << "\nAbout to copy d to d...";
        //     hipMemcpy(dstPtrIntermF32, polyMatrices, FARNEBACK_OUTPUT_FRAME_SIZE * 5 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipDeviceSynchronize();
        //     std::cerr << "\nCopy d to d complete...";
        //     return RPP_SUCCESS;
        // }

        // if (flags_ & OPTFLOW_FARNEBACK_GAUSSIAN)
        // {
        //     Mat g = getGaussianKernel(winSize_, winSize_/2*0.3f, CV_32F);
        //     device::optflow_farneback::setGaussianBlurKernel(g.ptr<float>(winSize_/2), winSize_/2);
        // }

        for (int i = 0; i < numIterations; i++)
        // for (int i = 0; i < 1; i++)  // TEMPORARILY having only 1 iteration
        {
            hipDeviceSynchronize();

            // if (flags_ & OPTFLOW_FARNEBACK_GAUSSIAN)
            //     updateFlow_gaussianBlur(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize_, i < numIters_-1, streams);
            // else
                // updateFlow_boxFilter(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize_, i < numIters_-1, streams);



            // INSIDE UPDATEFLOW_BOXFILTER
            // if (deviceSupports(FEATURE_SET_COMPUTE_12))
            //     device::optflow_farneback::boxFilter5Gpu(M, blockSize/2, bufM, StreamAccessor::getStream(streams[0]));
            // else
            //     device::optflow_farneback::boxFilter5Gpu_CC11(M, blockSize/2, bufM, StreamAccessor::getStream(streams[0]));
            // swap(M, bufM);

            // for (int i = 1; i < 5; ++i)
            //     streams[i].waitForCompletion();
            // device::optflow_farneback::updateFlowGpu(M, flowx, flowy, StreamAccessor::getStream(streams[0]));

            // if (updateMatrices)
            //     device::optflow_farneback::updateMatricesGpu(flowx, flowy, R0, R1, M, StreamAccessor::getStream(streams[0]));



            // std::cerr << "\nAbout to run 5 box_filter_tensor...";

            // ---------------- TEMPORARILY COMMENTING box filtering
            hip_exec_box_filter_f32_tensor(polyMatrices, mVecCompDescPtr, polyMatricesBlurred, mVecCompDescPtr, windowSize, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hip_exec_box_filter_f32_tensor(polyMatrices + FARNEBACK_OUTPUT_FRAME_SIZE, mVecCompDescPtr, polyMatricesBlurred + FARNEBACK_OUTPUT_FRAME_SIZE, mVecCompDescPtr, windowSize, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hip_exec_box_filter_f32_tensor(polyMatrices + (2 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompDescPtr, polyMatricesBlurred + (2 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompDescPtr, windowSize, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hip_exec_box_filter_f32_tensor(polyMatrices + (3 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompDescPtr, polyMatricesBlurred + (3 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompDescPtr, windowSize, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hip_exec_box_filter_f32_tensor(polyMatrices + (4 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompDescPtr, polyMatricesBlurred + (4 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompDescPtr, windowSize, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);
            hipDeviceSynchronize();
            Rpp32f *temp;
            temp = polyMatrices;
            polyMatrices = polyMatricesBlurred;
            polyMatricesBlurred = temp;


            // hipMemcpy(polyMatricesBlurred, polyMatrices, FARNEBACK_OUTPUT_FRAME_SIZE * 5 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);

            // -------------------- stage output dump check --------------------
            // if (i == 0)
            // {
            //     hipDeviceSynchronize();
            //     std::cerr << "\nAbout to copy d to d...";
            //     hipMemcpy(dstPtrIntermF32, polyMatrices, FARNEBACK_OUTPUT_FRAME_SIZE * 5 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
            //     hipDeviceSynchronize();
            //     std::cerr << "\nCopy d to d complete...";
            //     return RPP_SUCCESS;
            // }

            // std::cerr << "\nAbout to run farneback_motion_vectors_update_tensor...";
            hip_exec_farneback_motion_vectors_update_tensor(polyMatrices, mVecCurrCompX, mVecCurrCompY, mVecCompDescPtr, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);

            // -------------------- stage output dump check --------------------
            hipDeviceSynchronize();
            // if (i == 2)
            // {
            //     std::cerr << "\nAbout to copy d to d...";
            //     hipMemcpy(dstPtrIntermF32, mVecCurrCompX, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
            //     hipDeviceSynchronize();
            //     std::cerr << "\nCopy d to d complete...";
            //     return RPP_SUCCESS;
            // }

            if (i < numIterations - 1)
            {
                // std::cerr << "\nAbout to run farneback_matrices_update_tensor...";
                hip_exec_farneback_matrices_update_tensor(mVecCurrCompX, mVecCurrCompY, polyResPrev, polyResCurr, polyMatrices, mVecCompDescPtr, border, roiTensorPtrPyramid, RpptRoiType::XYWH, handle);

                // -------------------- stage output dump check --------------------
                hipDeviceSynchronize();
                // if (i == 1)
                // {
                //     std::cerr << "\nAbout to copy d to d...";
                //     hipMemcpy(dstPtrIntermF32, polyMatrices, FARNEBACK_OUTPUT_FRAME_SIZE * 5 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
                //     hipDeviceSynchronize();
                //     std::cerr << "\nCopy d to d complete...";
                //     return RPP_SUCCESS;
                // }
            }

        }

        // -------------------- stage output dump check --------------------
        // hipDeviceSynchronize();
        // std::cerr << "\nAbout to copy d to d...";
        // hipMemcpy(dstPtrIntermF32, mVecCurrCompX, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        // hipDeviceSynchronize();
        // std::cerr << "\nCopy d to d complete...";
        // return RPP_SUCCESS;

        Rpp32f *temp;

        temp = mVecPrevCompX;
        mVecPrevCompX = mVecCurrCompX;
        mVecCurrCompX = temp;

        temp = mVecPrevCompY;
        mVecPrevCompY = mVecCurrCompY;
        mVecCurrCompY = temp;

        // -------------------- stage output dump check --------------------
        // hipDeviceSynchronize();
        // if (k == 0)
        // {
        //     std::cerr << "\nAbout to copy d to d...";
        //     hipMemcpy(dstPtrIntermF32, mVecPrevCompX, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipMemcpy(dstPtrIntermF32 + FARNEBACK_OUTPUT_FRAME_SIZE, mVecPrevCompY, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        //     hipDeviceSynchronize();
        //     std::cerr << "\nCopy d to d complete...";
        //     return RPP_SUCCESS;
        // }

        // prevFlowX = curFlowX;
        // prevFlowY = curFlowY;

    }

    // flowx = curFlowX;
    // flowy = curFlowY;

    // if (!stream)
    //     streams[0].waitForCompletion();

    hipDeviceSynchronize();

    hipMemcpy(mVecCompX, mVecPrevCompX, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
    hipMemcpy(mVecCompY, mVecPrevCompY, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);

    // -------------------- stage output dump check --------------------
    // hipDeviceSynchronize();
    // std::cerr << "\nAbout to copy d to d...";
    // hipMemcpy(dstPtrIntermF32, mVecPrevCompX, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
    // hipMemcpy(dstPtrIntermF32 + FARNEBACK_OUTPUT_FRAME_SIZE, mVecPrevCompY, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
    // hipDeviceSynchronize();
    // std::cerr << "\nCopy d to d complete...";
    // return RPP_SUCCESS;



    // hip_exec_saturate_left_tensor(mVecCompX, mVecCompDescPtr, handle);
    // hip_exec_saturate_left_tensor(mVecCompY, mVecCompDescPtr, handle);
    // hipDeviceSynchronize();















    // Whole loop

    // for (int k = numLevelsCropped; k >= 0; k--)
    // {
    //     streams[0].waitForCompletion();

    //     scale = 1;
    //     for (int i = 0; i < k; i++)
    //         scale *= pyrScale_;

    //     double sigma = (1./scale - 1) * 0.5;
    //     int smoothSize = cvRound(sigma*5) | 1;
    //     smoothSize = std::max(smoothSize, 3);

    //     int width = cvRound(size.width*scale);
    //     int height = cvRound(size.height*scale);

    //     if (fastPyramids_)
    //     {
    //         width = pyramid0_[k].cols;
    //         height = pyramid0_[k].rows;
    //     }

    //     if (k > 0)
    //     {
    //         curFlowX.create(height, width, CV_32F);
    //         curFlowY.create(height, width, CV_32F);
    //     }
    //     else
    //     {
    //         curFlowX = flowx0;
    //         curFlowY = flowy0;
    //     }

    //     if (!prevFlowX.data)
    //     {
    //         if (flags_ & OPTFLOW_USE_INITIAL_FLOW)
    //         {
    //             cuda::resize(flowx0, curFlowX, Size(width, height), 0, 0, INTER_LINEAR, streams[0]);
    //             cuda::resize(flowy0, curFlowY, Size(width, height), 0, 0, INTER_LINEAR, streams[1]);
    //             curFlowX.convertTo(curFlowX, curFlowX.depth(), scale, streams[0]);
    //             curFlowY.convertTo(curFlowY, curFlowY.depth(), scale, streams[1]);
    //         }
    //         else
    //         {
    //             curFlowX.setTo(0, streams[0]);
    //             curFlowY.setTo(0, streams[1]);
    //         }
    //     }
    //     else
    //     {
    //         cuda::resize(prevFlowX, curFlowX, Size(width, height), 0, 0, INTER_LINEAR, streams[0]);
    //         cuda::resize(prevFlowY, curFlowY, Size(width, height), 0, 0, INTER_LINEAR, streams[1]);
    //         curFlowX.convertTo(curFlowX, curFlowX.depth(), 1./pyrScale_, streams[0]);
    //         curFlowY.convertTo(curFlowY, curFlowY.depth(), 1./pyrScale_, streams[1]);
    //     }

    //     GpuMat M = allocMatFromBuf(5*height, width, CV_32F, M_);
    //     GpuMat bufM = allocMatFromBuf(5*height, width, CV_32F, bufM_);
    //     GpuMat R[2] =
    //     {
    //         allocMatFromBuf(5*height, width, CV_32F, R_[0]),
    //         allocMatFromBuf(5*height, width, CV_32F, R_[1])
    //     };

    //     if (fastPyramids_)
    //     {
    //         device::optflow_farneback::polynomialExpansionGpu(pyramid0_[k], polyN_, R[0], StreamAccessor::getStream(streams[0]));
    //         device::optflow_farneback::polynomialExpansionGpu(pyramid1_[k], polyN_, R[1], StreamAccessor::getStream(streams[1]));
    //     }
    //     else
    //     {
    //         GpuMat blurredFrame[2] =
    //         {
    //             allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[0]),
    //             allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[1])
    //         };
    //         GpuMat pyrLevel[2] =
    //         {
    //             allocMatFromBuf(height, width, CV_32F, pyrLevel_[0]),
    //             allocMatFromBuf(height, width, CV_32F, pyrLevel_[1])
    //         };

    //         Mat g = getGaussianKernel(smoothSize, sigma, CV_32F);
    //         device::optflow_farneback::setGaussianBlurKernel(g.ptr<float>(smoothSize/2), smoothSize/2);

    //         for (int i = 0; i < 2; i++)
    //         {
    //             device::optflow_farneback::gaussianBlurGpu(
    //                     frames_[i], smoothSize/2, blurredFrame[i], BORDER_REFLECT101, StreamAccessor::getStream(streams[i]));
    //             cuda::resize(blurredFrame[i], pyrLevel[i], Size(width, height), 0.0, 0.0, INTER_LINEAR, streams[i]);
    //             device::optflow_farneback::polynomialExpansionGpu(pyrLevel[i], polyN_, R[i], StreamAccessor::getStream(streams[i]));
    //         }
    //     }

    //     streams[1].waitForCompletion();
    //     device::optflow_farneback::updateMatricesGpu(curFlowX, curFlowY, R[0], R[1], M, StreamAccessor::getStream(streams[0]));

    //     if (flags_ & OPTFLOW_FARNEBACK_GAUSSIAN)
    //     {
    //         Mat g = getGaussianKernel(winSize_, winSize_/2*0.3f, CV_32F);
    //         device::optflow_farneback::setGaussianBlurKernel(g.ptr<float>(winSize_/2), winSize_/2);
    //     }
    //     for (int i = 0; i < numIters_; i++)
    //     {
    //         if (flags_ & OPTFLOW_FARNEBACK_GAUSSIAN)
    //             updateFlow_gaussianBlur(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize_, i < numIters_-1, streams);
    //         else
    //             updateFlow_boxFilter(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize_, i < numIters_-1, streams);
    //     }

    //     prevFlowX = curFlowX;
    //     prevFlowY = curFlowY;
    // }

    // flowx = curFlowX;
    // flowy = curFlowY;

    // if (!stream)
    //     streams[0].waitForCompletion();

    hipHostFree(&buf);
    hipHostFree(&stdDevPtrForGaussian);
    hipHostFree(&border);
    hipHostFree(&pyramidImgPatchPtr);
    hipHostFree(&roiTensorPtrSrc);
    hipHostFree(&roiTensorPtrPyramid);

    return RPP_SUCCESS;
}
