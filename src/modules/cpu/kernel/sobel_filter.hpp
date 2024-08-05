/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include "rpp_cpu_filter.hpp"

template<typename T>
inline void sobel_filter_unidirection_generic_tensor(T **srcPtrTemp, T *dstPtrTemp, Rpp32s columnIndex,
                                                     Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit,
                                                     Rpp32f *filterTensor, Rpp32u channels = 1)
{
    Rpp32f accum = 0.0f;
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);
    if constexpr (std::is_same<T, Rpp8s>::value)
    {
        for (int i = 0; i < rowKernelLoopLimit; i++)
            for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
                accum += static_cast<Rpp32f>(srcPtrTemp[i][k] + 128) * filterTensor[i * kernelSize + j];
    }
    else
    {
        for (int i = 0; i < rowKernelLoopLimit; i++)
            for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
                accum += static_cast<Rpp32f>(srcPtrTemp[i][k]) * filterTensor[i * kernelSize + j];
    }

    saturate_pixel(accum, dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 kernel, process them separately
template<typename T>
inline void process_left_border_columns_pln_pln(T **srcPtrTemp, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f *filterTensor)
{
    for (int k = 0; k < padLength; k++)
    {
        sobel_filter_unidirection_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 1);
        dstPtrTemp++;
    }
}

template<typename T>
inline void sobel_filter_bidirection_generic_tensor(T **srcPtrTemp, T *dstPtrTemp, Rpp32s columnIndex,
                                                    Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit,
                                                    Rpp32f *filterXTensor, Rpp32f *filterYTensor, Rpp32u channels = 1)
{
    Rpp32f accumX = 0.0f;
    Rpp32f accumY = 0.0f;
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);
    if constexpr (std::is_same<T, Rpp8s>::value)
    {
        for (int i = 0; i < rowKernelLoopLimit; i++)
        {
            for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
            {
                accumX += static_cast<Rpp32f>(srcPtrTemp[i][k] + 128) * filterXTensor[i * kernelSize + j];
                accumY += static_cast<Rpp32f>(srcPtrTemp[i][k] + 128) * filterYTensor[i * kernelSize + j];
            }
        }
    }
    else
    {
        for (int i = 0; i < rowKernelLoopLimit; i++)
        {
            for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
            {
                accumX += static_cast<Rpp32f>(srcPtrTemp[i][k]) * filterXTensor[i * kernelSize + j];
                accumY += static_cast<Rpp32f>(srcPtrTemp[i][k]) * filterYTensor[i * kernelSize + j];
            }
        }
    }

    // saturate the values of accumX and accumY to the range of datatype
    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
    {
        accumX = RPPPIXELCHECK(accumX);
        accumY = RPPPIXELCHECK(accumY);
    }
    else
    {
        accumX = RPPPIXELCHECKF32(accumX);
        accumY = RPPPIXELCHECKF32(accumY);
    }

    Rpp32f accum = sqrt((accumX * accumX) + (accumY * accumY));
    saturate_pixel(accum, dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 kernel, process them separately
template<typename T>
inline void process_left_border_columns_pln_pln(T **srcPtrTemp, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f *filterXTensor, Rpp32f *filterYTensor)
{
    for (int k = 0; k < padLength; k++)
    {
        sobel_filter_bidirection_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterXTensor, filterYTensor, 1);
        dstPtrTemp++;
    }
}

Rpp32f sobel3x3X[9] = {-1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1};
Rpp32f sobel3x3Y[9] = {-1, -2, -1,
                       0, 0, 0,
                       1, 2, 1};
Rpp32f sobel5x5X[25] = {-1,  -2,   0,   2,   1,
                        -4,  -8,   0,   8,   4,
                        -6, -12,   0,  12,   6,
                        -4,  -8,   0,   8,   4,
                        -1,  -2,   0,   2,   1};
Rpp32f sobel5x5Y[25] = {-1,  -4,  -6,  -4,  -1,
                        -2,  -8, -12,  -8,  -2,
                         0,   0,   0,   0,   0,
                         2,   8,  12,   8,   2,
                         1,   4,   6,   4,   1};
Rpp32f sobel7x7X[49] = {-1,   -4,   -5,    0,    5,    4,    1,
                        -6,  -24,  -30,    0,   30,   24,    6,
                        -15,  -60,  -75,    0,   75,   60,   15,
                        -20,  -80, -100,    0,  100,   80,   20,
                        -15,  -60,  -75,    0,   75,   60,   15,
                        -6,  -24,  -30,    0,   30,   24,    6,
                        -1,   -4,   -5,    0,    5,    4,    1};
Rpp32f sobel7x7Y[49] = {-1,   -6,  -15,  -20,  -15,   -6,   -1,
                        -4,  -24,  -60,  -80,  -60,  -24,   -4,
                        -5,  -30,  -75, -100,  -75,  -30,   -5,
                         0,    0,    0,    0,    0,    0,    0,
                         5,   30,   75,  100,   75,   30,    5,
                         4,   24,   60,   80,   60,   24,    4,
                         1,    6,   15,   20,   15,    6,    1};

// load function for 3x3 kernel size
inline void rpp_load_sobel_filter_3x3_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load16_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[1], &pRow[2]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        rpp_load16_u8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    else
    {
        pRow[4] = avx_p0;
        pRow[5] = avx_p0;
    }
}

inline void rpp_load_sobel_filter_3x3_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load16_i8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[1], &pRow[2]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        rpp_load16_i8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    else
    {
        pRow[4] = avx_p0;
        pRow[5] = avx_p0;
    }
}

inline void rpp_load_sobel_filter_3x3_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load16_f32_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[1], &pRow[2]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        rpp_load16_f32_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    else
    {
        pRow[4] = avx_p0;
        pRow[5] = avx_p0;
    }
}

// load function for 3x3 kernel size
inline void rpp_load_sobel_filter_3x3_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load16_f16_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[1], &pRow[2]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        rpp_load16_f16_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    else
    {
        pRow[4] = avx_p0;
        pRow[5] = avx_p0;
    }
}

// load function for 5x5 kernel size
inline void rpp_load_sobel_filter_5x5_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[2], &pRow[4]);

    for (int k = 3; k < rowKernelLoopLimit; k++)
        rpp_load16_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_sobel_filter_5x5_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_i8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[2], &pRow[4]);

    for (int k = 3; k < rowKernelLoopLimit; k++)
        rpp_load16_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_sobel_filter_5x5_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_f32_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[2], &pRow[4]);

    for (int k = 3; k < rowKernelLoopLimit; k++)
        rpp_load16_f32_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_sobel_filter_5x5_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_f16_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[2], &pRow[4]);

    for (int k = 3; k < rowKernelLoopLimit; k++)
        rpp_load16_f16_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

// load function for 7x7 kernel size
inline void rpp_load_sobel_filter_7x7_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    rpp_load16_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        rpp_load16_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_sobel_filter_7x7_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    rpp_load16_i8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        rpp_load16_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_sobel_filter_7x7_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    rpp_load16_f32_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        rpp_load16_f32_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_sobel_filter_7x7_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_f16_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        rpp_load16_f16_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_sobel_store16(Rpp8u *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);
}

inline void rpp_sobel_store16(Rpp8s *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
}

inline void rpp_sobel_store16(Rpp32f *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
}

inline void rpp_sobel_store16(Rpp16f *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
}

inline void rpp_sobel_store8(Rpp8u *dstPtrTemp, __m256 *pDst)
{
    rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);
}

inline void rpp_sobel_store8(Rpp8s *dstPtrTemp, __m256 *pDst)
{
    rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
}

inline void rpp_sobel_store8(Rpp32f *dstPtrTemp, __m256 *pDst)
{
    rpp_store8_f32_to_f32_avx(dstPtrTemp, pDst);
}

inline void rpp_sobel_store8(Rpp16f *dstPtrTemp, __m256 *pDst)
{
    rpp_store8_f32_to_f16_avx(dstPtrTemp, pDst);
}

template<typename T>
RppStatus sobel_filter_host_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u sobelType,
                                   Rpp32u kernelSize,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
        Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;
        bool combined = (sobelType == 2);
        Rpp32f *filter, *filterX, *filterY;

#if __AVX2__
        __m256 pMax, pMin;
        if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
        {
            pMax = avx_p255;
            pMin = avx_p0;
        }
        else
        {
            pMax = avx_p1;
            pMin = avx_p0;
        }
#endif

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + roi.xywhROI.xy.x;
        dstPtrChannel = dstPtrImage;
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && (srcDescPtr->c == 1))
        {
            if (kernelSize == 3)
            {
                T *srcPtrRow[3], *dstPtrRow;
                for (int i = 0; i < 3; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;

                if (combined)
                {
                    filterX = sobel3x3X;
                    filterY = sobel3x3Y;
#if __AVX2__
                    __m256 pFilterX[9], pFilterY[9];
                    for (int i = 0; i < 9; i++)
                    {
                        pFilterX[i] = _mm256_set1_ps(filterX[i]);
                        pFilterY[i] = _mm256_set1_ps(filterY[i]);
                    }
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                        {
                            __m256 pRow[6], pDst[2], pDstX[2], pDstY[2];
                            rpp_load_sobel_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            for (int k = 0; k < 2; k++)
                            {
                                pDstX[k] = avx_p0;
                                pDstY[k] = avx_p0;
                                pDst[k] = avx_p0;
                            }
                            for (int k = 0; k < 3; k++)
                            {
                                __m256 pTemp[3], pRowShift[2];
                                Rpp32s filterIndex =  k * 3;
                                Rpp32s rowIndex = k * 2;

                                pRowShift[0] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1);
                                pRowShift[1] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2);
                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilterX[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterX[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterX[filterIndex + 2]);
                                pDstX[0] = _mm256_add_ps(pDstX[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilterY[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterY[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterY[filterIndex + 2]);
                                pDstY[0] = _mm256_add_ps(pDstY[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                                pRowShift[0] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 1), avx_pxMaskRotate0To1);
                                pRowShift[1] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 3), avx_pxMaskRotate0To2);
                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilterX[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterX[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterX[filterIndex + 2]);
                                pDstX[1] = _mm256_add_ps(pDstX[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilterY[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterY[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterY[filterIndex + 2]);
                                pDstY[1] = _mm256_add_ps(pDstY[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));
                            }
                            pDstX[0] = _mm256_min_ps(_mm256_max_ps(pDstX[0], pMin), pMax);
                            pDstY[0] = _mm256_min_ps(_mm256_max_ps(pDstY[0], pMin), pMax);
                            pDstX[0] = _mm256_mul_ps(pDstX[0], pDstX[0]);
                            pDstY[0] = _mm256_mul_ps(pDstY[0], pDstY[0]);
                            pDst[0] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[0], pDstY[0]));
                            pDst[0] = _mm256_min_ps(_mm256_max_ps(pDst[0], pMin), pMax);

                            pDstX[1] = _mm256_min_ps(_mm256_max_ps(pDstX[1], pMin), pMax);
                            pDstY[1] = _mm256_min_ps(_mm256_max_ps(pDstY[1], pMin), pMax);
                            pDstX[1] = _mm256_mul_ps(pDstX[1], pDstX[1]);
                            pDstY[1] = _mm256_mul_ps(pDstY[1], pDstY[1]);
                            pDst[1] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[1], pDstY[1]));
                            pDst[1] = _mm256_min_ps(_mm256_max_ps(pDst[1], pMin), pMax);

                            rpp_sobel_store16(dstPtrTemp, pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 14);
                            dstPtrTemp += 14;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_bidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
                else
                {
                    filter = (!sobelType) ? sobel3x3X : sobel3x3Y;
#if __AVX2__
                    __m256 pFilter[9];
                    for (int i = 0; i < 9; i++)
                        pFilter[i] = _mm256_set1_ps(filter[i]);
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                        {
                            __m256 pRow[6], pDst[2];
                            rpp_load_sobel_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            pDst[0] = avx_p0;
                            pDst[1] = avx_p0;
                            for (int k = 0; k < 3; k++)
                            {
                                __m256 pTemp[3];
                                Rpp32s filterIndex =  k * 3;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));
                            }
                            pDst[0] = _mm256_min_ps(_mm256_max_ps(pDst[0], pMin), pMax);
                            pDst[1] = _mm256_min_ps(_mm256_max_ps(pDst[1], pMin), pMax);
                            rpp_sobel_store16(dstPtrTemp, pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 14);
                            dstPtrTemp += 14;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_unidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
            }
            else if (kernelSize == 5)
            {
                T *srcPtrRow[5], *dstPtrRow;
                for (int i = 0; i < 5; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;

                if (combined)
                {
                    filterX = sobel5x5X;
                    filterY = sobel5x5Y;
#if __AVX2__
                    __m256 pFilterX[25], pFilterY[25];
                    for (int i = 0; i < 25; i++)
                    {
                        pFilterX[i] = _mm256_set1_ps(filterX[i]);
                        pFilterY[i] = _mm256_set1_ps(filterY[i]);
                    }
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2],  srcPtrRow[3],  srcPtrRow[4]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                        {
                            __m256 pRow[10], pDst[2], pDstX[2], pDstY[2];
                            rpp_load_sobel_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            for (int k = 0; k < 2; k++)
                            {
                                pDstX[k] = avx_p0;
                                pDstY[k] = avx_p0;
                                pDst[k] = avx_p0;
                            }
                            for (int k = 0; k < 5; k++)
                            {
                                __m256 pTemp[5], pRowShift[4];
                                Rpp32s filterIndex =  k * 5;
                                Rpp32s rowIndex = k * 2;

                                pRowShift[0] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1);
                                pRowShift[1] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2);
                                pRowShift[2] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3);
                                pRowShift[3] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4);
                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilterX[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterX[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterX[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(pRowShift[2], pFilterX[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(pRowShift[3], pFilterX[filterIndex + 4]);
                                pDstX[0] = _mm256_add_ps(pDstX[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilterY[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterY[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterY[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(pRowShift[2], pFilterY[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(pRowShift[3], pFilterY[filterIndex + 4]);
                                pDstY[0] = _mm256_add_ps(pDstY[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));

                                pRowShift[0] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 1), avx_pxMaskRotate0To1);
                                pRowShift[1] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 3), avx_pxMaskRotate0To2);
                                pRowShift[2] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 7), avx_pxMaskRotate0To3);
                                pRowShift[3] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 15), avx_pxMaskRotate0To4);
                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilterX[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterX[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterX[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(pRowShift[2], pFilterX[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(pRowShift[3], pFilterX[filterIndex + 4]);
                                pDstX[1] = _mm256_add_ps(pDstX[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilterY[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterY[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterY[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(pRowShift[2], pFilterY[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(pRowShift[3], pFilterY[filterIndex + 4]);
                                pDstY[1] = _mm256_add_ps(pDstY[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));
                            }
                            pDstX[0] = _mm256_min_ps(_mm256_max_ps(pDstX[0], pMin), pMax);
                            pDstY[0] = _mm256_min_ps(_mm256_max_ps(pDstY[0], pMin), pMax);
                            pDstX[0] = _mm256_mul_ps(pDstX[0], pDstX[0]);
                            pDstY[0] = _mm256_mul_ps(pDstY[0], pDstY[0]);
                            pDst[0] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[0], pDstY[0]));
                            pDst[0] = _mm256_min_ps(_mm256_max_ps(pDst[0], pMin), pMax);

                            pDstX[1] = _mm256_min_ps(_mm256_max_ps(pDstX[1], pMin), pMax);
                            pDstY[1] = _mm256_min_ps(_mm256_max_ps(pDstY[1], pMin), pMax);
                            pDstX[1] = _mm256_mul_ps(pDstX[1], pDstX[1]);
                            pDstY[1] = _mm256_mul_ps(pDstY[1], pDstY[1]);
                            pDst[1] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[1], pDstY[1]));
                            pDst[1] = _mm256_min_ps(_mm256_max_ps(pDst[1], pMin), pMax);

                            rpp_sobel_store16(dstPtrTemp, pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                            dstPtrTemp += 12;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_bidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
                else
                {
                    filter = (!sobelType) ? sobel5x5X : sobel5x5Y;
#if __AVX2__
                    __m256 pFilter[25];
                    for (int i = 0; i < 25; i++)
                        pFilter[i] = _mm256_set1_ps(filter[i]);
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                        {
                            __m256 pRow[10], pDst[2];
                            rpp_load_sobel_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            pDst[0] = avx_p0;
                            pDst[1] = avx_p0;
                            for (int k = 0; k < 5; k++)
                            {
                                __m256 pTemp[5];
                                Rpp32s filterIndex =  k * 5;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));
                            }
                            pDst[0] = _mm256_min_ps(_mm256_max_ps(pDst[0], pMin), pMax);
                            pDst[1] = _mm256_min_ps(_mm256_max_ps(pDst[1], pMin), pMax);
                            rpp_sobel_store16(dstPtrTemp, pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                            dstPtrTemp += 12;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_unidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
            }
            else if (kernelSize == 7)
            {
                T *srcPtrRow[7], *dstPtrRow;
                for (int i = 0; i < 7; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;

                if (combined)
                {
                    filterX = sobel7x7X;
                    filterY = sobel7x7Y;
#if __AVX2__
                    __m256 pFilterX[49], pFilterY[49];
                    for (int i = 0; i < 49; i++)
                    {
                        pFilterX[i] = _mm256_set1_ps(filterX[i]);
                        pFilterY[i] = _mm256_set1_ps(filterY[i]);
                    }
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[7] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2],  srcPtrRow[3],  srcPtrRow[4], srcPtrRow[5], srcPtrRow[6]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            __m256 pRow[14], pDst, pDstX, pDstY;
                            rpp_load_sobel_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            pDstX = avx_p0;
                            pDstY = avx_p0;
                            pDst = avx_p0;
                            for (int k = 0; k < 7; k++)
                            {
                                __m256 pTemp[7], pRowShift[6];
                                Rpp32s filterIndex =  k * 7;
                                Rpp32s rowIndex = k * 2;

                                pRowShift[0] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1);
                                pRowShift[1] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2);
                                pRowShift[2] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3);
                                pRowShift[3] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4);
                                pRowShift[4] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 31), avx_pxMaskRotate0To5);
                                pRowShift[5] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6);
                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilterX[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterX[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterX[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(pRowShift[2], pFilterX[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(pRowShift[3], pFilterX[filterIndex + 4]);
                                pTemp[5] = _mm256_mul_ps(pRowShift[4], pFilterX[filterIndex + 5]);
                                pTemp[6] = _mm256_mul_ps(pRowShift[5], pFilterX[filterIndex + 6]);
                                pDstX = _mm256_add_ps(pDstX, _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(_mm256_add_ps(pTemp[3], pTemp[4]), _mm256_add_ps(pTemp[5], pTemp[6]))));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilterY[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterY[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterY[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(pRowShift[2], pFilterY[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(pRowShift[3], pFilterY[filterIndex + 4]);
                                pTemp[5] = _mm256_mul_ps(pRowShift[4], pFilterY[filterIndex + 5]);
                                pTemp[6] = _mm256_mul_ps(pRowShift[5], pFilterY[filterIndex + 6]);
                                pDstY = _mm256_add_ps(pDstY, _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(_mm256_add_ps(pTemp[3], pTemp[4]), _mm256_add_ps(pTemp[5], pTemp[6]))));
                            }
                            pDstX = _mm256_min_ps(_mm256_max_ps(pDstX, pMin), pMax);
                            pDstY = _mm256_min_ps(_mm256_max_ps(pDstY, pMin), pMax);
                            pDstX = _mm256_mul_ps(pDstX, pDstX);
                            pDstY = _mm256_mul_ps(pDstY, pDstY);
                            pDst =  _mm256_sqrt_ps(_mm256_add_ps(pDstX, pDstY));
                            pDst = _mm256_min_ps(_mm256_max_ps(pDst, pMin), pMax);

                            rpp_sobel_store8(dstPtrTemp, &pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            dstPtrTemp += 8;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_bidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
                else
                {
                    filter = (!sobelType) ? sobel7x7X : sobel7x7Y;
#if __AVX2__
                    __m256 pFilter[49];
                    for (int i = 0; i < 49; i++)
                        pFilter[i] = _mm256_set1_ps(filter[i]);
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[7] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4], srcPtrRow[5], srcPtrRow[6]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            __m256 pRow[14], pDst;
                            rpp_load_sobel_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            pDst = avx_p0;
                            for (int k = 0; k < 7; k++)
                            {
                                __m256 pTemp[7];
                                Rpp32s filterIndex =  k * 7;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 31), avx_pxMaskRotate0To5), pFilter[filterIndex + 5]);
                                pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 6]);
                                pDst =  _mm256_add_ps(pDst, _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(_mm256_add_ps(pTemp[3], pTemp[4]), _mm256_add_ps(pTemp[5], pTemp[6]))));
                            }
                            pDst = _mm256_min_ps(_mm256_max_ps(pDst, pMin), pMax);
                            rpp_sobel_store8(dstPtrTemp, &pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            dstPtrTemp += 8;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_unidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}
