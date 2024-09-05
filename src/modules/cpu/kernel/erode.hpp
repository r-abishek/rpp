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
#include "rpp_cpu_common.hpp"
#include "rpp_cpu_filter.hpp"

// generic raw c code for box filter 
template<typename T>
inline void erode_generic_tensor(T **srcPtrTemp, T *dstPtrTemp, Rpp32s columnIndex,
                                 Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit,
                                 Rpp32f kernelSizeInverseSquare, Rpp32u channels = 1)
{
    T result = 255;
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);
    for (int i = 0; i < rowKernelLoopLimit; i++)
        for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
            result  = std::min<T>(result, srcPtrTemp[i][k]);
    *dstPtrTemp = result;
}

// process padLength number of columns in each row for PLN-PLN case
// left border pixels in image which does not have required pixels in 3x3/5x5/7x7/9x9 box, process them separately
template<typename T>
inline void process_left_border_columns_pln_pln(T **srcPtrTemp, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f kernelSizeInverseSquare)
{
    for (int k = 0; k < padLength; k++)
    {
        erode_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
        dstPtrTemp++;
    }
}

// process padLength * 3 number of columns in each row for PKD-PKD case
// left border pixels in image which does not have required pixels in 3x3/5x5/7x7/9x9 box, process them separately
template<typename T>
inline void process_left_border_columns_pkd_pkd(T **srcPtrTemp, T **srcPtrRow, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f kernelSizeInverseSquare)
{
    for (int c = 0; c < 3; c++)
    {
        T *dstPtrTempChannel = dstPtrTemp + c;
        for (int k = 0; k < padLength; k++)
        {
            erode_generic_tensor(srcPtrTemp, dstPtrTempChannel, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
            dstPtrTempChannel += 3;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }
    // reset source to initial position
    for (int k = 0; k < kernelSize; k++)
        srcPtrTemp[k] = srcPtrRow[k];
}

// process padLength * 3 number of columns in each row for PKD-PLN case
// left border pixels in image which does not have required pixels in 3x3/5x5/7x7/9x9 box, process them separately
template<typename T>
inline void process_left_border_columns_pkd_pln(T **srcPtrTemp, T **srcPtrRow, T **dstPtrTempChannels, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f kernelSizeInverseSquare)
{
    for (int c = 0; c < 3; c++)
    {
        for (int k = 0; k < padLength; k++)
        {
            erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[c], k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
            dstPtrTempChannels[c] += 1;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }

    // reset source to initial position
    for (int k = 0; k < kernelSize; k++)
        srcPtrTemp[k] = srcPtrRow[k];
}

// -------------------- Set 0 erode compute functions --------------------

// unpack lower half of 3 256 bit registers and add (used for 3x3 kernel size U8/I8 variants)
inline void unpacklo_and_min_3x3_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[2], avx_px0));
}

// unpack higher half of 3 256 bit registers and add (used for 3x3 kernel size U8/I8 variants)
inline void unpackhi_and_min_3x3_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[2], avx_px0));
}

// unpack lower half of 5 256 bit registers and add (used for 5x5 kernel size U8/I8 variants)
inline void unpacklo_and_min_5x5_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[2], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[3], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[4], avx_px0));
}

// unpack higher half of 5 256 bit registers and add (used for 5x5 kernel size U8/I8 variants)
inline void unpackhi_and_min_5x5_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[2], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[3], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[4], avx_px0));
}

// unpack lower half of 7 256 bit registers and add (used for 7x7 kernel size U8/I8 variants)
inline void unpacklo_and_min_7x7_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[2], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[3], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[4], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[5], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[6], avx_px0));
}

// unpack higher half of 7 256 bit registers and add (used for 7x7 kernel size U8/I8 variants)
inline void unpackhi_and_min_7x7_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[2], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[3], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[4], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[5], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[6], avx_px0));
}

// unpack lower half of 9 256 bit registers and add (used for 9x9 kernel size U8/I8 variants)
inline void unpacklo_and_min_9x9_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[2], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[3], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[4], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[5], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[6], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[7], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[8], avx_px0));
}

// unpack higher half of 9 256 bit registers and add (used for 9x9 kernel size U8/I8 variants)
inline void unpackhi_and_min_9x9_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[2], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[3], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[4], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[5], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[6], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[7], avx_px0));
    pxDst[0] = _mm256_min_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[8], avx_px0));
}

// #undef __AVX2__
// #define __AVX2__ 0

template<typename T>
RppStatus erode_char_host_tensor(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u kernelSize,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();
    static_assert((std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value), "T must be Rpp8u or Rpp8s");

    // if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
    //     return erode_generic_host_tensor(srcPtr, srcDescPtr, dstPtr, dstDescPtr, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);

    // set the required masks array needed for shuffle operations 
#if __AVX2__
    __m128i pxMaskPln[7] = {xmm_pxMaskRotate0To1, xmm_pxMaskRotate0To3, xmm_pxMaskRotate0To5, xmm_pxMaskRotate0To7, xmm_pxMaskRotate0To9, xmm_pxMaskRotate0To11, xmm_pxMaskRotate0To13};
    __m128i pxMaskPkd[7] = {xmm_pxMaskRotate0To5, xmm_pxMaskRotate0To11, xmm_pxMaskRotate0To1, xmm_pxMaskRotate0To7, xmm_pxMaskRotate0To13, xmm_pxMaskRotate0To3, xmm_pxMaskRotate0To9};
#endif

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
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
        Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

        Rpp32f kernelSizeInverseSquare = 1.0 / (kernelSize * kernelSize);
#if __AVX2__
        // set the register order needed for blend operations 
        Rpp32u blendRegisterOrder[7] = {0, 0, 1, 1, 1, 2, 2};
        if (srcDescPtr->layout == RpptLayout::NCHW)
            std::fill_n(blendRegisterOrder, 7, 0);
#endif
        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if (kernelSize == 3)
        {
            T *srcPtrRow[3], *dstPtrRow;
            for (int i = 0; i < 3; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude 2 * padLength number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                        {
                            __m256i pxRow[3], pxRowHalf[2], pxResult;
                            rpp_load_erode_char_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                            // unpack lower half and higher half of each of 3 loaded row values from 8 bit to 16 bit and add
                            unpacklo_and_min_3x3_host(pxRow, &pxRowHalf[0]);
                            unpackhi_and_min_3x3_host(pxRow, &pxRowHalf[1]);

                            // perform blend and shuffle operations to get required order and add them
                            __m128i pxTemp[4];
                            extract_4sse_registers(pxRowHalf, pxTemp);
                            blend_shuffle_min_3x3_host<1, 3>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_3x3_host<1, 3>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_3x3_host<1, 3>(&pxTemp[2], pxMaskPln, blendRegisterOrder);
                            
                            __m128i pxDst[2];
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                            
                            pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            if constexpr (std::is_same<T, Rpp8s>::value)
                                pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                            _mm256_storeu_si256((__m256i *)dstPtrTemp, pxResult);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                            dstPtrTemp += 24;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    dstPtrTemp += padLength * 3;
#if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxRow[3], pxRowHalf[2], pxResult;
                        rpp_load_erode_char_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        // unpack lower half and higher half of each of 3 loaded row values from 8 bit to 16 bit and add
                        unpacklo_and_min_3x3_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_3x3_host(pxRow, &pxRowHalf[1]);

                        // perform blend and shuffle operations for the first 8 output values to get required order and add them
                        __m128i pxTemp[4];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_3x3_host<7, 63>(&pxTemp[0], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_3x3_host<7, 63>(&pxTemp[1], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_3x3_host<7, 63>(&pxTemp[2], pxMaskPkd, blendRegisterOrder);

                        __m128i pxDst[2];
                        pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);

                        pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                        _mm256_storeu_si256((__m256i *)dstPtrTemp, pxResult);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        dstPtrTemp += 24;
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
                T *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
#if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxRow[3], pxRowHalf[2];
                        rpp_load_erode_char_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        // unpack lower half and higher half of each of 3 loaded row values from 8 bit to 16 bit and add
                        unpacklo_and_min_3x3_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_3x3_host(pxRow, &pxRowHalf[1]);

                        // perform blend and shuffle operations for the first 8 output values to get required order and add them
                        __m128i pxTemp[4];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_3x3_host<7, 63>(&pxTemp[0], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_3x3_host<7, 63>(&pxTemp[1], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_3x3_host<7, 63>(&pxTemp[2], pxMaskPkd, blendRegisterOrder);

                        __m128i pxDst[2];
                        pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        if constexpr (std::is_same<T, Rpp8s>::value)
                        {
                            pxDst[0] = _mm_sub_epi8(pxDst[0], xmm_pxConvertI8);
                            pxDst[1] = _mm_sub_epi8(pxDst[1], xmm_pxConvertI8);
                        }

                        // convert from PKD3 to PLN3 and store channelwise
                        __m128i pxDstChn[3];
                        rpp_convert24_pkd3_to_pln3(pxDst[0], pxDst[1], pxDstChn);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[0]), pxDstChn[0]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[1]), pxDstChn[1]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[2]), pxDstChn[2]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        increment_row_ptrs(dstPtrTempChannels, kernelSize, 8);
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                    {
                        int channel = c % 3;
                        erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3][3] = {
                                            {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]},
                                            {srcPtrRow[0] + srcDescPtr->strides.cStride, srcPtrRow[1] + srcDescPtr->strides.cStride, srcPtrRow[2] + srcDescPtr->strides.cStride},
                                            {srcPtrRow[0] + 2 * srcDescPtr->strides.cStride, srcPtrRow[1] + 2 * srcDescPtr->strides.cStride, srcPtrRow[2] + 2 * srcDescPtr->strides.cStride}
                                          };

                    T *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[3], pxRowHalf[2];
                            rpp_load_erode_char_3x3_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);

                            // unpack lower half and higher half of each of 3 loaded row values from 8 bit to 16 bit and add
                            unpacklo_and_min_3x3_host(pxRow, &pxRowHalf[0]);
                            unpackhi_and_min_3x3_host(pxRow, &pxRowHalf[1]);

                            // perform blend and shuffle operations for the first 8 output values to get required order and add them
                            __m128i pxTemp[4];
                            extract_4sse_registers(pxRowHalf, pxTemp);
                            blend_shuffle_min_3x3_host<1, 3>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_3x3_host<1, 3>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_3x3_host<1, 3>(&pxTemp[2], pxMaskPln, blendRegisterOrder);

                            __m128i pxDst[2];
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);

                            pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                        }
                        if constexpr (std::is_same<T, Rpp8s>::value)
                        {
                            pxResultPln[0] = _mm256_sub_epi8(pxResultPln[0], avx_pxConvertI8);
                            pxResultPln[1] = _mm256_sub_epi8(pxResultPln[1], avx_pxConvertI8);
                            pxResultPln[2] = _mm256_sub_epi8(pxResultPln[2], avx_pxConvertI8);
                        }

                        __m128i pxResultPkd[6];
                        // convert result from pln to pkd format and store in output buffer
                        rpp_convert72_pln3_to_pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 48), pxResultPkd[4]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 60), pxResultPkd[5]);
                        dstPtrTemp += 72;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
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

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    srcPtrRow[3] = srcPtrRow[2] + srcDescPtr->strides.hStride;
                    srcPtrRow[4] = srcPtrRow[3] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for (int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1 : 0;
                        T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                        {
                            __m256i pxRow[5], pxRowHalf[2], pxResult;
                            rpp_load_erode_char_5x5_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                            // pack lower and higher half of each of 5 loaded row values from 8 bit to 16 bit and add
                            unpacklo_and_min_5x5_host(pxRow, &pxRowHalf[0]);
                            unpackhi_and_min_5x5_host(pxRow, &pxRowHalf[1]);

                            __m128i pxTemp[4], pxDst[2];
                            extract_4sse_registers(pxRowHalf, pxTemp);
                            blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[2], pxMaskPln, blendRegisterOrder);
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                            pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            if constexpr (std::is_same<T, Rpp8s>::value)
                                pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                            _mm256_storeu_si256((__m256i *)dstPtrTemp, pxResult);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                            dstPtrTemp += 24;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 18) * 18;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    dstPtrTemp += padLength * 3;
#if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 18)
                    {
                        __m256i pxRow[5], pxRowHalf[2], pxResult;
                        rpp_load_erode_char_5x5_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        // pack lower and higher half of each of 5 loaded row values from 8 bit to 16 bit and add
                        unpacklo_and_min_5x5_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_5x5_host(pxRow, &pxRowHalf[1]);

                        __m128i pxTemp[5], pxDst[2];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        pxTemp[4] = xmm_px0;
                        blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[0], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[1], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[2], pxMaskPkd, blendRegisterOrder);
                        pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                        _mm256_storeu_si256((__m256i *)dstPtrTemp, pxResult);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 18);
                        dstPtrTemp += 18;
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 18) * 18;
                T *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                    T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
#if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 18)
                    {
                        __m256i pxRow[5], pxRowHalf[2];
                        rpp_load_erode_char_5x5_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        // pack lower and higher half of each of 5 loaded row values from 8 bit to 16 bit and add
                        unpacklo_and_min_5x5_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_5x5_host(pxRow, &pxRowHalf[1]);

                        __m128i pxTemp[5], pxDst[2];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        pxTemp[4] = xmm_px0;
                        blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[0], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[1], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_5x5_host<7, 63, 1, 15>(&pxTemp[2], pxMaskPkd, blendRegisterOrder);
                        pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                        if constexpr (std::is_same<T, Rpp8s>::value)
                        {
                            pxDst[0] = _mm_sub_epi8(pxDst[0], xmm_pxConvertI8);
                            pxDst[1] = _mm_sub_epi8(pxDst[1], xmm_pxConvertI8);
                        }

                        // convert from PKD3 to PLN3 and store channelwise
                        __m128i pxDstChn[3];
                        rpp_convert24_pkd3_to_pln3(pxDst[0], pxDst[1], pxDstChn);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[0]), pxDstChn[0]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[1]), pxDstChn[1]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[2]), pxDstChn[2]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 18);
                        increment_row_ptrs(dstPtrTempChannels, kernelSize, 6);
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                    {
                        int channel = c % 3;
                        erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength * 3 number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3][5];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 5; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    T *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 5x5 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[5], pxRowHalf[2], pxResult;
                            rpp_load_erode_char_5x5_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);

                            // pack lower and higher half of each of 5 loaded row values from 8 bit to 16 bit and add
                            unpacklo_and_min_5x5_host(pxRow, &pxRowHalf[0]);
                            unpackhi_and_min_5x5_host(pxRow, &pxRowHalf[1]);

                            __m128i pxTemp[4], pxDst[2];
                            extract_4sse_registers(pxRowHalf, pxTemp);
                            blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_5x5_host<1, 3, 7, 15>(&pxTemp[2], pxMaskPln, blendRegisterOrder);
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                            pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                        }
                        if constexpr (std::is_same<T, Rpp8s>::value)
                        {
                            pxResultPln[0] = _mm256_sub_epi8(pxResultPln[0], avx_pxConvertI8);
                            pxResultPln[1] = _mm256_sub_epi8(pxResultPln[1], avx_pxConvertI8);
                            pxResultPln[2] = _mm256_sub_epi8(pxResultPln[2], avx_pxConvertI8);
                        }

                        __m128i pxResultPkd[6];
                        // convert result from pln to pkd format and store in output buffer
                        rpp_convert72_pln3_to_pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 48), pxResultPkd[4]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 60), pxResultPkd[5]);
                        dstPtrTemp += 72;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
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

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 7; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for (int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1 : 0;
                        T *srcPtrTemp[7];
                        for (int k = 0; k < 7; k++)
                            srcPtrTemp[k] = srcPtrRow[k];
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                        {
                            __m256i pxRow[7], pxRowHalf[2], pxResult;
                            rpp_load_erode_char_7x7_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                            // unpack lower and higher half of each of 7 loaded row values from 8 bit to 16 bit and add
                            unpacklo_and_min_7x7_host(pxRow, &pxRowHalf[0]);
                            unpackhi_and_min_7x7_host(pxRow, &pxRowHalf[1]);

                            __m128i pxTemp[4], pxDst[2];
                            extract_4sse_registers(pxRowHalf, pxTemp);
                            blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[2], pxMaskPln, blendRegisterOrder);
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                            pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            if constexpr (std::is_same<T, Rpp8s>::value)
                                pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                            _mm256_storeu_si256((__m256i *)dstPtrTemp, pxResult);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                            dstPtrTemp += 24;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3][7];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 7; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    T *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 7x7 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[7], pxRowHalf[2], pxResult;
                            rpp_load_erode_char_7x7_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);

                            // unpack lower and higher half of each of 7 loaded row values from 8 bit to 16 bit and add
                            unpacklo_and_min_7x7_host(pxRow, &pxRowHalf[0]);
                            unpackhi_and_min_7x7_host(pxRow, &pxRowHalf[1]);

                            __m128i pxTemp[4], pxDst[2];
                            extract_4sse_registers(pxRowHalf, pxTemp);
                            blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_7x7_host<1, 3, 7, 15, 31, 63>(&pxTemp[2], pxMaskPln, blendRegisterOrder);
                            pxDst[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            pxDst[1] = _mm_packus_epi16(pxTemp[2], xmm_px0);
                            pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                        }
                        if constexpr (std::is_same<T, Rpp8s>::value)
                        {
                            pxResultPln[0] = _mm256_sub_epi8(pxResultPln[0], avx_pxConvertI8);
                            pxResultPln[1] = _mm256_sub_epi8(pxResultPln[1], avx_pxConvertI8);
                            pxResultPln[2] = _mm256_sub_epi8(pxResultPln[2], avx_pxConvertI8);
                        }

                        __m128i pxResultPkd[6];
                        // convert result from pln to pkd format and store in output buffer
                        rpp_convert72_pln3_to_pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 48), pxResultPkd[4]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 60), pxResultPkd[5]);
                        dstPtrTemp += 72;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - 2 * padLength * 3) / 12) * 12;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[7];
                    for (int k = 0; k < 7; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    dstPtrTemp += padLength * 3;
#if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256i pxRow[7], pxRowHalf[2];
                        rpp_load_erode_char_7x7_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        // unpack lower and higher half of each of 7 loaded row values from 8 bit to 16 bit and add
                        unpacklo_and_min_7x7_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_7x7_host(pxRow, &pxRowHalf[1]);

                        __m128i pxTemp[4], pxResult;
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_7x7_host<7, 63, 1, 15, 127, 3>(&pxTemp[0], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_7x7_host<7, 63, 1, 15, 127, 3>(&pxTemp[1], pxMaskPkd, blendRegisterOrder);
                        pxResult = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResult = _mm_sub_epi8(pxResult, xmm_pxConvertI8);

                        _mm_storeu_si128((__m128i*)dstPtrTemp, pxResult);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                        dstPtrTemp += 12;
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - 2 * padLength * 3) / 12) * 12;
                T *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[7] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4], srcPtrRow[5], srcPtrRow[6]};
                    T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
#if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256i pxRow[7], pxRowHalf[2];
                        rpp_load_erode_char_7x7_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        // unpack lower and higher half of each of 7 loaded row values from 8 bit to 16 bit and add
                        unpacklo_and_min_7x7_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_7x7_host(pxRow, &pxRowHalf[1]);

                        __m128i pxTemp[4], pxResult[2];
                        extract_4sse_registers(pxRowHalf, pxTemp);
                        blend_shuffle_min_7x7_host<7, 63, 1, 15, 127, 3>(&pxTemp[0], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_7x7_host<7, 63, 1, 15, 127, 3>(&pxTemp[1], pxMaskPkd, blendRegisterOrder);
                        pxResult[0] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                        pxResult[1] = xmm_px0;
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResult[0] = _mm_sub_epi8(pxResult[0], xmm_pxConvertI8);

                        // convert from PKD3 to PLN3 and store channelwise
                        __m128i pxDstChn[3];
                        rpp_convert24_pkd3_to_pln3(pxResult[0], pxResult[1], pxDstChn);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[0]), pxDstChn[0]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[1]), pxDstChn[1]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[2]), pxDstChn[2]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                        increment_row_ptrs(dstPtrTempChannels, kernelSize, 4);
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                    {
                        int channel = c % 3;
                        erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
                }
            }
        }
        else if (kernelSize == 9)
        {
            T *srcPtrRow[9], *dstPtrRow;
            for (int i = 0; i < 9; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 9; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[9];
                        for (int k = 0; k < 9; k++)
                            srcPtrTemp[k] = srcPtrRow[k];
                        T *dstPtrTemp = dstPtrRow;

                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                        {
                            __m256i pxRow[9], pxRowHalf[2];
                            rpp_load_erode_char_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                            // unpack lower half and higher half of each of 9 loaded row values from 8 bit to 16 bit and add
                            unpacklo_and_min_9x9_host(pxRow, &pxRowHalf[0]);
                            unpackhi_and_min_9x9_host(pxRow, &pxRowHalf[1]);

                            __m128i pxTemp[3], pxDst;
                            extract_3sse_registers(pxRowHalf, pxTemp);
                            blend_shuffle_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                            pxDst = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            if constexpr (std::is_same<T, Rpp8s>::value)
                                pxDst = _mm_sub_epi8(pxDst, xmm_pxConvertI8);

                            _mm_storeu_si128((__m128i *)dstPtrTemp, pxDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 16);
                            dstPtrTemp += 16;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 64) * 64;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    dstPtrTemp += padLength * 3;
#if __AVX2__
                    // load first 32 elements elements
                    __m256i pxRow[9];
                    if (alignedLength)
                        rpp_load_erode_char_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                    {
                        __m256i pxRowHalf[2], pxResult;
                        unpacklo_and_min_9x9_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_9x9_host(pxRow, &pxRowHalf[1]);

                        // get the accumalated result for first 8 elements
                        __m128i px128[8], pxTemp[7], pxDst[2];
                        extract_4sse_registers(pxRowHalf, &px128[0]);
                        blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[0], pxMaskPkd, blendRegisterOrder);

                        // compute for next 8 elements
                        increment_row_ptrs(srcPtrTemp, kernelSize, 32);
                        rpp_load_erode_char_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);
                        unpacklo_and_min_9x9_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_9x9_host(pxRow, &pxRowHalf[1]);

                        // get the accumalated result for next 24 elements
                        extract_4sse_registers(pxRowHalf, &px128[4]);
                        blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[1], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[2], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[3], pxMaskPkd, blendRegisterOrder);

                        // compute final result
                        pxDst[0] = _mm_packus_epi16(px128[0], px128[1]);
                        pxDst[1] = _mm_packus_epi16(px128[2], px128[3]);
                        pxResult = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            pxResult = _mm256_sub_epi8(pxResult, avx_pxConvertI8);

                        _mm256_storeu_si256((__m256i *)dstPtrTemp, pxResult);
                        dstPtrTemp += 32;
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        erode_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3][9];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 9; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m128i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[9], pxRowHalf[2];
                            rpp_load_erode_char_9x9_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);

                            // unpack lower half and higher half of each of 9 loaded row values from 8 bit to 16 bit and add
                            unpacklo_and_min_9x9_host(pxRow, &pxRowHalf[0]);
                            unpackhi_and_min_9x9_host(pxRow, &pxRowHalf[1]);

                            __m128i pxTemp[3], pxDst;
                            extract_3sse_registers(pxRowHalf, pxTemp);
                            blend_shuffle_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(&pxTemp[0], pxMaskPln, blendRegisterOrder);
                            blend_shuffle_min_9x9_host<1, 3, 7, 15, 31, 63, 127>(&pxTemp[1], pxMaskPln, blendRegisterOrder);
                            pxResultPln[c] = _mm_packus_epi16(pxTemp[0], pxTemp[1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 16);
                        }
                        if constexpr (std::is_same<T, Rpp8s>::value)
                        {
                            pxResultPln[0] = _mm_sub_epi8(pxResultPln[0], xmm_pxConvertI8);
                            pxResultPln[1] = _mm_sub_epi8(pxResultPln[1], xmm_pxConvertI8);
                            pxResultPln[2] = _mm_sub_epi8(pxResultPln[2], xmm_pxConvertI8);
                        }

                        __m128i pxResultPkd[4];
                        rpp_convert48_pln3_to_pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp), pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        dstPtrTemp += 48;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            erode_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 64) * 64;
                T *dstPtrChannels[3];
                for (int c = 0; c < 3; c++)
                    dstPtrChannels[c] = dstPtrChannel + c * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        // load first 32 elements elements
                        __m256i pxRow[9], pxRowHalf[2];
                        rpp_load_erode_char_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        // get the accumalated result for first 8 elements
                        unpacklo_and_min_9x9_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_9x9_host(pxRow, &pxRowHalf[1]);

                        // get the accumalated result for first 8 elements
                        __m128i px128[8], pxTemp[7], pxDst[2];
                        extract_4sse_registers(pxRowHalf, &px128[0]);
                        blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[0], pxMaskPkd, blendRegisterOrder);

                        // compute for next 8 elements
                        increment_row_ptrs(srcPtrTemp, kernelSize, 32);
                        rpp_load_erode_char_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);
                        unpacklo_and_min_9x9_host(pxRow, &pxRowHalf[0]);
                        unpackhi_and_min_9x9_host(pxRow, &pxRowHalf[1]);

                        // get the accumalated result for next 24 elements
                        extract_4sse_registers(pxRowHalf, &px128[4]);
                        blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[1], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[2], pxMaskPkd, blendRegisterOrder);
                        blend_shuffle_min_9x9_host<7, 63, 1, 15, 127, 3, 31>(&px128[3], pxMaskPkd, blendRegisterOrder);
                        pxDst[0] = _mm_packus_epi16(px128[0], px128[1]);
                        pxDst[1] = _mm_packus_epi16(px128[2], px128[3]);
                        if constexpr (std::is_same<T, Rpp8s>::value)
                        {
                            pxDst[0] = _mm_sub_epi8(pxDst[0], xmm_pxConvertI8);
                            pxDst[1] = _mm_sub_epi8(pxDst[1], xmm_pxConvertI8);
                        }

                        // convert from PKD3 to PLN3 and store
                        __m128i pxDstChn[3];
                        rpp_convert24_pkd3_to_pln3(pxDst[0], pxDst[1], pxDstChn);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[0]), pxDstChn[0]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[1]), pxDstChn[1]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[2]), pxDstChn[2]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, -8);
                        increment_row_ptrs(dstPtrTempChannels, 3, 8);
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        int channel = vectorLoopCount % 3;
                        erode_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
                }
            }
        }
    }

    return RPP_SUCCESS;
}