/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software OR associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, OR/or sell
copies of the Software, OR to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice OR this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "host_tensor_executors.hpp"


inline void compute_posterize_32_host(__m256i& p, __m256i& pPosterizeMask)
{
    p = _mm256_and_si256(p, pPosterizeMask);    // brightness adjustment
}

inline void compute_posterize_96_host(__m256i *p, __m256i& pPosterizeMask)
{
    p[0] = _mm256_and_si256(p[0], pPosterizeMask);    // brightness adjustment
    p[1] = _mm256_and_si256(p[1], pPosterizeMask);    // brightness adjustment
    p[2] = _mm256_and_si256(p[2], pPosterizeMask);    // brightness adjustment
}

// Both u8 and i8 use the same function for execution.
// Bitwise operation in the context of posterize gives the same output irrespective of sign
// So additional operations for conversion b/w u8 and i8 is avoided
// Inputs although in int8 is interpreted in calculation hence in uint8 as its the underlying bit representation that matters
RppStatus posterize_char_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u *posterizeLevelBits,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
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

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;

        Rpp8u posterizeBitsMask = ((1 << posterizeLevelBits[batchCount]) - 1) << (8 - posterizeLevelBits[batchCount]);
        __m256i pPosterizeBitsMask = _mm256_set1_epi8(posterizeBitsMask);

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p[3];
                    rpp_simd_load(rpp_load96_u8pkd3_to_u8pln3, srcPtrTemp, p);
                    compute_posterize_96_host(p, pPosterizeBitsMask);  // brightness adjustment
                    rpp_simd_store(rpp_store96_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = srcPtrTemp[0] & posterizeBitsMask;
                    *dstPtrTempG = srcPtrTemp[1] & posterizeBitsMask;
                    *dstPtrTempB = srcPtrTemp[2] & posterizeBitsMask;

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i p[3];
                    rpp_simd_load(rpp_load96_u8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_posterize_96_host(p, pPosterizeBitsMask);  // brightness adjustment
                    rpp_simd_store(rpp_store96_u8pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (*srcPtrTempR) & posterizeBitsMask;
                    dstPtrTemp[1] = (*srcPtrTempG) & posterizeBitsMask;
                    dstPtrTemp[2] = (*srcPtrTempB) & posterizeBitsMask;

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~31;
#endif

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i p[3];

                    rpp_simd_load(rpp_load96_u8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_posterize_96_host(p, pPosterizeBitsMask);
                    rpp_simd_store(rpp_store96_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR = *srcPtrTempR & posterizeBitsMask;
                    *dstPtrTempG = *srcPtrTempG & posterizeBitsMask;
                    *dstPtrTempB = *srcPtrTempB & posterizeBitsMask;

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~31;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                    {
                        __m256i p;

                        p = _mm256_loadu_si256((const __m256i *)srcPtrTemp);    // simd loads
                        compute_posterize_32_host(p, pPosterizeBitsMask);  // brightness adjustment
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, p);    // simd stores

                        srcPtrTemp +=32;
                        dstPtrTemp +=32;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (*srcPtrTemp) & posterizeBitsMask;

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus posterize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32u *posterizeLevelBits,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle) {
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        Rpp32f posterizeBitsFactor = 255.0/(1 << (8 - posterizeLevelBits[batchCount]));
        __m256 pPosterizeBitsFactor = _mm256_set1_ps(posterizeBitsFactor);

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    p[0] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[0], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    p[1] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[1], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    p[2] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[2], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = std::floor(srcPtrTemp[0] * posterizeBitsFactor)/posterizeBitsFactor;
                    *dstPtrTempG = std::floor(srcPtrTemp[1] * posterizeBitsFactor)/posterizeBitsFactor;
                    *dstPtrTempB = std::floor(srcPtrTemp[2] * posterizeBitsFactor)/posterizeBitsFactor;

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    p[0] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[0], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    p[1] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[1], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    p[2] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[2], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = std::floor(*srcPtrTempR * posterizeBitsFactor)/posterizeBitsFactor;
                    dstPtrTemp[1] = std::floor(*srcPtrTempG * posterizeBitsFactor)/posterizeBitsFactor;
                    dstPtrTemp[2] = std::floor(*srcPtrTempB * posterizeBitsFactor)/posterizeBitsFactor;

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                        p[0] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[0], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = std::floor(*srcPtrTemp * posterizeBitsFactor)/posterizeBitsFactor;

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus posterize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32u *posterizeLevelBits,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle) {
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr  batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr  batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage  (roi.xywhROI.xy.y * srcDescPtr->strides.hStride)  (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        Rpp32f posterizeBitsFactor = 255.0/(1 << (8 - posterizeLevelBits[batchCount]));
        __m256 pPosterizeBitsFactor = _mm256_set1_ps(posterizeBitsFactor);

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR  dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG  dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount = vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    p[0] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[0], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    p[1] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[1], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    p[2] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[2], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp = vectorIncrement;
                    dstPtrTempR = vectorIncrementPerChannel;
                    dstPtrTempG = vectorIncrementPerChannel;
                    dstPtrTempB = vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount = 3)
                {
                    *dstPtrTempR = (Rpp16f)(std::floor((Rpp32f)srcPtrTemp[0] * posterizeBitsFactor)/posterizeBitsFactor);
                    *dstPtrTempG = (Rpp16f)(std::floor((Rpp32f)srcPtrTemp[1] * posterizeBitsFactor)/posterizeBitsFactor);
                    *dstPtrTempB = (Rpp16f)(std::floor((Rpp32f)srcPtrTemp[2] * posterizeBitsFactor)/posterizeBitsFactor);

                    srcPtrTemp = 3;
                    dstPtrTempR;
                    dstPtrTempG;
                    dstPtrTempB;
                }

                srcPtrRow = srcDescPtr->strides.hStride;
                dstPtrRowR = dstDescPtr->strides.hStride;
                dstPtrRowG = dstDescPtr->strides.hStride;
                dstPtrRowB = dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR  srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG  srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount = vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    p[0] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[0], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    p[1] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[1], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    p[2] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[2], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR = vectorIncrementPerChannel;
                    srcPtrTempG = vectorIncrementPerChannel;
                    srcPtrTempB = vectorIncrementPerChannel;
                    dstPtrTemp = vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount)
                {
                    dstPtrTemp[0] = (Rpp16f)(std::floor((Rpp32f)*srcPtrTempR * posterizeBitsFactor)/posterizeBitsFactor);
                    dstPtrTemp[1] = (Rpp16f)(std::floor((Rpp32f)*srcPtrTempG * posterizeBitsFactor)/posterizeBitsFactor);
                    dstPtrTemp[2] = (Rpp16f)(std::floor((Rpp32f)*srcPtrTempB * posterizeBitsFactor)/posterizeBitsFactor);

                    srcPtrTempR;
                    srcPtrTempG;
                    srcPtrTempB;
                    dstPtrTemp = 3;
                }

                srcPtrRowR = srcDescPtr->strides.hStride;
                srcPtrRowG = srcDescPtr->strides.hStride;
                srcPtrRowB = srcDescPtr->strides.hStride;
                dstPtrRow = dstDescPtr->strides.hStride;
            }
        }
        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            for(int c = 0; c < layoutParams.channelParam; c)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount = vectorIncrementPerChannel)
                    {
                        __m256 p[1];

                        rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, p);    // simd loads
                        p[0] = _mm256_div_ps(_mm256_floor_ps(_mm256_mul_ps(p[0], pPosterizeBitsFactor)), pPosterizeBitsFactor);
                        rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp = vectorIncrementPerChannel;
                        dstPtrTemp = vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount)
                    {
                        *dstPtrTemp = (Rpp16f)(std::floor((Rpp32f)*srcPtrTemp * posterizeBitsFactor)/posterizeBitsFactor);

                        srcPtrTemp;
                        dstPtrTemp;
                    }

                    srcPtrRow = srcDescPtr->strides.hStride;
                    dstPtrRow = dstDescPtr->strides.hStride;
                }

                srcPtrChannel = srcDescPtr->strides.cStride;
                dstPtrChannel = dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}