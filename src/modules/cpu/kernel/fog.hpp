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
#include "fog_mask.hpp"

RppStatus fog_u8_u8_host_tensor(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *fogAlphaMask,
                                Rpp32f *fogIntensityMask,
                                Rpp32f *gammaTensor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle)
{   
    Rpp32u numThreads = handle.GetNumThreads();
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u fogWidth = srcDescPtr->w;
    Rpp32u fogHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f gamma = gammaTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        fogAlphaMaskPtr = fogAlphaMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        fogIntensityMaskPtr = fogIntensityMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        __m256 pGamma = _mm256_set1_ps(gamma);

        // Fog without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {   
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp + gamma));
                    *dstPtrTempR = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(srcPtrTemp[0]) * alphaFactor + intensityFactor)));
                    *dstPtrTempG = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(srcPtrTemp[1]) * alphaFactor + intensityFactor)));
                    *dstPtrTempB = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(srcPtrTemp[2]) * alphaFactor + intensityFactor)));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp + gamma));
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtrTempR) * alphaFactor + intensityFactor)));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtrTempG) * alphaFactor + intensityFactor)));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtrTempB) * alphaFactor + intensityFactor)));
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp + gamma));
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtrTemp) * alphaFactor + intensityFactor)));                                                
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength & ~15) - 16;

            Rpp8u *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads

                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrChannel, p);    // simd loads
                        compute_fog_16_host(p, pFogAlphaMask, pFogIntensityMask, pGamma);    // fog adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrChannel, p);    // simd stores

                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp + gamma));
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtrChannel) * alphaFactor + intensityFactor)));
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp++;
                    dstPtrTemp++;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fog_f16_f16_host_tensor(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f* fogAlphaMask,
                                  Rpp32f* fogIntensityMask,
                                  Rpp32f *gammaTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u fogWidth = srcDescPtr->w;
    Rpp32u fogHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f gamma = gammaTensor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        fogAlphaMaskPtr = fogAlphaMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        fogIntensityMaskPtr = fogIntensityMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pGamma = _mm256_set1_ps(gamma);

        // Fog without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f16pkd3_to_f16pln3_avx, srcPtrTemp, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma));
                    *dstPtrTempR = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(srcPtrTemp[0]) * alphaFactor + intensityFactor));
                    *dstPtrTempG = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(srcPtrTemp[1]) * alphaFactor + intensityFactor));
                    *dstPtrTempB = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(srcPtrTemp[2]) * alphaFactor + intensityFactor));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f16pln3_to_f16pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma));
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(*srcPtrTempR) * alphaFactor + intensityFactor));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(*srcPtrTempG) * alphaFactor + intensityFactor));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(*srcPtrTempB) * alphaFactor + intensityFactor));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f16pkd3_to_f16pln3_avx, srcPtrTemp, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma));
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(*srcPtrTemp) * alphaFactor + intensityFactor));
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#else
            alignedLength = bufferLength & ~3;
#endif

            Rpp16f *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]

                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        Rpp32f srcPtrChannel_ps[8], dstPtrChannel_ps[8];

                        __m256 p;
                        rpp_simd_load(rpp_load8_f16_to_f16_avx, srcPtrChannel, &p);    // simd loads
                        compute_fog_8_host(&p, &pFogAlphaMask, &pFogIntensityMask, pGamma);    // fog adjustment
                        rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrChannel, &p);    // simd stores

                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma));
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>(*srcPtrChannel) * alphaFactor + intensityFactor));
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp++;
                    dstPtrTemp++;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fog_f32_f32_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f* fogAlphaMask,
                                  Rpp32f* fogIntensityMask,
                                  Rpp32f *gammaTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u fogWidth = srcDescPtr->w;
    Rpp32u fogHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f gamma = gammaTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        fogAlphaMaskPtr = fogAlphaMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        fogIntensityMaskPtr = fogIntensityMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pGamma = _mm256_set1_ps(gamma);

        // Fog without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {                    
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma));
                    *dstPtrTempR = RPPPIXELCHECKF32((srcPtrTemp[0]) * alphaFactor + intensityFactor);
                    *dstPtrTempG = RPPPIXELCHECKF32((srcPtrTemp[1]) * alphaFactor + intensityFactor);
                    *dstPtrTempB = RPPPIXELCHECKF32((srcPtrTemp[2]) * alphaFactor + intensityFactor);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma));
                    dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtrTempR) * alphaFactor + intensityFactor);
                    dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtrTempG) * alphaFactor + intensityFactor);
                    dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtrTempB) * alphaFactor + intensityFactor);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    rpp_multiply24_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma));
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32((*srcPtrTemp) * alphaFactor + intensityFactor);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {

            alignedLength = bufferLength & ~7;

            Rpp32f *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]

                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256 p;   
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrChannel, &p);    // simd loads
                        compute_fog_8_host(&p, &pFogAlphaMask, &pFogIntensityMask, pGamma);    // fog adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrChannel, &p);    // simd stores
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma));
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = RPPPIXELCHECKF32((*srcPtrChannel) * alphaFactor + intensityFactor);
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp++;
                    dstPtrTemp++;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }
    }

    return RPP_SUCCESS;
}


RppStatus fog_i8_i8_host_tensor(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f* fogAlphaMask,
                                Rpp32f* fogIntensityMask,
                                Rpp32f *gammaTensor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u fogWidth = srcDescPtr->w;
    Rpp32u fogHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f gamma = gammaTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        std::random_device rd;  // Random number engine seed
        std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roi.xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roi.xywhROI.roiHeight);

        RppiPoint maskLoc;
        maskLoc.x = distribX(gen);
        maskLoc.y = distribY(gen);

        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        fogAlphaMaskPtr = fogAlphaMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        fogIntensityMaskPtr = fogIntensityMask + ((srcDescPtr->w * maskLoc.y) + maskLoc.x);
        
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        __m256 pGamma = _mm256_set1_ps(gamma);

        // Fog without fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma)) - 128.0f;
                    *dstPtrTempR = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(srcPtrTemp[0]) + 128.0f) * alphaFactor + intensityFactor));
                    *dstPtrTempG = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(srcPtrTemp[1]) + 128.0f) * alphaFactor + intensityFactor));
                    *dstPtrTempB = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(srcPtrTemp[2]) + 128.0f) * alphaFactor + intensityFactor));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma)) - 128.0f;
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(*srcPtrTempR) + 128.0f) * alphaFactor + intensityFactor));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(*srcPtrTempR) + 128.0f) * alphaFactor + intensityFactor));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(*srcPtrTempR) + 128.0f) * alphaFactor + intensityFactor));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask, pGamma);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma)) - 128.0f;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(*srcPtrTemp) + 128.0f) * alphaFactor + intensityFactor));
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }

        // Fog without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength & ~15) - 16;

            Rpp8s *srcPtrRow, *dstPtrRow;
            Rpp32f *fogAlphaMaskPtrRow, *fogIntensityMaskPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            fogAlphaMaskPtrRow = fogAlphaMaskPtr;
            fogIntensityMaskPtrRow = fogIntensityMaskPtr;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                Rpp32f *fogAlphaMaskPtrTemp, *fogIntensityMaskPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                fogAlphaMaskPtrTemp = fogAlphaMaskPtrRow;
                fogIntensityMaskPtrTemp = fogIntensityMaskPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask[2], pFogIntensityMask[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads

                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrChannel, p);    // simd loads
                        compute_fog_16_host(p, pFogAlphaMask, pFogIntensityMask, pGamma);    // fog adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrChannel, p);    // simd stores

                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    Rpp32f alphaFactor = ( 1 - (*fogAlphaMaskPtrTemp + gamma));
                    Rpp32f intensityFactor = ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp + gamma)) - 128.0f;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = static_cast<Rpp8s>(RPPPIXELCHECKI8((static_cast<Rpp32f>(*srcPtrChannel) + 128.0f) * alphaFactor + intensityFactor));
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp++;
                    dstPtrTemp++;
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fogWidth;
                fogIntensityMaskPtrRow += fogWidth;
            }
        }
    }

    return RPP_SUCCESS;
}