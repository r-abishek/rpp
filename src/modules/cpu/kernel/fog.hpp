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
                                Rpp32f *alphaTensor,
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
        
        Rpp32u fog_width = 0;
        Rpp32u fog_height = 0;
        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        if(roi.xywhROI.roiWidth<=224 && roi.xywhROI.roiHeight<=224)
        {
            fog_width = 224;
            fog_height =224;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_224_224[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_224_224[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else if(roi.xywhROI.roiWidth <= 640 && roi.xywhROI.roiHeight <= 480)
        {
            fog_width = 640;
            fog_height = 480;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_640_480[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_640_480[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else if(roi.xywhROI.roiWidth <= 1280 && roi.xywhROI.roiHeight <= 720)
        {
            fog_width = 1280;
            fog_height = 720;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_1280_720[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_1280_720[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else
        {
            fog_width = 1920;
            fog_height = 1080;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_1920_1080[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_1920_1080[(fog_width * maskLoc.y) + maskLoc.x];

        }

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {

                    __m256 pfogAlphaMask[2], pfogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pfogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pfogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_fog_48_host(p, pfogAlphaMask, pfogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {

                    
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) srcPtrTemp[0]) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) srcPtrTemp[1]) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) srcPtrTemp[2]) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))));

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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {

                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrTempR) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrTempG) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrTempB) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))));
                    
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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {

                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrTemp) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))));
                        
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                        compute_fog_16_host(p, pFogAlphaMask, pFogIntensityMask);    // fog adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrChannel, p);    // simd stores

                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = (Rpp8u) RPPPIXELCHECK(std::nearbyintf(((Rpp32f) *srcPtrChannel) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))));
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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fog_f16_f16_host_tensor(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *alphaTensor,
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


        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u fog_width = 0;
        Rpp32u fog_height = 0;
        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        if(roi.xywhROI.roiWidth<=224 && roi.xywhROI.roiHeight<=224)
        {
            fog_width = 224;
            fog_height =224;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_224_224[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_224_224[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else if(roi.xywhROI.roiWidth <= 640 && roi.xywhROI.roiHeight <= 480)
        {
            fog_width = 640;
            fog_height = 480;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_640_480[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_640_480[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else if(roi.xywhROI.roiWidth <= 1280 && roi.xywhROI.roiHeight <= 720)
        {
            fog_width = 1280;
            fog_height = 720;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_1280_720[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_1280_720[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else
        {
            fog_width = 1920;
            fog_height = 1080;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_1920_1080[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_1920_1080[(fog_width * maskLoc.y) + maskLoc.x];

        }


        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) srcPtrTemp[0]) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) srcPtrTemp[1]) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) srcPtrTemp[2]) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));

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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }

                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrTempR) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrTempG) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrTempB) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));

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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrTemp) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {

                    __m256 pFogAlphaMask, pFogIntensityMask;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads

                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        Rpp32f srcPtrChannel_ps[8], dstPtrChannel_ps[8];
                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                            srcPtrChannel_ps[cnt] = (Rpp32f) srcPtrChannel[cnt];

                        __m256 p;
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrChannel_ps, &p);    // simd loads
                        rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                        compute_fog_8_host(&p, &pFogAlphaMask, &pFogIntensityMask);    // fog adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrChannel_ps, &p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                            dstPtrChannel[cnt] = (Rpp16f) dstPtrChannel_ps[cnt];
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f) *srcPtrChannel) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fog_f32_f32_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *alphaTensor,
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

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u fog_width = 0;
        Rpp32u fog_height = 0;
        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        if(roi.xywhROI.roiWidth<=224 && roi.xywhROI.roiHeight<=224)
        {
            fog_width = 224;
            fog_height =224;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_224_224[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_224_224[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else if(roi.xywhROI.roiWidth <= 640 && roi.xywhROI.roiHeight <= 480)
        {
            fog_width = 640;
            fog_height = 480;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_640_480[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_640_480[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else if(roi.xywhROI.roiWidth <= 1280 && roi.xywhROI.roiHeight <= 720)
        {
            fog_width = 1280;
            fog_height = 720;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_1280_720[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_1280_720[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else
        {
            fog_width = 1920;
            fog_height = 1080;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_1920_1080[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_1920_1080[(fog_width * maskLoc.y) + maskLoc.x];

        }


        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;


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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {                    
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32((srcPtrTemp[0]) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                    *dstPtrTempG = RPPPIXELCHECKF32((srcPtrTemp[1]) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                    *dstPtrTempB = RPPPIXELCHECKF32((srcPtrTemp[2]) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));

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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtrTempR) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                    dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtrTempG) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                    dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtrTempB) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));

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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {

                    __m256 pFogAlphaMask, pFogIntensityMask, p[3];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    rpp_multiply24_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                    compute_fog_24_host(p, &pFogAlphaMask, &pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32((*srcPtrTemp) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pFogAlphaMask, pFogIntensityMask;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogAlphaMaskPtrTemp, &pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, fogIntensityMaskPtrTemp, &pFogIntensityMask);    // simd loads

                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {

                        __m256 p;
                        
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrChannel, &p);    // simd loads
                        rpp_multiply8_constant(&pFogIntensityMask, avx_p1op255);    // u8 normalization to range[0,1]
                        compute_fog_8_host(&p, &pFogAlphaMask, &pFogIntensityMask );    // fog adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrChannel, &p);    // simd stores
                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = RPPPIXELCHECKF32((*srcPtrChannel) * ( 1 - *fogAlphaMaskPtrTemp) + ((*fogIntensityMaskPtrTemp * ONE_OVER_255) * (*fogAlphaMaskPtrTemp)));
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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
            }
        }
    }

    return RPP_SUCCESS;
}


RppStatus fog_i8_i8_host_tensor(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *alphaTensor,
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

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u fog_width = 0;
        Rpp32u fog_height = 0;
        Rpp32f *fogAlphaMaskPtr, *fogIntensityMaskPtr;
        if(roi.xywhROI.roiWidth<=224 && roi.xywhROI.roiHeight<=224)
        {
            fog_width = 224;
            fog_height =224;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_224_224[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_224_224[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else if(roi.xywhROI.roiWidth <= 640 && roi.xywhROI.roiHeight <= 480)
        {
            fog_width = 640;
            fog_height = 480;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_640_480[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_640_480[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else if(roi.xywhROI.roiWidth <= 1280 && roi.xywhROI.roiHeight <= 720)
        {
            fog_width = 1280;
            fog_height = 720;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_1280_720[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_1280_720[(fog_width * maskLoc.y) + maskLoc.x];

        }
        else
        {
            fog_width = 1920;
            fog_height = 1080;
            std::random_device rd;  // Random number engine seed
            std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
            std::uniform_int_distribution<> distribX(0, fog_width - roi.xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(0, fog_height - roi.xywhROI.roiHeight);

            RppiPoint maskLoc;
            maskLoc.x = distribX(gen);
            maskLoc.y = distribY(gen);

            fogAlphaMaskPtr = &fogAlphaMask_1920_1080[(fog_width * maskLoc.y) + maskLoc.x];
            fogIntensityMaskPtr = &fogIntensityMask_1920_1080[(fog_width * maskLoc.y) + maskLoc.x];

        }

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;


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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {

                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) srcPtrTemp[0] + 128.0f) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))) - 128.0f);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) srcPtrTemp[1] + 128.0f) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))) - 128.0f);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) srcPtrTemp[2] + 128.0f) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))) - 128.0f);

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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {

                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrTempR + 128.0f) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))) - 128.0f);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrTempG + 128.0f) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))) - 128.0f);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrTempB + 128.0f) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))) - 128.0f);

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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {

                    __m256 pFogAlphaMask[2], pFogIntensityMask[2], p[6];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogAlphaMaskPtrTemp, pFogAlphaMask);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, fogIntensityMaskPtrTemp, pFogIntensityMask);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_fog_48_host(p, pFogAlphaMask, pFogIntensityMask);    // fog adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrTemp + 128.0f) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))) - 128.0f);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    fogAlphaMaskPtrTemp++;
                    fogIntensityMaskPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
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
                        compute_fog_16_host(p, pFogAlphaMask, pFogIntensityMask);    // fog adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrChannel, p);    // simd stores

                        srcPtrChannel += srcDescPtr->strides.cStride;
                        dstPtrChannel += dstDescPtr->strides.cStride;
                    }
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                    fogAlphaMaskPtrTemp += vectorIncrementPerChannel;
                    fogIntensityMaskPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    srcPtrChannel = srcPtrTemp;
                    dstPtrChannel = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrChannel = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) *srcPtrChannel + 128.0f) * ( 1 - *fogAlphaMaskPtrTemp) + (*fogIntensityMaskPtrTemp * (*fogAlphaMaskPtrTemp))) - 128.0f);
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
                fogAlphaMaskPtrRow += fog_width;
                fogIntensityMaskPtrRow += fog_width;
            }
        }
    }

    return RPP_SUCCESS;
}