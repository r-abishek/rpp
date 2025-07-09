/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "host_tensor_executors.hpp"
#include <random>

inline void generate_channel_masks(std::vector<std::vector<bool>> &channelMasks,
                                   Rpp32f *dropProb,
                                   Rpp32u batchSize,
                                   Rpp32u numChannels)
{
    std::mt19937 rng(std::random_device{}());

    for (Rpp32u batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        std::bernoulli_distribution keepDist(1.0f - dropProb[batchIdx]);
        bool anyKept = false;

        for (Rpp32u c = 0; c < numChannels; c++)
        {
            channelMasks[batchIdx][c] = keepDist(rng);
            anyKept |= channelMasks[batchIdx][c];
        }

        if (!anyKept)
            channelMasks[batchIdx][rng() % numChannels] = true;
    }
}

template<typename T>
RppStatus channel_dropout_host_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *dropProb,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    // Generate channel mask for this batch
    std::vector<std::vector<bool>> channelMasks(dstDescPtr->n, std::vector<bool>(srcDescPtr->c));
    generate_channel_masks(channelMasks, dropProb, dstDescPtr->n, srcDescPtr->c);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        const std::vector<bool> &channelMask = channelMasks[batchCount];
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        T *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        T *srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + 
                          (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        T *dstPtrChannel = dstPtrImage;
        // Channel dropout with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for (Rpp32s c = 0; c < srcDescPtr->c; c++)
            {
                T *dstPtrRow = dstPtrChannel;
                for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    T *srcPtrPixelRow = srcPtrChannel + i * srcDescPtr->strides.hStride;

                    for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                    {
                        Rpp32s pixelIdx = j * srcDescPtr->c + c;
                        if (channelMask[c])
                            *dstPtrTemp = srcPtrPixelRow[pixelIdx];
                        else
                            *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }

                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
        // Channel dropout with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *dstPtrTemp = dstPtrChannel + i * dstDescPtr->strides.hStride;

                for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    for (Rpp32s c = 0; c < srcDescPtr->c; c++)
                    {
                        Rpp32s srcIdx = c * srcDescPtr->strides.cStride +   // Channel offset
                                        i * srcDescPtr->strides.hStride +    // Row offset
                                        j;                                   // Column offset

                        if (channelMask[c])
                            *dstPtrTemp = srcPtrChannel[srcIdx];
                        else
                            *dstPtrTemp = 0;

                        dstPtrTemp++;  // Because NHWC layout
                    }
                }
            }
        }
        // Channel Dropout without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *dstPtrRow = dstPtrChannel;
            T *srcPtrRow = srcPtrChannel;

            for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *dstPtrTemp = dstPtrRow;
                T *srcPtrTemp = srcPtrRow;

                for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    for (Rpp32s c = 0; c < srcDescPtr->c; c++)
                    {
                        dstPtrTemp[c] = channelMask[c] ? srcPtrTemp[c] : 0;
                    }
                    dstPtrTemp += srcDescPtr->c;
                    srcPtrTemp += srcDescPtr->c;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Channel dropout without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for (Rpp32s c = 0; c < srcDescPtr->c; c++)
            {
                if (channelMask[c])
                {
                    // Copy entire channel if active
                    for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        memcpy(dstPtrChannel + (i * dstDescPtr->strides.hStride),
                            srcPtrChannel + (i * srcDescPtr->strides.hStride),
                            roi.xywhROI.roiWidth * sizeof(T));
                    }
                }
                else
                {
                    // Zero out channel if dropped
                    for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        memset(dstPtrChannel + (i * dstDescPtr->strides.hStride),
                            0,
                            roi.xywhROI.roiWidth * sizeof(T));
                    }
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
        // Channel dropout single channel without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            if (channelMask[0])
            {
                for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrChannel + (i * dstDescPtr->strides.hStride),
                        srcPtrChannel + (i * srcDescPtr->strides.hStride),
                        roi.xywhROI.roiWidth * sizeof(T));
                }
            }
            else
            {
                for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memset(dstPtrChannel + (i * dstDescPtr->strides.hStride),
                        0,
                        roi.xywhROI.roiWidth * sizeof(T));
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus channel_dropout_host_tensor<Rpp8u>(Rpp8u*,
                                                      RpptDescPtr,
                                                      Rpp8u*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);
template RppStatus channel_dropout_host_tensor<Rpp32f>(Rpp32f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       RppLayoutParams,
                                                       rpp::Handle&);
template RppStatus channel_dropout_host_tensor<Rpp16f>(Rpp16f*,
                                                       RpptDescPtr,
                                                       Rpp16f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       RppLayoutParams,
                                                       rpp::Handle&);
template RppStatus channel_dropout_host_tensor<Rpp8s>(Rpp8s*,
                                                      RpptDescPtr,
                                                      Rpp8s*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);