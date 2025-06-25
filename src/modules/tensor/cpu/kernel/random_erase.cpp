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

template <typename T>
RppStatus random_erase_host_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptRoiLtrb *anchorBoxInfoTensor,
                                   Rpp32u *numBoxesTensor,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    // Random number generator setup (thread-safe)
    std::random_device rd;
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        std::mt19937 gen(rd() + batchCount * 4096 + omp_get_thread_num());
        std::uniform_int_distribution<int> dist_int(0, std::numeric_limits<T>::max());
        std::uniform_real_distribution<float> dist_float(0.0f, 1.0f);

        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u numBoxes = numBoxesTensor[batchCount];
        RpptRoiLtrb *anchorBoxInfo = anchorBoxInfoTensor + batchCount * numBoxes;

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        T userPixel3[3];
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier * sizeof(T);

        // Erase with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            T *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcPtrTemp = srcPtrRow;
                T *dstPtrTempR = dstPtrRowR;
                T *dstPtrTempG = dstPtrRowG;
                T *dstPtrTempB = dstPtrRowB;

                for (int j = 0; j < roi.xywhROI.roiWidth;)
                {
                    bool isErase = false;
                    Rpp32u bufferLength = 0;

                    for(int count = 0; count < numBoxes; count++)
                    {
                        Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
                        Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
                        Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth));
                        Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight));
                        if(i >= y1 && i <= y2 && j >= x1 && j <= x2)
                        {
                            isErase = true;
                            bufferLength = x2 - x1 + 1;
                            break;
                        }
                    }
                    if(isErase && bufferLength)
                    {
                        for (int k = 0; k < bufferLength; k++)
                        {
                            if constexpr (std::is_floating_point<T>::value) {
                                *dstPtrTempR++ = static_cast<T>(dist_float(gen));
                                *dstPtrTempG++ = static_cast<T>(dist_float(gen));
                                *dstPtrTempB++ = static_cast<T>(dist_float(gen));
                            } else {
                                *dstPtrTempR++ = static_cast<T>(dist_int(gen));
                                *dstPtrTempG++ = static_cast<T>(dist_int(gen));
                                *dstPtrTempB++ = static_cast<T>(dist_int(gen));
                            }
                            srcPtrTemp += 3;
                        }
                        j += bufferLength;
                    }
                    else
                    {
                        *dstPtrTempR++ = *srcPtrTemp++;
                        *dstPtrTempG++ = *srcPtrTemp++;
                        *dstPtrTempB++ = *srcPtrTemp++;
                        j++;
                    }
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Erase with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                for (int j = 0; j < roi.xywhROI.roiWidth;)
                {
                    bool isErase = false;
                    Rpp32u bufferLengthPerChannel = 0;
                    for(int count = 0; count < numBoxes; count++)
                    {
                        Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
                        Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
                        Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth));
                        Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight));

                        if(i >= y1 && i <= y2 && j >= x1 && j <= x2)
                        {
                            isErase = true;
                            bufferLengthPerChannel = x2 - x1 + 1;
                            break;
                        }
                    }
                    if(isErase && bufferLengthPerChannel)
                    {
                        for (int k = 0; k < bufferLengthPerChannel; k++)
                        {
                            if constexpr (std::is_floating_point<T>::value) {
                                *dstPtrTemp++ = static_cast<T>(dist_float(gen));  // R
                                *dstPtrTemp++ = static_cast<T>(dist_float(gen));  // G
                                *dstPtrTemp++ = static_cast<T>(dist_float(gen));  // B
                            } else {
                                *dstPtrTemp++ = static_cast<T>(dist_int(gen));
                                *dstPtrTemp++ = static_cast<T>(dist_int(gen));
                                *dstPtrTemp++ = static_cast<T>(dist_int(gen));
                            }
                            srcPtrTempR++;
                            srcPtrTempG++;
                            srcPtrTempB++;
                        }
                        j += bufferLengthPerChannel;
                    }
                    else
                    {
                        dstPtrTemp[0] = *srcPtrTempR++;
                        dstPtrTemp[1] = *srcPtrTempG++;
                        dstPtrTemp[2] = *srcPtrTempB++;
                        dstPtrTemp += 3;
                        j++;
                    }
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Erase without fused output-layout toggle 3 channel(NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            // To copy ROI region in Image
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                T *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrRow, srcPtrRow, bufferLength);
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }

            for(int count = 0; count < numBoxes; count++)
            {
                Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
                Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
                Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth));
                Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight));

                Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxWidth = x2 - x1 + 1;

                T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrImage + pixelLocation;
                dstPtrTempG = dstPtrTempR + dstDescPtr->strides.cStride;
                dstPtrTempB = dstPtrTempG + dstDescPtr->strides.cStride;
                for (int i = 0; i < boxHeight; i++)
                {
                    for (int j = 0; j < boxWidth; j++)
                    {
                        if constexpr (std::is_floating_point<T>::value) {
                            dstPtrTempR[j] = static_cast<T>(dist_float(gen));
                            dstPtrTempG[j] = static_cast<T>(dist_float(gen));
                            dstPtrTempB[j] = static_cast<T>(dist_float(gen));
                        } else {
                            dstPtrTempR[j] = static_cast<T>(dist_int(gen));
                            dstPtrTempG[j] = static_cast<T>(dist_int(gen));
                            dstPtrTempB[j] = static_cast<T>(dist_int(gen));
                        }
                    }
                    dstPtrTempR += dstDescPtr->strides.hStride;
                    dstPtrTempG += dstDescPtr->strides.hStride;
                    dstPtrTempB += dstDescPtr->strides.hStride;
                }
            }
        }
        // Erase without fused output-layout toggle 1 channel(NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            // To copy ROI region in Image
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                memcpy(dstPtrChannel, srcPtrChannel, bufferLength);
                srcPtrChannel += srcDescPtr->strides.hStride;
                dstPtrChannel += dstDescPtr->strides.hStride;
            }

            for (int count = 0; count < numBoxes; count++)
            {
                Rpp32u x1 = (Rpp32u)RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth);
                Rpp32u y1 = (Rpp32u)RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight);
                Rpp32u x2 = (Rpp32u)RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth);
                Rpp32u y2 = (Rpp32u)RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight);

                Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxWidth = x2 - x1 + 1;

                T *dstPtrTemp;
                dstPtrTemp = dstPtrImage + pixelLocation;

                for (int i = 0; i < boxHeight; i++)
                {
                    for (int j = 0; j < boxWidth; j++)
                    {
                        if constexpr (std::is_floating_point<T>::value)
                            dstPtrTemp[j] = static_cast<T>(dist_float(gen));
                        else
                            dstPtrTemp[j] = static_cast<T>(dist_int(gen));
                    }
                    dstPtrTemp += dstDescPtr->strides.hStride;
                }
            }
        }

        // Erase without fused output-layout toggle 3 channel(NHWC -> NHWC)
        else
        {
            // To copy ROI region in Image
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                memcpy(dstPtrChannel, srcPtrChannel, bufferLength);
                srcPtrChannel += srcDescPtr->strides.hStride;
                dstPtrChannel += dstDescPtr->strides.hStride;
            }

            for(int count = 0; count < numBoxes; count++)
            {
                Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
                Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
                Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth));
                Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight));
                Rpp32u countMul3 = count * 3;

                Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxWidth = x2 - x1 + 1;
                T *dstPtrTemp;
                dstPtrTemp = dstPtrImage + pixelLocation;

                for (int i = 0; i < boxHeight; i++)
                {
                    T *dstPtrRow = dstPtrTemp;
                    for (int j = 0; j < boxWidth; j++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            if constexpr (std::is_floating_point<T>::value)
                                dstPtrRow[c] = static_cast<T>(dist_float(gen));
                            else
                                dstPtrRow[c] = static_cast<T>(dist_int(gen));
                        }
                        dstPtrRow += srcDescPtr->c;
                    }
                    dstPtrTemp += dstDescPtr->strides.hStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus random_erase_host_tensor<Rpp8u>(Rpp8u*,
                                                   RpptDescPtr,
                                                   Rpp8u*,
                                                   RpptDescPtr,
                                                   RpptRoiLtrb*,
                                                   Rpp32u*,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   RppLayoutParams,
                                                   rpp::Handle&);

template RppStatus random_erase_host_tensor<Rpp16f>(Rpp16f*,
                                                    RpptDescPtr,
                                                    Rpp16f*,
                                                    RpptDescPtr,
                                                    RpptRoiLtrb*,
                                                    Rpp32u*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    RppLayoutParams,
                                                    rpp::Handle&);

template RppStatus random_erase_host_tensor<Rpp32f>(Rpp32f*,
                                                    RpptDescPtr,
                                                    Rpp32f*,
                                                    RpptDescPtr,
                                                    RpptRoiLtrb*,
                                                    Rpp32u*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    RppLayoutParams,
                                                    rpp::Handle&);
       
template RppStatus random_erase_host_tensor<Rpp8s>(Rpp8s*,
                                                   RpptDescPtr,
                                                   Rpp8s*,
                                                   RpptDescPtr,
                                                   RpptRoiLtrb*,
                                                   Rpp32u*,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   RppLayoutParams,
                                                   rpp::Handle&);
