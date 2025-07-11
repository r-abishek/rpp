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

inline uint generate_seed(uint x, uint y, uint z)
{
    return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
}

inline float generate_random_float(uint seed)
{
    seed = (1103515245 * seed + 12345);
    return ((seed / 65536) % 32768) / 32768.0f;
}

inline uint generate_random_int(uint seed)
{
    seed = (1103515245 * seed + 12345);
    return (seed >> 16) & 0xFF;
}

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

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        uint seed = 0;
        Rpp32u numBoxes = numBoxesTensor[batchCount];
        RpptRoiLtrb *anchorBoxInfo = anchorBoxInfoTensor + batchCount * numBoxes;

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier * sizeof(T);

        // Erase with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            T *srcPtrRow = srcPtrChannel;
            T *dstPtrRowR = dstPtrChannel;
            T *dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            T *dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
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
                            uint seed = generate_seed(j + k, i, batchCount);
                            if constexpr (std::is_floating_point<T>::value || std::is_same<T, Rpp16f>::value) {
                                *dstPtrTempR++ = static_cast<T>(generate_random_float(seed + 0));
                                *dstPtrTempG++ = static_cast<T>(generate_random_float(seed + 1));
                                *dstPtrTempB++ = static_cast<T>(generate_random_float(seed + 2));
                            } else {
                                *dstPtrTempR++ = static_cast<T>(generate_random_int(seed + 0));
                                *dstPtrTempG++ = static_cast<T>(generate_random_int(seed + 1));
                                *dstPtrTempB++ = static_cast<T>(generate_random_int(seed + 2));
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
                            seed = generate_seed(j + k, i, batchCount);
                            if constexpr (std::is_floating_point<T>::value || std::is_same<T, Rpp16f>::value) {
                                *dstPtrTemp++ = static_cast<T>(generate_random_float(seed + 0));  // R
                                *dstPtrTemp++ = static_cast<T>(generate_random_float(seed + 1));  // G
                                *dstPtrTemp++ = static_cast<T>(generate_random_float(seed + 2));  // B
                            } else {
                                *dstPtrTemp++ = static_cast<T>(generate_random_int(seed + 0));
                                *dstPtrTemp++ = static_cast<T>(generate_random_int(seed + 1));
                                *dstPtrTemp++ = static_cast<T>(generate_random_int(seed + 2));
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
                        uint seed = generate_seed(x1 + j, y1 + i, batchCount);
                        if constexpr (std::is_floating_point<T>::value || std::is_same<T, Rpp16f>::value) {
                            dstPtrTempR[j] = static_cast<T>(generate_random_float(seed + 0));
                            dstPtrTempG[j] = static_cast<T>(generate_random_float(seed + 1));
                            dstPtrTempB[j] = static_cast<T>(generate_random_float(seed + 2));
                        } else {
                            dstPtrTempR[j] = static_cast<T>(generate_random_int(seed + 0));
                            dstPtrTempG[j] = static_cast<T>(generate_random_int(seed + 1));
                            dstPtrTempB[j] = static_cast<T>(generate_random_int(seed + 2));
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
                        uint seed = generate_seed(x1 + j, y1 + i, batchCount);
                        if constexpr (std::is_floating_point<T>::value || std::is_same<T, Rpp16f>::value)
                            dstPtrTemp[j] = static_cast<T>(generate_random_float(seed));
                        else
                            dstPtrTemp[j] = static_cast<T>(generate_random_int(seed));
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
                        uint seed = generate_seed(x1 + j, y1 + i, batchCount);
                        if constexpr (std::is_floating_point<T>::value || std::is_same<T, Rpp16f>::value)
                        {
                            dstPtrRow[0] = static_cast<T>(generate_random_float(seed + 0));
                            dstPtrRow[1] = static_cast<T>(generate_random_float(seed + 1));
                            dstPtrRow[2] = static_cast<T>(generate_random_float(seed + 2));
                        }
                        else
                        {
                            dstPtrRow[0] = static_cast<T>(generate_random_int(seed + 0));
                            dstPtrRow[1] = static_cast<T>(generate_random_int(seed + 1));
                            dstPtrRow[2] = static_cast<T>(generate_random_int(seed + 2));
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
