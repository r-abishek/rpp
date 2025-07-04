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

template <typename T>
RppStatus coarse_dropout_host_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     T *colorsTensor,
                                     Rpp32u *numBoxesTensor,
                                     int maxBoxesPerImage, 
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

        Rpp32u numBoxes = numBoxesTensor[batchCount];
        RpptRoiLtrb *anchorBoxInfo = anchorBoxInfoTensor + batchCount * maxBoxesPerImage;
        T *colors = colorsTensor + batchCount * maxBoxesPerImage * srcDescPtr->c;

        T *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        T *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        T *srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        T *dstPtrChannel = dstPtrImage;
        T userPixel3[3];
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier * sizeof(T);

        // NHWC -> NCHW
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
                    bool erased = false;
                    for(int count = 0; count < numBoxes; count++)
                    {
                        Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
                        Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
                        Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth));
                        Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight));
                        Rpp32u countMul3 = count * 3;
                        T userPixelR = colors[countMul3];
                        T userPixelG = colors[countMul3 + 1];
                        T userPixelB = colors[countMul3 + 2];
                        if(i >= y1 && i <= y2 && j == x1)
                        {
                            Rpp32u bufferLength = x2 - x1 + 1;
                            std::fill_n(dstPtrTempR, bufferLength, userPixelR);
                            std::fill_n(dstPtrTempG, bufferLength, userPixelG);
                            std::fill_n(dstPtrTempB, bufferLength, userPixelB);
                            srcPtrTemp += 3 * bufferLength;
                            dstPtrTempR += bufferLength;
                            dstPtrTempG += bufferLength;
                            dstPtrTempB += bufferLength;
                            j += bufferLength;
                            erased = true;
                            break;
                        }
                    }
                    if (!erased)
                    {
                        *dstPtrTempR++ = srcPtrTemp[0];
                        *dstPtrTempG++ = srcPtrTemp[1];
                        *dstPtrTempB++ = srcPtrTemp[2];
                        srcPtrTemp += 3;
                        j++;
                    }
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // NCHW -> NHWC
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *srcPtrRowR = srcPtrChannel;
            T *srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            T *srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            T *dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcPtrTempR = srcPtrRowR;
                T *srcPtrTempG = srcPtrRowG;
                T *srcPtrTempB = srcPtrRowB;
                T *dstPtrTemp = dstPtrRow;

                for (int j = 0; j < roi.xywhROI.roiWidth;)
                {
                    bool erased = false;
                    for(int count = 0; count < numBoxes; count++)
                    {
                        Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
                        Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
                        Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth));
                        Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight));
                        Rpp32u countMul3 = count * 3;
                        T userPixel3[3] = {colors[countMul3], colors[countMul3 + 1], colors[countMul3 + 2]};
                        if(i >= y1 && i <= y2 && j == x1)
                        {
                            Rpp32u bufferLength = x2 - x1 + 1;
                            for (int k = 0; k < bufferLength; k++)
                            {
                                memcpy(dstPtrTemp, userPixel3, sizeof(T) * 3);
                                dstPtrTemp += 3;
                            }
                            srcPtrTempR += bufferLength;
                            srcPtrTempG += bufferLength;
                            srcPtrTempB += bufferLength;
                            j += bufferLength;
                            erased = true;
                            break;
                        }
                    }
                    if (!erased)
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
        // NCHW -> NCHW (3 channel)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                T *srcPtrRow = srcPtrChannel;
                T *dstPtrRow = dstPtrChannel;
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
                T *dstPtrTempR = dstPtrImage + pixelLocation;
                T *dstPtrTempG = dstPtrTempR + dstDescPtr->strides.cStride;
                T *dstPtrTempB = dstPtrTempG + dstDescPtr->strides.cStride;
                Rpp32u countMul3 = count * 3;
                T userPixelR = colors[countMul3];
                T userPixelG = colors[countMul3 + 1];
                T userPixelB = colors[countMul3 + 2];
                for (int i = 0; i < boxHeight; i++)
                {
                    std::fill_n(dstPtrTempR, boxWidth, userPixelR);
                    std::fill_n(dstPtrTempG, boxWidth, userPixelG);
                    std::fill_n(dstPtrTempB, boxWidth, userPixelB);
                    dstPtrTempR += dstDescPtr->strides.hStride;
                    dstPtrTempG += dstDescPtr->strides.hStride;
                    dstPtrTempB += dstDescPtr->strides.hStride;
                }
            }
        }
        // NCHW -> NCHW (1 channel)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
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
                T *dstPtrTemp = dstPtrImage + pixelLocation;
                T userPixel = colors[count];
                for (int i = 0; i < boxHeight; i++)
                {
                    std::fill_n(dstPtrTemp, boxWidth, userPixel);
                    dstPtrTemp += dstDescPtr->strides.hStride;
                }
            }
        }
        // NHWC -> NHWC (default)
        else
        {
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
                T userPixel3[3] = {colors[countMul3], colors[countMul3 + 1], colors[countMul3 + 2]};
                Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxWidth = x2 - x1 + 1;
                T *dstPtrTemp = dstPtrImage + pixelLocation;
                for (int i = 0; i < boxHeight; i++)
                {
                    T *dstPtrRow = dstPtrTemp;
                    for (int j = 0; j < boxWidth; j++)
                    {
                        memcpy(dstPtrRow, userPixel3, sizeof(T) * 3);
                        dstPtrRow += 3;
                    }
                    dstPtrTemp += dstDescPtr->strides.hStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

template RppStatus coarse_dropout_host_tensor<Rpp8u>(Rpp8u*,
                                                     RpptDescPtr,
                                                     Rpp8u*,
                                                     RpptDescPtr,
                                                     RpptRoiLtrb*,
                                                     Rpp8u*,
                                                     Rpp32u*,
                                                     int,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     RppLayoutParams,
                                                     rpp::Handle&);

template RppStatus coarse_dropout_host_tensor<Rpp16f>(Rpp16f*,
                                                      RpptDescPtr,
                                                      Rpp16f*,
                                                      RpptDescPtr,
                                                      RpptRoiLtrb*,
                                                      Rpp16f*,
                                                      Rpp32u*,
                                                      int, 
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);

template RppStatus coarse_dropout_host_tensor<Rpp32f>(Rpp32f*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptDescPtr,
                                                      RpptRoiLtrb*,
                                                      Rpp32f*,
                                                      Rpp32u*,
                                                      int, 
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);

template RppStatus coarse_dropout_host_tensor<Rpp8s>(Rpp8s*,
                                                     RpptDescPtr,
                                                     Rpp8s*,
                                                     RpptDescPtr,
                                                     RpptRoiLtrb*,
                                                     Rpp8s*,
                                                     Rpp32u*,
                                                     int, 
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     RppLayoutParams,
                                                     rpp::Handle&);