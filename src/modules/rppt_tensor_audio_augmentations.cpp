/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_audio_augmentations.h"
#include "cpu/host_tensor_audio_augmentations.hpp"

/******************** non_silent_region_detection ********************/

RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp32s *srcSize,
                                                Rpp32f *detectedIndexTensor,
                                                Rpp32f *detectionLengthTensor,
                                                Rpp32f cutOffDB,
                                                Rpp32s windowLength,
                                                Rpp32f referencePower,
                                                Rpp32s resetInterval)
{
    non_silent_region_detection_host_tensor((Rpp32f*)srcPtr,
                                            srcDescPtr,
                                            srcSize,
                                            detectedIndexTensor,
                                            detectionLengthTensor,
                                            cutOffDB,
                                            windowLength,
                                            referencePower,
                                            resetInterval);

    return RPP_SUCCESS;
}

/******************** to_decibels ********************/

RppStatus rppt_to_decibels_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptImagePatchPtr srcDims,
                                Rpp32f cutOffDB,
                                Rpp32f multiplier,
                                Rpp32f referenceMagnitude)
{
    to_decibels_host_tensor((Rpp32f*)srcPtr,
                            srcDescPtr,
                            (Rpp32f*)dstPtr,
                            dstDescPtr,
                            srcDims,
                            cutOffDB,
                            multiplier,
                            referenceMagnitude);

    return RPP_SUCCESS;
}

/******************** pre_emphasis_filter ********************/

RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        RppPtr_t dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32s *srcLengthTensor,
                                        Rpp32f *coeffTensor,
                                        RpptAudioBorderType borderType)
{
    pre_emphasis_filter_host_tensor((Rpp32f*)srcPtr,
                                    srcDescPtr,
                                    (Rpp32f*)dstPtr,
                                    dstDescPtr,
                                    srcLengthTensor,
                                    coeffTensor,
                                    borderType);

    return RPP_SUCCESS;
}

/******************** down_mixing ********************/

RppStatus rppt_down_mixing_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32s *srcLengthTensor,
                                Rpp32s *channelsTensor,
                                bool  normalizeWeights)
{
    down_mixing_host_tensor((Rpp32f*)srcPtr,
                            srcDescPtr,
                            (Rpp32f*)dstPtr,
                            dstDescPtr,
                            srcLengthTensor,
                            channelsTensor,
                            normalizeWeights);

    return RPP_SUCCESS;
}

/******************** slice ********************/

RppStatus rppt_slice_host(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32s *srcLengthTensor,
                          Rpp32f *anchorTensor,
                          Rpp32f *shapeTensor,
                          Rpp32f *fillValues)
{
    slice_host_tensor((Rpp32f*)srcPtr,
                      srcDescPtr,
                      (Rpp32f*)dstPtr,
                      dstDescPtr,
                      srcLengthTensor,
                      anchorTensor,
                      shapeTensor,
                      fillValues);

    return RPP_SUCCESS;
}

/******************** mel_filter_bank ********************/

RppStatus rppt_mel_filter_bank_host(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptImagePatchPtr srcDims,
                                    Rpp32f maxFreq,
                                    Rpp32f minFreq,
                                    RpptMelScaleFormula melFormula,
                                    Rpp32s numFilter,
                                    Rpp32f sampleRate,
                                    bool normalize)
{
    mel_filter_bank_host_tensor((Rpp32f*)srcPtr,
                                srcDescPtr,
                                (Rpp32f*)dstPtr,
                                dstDescPtr,
                                srcDims,
                                maxFreq,
                                minFreq,
                                melFormula,
                                numFilter,
                                sampleRate,
                                normalize);

    return RPP_SUCCESS;
}

/******************** spectrogram ********************/

RppStatus rppt_spectrogram_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32s *srcLengthTensor,
                                bool centerWindows,
                                bool reflectPadding,
                                Rpp32f *windowFunction,
                                Rpp32s nfft,
                                Rpp32s power,
                                Rpp32s windowLength,
                                Rpp32s windowStep,
                                RpptSpectrogramLayout layout)
{
    spectrogram_host_tensor((Rpp32f*)srcPtr,
                            srcDescPtr,
                            (Rpp32f*)dstPtr,
                            dstDescPtr,
                            srcLengthTensor,
                            centerWindows,
                            reflectPadding,
                            windowFunction,
                            nfft,
                            power,
                            windowLength,
                            windowStep,
                            layout);

    return RPP_SUCCESS;
}

/******************** resample ********************/

RppStatus rppt_resample_host(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32f *inRateTensor,
                             Rpp32f *outRateTensor,
                             Rpp32s *srcLengthTensor,
                             Rpp32s *channelsTensor,
                             Rpp32f quality)
{
    resample_host_tensor((Rpp32f*)srcPtr,
                         srcDescPtr,
                         (Rpp32f*)dstPtr,
                         dstDescPtr,
                         inRateTensor,
                         outRateTensor,
                         srcLengthTensor,
                         channelsTensor,
                         quality);

    return RPP_SUCCESS;
}

/******************** normalize_audio ********************/

RppStatus rppt_normalize_audio_host(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32s *srcLengthTensor,
                                    Rpp32s *channelsTensor,
                                    Rpp32s axisMask,
                                    Rpp32f mean,
                                    Rpp32f stdDev,
                                    Rpp32f scale,
                                    Rpp32f shift,
                                    Rpp32f epsilon,
                                    Rpp32s ddof,
                                    Rpp32s numOfDims)
{
    normalize_audio_host_tensor((Rpp32f*)srcPtr,
                                srcDescPtr,
                                (Rpp32f*)dstPtr,
                                dstDescPtr,
                                srcLengthTensor,
                                channelsTensor,
                                axisMask,
                                mean,
                                stdDev,
                                scale,
                                shift,
                                epsilon,
                                ddof,
                                numOfDims);
    return RPP_SUCCESS;
}
