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

#ifndef RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
#define RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** non_silent_region_detection ********************/

// Non Silent Region Detection augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[in] srcLengthTensor source audio buffer length
// *param[out] detectedIndex beginning index of non silent region (1D tensor of size batchSize)
// *param[out] detectionLength length of non silent region (1D tensor of size batchSize)
// *param[in] cutOffDB threshold(dB) below which the signal is considered silent
// *param[in] windowLength size of the sliding window used to calculate of the short-term power of the signal
// *param[in] referencePower reference power that is used to convert the signal to dB.
// *param[in] resetInterval number of samples after which the moving mean average is recalculated to avoid loss of precision
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, Rpp32s *srcLengthTensor, Rpp32f *detectedIndexTensor, Rpp32f *detectionLengthTensor, Rpp32f cutOffDB, Rpp32s windowLength, Rpp32f referencePower, Rpp32s resetInterval);

/******************** to_decibels ********************/

// To Decibels augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcDims source dimensions
// *param[in] cutOffDB  minimum or cut-off ratio in dB
// *param[in] multiplier factor by which the logarithm is multiplied
// *param[in] referenceMagnitude Reference magnitude if not provided maximum value of input used as reference
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_to_decibels_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f cutOffDB, Rpp32f multiplier, Rpp32f referenceMagnitude);

/******************** pre_emphasis_filter ********************/

// Pre Emphasis Filter augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor source audio buffer length (1D tensor of size batchSize)
// *param[in] coeffTensor preemphasis coefficient (1D tensor of size batchSize)
// *param[in] borderType border value policy
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32f *coeffTensor, RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP);

/******************** down_mixing ********************/

// Downmixing augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor source audio buffer length (1D tensor of size batchSize)
// *param[in] channelsTensor number of channels in audio buffer (1D tensor of size batchSize)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_down_mixing_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor, bool normalizeWeights = false);

/******************** slice ********************/

// Slice augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcDimsTensor (1D tensor of size batchSize * 2)
// *param[in] anchorTensor starting index of the slice (1D tensor of size batchSize * 2)
// *param[in] shapeTensor length of the slice (1D tensor of size batchSize * 2)
// *param[in] fillValues fill values based on out of Bound policy
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_slice_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcDimsTensor, Rpp32f *anchorTensor, Rpp32f *shapeTensor, Rpp32f *fillValues);

/******************** mel_filter_bank ********************/

// Mel Filter Bank augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcDims source dimensions
// *param[in] maxFreq maximum frequency if not provided maxFreq = sampleRate / 2
// *param[in] minFreq minimum frequency
// *param[in] melFormula Formula that will be used to convert frequencies from hertz to mel and from mel to hertz
// *param[in] numFilter Number of mel filters
// *param[in] sampleRate Sampling rate of the audio signal
// *param[in] normalize Boolean value that determines whether to normalize the triangular filter weights by the width of their frequency bands
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_mel_filter_bank_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f minFreq, Rpp32f maxFreq, RpptMelScaleFormula melFormula, Rpp32s numFilter, Rpp32f sampleRate, bool normalize);

/******************** spectrogram ********************/

// Spectrogram augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor source audio buffer length (1D tensor of size batchSize)
// *param[in] centerWindows Indicates whether extracted windows should be padded so that the window function is centered at multiples of window_step
// *param[in] reflectPadding Indicates the padding policy when sampling outside the bounds of the signal
// *param[in] windowFunction Samples of the window function that will be multiplied to each extracted window when calculating the STFT
// *param[in] nfft Size of the FFT
// *param[in] power Exponent of the magnitude of the spectrum
// *param[in] windowLength Window size in number of samples
// *param[in] windowStep Step betweeen the STFT windows in number of samples
// *param[in] layout output layout of spectrogram
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_spectrogram_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, bool centerWindows, bool reflectPadding, Rpp32f *windowFunction, Rpp32s nfft, Rpp32s power, Rpp32s windowLength, Rpp32s windowStep, RpptSpectrogramLayout layout);

/******************** resample ********************/

// Resample augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] inRate Input sampling rate (1D tensor of size batchSize)
// *param[in] outRate Output sampling rate (1D tensor of size batchSize)
// *param[in] srcLengthTensor source audio buffer length (1D tensor of size batchSize)
// *param[in] channelsTensor number of channels in audio buffer (1D tensor of size batchSize)
// *param[in] quality Resampling quality, where 0 is the lowest, and 100 is the highest
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_resample_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *inRateTensor, Rpp32f *outRateTensor, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor, Rpp32f quality);

/******************** normalize_audio ********************/

// Normalize augmentation for a NHW layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor source audio buffer length (1D tensor of size batchSize)
// *param[in] channelsTensor number of channels in audio buffer (1D tensor of size batchSize)
// *param[in] axisMask axis along which normalization needs to be done
// *param[in] mean mean value to be subtracted from input
// *param[in] stdDev standard deviation value to scale the input
// *param[in] scale scaling factor applied to output
// *param[in] shift value to which the mean will map in the output
// *param[in] epsilon  value that is added to the variance to avoid division by small number
// *param[in] ddof Delta Degrees of Freedom for Bessel’s correction
// *param[in] numOfDims number of dimensions of input
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_normalize_audio_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor, Rpp32s axisMask,
                                    Rpp32f mean, Rpp32f stdDev, Rpp32f scale, Rpp32f shift, Rpp32f epsilon, Rpp32s ddof, Rpp32s numOfDims);
#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_AUDIO_AUGMENTATIONS_H