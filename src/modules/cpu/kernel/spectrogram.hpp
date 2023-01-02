#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include <chrono>
#include <complex>

inline Rpp32f reduce_add_ps1(__m256 src)
{
    __m256 srcAdd = _mm256_add_ps(src, _mm256_permute2f128_ps(src, src, 1));
    srcAdd = _mm256_add_ps(srcAdd, _mm256_shuffle_ps(srcAdd, srcAdd, _MM_SHUFFLE(1, 0, 3, 2)));
    srcAdd = _mm256_add_ps(srcAdd, _mm256_shuffle_ps(srcAdd, srcAdd, _MM_SHUFFLE(2, 3, 0, 1)));
    Rpp32f *addResult = (Rpp32f *)&srcAdd;
    return addResult[0];
}

inline void hann_window(Rpp32f *output, Rpp32s windowSize)
{
    Rpp32f a = (2 * PI / windowSize);
    for (Rpp32s t = 0; t < windowSize; t++)
    {
        Rpp32f phase = a * (t + 0.5);
        output[t] = (0.5 * (1.0 - std::cos(phase)));
    }
}

inline Rpp32s get_num_windows(Rpp32s length, Rpp32s windowLength, Rpp32s windowStep, bool centerWindows)
{
    if (!centerWindows)
        length -= windowLength;
    return ((length / windowStep) + 1);
}

inline Rpp32s get_idx_reflect(Rpp32s idx, Rpp32s lo, Rpp32s hi)
{
    if (hi - lo < 2)
        return hi - 1;
    for (;;)
    {
        if (idx < lo)
            idx = 2 * lo - idx;
        else if (idx >= hi)
            idx = 2 * hi - 2 - idx;
        else
            break;
    }
    return idx;
}

RppStatus spectrogram_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
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
    Rpp32s windowCenterOffset = 0;
    bool vertical = (layout == RpptSpectrogramLayout::FT);
    if (centerWindows)
        windowCenterOffset = windowLength / 2;
    if (nfft == 0.0f)
        nfft = windowLength;
    Rpp32s numBins = nfft / 2 + 1;
    const Rpp32f mulFactor = (2.0f * PI) / nfft;
    Rpp32f cosf[numBins * nfft];
    Rpp32f sinf[numBins * nfft];
    for (Rpp32s k = 0; k < numBins; k++)
    {
        for (Rpp32s i = 0; i < nfft; i++)
        {
            cosf[k * nfft + i] = std::cos(k * i * mulFactor);
            sinf[k * nfft + i] = -std::sin(k * i * mulFactor);
        }
    }
    std::vector<Rpp32f> windowFn;
    windowFn.resize(windowLength);
    // Generate hanning window
    if (windowFunction == NULL)
        hann_window(windowFn.data(), windowLength);
    else
        memcpy(windowFn.data(), windowFunction, windowLength*sizeof(Rpp32f));
    Rpp32u hStride = dstDescPtr->strides.hStride;
    Rpp32s alignedNfftLength = nfft & ~7;
    Rpp32s alignedNbinsLength = numBins & ~7;
    Rpp32s alignedWindowLength = windowLength & ~7;
    // Get windows output
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(8)
    for (Rpp32s batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        Rpp32s numWindows = get_num_windows(bufferLength, windowLength, windowStep, centerWindows);
        Rpp32f windowOutput[numWindows * nfft];
        std::fill_n (windowOutput, numWindows * nfft, 0);
        for (int64_t w = 0; w < numWindows; w++)
        {
            int64_t windowStart = w * windowStep - windowCenterOffset;
            Rpp32f *windowOutputTemp = windowOutput + (w * nfft);
            if (windowStart < 0 || (windowStart + windowLength) > bufferLength)
            {
                for (Rpp32s t = 0; t < windowLength; t++)
                {
                    int64_t inIdx = windowStart + t;
                    if (reflectPadding)
                    {
                        inIdx = get_idx_reflect(inIdx, 0, bufferLength);
                        *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                    }
                    else
                    {
                        if (inIdx >= 0 && inIdx < bufferLength)
                            *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                        else
                            *windowOutputTemp++ = 0;
                    }
                }
            }
            else
            {
                Rpp32f *srcPtrWindowTemp = srcPtrTemp + windowStart;
                Rpp32f *windowFnTemp = windowFn.data();
                Rpp32s t = 0;
                for (; t < alignedWindowLength; t += 8)
                {
                    __m256 pSrc, pWindowFn;
                    pSrc = _mm256_loadu_ps(srcPtrWindowTemp);
                    pWindowFn = _mm256_loadu_ps(windowFnTemp);
                    pSrc = _mm256_mul_ps(pSrc, pWindowFn);
                    _mm256_storeu_ps(windowOutputTemp, pSrc);
                    srcPtrWindowTemp += 8;
                    windowFnTemp += 8;
                    windowOutputTemp += 8;
                }
                for (; t < windowLength; t++)
                    *windowOutputTemp++ = (*windowFnTemp++) * (*srcPtrWindowTemp++);
            }
        }
        // Generate specgram output
        for (int64_t w = 0; w < numWindows; w++)
        {
            Rpp32f fftReal[numBins];
            Rpp32f fftImag[numBins];
            std::fill_n (fftReal, numBins, 0);
            std::fill_n (fftImag, numBins, 0);
            // Compute FFT
            for (Rpp32s k = 0; k < numBins; k++)
            {
                Rpp32f *windowOutputTemp = windowOutput + (w * nfft);
                Rpp32f *cosfTemp = cosf + (k * nfft);
                Rpp32f *sinfTemp = sinf + (k * nfft);
                Rpp32f real = 0.0f;
                Rpp32f imag = 0.0f;
                __m256 pReal, pImag;
                pReal = avx_p0;
                pImag = avx_p0;
                Rpp32s i = 0;
                for (; i < alignedNfftLength; i += 8)
                {
                    __m256 pSrc, pSin, pCos;
                    pSrc = _mm256_loadu_ps(windowOutputTemp);
                    pCos = _mm256_loadu_ps(cosfTemp);
                    pSin = _mm256_loadu_ps(sinfTemp);
                    pReal = _mm256_add_ps(pReal, _mm256_mul_ps(pSrc, pCos));
                    pImag = _mm256_add_ps(pImag, _mm256_mul_ps(pSrc, pSin));
                    windowOutputTemp += 8;
                    cosfTemp += 8;
                    sinfTemp += 8;
                }
                real = reduce_add_ps1(pReal);
                imag = reduce_add_ps1(pImag);
                for (; i < nfft; i++)
                {
                    Rpp32f x = *windowOutputTemp++;
                    real += x * *cosfTemp++;
                    imag += x * *sinfTemp++;
                }
                fftReal[k] = real;
                fftImag[k] = imag;
            }
            Rpp32f *fftRealTemp = fftReal;
            Rpp32f *fftImagTemp = fftImag;
            Rpp32s i = 0;
            Rpp32f *dstPtrBinTemp = dstPtrTemp + (w * hStride);
            for (; i < alignedNbinsLength; i += 8)
            {
                __m256 pReal, pImag, pTotal;
                pReal = _mm256_loadu_ps(fftRealTemp);
                pImag = _mm256_loadu_ps(fftImagTemp);
                pReal = _mm256_mul_ps(pReal, pReal);
                pImag = _mm256_mul_ps(pImag, pImag);
                pTotal = _mm256_add_ps(pReal, pImag);
                if (power == 1)
                    pTotal = _mm256_sqrt_ps(pTotal);
                if (vertical) {
                    Rpp32f *pTotalPtr = (Rpp32f *)&pTotal;
                    for (Rpp32s j = i; j < (i + 8); j++)
                        dstPtrTemp[j * hStride + w] = pTotalPtr[j - i];
                } else {
                    _mm256_storeu_ps(dstPtrBinTemp, pTotal);
                    dstPtrBinTemp += 8;
                }
                fftRealTemp += 8;
                fftImagTemp += 8;
            }
            for (; i < numBins; i++)
            {
                Rpp32f real = *fftRealTemp++;
                Rpp32f imag = *fftImagTemp++;
                Rpp32f total = (real * real) + (imag * imag);
                if (power == 1)
                    total = std::sqrt(total);
                if (vertical) {
                    int64_t outIdx = (i * hStride + w);
                    dstPtrTemp[outIdx] = total;
                }
                else
                    *dstPtrBinTemp++ = total;
            }
        }
    }
    return RPP_SUCCESS;
}