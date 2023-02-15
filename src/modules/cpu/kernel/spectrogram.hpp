#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void hann_window(Rpp32f *output, Rpp32s windowSize)
{
    Rpp32f a = TWO_PI / windowSize;
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
    const Rpp32f mulFactor = TWO_PI / nfft;
    Rpp32u hStride = dstDescPtr->strides.hStride;
    Rpp32s alignedNfftLength = nfft & ~7;
    Rpp32s alignedNbinsLength = numBins & ~7;
    Rpp32s alignedWindowLength = windowLength & ~7;

    Rpp32f cosf[numBins * nfft];
    Rpp32f sinf[numBins * nfft];
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(8)
    for (Rpp32s k = 0; k < numBins; k++)
    {
        Rpp32f* cosfTemp = cosf + (k * nfft);
        Rpp32f* sinfTemp = sinf + (k * nfft);
        __m256 PNfftN = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
        __m256 pMulFactorN = _mm256_set1_ps(k*mulFactor);
        __m256 pAddFactorN = avx_p8;
        Rpp32s i = 0;
        for (; i < alignedNfftLength; i += 8)
        {
            __m256 pSrc = _mm256_mul_ps(PNfftN, pMulFactorN);
            __m256 pSin, pCos;
            sincos_ps(pSrc, &pSin, &pCos);
            pSin = _mm256_mul_ps(avx_pm1, pSin);
            PNfftN = _mm256_add_ps(PNfftN, pAddFactorN);
            _mm256_storeu_ps(cosfTemp, pCos);
            _mm256_storeu_ps(sinfTemp, pSin);
            cosfTemp += 8;
            sinfTemp += 8;
        }
        for (; i < nfft; i++)
        {
            *cosfTemp++ = std::cos(k * i * mulFactor);
            *sinfTemp++ = -std::sin(k * i * mulFactor);
        }
    }

    std::vector<Rpp32f> windowFn;
    windowFn.resize(windowLength);
    // Generate hanning window
    if (windowFunction == NULL)
        hann_window(windowFn.data(), windowLength);
    else
        memcpy(windowFn.data(), windowFunction, windowLength * sizeof(Rpp32f));

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
        for (Rpp64s w = 0; w < numWindows; w++)
        {
            Rpp64s windowStart = w * windowStep - windowCenterOffset;
            Rpp32f *windowOutputTemp = windowOutput + (w * nfft);
            if (windowStart < 0 || (windowStart + windowLength) > bufferLength)
            {
                for (Rpp32s t = 0; t < windowLength; t++)
                {
                    Rpp64s inIdx = windowStart + t;
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
        for (Rpp64s w = 0; w < numWindows; w++)
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
                real = rpp_horizontal_add_avx(pReal);
                imag = rpp_horizontal_add_avx(pImag);
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
                if (vertical)
                {
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
                if (vertical)
                {
                    Rpp64s outIdx = (i * hStride + w);
                    dstPtrTemp[outIdx] = total;
                }
                else
                    *dstPtrBinTemp++ = total;
            }
        }
    }
    return RPP_SUCCESS;
}