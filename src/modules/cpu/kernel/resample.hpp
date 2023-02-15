#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline Rpp64f hann(Rpp32f x)
{
    return 0.5 * (1 + std::cos(x * PI));
}

struct ResamplingWindow
{
    inline std::pair<Rpp32s, Rpp32s> input_range(Rpp32f x)
    {
        Rpp32s xc = ceilf(x);
        Rpp32s loc0 = xc - lobes;
        Rpp32s loc1 = xc + lobes;
        return {loc0, loc1};
    }

    inline Rpp32f operator()(Rpp32f x)
    {
        Rpp32f locRaw = x * scale + center;
        Rpp32s locFloor = floorf(locRaw);
        Rpp32f weight = locRaw - locFloor;
        locFloor = std::max(std::min(locFloor, lookupSize - 2), 0);
        Rpp32f curr = lookup[locFloor];
        Rpp32f next = lookup[locFloor + 1];
       return curr + weight * (next - curr);
    }

    inline __m128 operator()(__m128 x)
    {
        __m128 pLocRaw = _mm_add_ps(x * _mm_set1_ps(scale), _mm_set1_ps(center));
        __m128i pxLocFloor = _mm_cvtps_epi32(pLocRaw);
        __m128 pWeight = _mm_sub_ps(pLocRaw, _mm_cvtepi32_ps(pxLocFloor));
        pxLocFloor = _mm_max_epi32(_mm_min_epi32(pxLocFloor, pxLookupMax), xmm_px0);
        Rpp32s idx[4];
        _mm_storeu_si128((__m128i*)idx, pxLocFloor);
        __m128 pCurrent = _mm_setr_ps(lookup[idx[0]], lookup[idx[1]], lookup[idx[2]], lookup[idx[3]]);
        __m128 pNext = _mm_setr_ps(lookup[idx[0] + 1], lookup[idx[1] + 1], lookup[idx[2] + 1], lookup[idx[3] + 1]);
        return _mm_add_ps(pCurrent, _mm_mul_ps(pWeight, _mm_sub_ps(pNext, pCurrent)));
    }

    Rpp32f scale = 1, center = 1;
    Rpp32s lobes = 0, coeffs = 0;
    Rpp32s lookupSize = 0;
    __m128i pxLookupMax;
    std::vector<Rpp32f> lookup;
};

inline void windowed_sinc(ResamplingWindow &window, Rpp32s coeffs, Rpp32s lobes)
{
    Rpp32f scale = 2.0f * lobes / (coeffs - 1);
    Rpp32f scaleEnvelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.resize(coeffs + 5);
    window.lookupSize = window.lookup.size();
    window.pxLookupMax = _mm_set1_epi32(window.lookupSize - 2);

    window.center = ((coeffs - 1) * 0.5f) + 1;
    window.scale = 1 / scale;
    for (int i = 1, iMinusCenter = (1 - window.center); i <= coeffs; i++, iMinusCenter++)
        window.lookup[i] = sinc(iMinusCenter * scale) * hann(iMinusCenter * scaleEnvelope);
}

RppStatus resample_host_tensor(Rpp32f *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *inRateTensor,
                               Rpp32f *outRateTensor,
                               Rpp32s *srcLengthTensor,
                               Rpp32s *channelTensor,
                               Rpp32f quality)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(Rpp32s batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f inRate = inRateTensor[batchCount];
        Rpp32f outRate = outRateTensor[batchCount];
        Rpp32s srcLength = srcLengthTensor[batchCount];
        Rpp32s numChannels = channelTensor[batchCount];

        if(outRate == inRate)
        {
            // No need of Resampling, do a direct memcpy
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcLength * numChannels * sizeof(Rpp32f)));
        }
        else
        {
            ResamplingWindow window;
            Rpp32s lobes = std::round((quality * (0.007 * quality - 0.09) + 3));
            Rpp32s lookupSize = lobes * 64 + 1;
            windowed_sinc(window, lookupSize, lobes);
            Rpp64s outBegin = 0;
            Rpp64s outEnd = std::ceil(srcLength * outRate / inRate);
            Rpp64s inPos = 0;
            Rpp64s block = 1 << 10;
            Rpp64f scale = inRate / outRate;
            Rpp32f fScale = scale;

            if(numChannels == 1)
            {
                for (Rpp64s outBlock = outBegin; outBlock < outEnd; outBlock += block)
                {
                    Rpp64s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp64s inBlockFloor = std::floor(inBlockRaw);
                    Rpp32f inPos = inBlockRaw - inBlockFloor;
                    const Rpp32f *inBlockPtr = srcPtrTemp + inBlockFloor;

                    for (Rpp64s outPos = outBlock; outPos < blockEnd; outPos++, inPos += fScale)
                    {
                        Rpp32s loc0, loc1;
                        std::tie(loc0, loc1) = window.input_range(inPos);
                        if (loc0 + inBlockFloor < 0)
                            loc0 = -inBlockFloor;
                        if (loc1 + inBlockFloor >= srcLength)
                            loc1 = srcLength - inBlockFloor;
                        Rpp32f accum = 0;
                        Rpp32s i = loc0;

                        __m128 pAccum = xmm_p0;
                        Rpp32f locInWindow = i - inPos;
                        __m128 pLocInWindow = _mm_setr_ps(locInWindow, locInWindow + 1, locInWindow + 2, locInWindow + 3);
                        for (; i + 3 < loc1; i += 4)
                        {
                            __m128 pW = window(pLocInWindow);
                            pAccum = _mm_add_ps(pAccum, _mm_mul_ps(_mm_loadu_ps(inBlockPtr + i), pW));
                            pLocInWindow = _mm_add_ps(pLocInWindow, xmm_p4);
                        }
                        pAccum = _mm_add_ps(pAccum, _mm_shuffle_ps(pAccum, pAccum, _MM_SHUFFLE(1, 0, 3, 2)));
                        pAccum = _mm_add_ps(pAccum, _mm_shuffle_ps(pAccum, pAccum, _MM_SHUFFLE(0, 1, 0, 1)));
                        accum = _mm_cvtss_f32(pAccum);

                        locInWindow = i - inPos;
                        for (; i < loc1; i++, locInWindow++)
                        {
                            Rpp32f w = window(locInWindow);
                            accum += inBlockPtr[i] * w;
                        }

                        dstPtrTemp[outPos] = accum;
                    }
                }
            }
            else
            {
                std::vector<Rpp32f> channelAccum;
                channelAccum.resize(numChannels);
                for (Rpp64s outBlock = outBegin; outBlock < outEnd; outBlock += block)
                {
                    Rpp64s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp64s inBlockFloor = std::floor(inBlockRaw);

                    Rpp32f inPos = inBlockRaw - inBlockFloor;
                    const Rpp32f *inBlockPtr = srcPtrTemp + inBlockFloor * numChannels;
                    for (Rpp64s outPos = outBlock; outPos < blockEnd; outPos++, inPos += fScale)
                    {
                        Rpp32s loc0, loc1;
                        std::tie(loc0, loc1) = window.input_range(inPos);
                        if (loc0 + inBlockFloor < 0)
                            loc0 = -inBlockFloor;
                        if (loc1 + inBlockFloor >= srcLength)
                            loc1 = srcLength - inBlockFloor;

                        std::fill(channelAccum.begin() , channelAccum.end() , 0.0f);
                        Rpp32f locInWindow = loc0 - inPos;
                        Rpp32s ofs0 = loc0 * numChannels;
                        Rpp32s ofs1 = loc1 * numChannels;

                        for (Rpp32s inOfs = ofs0; inOfs < ofs1; inOfs += numChannels, locInWindow++)
                        {
                            Rpp32f w = window(locInWindow);
                            for (Rpp32s c = 0; c < numChannels; c++)
                                channelAccum[c] += inBlockPtr[inOfs + c] * w;
                        }

                        Rpp32s dstLoc = outPos * numChannels;
                        for (Rpp32s c = 0; c < numChannels; c++)
                            dstPtrTemp[dstLoc + c] = channelAccum[c];
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}
