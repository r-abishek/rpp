#include "rppdefs.h"
#include <omp.h>

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  Rpp32s *channelsTensor,
                                  bool normalizeWeights)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrCurrent = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrCurrent = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s channels = channelsTensor[batchCount];
        Rpp32s samples = srcLengthTensor[batchCount];

        if(channels == 1)
        {
            // No need of downmixing, do a direct memcpy
            memcpy(dstPtrCurrent, srcPtrCurrent, (size_t)(samples * sizeof(Rpp32f)));
        }
        else
        {
            std::vector<Rpp32f> weights;
            weights.resize(channels, 1.f / channels);
            std::vector<Rpp32f> normalizedWeights;

            if(normalizeWeights)
            {
                normalizedWeights.resize(channels);

                // Compute sum of the weights
                Rpp32f sum = 0.0;
                for(int i = 0; i < channels; i++)
                    sum += weights[i];

                // Normalize the weights
                Rpp32f invSum = 1.0 / sum;
                for(int i = 0; i < channels; i++)
                    normalizedWeights[i] = weights[i] * invSum;

                weights = normalizedWeights;
            }

            Rpp32u channelIncrement = 4;
            Rpp32u alignedChannels = (channels / 4) * 4;

            // use weights to downmix to mono
            Rpp32f *srcPtrRow = srcPtrCurrent;
            for(Rpp64s dstIdx = 0; dstIdx < samples; dstIdx++)
            {
                Rpp32f *srcPtrTemp = srcPtrRow;
                __m128 pDst = xmm_p0;
                int channelLoopCount = 0;
                for(; channelLoopCount < alignedChannels; channelLoopCount += channelIncrement)
                {
                    __m128 pSrc, pWeights;
                    pWeights = _mm_loadu_ps(&weights[channelLoopCount]);
                    pSrc = _mm_loadu_ps(srcPtrTemp);
                    pSrc = _mm_mul_ps(pSrc, pWeights);
                    pDst = _mm_add_ps(pDst, pSrc);
                    srcPtrTemp += channelIncrement;
                }
                dstPtrCurrent[dstIdx] = rpp_horizontal_add_sse(pDst);
                for(; channelLoopCount < channels; channelLoopCount++)
                {
                    dstPtrCurrent[dstIdx] += ((*srcPtrTemp) * weights[channelLoopCount]);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
