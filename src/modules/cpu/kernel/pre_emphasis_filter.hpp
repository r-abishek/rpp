#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus pre_emphasis_filter_host_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s *srcLengthTensor,
                                          Rpp32f *coeffTensor,
                                          Rpp32u borderType)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        Rpp32f coeff = coeffTensor[batchCount];

        if(borderType == RpptAudioBorderType::ZERO)
            dstPtrTemp[0] = srcPtrTemp[0];
        else if(borderType == RpptAudioBorderType::CLAMP)
            dstPtrTemp[0] = srcPtrTemp[0] * (1 - coeff);
        else if(borderType == RpptAudioBorderType::REFLECT)
            dstPtrTemp[0] = srcPtrTemp[0] - coeff * srcPtrTemp[1];

        int vectorIncrement = 8;
        int alignedLength = bufferLength & ~7;
        __m256 pCoeff = _mm256_set1_ps(coeff);

        int vectorLoopCount = 1;
        dstPtrTemp++;
        srcPtrTemp++;
        for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
        {
            __m256 pSrc[2];
            pSrc[0] = _mm256_loadu_ps(srcPtrTemp);
            pSrc[1] = _mm256_loadu_ps(srcPtrTemp - 1);
            pSrc[1] = _mm256_sub_ps(pSrc[0], _mm256_mul_ps(pSrc[1], pCoeff));
            _mm256_storeu_ps(dstPtrTemp, pSrc[1]);
            srcPtrTemp += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
        for(; vectorLoopCount < bufferLength; vectorLoopCount++)
            *dstPtrTemp = *srcPtrTemp - coeff * (*(srcPtrTemp - 1));
    }

    return RPP_SUCCESS;
}
