#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

void compute_2D_mean_axis1(Rpp32f *srcPtr,
                           Rpp32f *meanPtr,
                           Rpp32u *dims,
                           Rpp32u *stride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32u vectorLoopCount = dims[1]/8;
    __m256 pAddFactorN = _mm256_set1_ps(8);
    __m256 pStrideN = _mm256_set1_ps(stride[0]);
    // Outer loop with channels
    for(Rpp32u index_i = 0; index_i < dims[0]; index_i++)
    {
        meanPtr[index_i] = 0;
        __m256 pLengthN = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
        __m256 pMeanPtrN = avx_p0;
        Rpp32u index_j = 0;
        // Inner loop with length
        for( ; index_j < vectorLoopCount; index_j++)
        {
            __m256 pStrideJN = _mm256_mul_ps(pLengthN, pStrideN);
            __m256 pSrcPtrTempN = _mm256_i32gather_ps(srcPtrTemp, _mm256_cvtps_epi32(pStrideJN), 4);
            pMeanPtrN = _mm256_add_ps(pSrcPtrTempN, pMeanPtrN);
            pLengthN = _mm256_add_ps(pLengthN, pAddFactorN);
        }
        index_j = index_j * 8;
        for (; index_j < dims[1]; index_j++)
            meanPtr[index_i] += srcPtrTemp[stride[0]*index_j];
        meanPtr[index_i] += rpp_horizontal_add_avx(pMeanPtrN);
        meanPtr[index_i] = meanPtr[index_i] / dims[1];
        srcPtrTemp += stride[1];
    }
}

void compute_2D_mean_axis2(Rpp32f *srcPtr,
                           Rpp32f *meanPtr,
                           Rpp32u *dims,
                           Rpp32u *stride)
{
    Rpp32u vectorLoopCount = dims[1]/8;
    // Outer loop source length
    for(Rpp32u index_i = 0; index_i < dims[0]; index_i++)
    {
        Rpp32f *srcPtrTemp = srcPtr + (index_i * stride[1]);
        meanPtr[index_i] = 0;
        __m256 pMeanPtrN  = avx_p0;
        Rpp32u index_j = 0;
        // Inner loop channel
        for(; index_j < vectorLoopCount; index_j++)
        {
            //meanPtr[index_i] += (*(srcPtrTemp + index_j * stride[0]));
            __m256 pSrcPtrTempN = _mm256_loadu_ps(srcPtrTemp);
            pMeanPtrN = _mm256_add_ps(pSrcPtrTempN, pMeanPtrN);
            srcPtrTemp += 8;
        }
        index_j = index_j * 8;
        for(; index_j < dims[1]; index_j++)
            meanPtr[index_i] += *srcPtrTemp++;
        meanPtr[index_i] += rpp_horizontal_add_avx(pMeanPtrN);
        meanPtr[index_i] = meanPtr[index_i] / dims[1];
    }
}

void compute_2D_mean_axis3(Rpp32f *srcPtr,
                           Rpp32f *meanPtr,
                           Rpp32u *dims,
                           Rpp32u *stride)
{
    // Set total length and calculate rem
    Rpp32u vectorLoopCount = dims[1]/8;
    meanPtr[0] = 0;
    // Outer loop source length
    for(Rpp32u index_i = 0; index_i < dims[0]; index_i++)
    {
        Rpp32f *srcPtrTemp = srcPtr + (index_i * stride[1]);
        __m256 pMeanPtrN  = avx_p0;
        Rpp32u index_j = 0;
        // Inner loop channel
        for(; index_j < vectorLoopCount; index_j++)
        {
            __m256 pSrcPtrTempN = _mm256_loadu_ps(srcPtrTemp);
            pMeanPtrN = _mm256_add_ps(pSrcPtrTempN, pMeanPtrN);
            srcPtrTemp+=8;
        }
        index_j = index_j * 8;
        for(; index_j < dims[1]; index_j++)
            meanPtr[0] += *srcPtrTemp++;
        meanPtr[0] += rpp_horizontal_add_avx(pMeanPtrN);
    }
    meanPtr[0] = meanPtr[0] / (dims[0] * dims[1]);
}

void compute_2D_inv_std_dev_axis1(Rpp32f *srcPtr,
                                  Rpp32f *meanPtr,
                                  Rpp32f *stdDevPtr,
                                  Rpp32u *dims,
                                  Rpp32u *stride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32u vectorLoopCount = dims[1]/8;
    __m256 pAddFactorN = _mm256_set1_ps(8);
    __m256 pStrideN = _mm256_set1_ps(stride[0]);
    // Outer loop channels
    for(Rpp32u index_i = 0; index_i < dims[0]; index_i++)
    {
        stdDevPtr[index_i] = 0;
        __m256 pLengthN = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
        __m256 pMeanPtrN = _mm256_set1_ps(meanPtr[index_i]);
        __m256 pStdDevPtrN = avx_p0;
        Rpp32u index_j = 0;
        // Inner loop length
        for(; index_j < vectorLoopCount; index_j++)
        {
            __m256 pStrideJN = _mm256_mul_ps(pLengthN, pStrideN);
            __m256 pDiffN = _mm256_sub_ps(_mm256_i32gather_ps(srcPtrTemp, _mm256_cvtps_epi32(pStrideJN), 4), pMeanPtrN);
            pStdDevPtrN = _mm256_add_ps(pStdDevPtrN, _mm256_mul_ps(pDiffN, pDiffN));
            pLengthN = _mm256_add_ps(pLengthN, pAddFactorN);
        }
        index_j  = index_j * 8;
        for(; index_j < dims[1]; index_j++)
        {
            Rpp32f diff = (*(srcPtrTemp + index_j * stride[0]) - meanPtr[index_i]);
            stdDevPtr[index_i] += (diff * diff);
        }
        stdDevPtr[index_i] += rpp_horizontal_add_avx(pStdDevPtrN);
        stdDevPtr[index_i] = stdDevPtr[index_i] / dims[1];
        stdDevPtr[index_i] = (!stdDevPtr[index_i]) ? 0.0f : 1.0f / sqrt(stdDevPtr[index_i]);
        srcPtrTemp += stride[1];
    }
}

void compute_2D_inv_std_dev_axis2(Rpp32f *srcPtr,
                                  Rpp32f *meanPtr,
                                  Rpp32f *stdDevPtr,
                                  Rpp32u *dims,
                                  Rpp32u *stride)
{
    Rpp32u vectorLoopCount = dims[1]/8;
    // Outer loop source length
    for(Rpp32u index_i = 0; index_i < dims[0]; index_i++)
    {
        Rpp32f *srcPtrTemp = srcPtr + (index_i * stride[1]);
        stdDevPtr[index_i] = 0;
        __m256 pMeanPtrN = _mm256_set1_ps(meanPtr[index_i]);
        __m256 pStdDevPtrN = avx_p0;
        Rpp32u index_j = 0;
        // Inner loop channels
        for(; index_j < vectorLoopCount; index_j++)
        {
            __m256 pDiffN = _mm256_sub_ps(_mm256_loadu_ps(srcPtrTemp), pMeanPtrN);
            pStdDevPtrN = _mm256_add_ps(pStdDevPtrN, _mm256_mul_ps(pDiffN, pDiffN));
            srcPtrTemp += 8;
        }
        index_j = index_j * 8;
        for(; index_j < dims[1]; index_j++)
        {
            Rpp32f diff = (*srcPtrTemp++ - meanPtr[index_i]);
            stdDevPtr[index_i] += (diff * diff);
        }
        stdDevPtr[index_i] += rpp_horizontal_add_avx(pStdDevPtrN);
        stdDevPtr[index_i] = stdDevPtr[index_i] / dims[1];
        stdDevPtr[index_i] = (!stdDevPtr[index_i]) ? 0.0f : 1.0f / sqrt(stdDevPtr[index_i]);
    }
}

void compute_2D_inv_std_dev_axis3(Rpp32f *srcPtr,
                                  Rpp32f *meanPtr,
                                  Rpp32f *stdDevPtr,
                                  Rpp32u *dims,
                                  Rpp32u *stride)
{
    Rpp32u vectorLoopCount = dims[1]/8;
    stdDevPtr[0] = 0;
    // Outer loop source length
    for(Rpp32u index_i = 0; index_i < dims[0]; index_i++)
    {
        Rpp32f *srcPtrTemp = srcPtr + (index_i * stride[1]);
        __m256 pMeanPtrN = _mm256_set1_ps(meanPtr[0]);
        __m256 pStdDevPtrN = avx_p0;
        Rpp32u index_j = 0;
        // Inner loop channels
        for(; index_j < vectorLoopCount; index_j++)
        {
            __m256 pDiffN = _mm256_sub_ps(_mm256_loadu_ps(srcPtrTemp), pMeanPtrN);
            pStdDevPtrN = _mm256_add_ps(pStdDevPtrN, _mm256_mul_ps(pDiffN, pDiffN));
            srcPtrTemp += 8;
        }
        index_j = index_j * 8;
        for(; index_j < dims[1]; index_j++)
        {
            Rpp32f diff = (*srcPtrTemp++ - meanPtr[0]);
            stdDevPtr[0] += (diff * diff);
        }
        stdDevPtr[0] += rpp_horizontal_add_avx(pStdDevPtrN);
    }
    stdDevPtr[0] = stdDevPtr[0] / (dims[0] * dims[1]);
    stdDevPtr[0] = (!stdDevPtr[0]) ? 0.0f : 1.0f / sqrt(stdDevPtr[0]);
}

void normalize_2D_tensor_cpu(Rpp32f *srcPtr,
                            RpptDescPtr srcDescPtr,
                            Rpp32f *dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32f *meanPtr,
                            Rpp32f *invStdDevPtr,
                            Rpp32f shift,
                            Rpp32u *dims,
                            Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    for(Rpp32u index_i = 0; index_i < dims[0]; index_i++)
    {
        Rpp32f *srcPtrTempRow = srcPtrTemp;
        Rpp32f *dstPtrTempRow = dstPtrTemp;
        for(Rpp32u index_j = 0; index_j < dims[1]; index_j++)
        {
            *dstPtrTempRow++ = (*srcPtrTempRow++ - meanPtr[paramIdx]) * invStdDevPtr[paramIdx] + shift;
            paramIdx += paramStride[0];
        }
        paramIdx = (!paramStride[1]) ? 0 : paramIdx + paramStride[1];
        srcPtrTemp += (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? srcDescPtr->strides.hStride : srcDescPtr->strides.wStride;
        dstPtrTemp += (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? dstDescPtr->strides.hStride : dstDescPtr->strides.wStride;
    }
}

RppStatus normalize_audio_host_tensor(Rpp32f* srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcLengthTensor,
                                      Rpp32s *channelsTensor,
                                      Rpp32s axis_mask,
                                      Rpp32f mean,
                                      Rpp32f stdDev,
                                      Rpp32f scale,
                                      Rpp32f shift,
                                      Rpp32f epsilon,
                                      Rpp32s ddof,
                                      Rpp32u numOfDims)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32u srcAudioDims[numOfDims], srcReductionDims[numOfDims], srcStride[numOfDims], paramStride[numOfDims];
        srcAudioDims[0] = srcLengthTensor[batchCount];
        srcAudioDims[1] = channelsTensor[batchCount];

        if (axis_mask == 3)
        {
            srcStride[0] = srcDescPtr->strides.cStride;
            srcStride[1] = (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? srcDescPtr->strides.hStride : srcDescPtr->strides.wStride;
            srcReductionDims[0] = srcAudioDims[0];
            srcReductionDims[1] = srcAudioDims[1];
            paramStride[0] = paramStride[1] = 0;
        }
        else if (axis_mask == 1)
        {
            srcStride[0] = (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? srcDescPtr->strides.hStride : srcDescPtr->strides.wStride;
            srcStride[1] = srcDescPtr->strides.cStride;
            srcReductionDims[0] = srcAudioDims[1];
            srcReductionDims[1] = srcAudioDims[0];
            paramStride[0] = 1;
            paramStride[1] = 0;
        }
        else if (axis_mask == 2)
        {
            srcStride[0] = srcDescPtr->strides.cStride;
            srcStride[1] = (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? srcDescPtr->strides.hStride : srcDescPtr->strides.wStride;
            srcReductionDims[0] = srcAudioDims[0];
            srcReductionDims[1] = srcAudioDims[1];
            paramStride[0] = 0;
            paramStride[1] = 1;
        }

        Rpp32f meanTensor[srcReductionDims[0]];
        Rpp32f stdDevTensor[srcReductionDims[0]];
        std::fill_n(meanTensor, srcReductionDims[0], 0);
        std::fill_n(stdDevTensor, srcReductionDims[0], 0);
        meanTensor[0] = mean;
        stdDevTensor[0] = stdDev;

        if(!mean)
        {
            if (axis_mask == 1)
                compute_2D_mean_axis1(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
            else if (axis_mask == 2)
                compute_2D_mean_axis2(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
            else if (axis_mask == 3)
                compute_2D_mean_axis3(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
        }

        if(!stdDev)
        {
            if (axis_mask == 1)
                compute_2D_inv_std_dev_axis1(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);
            else if (axis_mask == 2)
                compute_2D_inv_std_dev_axis2(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);
            else if (axis_mask == 3)
                compute_2D_inv_std_dev_axis3(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);
        }

        normalize_2D_tensor_cpu(srcPtrTemp, srcDescPtr, dstPtrTemp, dstDescPtr, meanTensor, stdDevTensor, shift, srcAudioDims, paramStride);
    }
    return RPP_SUCCESS;
}