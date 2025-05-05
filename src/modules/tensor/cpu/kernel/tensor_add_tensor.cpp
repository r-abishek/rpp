#include "host_tensor_executors.hpp"
#include "broadcast.hpp"
#include "rpp_cpu_simd_math.hpp"

template<typename T>
void tensor_add_tensor_recursive(T *src1, T *src2, Rpp32u *src1Strides, Rpp32u *src2Strides, T *dst, Rpp32u *dstStrides, Rpp32u *dstShape, Rpp32u nDim)
{
    if (!nDim)
        *dst = *src1 + *src2;
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            tensor_add_tensor_recursive(src1, src2, src1Strides + 1, src2Strides + 1, dst, dstStrides + 1, dstShape + 1, nDim - 1);
            dst += *(dstStrides + 1);
            src1 += *(src1Strides + 1);
            src2 += *(src2Strides + 1);
        }
    }
}


RppStatus tensor_add_tensor_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                                Rpp32f *srcPtr2,
                                                RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                Rpp32f *dstPtr,
                                                RpptGenericDescPtr dstGenericDescPtr,
                                                Rpp32u *srcPtr1roiTensor,
                                                Rpp32u *srcPtr2roiTensor,
                                                rpp::Handle& handle) {

    checkEqualBatchSize(srcPtr1GenericDescPtr, srcPtr2GenericDescPtr);
    BroadcastDstShape(srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstGenericDescPtr);
    RpptGenericDesc src1BroadcastDesc, src2BroadcastDesc, dstBroadcastDesc;
    RpptGenericDescPtr src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr;
    src1BroadcastDescPtr = &src1BroadcastDesc;
    src2BroadcastDescPtr = &src2BroadcastDesc;
    dstBroadcastDescPtr = &dstBroadcastDesc;
    src1BroadcastDesc = *srcPtr1GenericDescPtr;
    src2BroadcastDesc = *srcPtr2GenericDescPtr;
    dstBroadcastDesc = *dstGenericDescPtr;
    GroupShapes(src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src1BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src2BroadcastDescPtr, dstBroadcastDescPtr);

    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u src1NDim = srcPtr1GenericDescPtr->numDims - 1;
    Rpp32u src2NDim = srcPtr2GenericDescPtr->numDims - 1;
    Rpp32u broadcastNDim = dstBroadcastDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstBroadcastDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u *src1begin = src1roi;

        Rpp32u *src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u *src2begin = src2roi;

        Rpp32f *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        Rpp32f *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for(int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for(int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u *length = dstBroadcastDescPtr->dims + 1;
        Rpp32u *src1length = src1BroadcastDescPtr->dims + 1;
        Rpp32u *src2length = src2BroadcastDescPtr->dims + 1;
        printf("Length, src1Length, and src2Length is %d %d %d\n", length[0], src1length[0], src2length[0]);

        Rpp32u vectorIncrement = 16;
        printf("broadcastNDim is %d\n", broadcastNDim);
        if (broadcastNDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~15;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
                printf("src1shape is %d\n", src1shape);
            }
            else if (src2shape == 1)
            {
                printf("src2shape is %d\n", src2shape);
            }
            else
            {
                printf("src1shape and src2shape are %d %d\n", src1shape, src2shape);
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    printf("Inside loop with vectorLoopCount %d\n", vectorLoopCount);
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    p2[0] = _mm256_add_ps(p1[0], p2[0]);
                    p2[1] = _mm256_add_ps(p1[1], p2[1]);
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p2);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++)
                {
                    printf("Inside loop with vectorLoopCount %d\n", vectorLoopCount);
                    *dstPtrTemp = *srcPtrTemp1 + *srcPtrTemp2;
                    srcPtrTemp1++;
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            }
        }
        else
            tensor_add_tensor_recursive(srcPtrTemp1, srcPtrTemp2, src1BroadcastDescPtr->strides, src2BroadcastDescPtr->strides, dstPtrTemp, dstBroadcastDescPtr->strides, length, broadcastNDim);
    }

    return RPP_SUCCESS;
}

RppStatus tensor_add_tensor_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                                Rpp16f *srcPtr2,
                                                RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                Rpp16f *dstPtr,
                                                RpptGenericDescPtr dstGenericDescPtr,
                                                Rpp32u *srcPtr1roiTensor,
                                                Rpp32u *srcPtr2roiTensor,
                                                rpp::Handle& handle) {

    checkEqualBatchSize(srcPtr1GenericDescPtr, srcPtr2GenericDescPtr);
    BroadcastDstShape(srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstGenericDescPtr);
    RpptGenericDesc src1BroadcastDesc, src2BroadcastDesc, dstBroadcastDesc;
    RpptGenericDescPtr src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr;
    src1BroadcastDescPtr = &src1BroadcastDesc;
    src2BroadcastDescPtr = &src2BroadcastDesc;
    dstBroadcastDescPtr = &dstBroadcastDesc;
    src1BroadcastDesc = *srcPtr1GenericDescPtr;
    src2BroadcastDesc = *srcPtr2GenericDescPtr;
    dstBroadcastDesc = *dstGenericDescPtr;
    GroupShapes(src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src1BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src2BroadcastDescPtr, dstBroadcastDescPtr);

    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u src1NDim = srcPtr1GenericDescPtr->numDims - 1;
    Rpp32u src2NDim = srcPtr2GenericDescPtr->numDims - 1;
    Rpp32u broadcastNDim = dstBroadcastDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstBroadcastDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *src1roi = srcPtr1roiTensor + batchCount * src1NDim * 2;
        Rpp32u *src1begin = src1roi;

        Rpp32u *src2roi = srcPtr2roiTensor + batchCount * src2NDim * 2;
        Rpp32u *src2begin = src2roi;

        Rpp16f *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        Rpp16f *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];

        for(int i = 0; i < src1NDim; i++)
            srcPtrTemp1 += src1begin[i] * srcPtr1GenericDescPtr->strides[i + 1];

        for(int i = 0; i < src2NDim; i++)
            srcPtrTemp2 += src2begin[i] * srcPtr2GenericDescPtr->strides[i + 1];

        Rpp16f *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u *length = dstBroadcastDescPtr->dims + 1;

        tensor_add_tensor_recursive(srcPtrTemp1, srcPtrTemp2, src1BroadcastDescPtr->strides, src2BroadcastDescPtr->strides, dstPtrTemp, dstBroadcastDescPtr->strides, length, broadcastNDim);
    }

    return RPP_SUCCESS;
}
