#include "host_tensor_executors.hpp"
#include "broadcast.hpp"
#include "rpp_cpu_simd_math.hpp"

template<typename T>
inline T add_op(T src1, T src2) { return src1 + src2; }

template<typename T>
inline T subtract_op(T src1, T src2) { return src1 - src2; }

template<typename T>
inline T multiply_op(T src1, T src2) { return src1 * src2; }

template<typename T>
inline T divide_op(T src1, T src2) { return src1 / src2; }

inline __m256 simd_add_ps(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
inline __m256 simd_subtract_ps(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
inline __m256 simd_multiply_ps(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
inline __m256 simd_divide_ps(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }

template<typename T, typename Operation>
inline void tensor_binary_op_recursive(T *src1, T *src2, Rpp32u *src1Strides, Rpp32u *src2Strides, T *dst, Rpp32u *dstStrides, Rpp32u *dstShape, Rpp32u nDim, Operation op)
{
    if (!nDim)
        *dst = op(*src1, *src2);
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            tensor_binary_op_recursive(src1, src2, src1Strides + 1, src2Strides + 1, dst, dstStrides + 1, dstShape + 1, nDim - 1, op);
            dst += *(dstStrides + 1);
            src1 += *(src1Strides + 1);
            src2 += *(src2Strides + 1);
        }
    }
}

template<typename Operation, typename SIMDOperation>
RppStatus tensor_binary_op_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                               Rpp32f *srcPtr2,
                                               RpptGenericDescPtr srcPtr1GenericDescPtr,
                                               RpptGenericDescPtr srcPtr2GenericDescPtr,
                                               Rpp32f *dstPtr,
                                               RpptGenericDescPtr dstGenericDescPtr,
                                               Operation op,
                                               SIMDOperation simd_op,
                                               Rpp32u *srcPtr1roiTensor,
                                               Rpp32u *srcPtr2roiTensor,
                                               rpp::Handle& handle) {
    printf("Inside tensor binary operations\n");
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

        Rpp32u vectorIncrement = 16;

        if (broadcastNDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~15;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
#if __AVX2__
                __m256 p1 = _mm256_set1_ps(srcPtrTemp1[0]);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p2[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    p2[0] = simd_op(p1, p2[0]);
                    p2[1] = simd_op(p1, p2[1]);
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p2);    // simd stores
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     *dstPtrTemp = op(*srcPtrTemp1, *srcPtrTemp2);
                     srcPtrTemp2++;
                     dstPtrTemp++;
                 }
            }
            else if (src2shape == 1)
            {
#if __AVX2__
                __m256 p2 = _mm256_set1_ps(srcPtrTemp2[0]);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    p1[0] = simd_op(p1[0], p2);
                    p1[1] = simd_op(p1[1], p2);
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p1);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     *dstPtrTemp = op(*srcPtrTemp1, *srcPtrTemp2);
                     srcPtrTemp1++;
                     dstPtrTemp++;
                 }
            }
            else
            {
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    p2[0] = simd_op(p1[0], p2[0]);
                    p2[1] = simd_op(p1[1], p2[1]);
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p2);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++)
                {
                    *dstPtrTemp = op(*srcPtrTemp1, *srcPtrTemp2);
                    srcPtrTemp1++;
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            }
        }
        else if (broadcastNDim == 2)
        {
            Rpp32u alignedLength = length[1] & ~15;
            Rpp32u src1shape = src1length[1];
            Rpp32u src2shape = src2length[1];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrTest1 = srcPtrTemp1;
                    Rpp32f *srcPtrTest2 = srcPtrTemp2;
                    Rpp32f *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
                    __m256 p1 = _mm256_set1_ps(srcPtrTest1[0]);
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p2[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTest2, p2);    // simd loads
                        p2[0] = simd_op(p1, p2[0]);
                        p2[1] = simd_op(p1, p2[1]);
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTest, p2);    // simd stores
                        srcPtrTest2 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        *dstPtrTest = op(*srcPtrTest1, *srcPtrTest2);
                        srcPtrTest2++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrTest1 = srcPtrTemp1;
                    Rpp32f *srcPtrTest2 = srcPtrTemp2;
                    Rpp32f *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
                    __m256 p2 = _mm256_set1_ps(srcPtrTest2[0]);
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTest1, p1);    // simd loads
                        p1[0] = simd_op(p1[0], p2);
                        p1[1] = simd_op(p1[1], p2);
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTest, p1);    // simd stores
                        srcPtrTest1 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        *dstPtrTest = op(*srcPtrTest1, *srcPtrTest2);
                        srcPtrTest1++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrTest1 = srcPtrTemp1;
                    Rpp32f *srcPtrTest2 = srcPtrTemp2;
                    Rpp32f *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[2], p2[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTest1, p1);    // simd loads
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTest2, p2);    // simd loads
                        p2[0] = simd_op(p1[0], p2[0]);
                        p2[1] = simd_op(p1[1], p2[1]);
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTest, p2);    // simd stores
                        srcPtrTest1 += vectorIncrement;
                        srcPtrTest2 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        *dstPtrTest = op(*srcPtrTest1, *srcPtrTest2);
                        srcPtrTest1++;
                        srcPtrTest2++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
        }
        else if (broadcastNDim == 3)
        {
            Rpp32u alignedLength = length[2] & ~15;
            Rpp32u src1shape = src1length[2];
            Rpp32u src2shape = src2length[2];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrTest1 = srcPtrTemp1;
                    Rpp32f *srcPtrTest2 = srcPtrTemp2;
                    Rpp32f *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp32f *srcPtrNew1 = srcPtrTest1;
                        Rpp32f *srcPtrNew2 = srcPtrTest2;
                        Rpp32f *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;
                        __m256 p1 = _mm256_set1_ps(srcPtrNew1[0]);
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p2[2];
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrNew2, p2);    // simd loads
                            p2[0] = simd_op(p1, p2[0]);
                            p2[1] = simd_op(p1, p2[1]);
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrNew, p2);    // simd stores
                            srcPtrNew2 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            *dstPtrNew = op(*srcPtrNew1, *srcPtrNew2);
                            srcPtrNew2++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrTest1 = srcPtrTemp1;
                    Rpp32f *srcPtrTest2 = srcPtrTemp2;
                    Rpp32f *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp32f *srcPtrNew1 = srcPtrTest1;
                        Rpp32f *srcPtrNew2 = srcPtrTest2;
                        Rpp32f *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;
                        __m256 p2 = _mm256_set1_ps(srcPtrNew2[0]);
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p1[2];
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrNew1, p1);    // simd loads
                            p1[0] = simd_op(p1[0], p2);
                            p1[1] = simd_op(p1[1], p2);
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrNew, p1);    // simd stores
                            srcPtrNew1 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            *dstPtrNew = op(*srcPtrNew1, *srcPtrNew2);
                            srcPtrNew1++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp32f *srcPtrTest1 = srcPtrTemp1;
                    Rpp32f *srcPtrTest2 = srcPtrTemp2;
                    Rpp32f *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp32f *srcPtrNew1 = srcPtrTest1;
                        Rpp32f *srcPtrNew2 = srcPtrTest2;
                        Rpp32f *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p1[2], p2[2];
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrNew1, p1);    // simd loads
                            rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrNew2, p2);    // simd loads
                            p2[0] = simd_op(p1[0], p2[0]);
                            p2[1] = simd_op(p1[1], p2[1]);
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrNew, p2);    // simd stores
                            srcPtrNew1 += vectorIncrement;
                            srcPtrNew2 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            *dstPtrNew = op(*srcPtrNew1, *srcPtrNew2);
                            srcPtrNew1++;
                            srcPtrNew2++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
        }
        else
            tensor_binary_op_recursive(srcPtrTemp1, srcPtrTemp2, src1BroadcastDescPtr->strides, src2BroadcastDescPtr->strides, dstPtrTemp, dstBroadcastDescPtr->strides, length, broadcastNDim, op);
    }

    return RPP_SUCCESS;
}

RppStatus tensor_binary_op_dispatch_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                                        Rpp32f *srcPtr2,
                                                        RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                        RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                        Rpp32f *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        RpptOp tensorOp,
                                                        Rpp32u *srcPtr1roiTensor,
                                                        Rpp32u *srcPtr2roiTensor,
                                                        rpp::Handle& handle) {
    switch(tensorOp) {
        case RPP_TENSOR_OP_ADD:
            tensor_binary_op_f32_f32_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, add_op<Rpp32f>, simd_add_ps, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_SUBTRACT:
            tensor_binary_op_f32_f32_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, subtract_op<Rpp32f>, simd_subtract_ps, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_MULTIPLY:
            tensor_binary_op_f32_f32_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, multiply_op<Rpp32f>, simd_multiply_ps, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_DIVIDE:
            tensor_binary_op_f32_f32_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, divide_op<Rpp32f>, simd_divide_ps, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
    }

    return RPP_SUCCESS;
}

template<typename Operation, typename SIMDOperation>
RppStatus tensor_binary_op_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                                Rpp16f *srcPtr2,
                                                RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                Rpp16f *dstPtr,
                                                RpptGenericDescPtr dstGenericDescPtr,
                                                Operation op,
                                                SIMDOperation simd_op,
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
        Rpp32u *src1length = src1BroadcastDescPtr->dims + 1;
        Rpp32u *src2length = src2BroadcastDescPtr->dims + 1;

        Rpp32u vectorIncrement = 16;

        if (broadcastNDim == 1)
        {
            Rpp32u alignedLength = length[0] & ~15;
            Rpp32u src1shape = src1length[0];
            Rpp32u src2shape = src2length[0];
            Rpp32u vectorLoopCount = 0;
            if (src1shape == 1)
            {
#if __AVX2__
                __m256 p1 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrTemp1)[0]));
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p2[2];
                    rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    p2[0] = simd_op(p1, p2[0]);
                    p2[1] = simd_op(p1, p2[1]);
                    rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p2);    // simd stores
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     *dstPtrTemp = op(*srcPtrTemp1, *srcPtrTemp2);
                     srcPtrTemp2++;
                     dstPtrTemp++;
                 }
            }
            else if (src2shape == 1)
            {
#if __AVX2__
                __m256 p2 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrTemp2)[0]));
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[2];
                    rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    p1[0] = simd_op(p1[0], p2);
                    p1[1] = simd_op(p1[1], p2);
                    rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p1);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                 for (; vectorLoopCount < length[0]; vectorLoopCount++)
                 {
                     *dstPtrTemp = op(*srcPtrTemp1, *srcPtrTemp2);
                     srcPtrTemp1++;
                     dstPtrTemp++;
                 }
            }
            else
            {
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTemp1, p1);    // simd loads
                    rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTemp2, p2);    // simd loads
                    p2[0] = simd_op(p1[0], p2[0]);
                    p2[1] = simd_op(p1[1], p2[1]);
                    rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p2);    // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0]; vectorLoopCount++)
                {
                    *dstPtrTemp = op(*srcPtrTemp1, *srcPtrTemp2);
                    srcPtrTemp1++;
                    srcPtrTemp2++;
                    dstPtrTemp++;
                }
            }
        }
        else if (broadcastNDim == 2)
        {
            Rpp32u alignedLength = length[1] & ~15;
            Rpp32u src1shape = src1length[1];
            Rpp32u src2shape = src2length[1];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrTest1 = srcPtrTemp1;
                    Rpp16f *srcPtrTest2 = srcPtrTemp2;
                    Rpp16f *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
                    __m256 p1 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrTest1)[0]));
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p2[2];
                        rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTest2, p2);    // simd loads
                        p2[0] = simd_op(p1, p2[0]);
                        p2[1] = simd_op(p1, p2[1]);
                        rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTest, p2);    // simd stores
                        srcPtrTest2 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        *dstPtrTest = op(*srcPtrTest1, *srcPtrTest2);
                        srcPtrTest2++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrTest1 = srcPtrTemp1;
                    Rpp16f *srcPtrTest2 = srcPtrTemp2;
                    Rpp16f *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
                    __m256 p2 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrTest2)[0]));
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[2];
                        rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTest1, p1);    // simd loads
                        p1[0] = simd_op(p1[0], p2);
                        p1[1] = simd_op(p1[1], p2);
                        rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTest, p1);    // simd stores
                        srcPtrTest1 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        *dstPtrTest = op(*srcPtrTest1, *srcPtrTest2);
                        srcPtrTest1++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrTest1 = srcPtrTemp1;
                    Rpp16f *srcPtrTest2 = srcPtrTemp2;
                    Rpp16f *dstPtrTest = dstPtrTemp;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p1[2], p2[2];
                        rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTest1, p1);    // simd loads
                        rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrTest2, p2);    // simd loads
                        p2[0] = simd_op(p1[0], p2[0]);
                        p2[1] = simd_op(p1[1], p2[1]);
                        rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTest, p2);    // simd stores
                        srcPtrTest1 += vectorIncrement;
                        srcPtrTest2 += vectorIncrement;
                        dstPtrTest += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[1]; vectorLoopCount++)
                    {
                        *dstPtrTest = op(*srcPtrTest1, *srcPtrTest2);
                        srcPtrTest1++;
                        srcPtrTest2++;
                        dstPtrTest++;
                    }
                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
        }
        else if (broadcastNDim == 3)
        {
            Rpp32u alignedLength = length[2] & ~15;
            Rpp32u src1shape = src1length[2];
            Rpp32u src2shape = src2length[2];
            if(src1shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrTest1 = srcPtrTemp1;
                    Rpp16f *srcPtrTest2 = srcPtrTemp2;
                    Rpp16f *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp16f *srcPtrNew1 = srcPtrTest1;
                        Rpp16f *srcPtrNew2 = srcPtrTest2;
                        Rpp16f *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;
                        __m256 p1 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrNew1)[0]));
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p2[2];
                            rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrNew2, p2);    // simd loads
                            p2[0] = simd_op(p1, p2[0]);
                            p2[1] = simd_op(p1, p2[1]);
                            rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrNew, p2);    // simd stores
                            srcPtrNew2 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            *dstPtrNew = op(*srcPtrNew1, *srcPtrNew2);
                            srcPtrNew2++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else if (src2shape == 1)
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrTest1 = srcPtrTemp1;
                    Rpp16f *srcPtrTest2 = srcPtrTemp2;
                    Rpp16f *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp16f *srcPtrNew1 = srcPtrTest1;
                        Rpp16f *srcPtrNew2 = srcPtrTest2;
                        Rpp16f *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;
                        __m256 p2 = _mm256_cvtph_ps(_mm_set1_epi16(reinterpret_cast<Rpp16s*>(srcPtrNew2)[0]));
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p1[2];
                            rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrNew1, p1);    // simd loads
                            p1[0] = simd_op(p1[0], p2);
                            p1[1] = simd_op(p1[1], p2);
                            rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrNew, p1);    // simd stores
                            srcPtrNew1 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            *dstPtrNew = op(*srcPtrNew1, *srcPtrNew2);
                            srcPtrNew1++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
            else
            {
                for (int i = 0; i < length[0]; i++)
                {
                    Rpp16f *srcPtrTest1 = srcPtrTemp1;
                    Rpp16f *srcPtrTest2 = srcPtrTemp2;
                    Rpp16f *dstPtrTest = dstPtrTemp;

                    for (int j = 0; j < length[1]; j++)
                    {
                        Rpp16f *srcPtrNew1 = srcPtrTest1;
                        Rpp16f *srcPtrNew2 = srcPtrTest2;
                        Rpp16f *dstPtrNew = dstPtrTest;

                        int vectorLoopCount = 0;
#if __AVX2__
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 p1[2], p2[2];
                            rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrNew1, p1);    // simd loads
                            rpp_simd_load(rpp_load16_f16_to_f32_avx, srcPtrNew2, p2);    // simd loads
                            p2[0] = simd_op(p1[0], p2[0]);
                            p2[1] = simd_op(p1[1], p2[1]);
                            rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrNew, p2);    // simd stores
                            srcPtrNew1 += vectorIncrement;
                            srcPtrNew2 += vectorIncrement;
                            dstPtrNew += vectorIncrement;
                        }
#endif
                        for (; vectorLoopCount < length[2]; vectorLoopCount++)
                        {
                            *dstPtrNew = op(*srcPtrNew1, *srcPtrNew2);
                            srcPtrNew1++;
                            srcPtrNew2++;
                            dstPtrNew++;
                        }

                        srcPtrTest1 += src1BroadcastDescPtr->strides[2];
                        srcPtrTest2 += src2BroadcastDescPtr->strides[2];
                        dstPtrTest += dstBroadcastDescPtr->strides[2];
                    }

                    srcPtrTemp1 += src1BroadcastDescPtr->strides[1];
                    srcPtrTemp2 += src2BroadcastDescPtr->strides[1];
                    dstPtrTemp += dstBroadcastDescPtr->strides[1];
                }
            }
        }
        else
            tensor_binary_op_recursive(srcPtrTemp1, srcPtrTemp2, src1BroadcastDescPtr->strides, src2BroadcastDescPtr->strides, dstPtrTemp, dstBroadcastDescPtr->strides, length, broadcastNDim, op);
    }

    return RPP_SUCCESS;
}

RppStatus tensor_binary_op_dispatch_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                                        Rpp16f *srcPtr2,
                                                        RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                        RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                        Rpp16f *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        RpptOp tensorOp,
                                                        Rpp32u *srcPtr1roiTensor,
                                                        Rpp32u *srcPtr2roiTensor,
                                                        rpp::Handle& handle) {
    switch(tensorOp) {
        case RPP_TENSOR_OP_ADD:
            tensor_binary_op_f16_f16_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, add_op<Rpp16f>, simd_add_ps, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_SUBTRACT:
            tensor_binary_op_f16_f16_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, subtract_op<Rpp16f>, simd_subtract_ps, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_MULTIPLY:
            tensor_binary_op_f16_f16_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, multiply_op<Rpp16f>, simd_multiply_ps, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_DIVIDE:
            tensor_binary_op_f16_f16_host_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, divide_op<Rpp16f>, simd_divide_ps, srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
    }

    return RPP_SUCCESS;
}