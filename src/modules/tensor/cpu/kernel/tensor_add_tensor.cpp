#include "host_tensor_executors.hpp"
#include "broadcast.hpp"
#include "rpp_cpu_simd_math.hpp"

void tensor_add_tensor_recursive(Rpp32f *src1, Rpp32f *src2, Rpp32u *src1Strides, Rpp32u *src2Strides, Rpp32f *dst, Rpp32u *dstStrides, Rpp32u *dstShape, Rpp32u nDim)
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
    printf("src1 : ");
    for(int i1 = 0; i1 < srcPtr1GenericDescPtr->numDims; i1++) {
        printf("%d ", srcPtr1GenericDescPtr->dims[i1]);
    }
    printf("\n");
    printf("src2 : ");
    for(int i1 = 0; i1 < srcPtr2GenericDescPtr->numDims; i1++) {
        printf("%d ", srcPtr2GenericDescPtr->dims[i1]);
    }
    printf("\n");
    printf("dst : ");
    for(int i1 = 0; i1 < dstGenericDescPtr->numDims; i1++) {
        printf("%d ", dstGenericDescPtr->dims[i1]);
    }
    printf("\n");
    printf("src1 : ");
    for(int i1 = 0; i1 < src1BroadcastDescPtr->numDims; i1++) {
        printf("%d ", src1BroadcastDescPtr->dims[i1]);
    }
    printf("\n");
    printf("src2 : ");
    for(int i1 = 0; i1 < src2BroadcastDescPtr->numDims; i1++) {
        printf("%d ", src2BroadcastDescPtr->dims[i1]);
    }
    printf("\n");
    printf("dst : ");
    for(int i1 = 0; i1 < dstBroadcastDescPtr->numDims; i1++) {
        printf("%d ", dstBroadcastDescPtr->dims[i1]);
    }
    printf("\n");
    GroupShapes(src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr);
    printf("src1 : ");
    for(int i1 = 0; i1 < src1BroadcastDescPtr->numDims; i1++) {
        printf("%d ", src1BroadcastDescPtr->dims[i1]);
    }
    printf("\n");
    printf("src2 : ");
    for(int i1 = 0; i1 < src2BroadcastDescPtr->numDims; i1++) {
        printf("%d ", src2BroadcastDescPtr->dims[i1]);
    }
    printf("\n");
    printf("dst : ");
    for(int i1 = 0; i1 < dstBroadcastDescPtr->numDims; i1++) {
        printf("%d ", dstBroadcastDescPtr->dims[i1]);
    }
    printf("\n");
    StridesForBroadcasting(src1BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src2BroadcastDescPtr, dstBroadcastDescPtr);
    printf("Strides src1 : ");
    for(int i1 = 0; i1 < src1BroadcastDescPtr->numDims; i1++) {
        printf("%d ", src1BroadcastDescPtr->strides[i1]);
    }
    printf("\n");
    printf("Strides src2 : ");
    for(int i1 = 0; i1 < src2BroadcastDescPtr->numDims; i1++) {
        printf("%d ", src2BroadcastDescPtr->strides[i1]);
    }
    printf("\n");
    printf("Strides dst : ");
    for(int i1 = 0; i1 < dstBroadcastDescPtr->numDims; i1++) {
        printf("%d ", dstBroadcastDescPtr->strides[i1]);
    }
    printf("\n");
    //exit(0);

    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = dstBroadcastDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstBroadcastDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        //Rpp32u *roi = srcPtr1roiTensor + batchCount * nDim * 2;
        Rpp32u *length = dstBroadcastDescPtr->dims + 1;

        Rpp32f *srcPtrTemp1 = srcPtr1 + batchCount * src1BroadcastDescPtr->strides[0];
        Rpp32f *srcPtrTemp2 = srcPtr2 + batchCount * src2BroadcastDescPtr->strides[0];
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstBroadcastDescPtr->strides[0];
        tensor_add_tensor_recursive(srcPtrTemp1, srcPtrTemp2, src1BroadcastDescPtr->strides, src2BroadcastDescPtr->strides, dstPtrTemp, dstBroadcastDescPtr->strides, length, nDim);
    }

    return RPP_SUCCESS;
}