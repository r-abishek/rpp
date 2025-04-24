#include "host_tensor_executors.hpp"
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
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = dstGenericDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *roi = srcPtr1roiTensor + batchCount * nDim * 2;
        Rpp32u *length = &roi[nDim];

        Rpp32f *srcPtrTemp1 = srcPtr1 + batchCount * srcPtr1GenericDescPtr->strides[0];
        Rpp32f *srcPtrTemp2 = srcPtr2 + batchCount * srcPtr2GenericDescPtr->strides[0];
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];
        tensor_add_tensor_recursive(srcPtrTemp1, srcPtrTemp2, srcPtr1GenericDescPtr->strides, srcPtr2GenericDescPtr->strides, dstPtrTemp, dstGenericDescPtr->strides, length, nDim);
    }

    return RPP_SUCCESS;
}