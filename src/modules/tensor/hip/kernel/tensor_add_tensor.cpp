#include "hip_tensor_executors.hpp"
#include "rpp_hip_math.hpp"

template <typename T>
__global__ void tensor_add_tensor_nd_hip_tensor(T *srcPtr1,
                                                T *srcPtr2,
                                                uint *srcStrides1,
                                                uint *srcStrides2,
                                                uint *srcDims1,
                                                uint *srcDims2,
                                                uint numDims,
                                                T *dstPtr,
                                                uint *dstStrides,
                                                uint *dstDims,
                                                Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    if(id_x >= dstStrides[0])
        return;

    uint *roi = roiTensor + id_z * numDims * 2;
    uint *begin = roi;
    uint *length = &roi[numDims];
    uint dstIdx = (id_z * *dstStrides++);
    uint srcIdx1 = (id_z * *srcStrides1++);
    uint srcIdx2 = (id_z * *srcStrides2++);
    uint srcCoords1[RPPT_MAX_DIMS], srcCoords2[RPPT_MAX_DIMS], dstCoords[RPPT_MAX_DIMS];

    for (int i = 0; i < numDims; i++)
    {
        srcCoords1[i] = (id_x / srcStrides1[i]) % srcDims1[i];
        srcCoords2[i] = (id_x / srcStrides2[i]) % srcDims2[i];
        dstCoords[i] = (id_x / dstStrides[i]) % dstDims[i];
        if(dstCoords[i] >= length[i])
            return;
    }

    for (int i = 0; i < numDims; i++)
    {
        dstIdx += (dstCoords[i] * dstStrides[i]);
        srcIdx1 += (begin[i] + (srcCoords1[i] * srcStrides1[i]));
        srcIdx2 += (begin[i] + (srcCoords2[i] * srcStrides2[i]));
    }

    d_float8 src1_f8, src2_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src1_f8);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src2_f8);
    rpp_hip_math_add8(&src1_f8, &src2_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
RppStatus hip_exec_tensor_add_tensor_generic_tensor(T *srcPtr1,
                                                    T *srcPtr2,
                                                    RpptGenericDescPtr srcGenericDescPtr1,
                                                    RpptGenericDescPtr srcGenericDescPtr2,
                                                    T *dstPtr,
                                                    RpptGenericDescPtr dstGenericDescPtr,
                                                    uint *roiTensor1,
                                                    uint *roiTensor2,
                                                    rpp::Handle& handle)
{
    Rpp32u numDims = srcGenericDescPtr1->numDims - 1; // exclude batchsize from input dims

    // interpret the input as 1D tensor
    int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3;
    int globalThreads_y = 1;
    int globalThreads_z = dstGenericDescPtr->dims[0];

    hipLaunchKernelGGL(tensor_add_tensor_nd_hip_tensor,
                        dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                        dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                        0,
                        handle.GetStream(),
                        srcPtr1,
                        srcPtr2,
                        srcGenericDescPtr1->strides,
                        srcGenericDescPtr2->strides,
                        srcGenericDescPtr1->dims + 1,
                        srcGenericDescPtr2->dims + 1,
                        srcGenericDescPtr1->numDims - 1,
                        dstPtr,
                        dstGenericDescPtr->strides,
                        dstGenericDescPtr->dims + 1,
                        roiTensor1);

    return RPP_SUCCESS;
}

template RppStatus hip_exec_tensor_add_tensor_generic_tensor<Rpp8u>(Rpp8u*,
                                                                    Rpp8u*,
                                                                    RpptGenericDescPtr,
                                                                    RpptGenericDescPtr,
                                                                    Rpp8u*,
                                                                    RpptGenericDescPtr,
                                                                    Rpp32u*,
                                                                    Rpp32u*,
                                                                    rpp::Handle&);

template RppStatus hip_exec_tensor_add_tensor_generic_tensor<Rpp32f>(Rpp32f*,
                                                                     Rpp32f*,
                                                                     RpptGenericDescPtr,
                                                                     RpptGenericDescPtr,
                                                                     Rpp32f*,
                                                                     RpptGenericDescPtr,
                                                                     Rpp32u*,
                                                                     Rpp32u*,
                                                                     rpp::Handle&);

template RppStatus hip_exec_tensor_add_tensor_generic_tensor<half>(half*,
                                                                   half*,
                                                                   RpptGenericDescPtr,
                                                                   RpptGenericDescPtr,
                                                                   half*,
                                                                   RpptGenericDescPtr,
                                                                   Rpp32u*,
                                                                   Rpp32u*,
                                                                   rpp::Handle&);

template RppStatus hip_exec_tensor_add_tensor_generic_tensor<Rpp8s>(Rpp8s*,
                                                                    Rpp8s*,
                                                                    RpptGenericDescPtr,
                                                                    RpptGenericDescPtr,
                                                                    Rpp8s*,
                                                                    RpptGenericDescPtr,
                                                                    Rpp32u*,
                                                                    Rpp32u*,
                                                                    rpp::Handle&);