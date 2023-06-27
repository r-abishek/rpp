#include <hip/hip_runtime.h>
#include <omp.h>
#include "rpp_hip_common.hpp"

__device__ void fmaf_scalar_hip_compute(d_float8 *val_f8, float2 *fmaddParams_f2)
{
    val_f8->f1[0] = fmaf(val_f8->f1[0], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[1] = fmaf(val_f8->f1[1], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[2] = fmaf(val_f8->f1[2], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[3] = fmaf(val_f8->f1[3], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[4] = fmaf(val_f8->f1[4], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[5] = fmaf(val_f8->f1[5], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[6] = fmaf(val_f8->f1[6], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[7] = fmaf(val_f8->f1[7], fmaddParams_f2->x, fmaddParams_f2->y);
}

// FIRST VERSION
// __global__ void fmadd_scalar_tensor(float *srcPtr,
//                                     uint3 srcStrides012,
//                                     float *dstPtr,
//                                     uint3 dstStrides012,
//                                     int dim1Max,
//                                     float *mul,
//                                     float *add,
//                                     RpptGenericROIPtr roiGenericPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // inner most dim vectorized
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // second to inner
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // outer most dim

//     RpptGenericROI roiGenericSrc = roiGenericPtrSrc[id_z];

//     if ((id_y >= roiGenericSrc.roiLength[2]) || (id_x >= roiGenericSrc.roiLength[3]))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStrides012.x) + ((id_y + roiGenericSrc.roiBegin[2]) * srcStrides012.z) + (id_x + roiGenericSrc.roiBegin[3]);
//     uint dstIdx = (id_z * dstStrides012.x) + (id_y * dstStrides012.z) + id_x;

//     float2 fmaddParams_f2 = make_float2(mul[id_z], add[id_z]);

//     d_float8 val_f8;
//     for(int dim1 = 0; dim1 < dim1Max; dim1++)
//     {
//         rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
//         fmaf_scalar_hip_compute(&val_f8, &fmaddParams_f2);
//         rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
//         srcIdx += srcStrides012.y;
//         dstIdx += dstStrides012.y;
//     }


//     // int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;        // inner most dim vectorized
//     // int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // second to inner
//     // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // outer most dim

//     // RpptGenericROI roiGenericSrc = roiGenericPtrSrc[id_z];

//     // if ((id_y >= roiGenericSrc.roiLength[2]) || (id_x >= roiGenericSrc.roiLength[3]))
//     // {
//     //     return;
//     // }

//     // uint srcIdx = (id_z * srcStrides012.x) + ((id_y + roiGenericSrc.roiBegin[2]) * srcStrides012.z) + (id_x + roiGenericSrc.roiBegin[3]);
//     // uint dstIdx = (id_z * dstStrides012.x) + (id_y * dstStrides012.z) + id_x;

//     // float2 fmaddParams_f2 = make_float2(mul[id_z], add[id_z]);

//     // for(int dim1 = 0; dim1 < dim1Max; dim1++)
//     // {
//     //     dstPtr[dstIdx] = srcPtr[srcIdx] * fmaddParams_f2.x + fmaddParams_f2.y;
//     //     srcIdx += srcStrides012.y;
//     //     dstIdx += dstStrides012.y;
//     // }
// }






// SECOND VERSION
__global__ void fmadd_scalar_ncdhw_tensor(float *srcPtr,
                                          uint3 srcStridesCDH,
                                          float *dstPtr,
                                          uint3 dstStridesCDH,
                                          int channels,
                                          float2 fmaddParams_f2,
                                          RpptGenericROI *roiGenericSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // outer most dim

    if ((id_z >= roiGenericSrc->roiLength[2]) || (id_y >= roiGenericSrc->roiLength[3]) || (id_x >= roiGenericSrc->roiLength[4]))
    {
        return;
    }

    uint srcIdx = ((id_z + roiGenericSrc->roiBegin[2]) * srcStridesCDH.y) + ((id_y + roiGenericSrc->roiBegin[3]) * srcStridesCDH.z) + (id_x + roiGenericSrc->roiBegin[4]);
    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;

    // float2 fmaddParams_f2 = make_float2(mul[id_z], add[id_z]);

    d_float8 val_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
        fmaf_scalar_hip_compute(&val_f8, &fmaddParams_f2);
        rpp_hip_pixel_check_0to1(&val_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        srcIdx += srcStridesCDH.x;
        dstIdx += dstStridesCDH.x;
    }
}

__global__ void fmadd_scalar_ndhwc_tensor(float *srcPtr,
                                          uint3 srcStridesCDH,
                                          float *dstPtr,
                                          uint3 dstStridesCDH,
                                          int channels,
                                          float2 fmaddParams_f2,
                                          RpptGenericROI *roiGenericSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // outer most dim

    if ((id_z >= roiGenericSrc->roiLength[2]) || (id_y >= roiGenericSrc->roiLength[3]) || (id_x >= roiGenericSrc->roiLength[4]))
    {
        return;
    }

    uint srcIdx = ((id_z + roiGenericSrc->roiBegin[2]) * srcStridesCDH.y) + ((id_y + roiGenericSrc->roiBegin[3]) * srcStridesCDH.z) + (id_x + roiGenericSrc->roiBegin[4]);
    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;

    // float2 fmaddParams_f2 = make_float2(mul[id_z], add[id_z]);

    // Audio has to_decibels and another funciton that use separate code for w = 1
    // NDHWC -> // Assuming that the operation is same across all channels


    d_float8 val_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
        fmaf_scalar_hip_compute(&val_f8, &fmaddParams_f2);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        srcIdx += srcStridesCDH.x;
        dstIdx += dstStridesCDH.x;
    }
}







RppStatus hip_exec_fmadd_scalar_tensor(Rpp32f *srcPtr,
                                       RpptGenericDescPtr srcGenericDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptGenericDescPtr dstGenericDescPtr,
                                       RpptGenericROIPtr roiGenericPtrSrc,
                                       Rpp32f *mulTensor,
                                       Rpp32f *addTensor,
                                       rpp::Handle& handle)
{
    if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
    {
        int localThreads_x = LOCAL_THREADS_X;
        int localThreads_y = LOCAL_THREADS_Y;
        int localThreads_z = LOCAL_THREADS_Z;
        int globalThreads_x = (dstGenericDescPtr->strides[3] + 7) >> 3;                // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[3];                           // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[2];                           // D - depth (z direction)

        std::cerr << "\nmulTensor[0] = " << mulTensor[0];
        std::cerr << "\nmulTensor[1] = " << mulTensor[1];
        std::cerr << "\naddTensor[0] = " << addTensor[0];
        std::cerr << "\naddTensor[1] = " << addTensor[1];

        // std::cerr << "\ndims = " << srcGenericDescPtr->dims[0] << ", " << srcGenericDescPtr->dims[1] << ", " << srcGenericDescPtr->dims[2] << ", " << srcGenericDescPtr->dims[3] << ", " << srcGenericDescPtr->dims[4];
        // std::cerr << "\ndims = " << dstGenericDescPtr->dims[0] << ", " << dstGenericDescPtr->dims[1] << ", " << dstGenericDescPtr->dims[2] << ", " << dstGenericDescPtr->dims[3] << ", " << dstGenericDescPtr->dims[4];
        // std::cerr << "\nstrides = " << srcGenericDescPtr->strides[0] << ", " << srcGenericDescPtr->strides[1] << ", " << srcGenericDescPtr->strides[2] << ", " << srcGenericDescPtr->strides[3] << ", " << srcGenericDescPtr->strides[4];
        // std::cerr << "\nstrides = " << dstGenericDescPtr->strides[0] << ", " << dstGenericDescPtr->strides[1] << ", " << dstGenericDescPtr->strides[2] << ", " << dstGenericDescPtr->strides[3] << ", " << dstGenericDescPtr->strides[4];

        // Rpp32f *mulParams = handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem;
        // Rpp32f *addParams = handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem;

//         omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstGenericDescPtr->dims[0])
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)  // N - batch
        {
            // std::cerr << "\nLaunching " << batchCount;
            hipLaunchKernelGGL(fmadd_scalar_ncdhw_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint3(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2], srcGenericDescPtr->strides[3]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                               dstGenericDescPtr->dims[1],                              // C - channel
                            //    make_float2(mulParams[batchCount], addParams[batchCount]),
                               make_float2(mulTensor[batchCount], addTensor[batchCount]),
                               &roiGenericPtrSrc[batchCount]);
        }
    }

    return RPP_SUCCESS;
}
