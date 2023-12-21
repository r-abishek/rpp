#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ __forceinline__ uint rpp_hip_mod(uint a, uint b)
{
    return (a % b);
}

__device__ uint compute_2d_paramindex(uint y, uint x, uint *paramShape, uint *paramStrides)
{
    uint yFactor =  ((paramShape[0] > 1)) ? (rpp_hip_mod(y, paramShape[0])) * paramStrides[0] : 0;
    uint xFactor =  ((paramShape[1] > 1)) ? (rpp_hip_mod(x, paramShape[1])) * paramStrides[1] : 0;
    uint paramIndex = yFactor + xFactor;
    return paramIndex;
}

__global__ void compute_mean_2d_hip_tensor(float *srcPtr,
                                           uint2 srcStridesNH,
                                           float *meanTensor,
                                           uint *roiTensor,
                                           uint maxParamVolume,
                                           uint axisMask)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * 4 + 2];
    uint height = roi[0];
    uint width = roi[1];
    if(axisMask == 1)
    {
        uint stride = srcStridesNH.y;
        uint srcIdx = id_z * srcStridesNH.x + id_x;
        uint dstIdx = id_z * maxParamVolume + id_x;
        if(id_x < width)
        {
            float accum = 0.0f;
            for(int i = 0; i < height; i++)
            {
                accum += srcPtr[srcIdx];
                srcIdx += stride;
            }
            meanTensor[dstIdx] = accum / static_cast<float>(height);
        }
    }
    else if(axisMask == 2)
    {
        uint stride = 1;
        uint srcIdx = id_z * srcStridesNH.x + id_x * srcStridesNH.y;
        uint dstIdx = id_z * maxParamVolume + id_x;
        if(id_x < height)
        {
            float accum = 0.0f;
            for(int i = 0; i < width; i++)
            {
                accum += srcPtr[srcIdx];
                srcIdx += stride;
            }
            meanTensor[dstIdx] = accum / static_cast<float>(width);
        }
    }
}

__global__ void normalize_2d_hip_tensor(float *srcPtr,
                                        uint2 srcStridesNH,
                                        float *dstPtr,
                                        uint2 dstStridesNH,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float scale,
                                        float shift,
                                        uint *roiTensor,
                                        uint *paramShapeTensor,
                                        uint *paramStridesTensor,
                                        uint maxParamVolume)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    uint *roi = &roiTensor[id_z * 4 + 2];
    uint height = roi[0];
    uint width = roi[1];

    if (id_x >= width || id_y >= height)
        return;

    uint *paramShape = &paramShapeTensor[id_z * 2];
    uint *paramStrides = &paramStridesTensor[id_z * 2];
    uint paramIndex = id_z * maxParamVolume;
    paramIndex += (maxParamVolume == 1) ? 0 : compute_2d_paramindex(id_y, id_x, paramShape, paramStrides);

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    float mean = meanTensor[paramIndex];
    float stdDev = stdDevTensor[paramIndex];
    float stdDevSquare = stdDev * stdDev;
    float invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    dstPtr[dstIdx] = fmaf((srcPtr[srcIdx] - mean), invStdDev, shift);
}

// __device__ void normalize_hip_compute(d_float8 *data_f8, d_float8 *mean_f8, d_float8 *invStdDev_f8, d_float8 *shift_f8)
// {
//     data_f8->f4[0] = ((data_f8->f4[0] - mean_f8->f4[0]) * invStdDev_f8->f4[1]) + shift_f8->f4[0];
//     data_f8->f4[1] = ((data_f8->f4[1] - mean_f8->f4[1]) * invStdDev_f8->f4[1]) + shift_f8->f4[1];
// }

// __device__ void load_normalize_params(d_uint8 *locParam_ui8, float *meanPtr, float *stdDevPtr, float scale, d_float8 *mean_f8,  d_float8 *invStdDev_f8)
// {
//     for(int i = 0; i < 8; i++)
//     {
//         mean_f8->f1[i] = meanPtr[locParam_ui8->ui1[i]];
//         float stdDev = stdDevPtr[locParam_ui8->ui1[i]];
//         float stdDevSquare = stdDev * stdDev;
//         float invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
//         invStdDev_f8->f1[i] = invStdDev;
//     }
// }

// __device__ void normalize_2d_paramlocs_hip_compute(uint id_y, uint id_x, d_uint8 *locParam_ui8, uint *paramShape, uint *paramStrides)
// {
//     d_uint8 increment_ui8, locDstx_ui8;
//     increment_ui8.ui4[0] = make_uint4(0, 1, 2, 3);
//     increment_ui8.ui4[1] = make_uint4(4, 5, 6, 7);
//     locDstx_ui8.ui4[0] = static_cast<uint4>(id_x) + increment_ui8.ui4[0];
//     locDstx_ui8.ui4[1] = static_cast<uint4>(id_x) + increment_ui8.ui4[1];

//     locParam_ui8->ui1[0] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[0], paramShape, paramStrides);
//     locParam_ui8->ui1[1] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[1], paramShape, paramStrides);
//     locParam_ui8->ui1[2] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[2], paramShape, paramStrides);
//     locParam_ui8->ui1[3] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[3], paramShape, paramStrides);
//     locParam_ui8->ui1[4] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[4], paramShape, paramStrides);
//     locParam_ui8->ui1[5] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[5], paramShape, paramStrides);
//     locParam_ui8->ui1[6] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[6], paramShape, paramStrides);
//     locParam_ui8->ui1[7] = compute_2d_paramindex(id_y, locDstx_ui8.ui1[7], paramShape, paramStrides);
// }

// __global__ void normalize_2d_hip_tensor(float *srcPtr,
//                                         uint2 srcStridesNH,
//                                         float *dstPtr,
//                                         uint2 dstStridesNH,
//                                         float *meanTensor,
//                                         float *stdDevTensor,
//                                         float scale,
//                                         float shift,
//                                         uint *roiTensor,
//                                         uint *paramShapeTensor,
//                                         uint *paramStridesTensor,
//                                         uint maxParamVolume)
// {
//     uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
//     uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
//     uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

//     uint *roi = &roiTensor[id_z * 4 + 2];
//     uint height = roi[0];
//     uint width = roi[1];

//     if (id_x >= width || id_y >= height)
//         return;

//     uint *paramShape = &paramShapeTensor[id_z * 2];
//     uint *paramStrides = &paramStridesTensor[id_z * 2];

//     d_uint8 locParam_ui8;
//     normalize_2d_paramlocs_hip_compute(id_y, id_x, &locParam_ui8, paramShape, paramStrides);

//     d_float8 mean_f8, invStdDev_f8, shift_f8;
//     float *meanPtr = &meanTensor[id_z * maxParamVolume];
//     float *stdDevPtr = &stdDevTensor[id_z * maxParamVolume];
//     load_normalize_params(&locParam_ui8, meanPtr, stdDevPtr, scale, &mean_f8, &invStdDev_f8);
//     shift_f8.f4[0] = static_cast<float4>(shift);
//     shift_f8.f4[1] = shift_f8.f4[0];

//     d_float8 data_f8;
//     uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
//     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &data_f8);
//     normalize_hip_compute(&data_f8, &mean_f8, &invStdDev_f8, &shift_f8);
//     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &data_f8);
// }

__device__ uint compute_3d_paramindex(uint z, uint y, uint x, uint *paramShape, uint *paramStrides)
{
    uint zFactor =  ((paramShape[0] > 1)) ? rpp_hip_mod(z, paramShape[0]) * paramStrides[0] : 0;
    uint yFactor =  ((paramShape[1] > 1)) ? rpp_hip_mod(y, paramShape[1]) * paramStrides[1] : 0;
    uint xFactor =  ((paramShape[2] > 1)) ? rpp_hip_mod(x, paramShape[2]) * paramStrides[2] : 0;
    uint paramIndex = zFactor + yFactor + xFactor;
    return paramIndex;
}

__global__ void normalize_3d_hip_tensor(float *srcPtr,
                                        uint2 srcStridesDH,
                                        float *dstPtr,
                                        uint2 dstStridesDH,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float scale,
                                        float shift,
                                        uint *roiTensor,
                                        uint *paramShapeTensor,
                                        uint *paramStridesTensor,
                                        uint maxParamVolume)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // depth

    uint *roi = roiTensor;
    uint width = roi[2];
    uint height = roi[1];
    uint depth = roi[0];

    if (id_x >= width || id_y >= height || id_z >= depth)
        return;

    uint *paramShape = paramShapeTensor;
    uint *paramStrides = paramStridesTensor;
    uint paramIndex = id_z * maxParamVolume;
    paramIndex += (maxParamVolume == 1) ? 0 : compute_3d_paramindex(id_z, id_y, id_x, paramShape, paramStrides);

    uint srcIdx = (id_z * srcStridesDH.x) + (id_y * srcStridesDH.y) + id_x;
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x;
    float mean = meanTensor[paramIndex];
    float stdDev = stdDevTensor[paramIndex];
    float stdDevSquare = stdDev * stdDev;
    float invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    dstPtr[dstIdx] = fmaf((srcPtr[srcIdx] - mean), invStdDev, shift);
}

__global__ void normalize_nd_hip_tensor(float *srcPtr,
                                        uint *srcMaxDims,
                                        uint *srcStrides,
                                        float *dstPtr,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float scale,
                                        float shift,
                                        uint *roiTensor,
                                        uint *paramShapeTensor,
                                        uint *paramStridesTensor,
                                        uint maxParamVolume,
                                        uint numDims)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * numDims * 2 + numDims];
    uint *paramShape = &paramShapeTensor[id_z * numDims];
    uint *paramStrides = &paramStridesTensor[id_z * numDims];

    uint paramIndex = id_z * maxParamVolume;
    if (maxParamVolume != 1)
    {
        for (int i = 0; i < numDims; i++)
        {
            uint coord = id_x / srcStrides[i] % srcMaxDims[i];
            if(coord > roi[i])
                return;
            paramIndex += rpp_hip_mod(coord, paramShape[i]) * paramStrides[i];
        }
    }

    float mean = meanTensor[paramIndex];
    float stdDev = stdDevTensor[paramIndex];
    float stdDevSquare = stdDev * stdDev;
    float invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    dstPtr[id_x] = fmaf((srcPtr[id_x] - mean), invStdDev, shift);
}

void normalize_setup(Rpp32u *roiTensor, Rpp32u batchSize, Rpp32u numDims, Rpp32u axisMask,
                     Rpp32u *paramShapeTensor, Rpp32u *paramStridesTensor, Rpp32u &maxParamVolume)
{
    maxParamVolume = 1;
    uint axisSet[RPPT_MAX_DIMS];
    for(int i = 0; i < numDims; i++)
        axisSet[i] = ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : 0;

    for(uint i = 0; i < batchSize; i++)
    {
        // calculate the param shape and param volume based on the axis mask
        Rpp32u paramVolume = 1;
        Rpp32u *roi = &roiTensor[numDims * 2 * i + numDims];
        Rpp32u *paramShape = &paramShapeTensor[i * numDims];
        for(uint j = 0; j < numDims; j++)
        {
            paramShape[j] = (axisSet[j]) ? 1 : roi[j];
            paramVolume *= paramShape[j];
        }
        maxParamVolume = std::max(maxParamVolume, paramVolume);

        // calculate the param strides from the param shape
        Rpp32u *paramStrides = &paramStridesTensor[i * numDims];
        Rpp32u val = 1;
        for(uint j = numDims - 1; j > 0; j--)
        {
            paramStrides[j] = val;
            val *= paramShape[j];
        }
        paramStrides[0] = val;
    }
}

__global__ void reduce_final_result_hip(float *partialSumTensor,
                                        uint numPartialSums,
                                        float *meanTensor,
                                        uint *roiTensor,
                                        uint axisMask,
                                        uint numDims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint *roi = &roiTensor[id_z * numDims * 2 + numDims];

    uint meanFactor;
    if(numDims == 3)
    {
        uint lengthZ = roi[0];
        uint lengthY = roi[1];
        uint lengthX = roi[2];

        if(axisMask == 3)
            meanFactor = lengthZ * lengthY;
        else if(axisMask == 6)
            meanFactor = lengthY * lengthX;
        else if(axisMask == 7)
            meanFactor = lengthZ * lengthY * lengthX;
    }
    else if(numDims == 2)
    {
        uint lengthY = roi[0];
        uint lengthX = roi[1];

        if(axisMask == 2)
            meanFactor = lengthX;
        else if(axisMask == 3)
            meanFactor = lengthY * lengthX;
    }

    __shared__ float partialSum_smem[16];
    partialSum_smem[hipThreadIdx_x] = 0.0f;

    float accum = 0.0f;
    while(id_x < numPartialSums)
    {
        uint srcIdx = (id_z * hipGridDim_y * numPartialSums) + (id_y * numPartialSums) + id_x;
        accum += partialSumTensor[srcIdx];
        id_x += hipBlockDim_x;
    }
    partialSum_smem[hipThreadIdx_x] = accum;
    __syncthreads();

    // Now do block level reduction sum
    for(int i = 8; i >= 1; i >>= 1)
    {
        if(hipThreadIdx_x < i)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + i];
        __syncthreads();
    }

    // Final store to dst
    if(hipThreadIdx_x == 0)
        meanTensor[id_z * hipGridDim_y + id_y] = partialSum_smem[0] / meanFactor;
}

__global__ void compute_mean_2d_hip_tensor(float *srcPtr,
                                           uint2 srcStridesNH,
                                           float *meanTensor,
                                           float *partialSumTensor,
                                           uint *roiTensor,
                                           uint maxParamVolume,
                                           uint axisMask)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * 4 + 2];
    uint height = roi[0];
    uint width = roi[1];

    // perform column wise sum
    if(axisMask == 1)
    {
        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        uint srcIdx = id_z * srcStridesNH.x + id_x;
        uint dstIdx = id_z * maxParamVolume + id_x;
        if(id_x < width)
        {
            float accum = 0.0f;
            for(int i = 0; i < height; i++)
            {
                accum += srcPtr[srcIdx];
                srcIdx += srcStridesNH.y;
            }
            meanTensor[dstIdx] = accum / static_cast<float>(height);
        }
    }
    // perform row wise sum
    else if(axisMask == 2)
    {
        id_x *= 8;
        __shared__ float partialRowSum_smem[256];
        partialRowSum_smem[hipThreadIdx_x] = 0.0f;

        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        int xAlignedLength =  width & ~7;      // alignedLength for vectorized global loads
        int xDiff = width - xAlignedLength;    // difference between roiWidth and alignedLength
        uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        if (id_x + 8 > width)
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
        src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
        partialRowSum_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                              src_f8.f1[1] +
                                              src_f8.f1[2] +
                                              src_f8.f1[3]);                                // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();

        // Now do block level reduction sum
        for(int i = 128; i > 0; i >>= 1)
        {
            if(hipThreadIdx_x < i)
                partialRowSum_smem[hipThreadIdx_x] += partialRowSum_smem[hipThreadIdx_x + i];
            __syncthreads();
        }

        // Final store to dst
        if(hipThreadIdx_x == 0)
        {
            uint paramIndex = (id_z * hipGridDim_y * hipGridDim_x) + (id_y * hipGridDim_x) + hipBlockIdx_x;
            partialSumTensor[paramIndex] = partialRowSum_smem[0];
        }
    }
    else if(axisMask == 3)
    {
        id_x *= 8;
        __shared__ float sh_mem[16][16];                               // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
        float *partialSumRowPtr_smem = &sh_mem[hipThreadIdx_y][0];     // float pointer to beginning of each row in Shared
        partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                           // initialization of Shared to 0.0f using all 16 x 16 threads

        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        int xAlignedLength =  width & ~7;       // alignedLength for vectorized global loads
        int xDiff = width - xAlignedLength;    // difference between roiWidth and alignedLength
        uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        if (id_x + 8 > width)
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
        src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
        partialSumRowPtr_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                                src_f8.f1[1] +
                                                src_f8.f1[2] +
                                                src_f8.f1[3]);                 // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();                                                        // syncthreads after Shared load

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
        {
            if (hipThreadIdx_x < threadMax)
                partialSumRowPtr_smem[hipThreadIdx_x] += partialSumRowPtr_smem[hipThreadIdx_x + threadMax];
            __syncthreads();
        }

        if (hipThreadIdx_x == 0)
        {
            // Reduction of 16 floats on 16 threads per block in y dimension
            for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
            {
                if (hipThreadIdx_y < threadMax)
                    partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                __syncthreads();
            }

            // Final store to dst
            if (hipThreadIdx_y == 0)
                partialSumTensor[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumRowPtr_smem[0];
        }
    }
}


__global__ void compute_mean_3d_hip_tensor(float *srcPtr,
                                           uint3 srcStridesNZY,
                                           float *meanTensor,
                                           uint *roiTensor,
                                           float *partialSumTensor,
                                           uint maxParamVolume,
                                           uint axisMask)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * 6 + 3];
    uint lengthZ = roi[0];
    uint lengthY = roi[1];
    uint lengthX = roi[2];

    // compute mean along z direction
    if(axisMask == 1)
    {
        if(id_x >= lengthX && id_y >= lengthY)
            return;

        uint srcIdx = id_z * srcStridesNZY.x + id_y * srcStridesNZY.z + id_x;
        uint dstIdx = id_z * maxParamVolume + id_y * lengthX + id_x;
        float accum = 0.0f;
        for(uint i = 0; i < lengthZ; i++)
        {
            accum += srcPtr[srcIdx];
            srcIdx += srcStridesNZY.y;
        }
        meanTensor[dstIdx] = accum / static_cast<float>(lengthZ);
    }
    // compute mean along y direction
    else if(axisMask == 2)
    {
        if(id_x >= lengthX && id_y >= lengthZ)
            return;

        uint srcIdx = id_z * srcStridesNZY.x + id_y * srcStridesNZY.y + id_x;
        uint dstIdx = id_z * maxParamVolume +  id_y * lengthX + id_x;
        float accum = 0.0f;
        for(uint i = 0; i < lengthY; i++)
        {
            accum += srcPtr[srcIdx];
            srcIdx += srcStridesNZY.z;
        }
        meanTensor[dstIdx] = accum / static_cast<float>(lengthY);
    }
    // compute mean along x direction
    else if(axisMask == 4)
    {
        __shared__ float sh_mem[32];
        sh_mem[hipThreadIdx_x] = 0.0f;

        if(hipBlockIdx_x >= lengthY || hipBlockIdx_y >= lengthZ)
            return;

        uint srcIdx = id_z * srcStridesNZY.x + hipBlockIdx_x * srcStridesNZY.z + hipBlockIdx_y * srcStridesNZY.y + hipThreadIdx_x;
        uint dstIdx = id_z * maxParamVolume +  hipBlockIdx_x + hipBlockIdx_y * lengthY;
        uint tid_x = hipThreadIdx_x;

        // perform block strided sum
        float accum = 0.0f;
        while(tid_x < lengthX)
        {
            accum += srcPtr[srcIdx];
            srcIdx += hipBlockDim_x;
            tid_x += hipBlockDim_x;
        }
        sh_mem[hipThreadIdx_x] = accum;
        __syncthreads();

        // perform reduction on shared memory sums
        for(int i = hipBlockDim_x / 2; i >= 1 ; i /= 2)
        {
            if(hipThreadIdx_x < i)
                sh_mem[hipThreadIdx_x] += sh_mem[hipThreadIdx_x + i];
            __syncthreads();
        }

        if(hipThreadIdx_x == 0)
            meanTensor[dstIdx] = sh_mem[0] / static_cast<float>(lengthX);
    }
    // compute mean along x-z direction
    else if(axisMask == 5)
    {
        __shared__ float sh_mem[32];
        sh_mem[hipThreadIdx_x] = 0.0f;

        if(hipBlockIdx_x >= lengthY)
            return;

        uint dstIdx = id_z * maxParamVolume +  hipBlockIdx_x;
        float accum = 0.0f;
        for (uint i = 0; i < lengthZ; i++)
        {
            uint tid_x = hipThreadIdx_x;
            uint srcIdx = (id_z * srcStridesNZY.x) + (i * srcStridesNZY.y) + (hipBlockIdx_x * srcStridesNZY.z);
            while (tid_x < lengthX)
            {
                accum += srcPtr[srcIdx + tid_x];
                tid_x += hipBlockDim_x;
            }
        }
        sh_mem[hipThreadIdx_x] = accum;
        __syncthreads();

        // perform reduction on shared memory sums
        for(int i = hipBlockDim_x / 2; i >= 1 ; i /= 2)
        {
            if(hipThreadIdx_x < i)
                sh_mem[hipThreadIdx_x] += sh_mem[hipThreadIdx_x + i];
            __syncthreads();
        }

        if(hipThreadIdx_x == 0)
            meanTensor[dstIdx] = sh_mem[0] / static_cast<float>(lengthX * lengthZ);
    }
    // compute mean along x-y direction
    else if(axisMask == 6 || axisMask == 7)
    {
        __shared__ float sh_mem[16][16];                                        // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
        float *partialSumRowPtr_smem = &sh_mem[hipThreadIdx_y][0];              // float pointer to beginning of each row in Shared
        partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                           // initialization of Shared to 0.0f using all 16 x 16 threads

        const uint maxLengthZ = srcStridesNZY.x / srcStridesNZY.y;
        uint z_index = id_z % maxLengthZ;
        uint batchIndex = id_z / maxLengthZ;

        roi = &roiTensor[batchIndex * 6 + 3];
        lengthZ = roi[0];
        lengthY = roi[1];
        lengthX = roi[2];
        if ((id_x >= lengthX) || (id_y >= lengthY) || (z_index >= lengthZ))
        {
            return;
        }

        uint srcIdx = (batchIndex * srcStridesNZY.x) + (z_index * srcStridesNZY.y) + (id_y * srcStridesNZY.z) + id_x;                                           // perform small work of vectorized float4 addition
        partialSumRowPtr_smem[hipThreadIdx_x] = srcPtr[srcIdx];                               // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();                                                                      // syncthreads after Shared load

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
        {
            if (hipThreadIdx_x < threadMax)
                partialSumRowPtr_smem[hipThreadIdx_x] += partialSumRowPtr_smem[hipThreadIdx_x + threadMax];
            __syncthreads();
        }

        if (hipThreadIdx_x == 0)
        {
            // Reduction of 16 floats on 16 threads per block in y dimension
            for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
            {
                if (hipThreadIdx_y < threadMax)
                    partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                __syncthreads();
            }

            // Final store to dst
            if (hipThreadIdx_y == 0)
            {
                uint dstIdx = (id_z * hipGridDim_y * hipGridDim_x) + (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x);
                partialSumTensor[dstIdx] = partialSumRowPtr_smem[0];
            }
        }
    }
    // compute mean along y-z direction
    else if(axisMask == 3)
    {
        __shared__ float sh_mem[16][16];                                        // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
        float *partialSumRowPtr_smem = &sh_mem[hipThreadIdx_z][0];              // float pointer to beginning of each row in Shared
        partialSumRowPtr_smem[hipThreadIdx_y] = 0.0f;                           // initialization of Shared to 0.0f using all 16 x 16 threads

        const uint maxLengthX = srcStridesNZY.z;
        uint x_index = id_x % maxLengthX;
        uint batchIndex = id_x / maxLengthX;

        roi = &roiTensor[batchIndex * 6 + 3];
        lengthZ = roi[0];
        lengthY = roi[1];
        lengthX = roi[2];
        if ((x_index >= lengthX) || (id_y >= lengthY) || (id_z >= lengthZ))
        {
            return;
        }

        uint srcIdx = (batchIndex * srcStridesNZY.x) + (id_z * srcStridesNZY.y) + (id_y * srcStridesNZY.z) + x_index;                                           // perform small work of vectorized float4 addition
        partialSumRowPtr_smem[hipThreadIdx_y] = srcPtr[srcIdx];                               // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();                                                                      // syncthreads after Shared load

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialSumRowPtr_smem[hipThreadIdx_y] += partialSumRowPtr_smem[hipThreadIdx_y + threadMax];
            __syncthreads();
        }

        if (hipThreadIdx_y == 0)
        {
            // Reduction of 16 floats on 16 threads per block in z dimension
            for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
            {
                if (hipThreadIdx_z < threadMax)
                    partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                __syncthreads();
            }

            // Final store to dst
            if (hipThreadIdx_z == 0)
            {
                uint dstIdx = (id_x * hipGridDim_z * hipGridDim_y) + (hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y);
                partialSumTensor[dstIdx] = partialSumRowPtr_smem[0];
            }
        }
    }
}

void set_kernel_launch_config_2d(RpptGenericDescPtr srcGenericDescPtr,
                                 int &globalThreads_x,
                                 int &globalThreads_y,
                                 int &globalThreads_z,
                                 int &localThreads_x,
                                 int &localThreads_y,
                                 int &localThreads_z,
                                 Rpp32u axisMask,
                                 Rpp32f *partialSumArr,
                                 rpp::Handle& handle)
{
    switch (axisMask)
    {
        // compute along Y direction
        case 1:
        {
            localThreads_x = 256;
            localThreads_y = 1;
            localThreads_z = 1;
            globalThreads_x = static_cast<int>(ceil((float)srcGenericDescPtr->dims[2] / localThreads_x));
            globalThreads_y = 1;
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along X direction
        case 2:
        {
            localThreads_x = 256;
            localThreads_y = 1;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)((srcGenericDescPtr->dims[2] + 7) >> 3) / 256));
            globalThreads_y = srcGenericDescPtr->dims[1];
            globalThreads_z = srcGenericDescPtr->dims[0];

            Rpp32u partialSumArrLength = srcGenericDescPtr->dims[0] * srcGenericDescPtr->dims[1] * globalThreads_x;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
        // compute along XY direction
        case 3:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)((srcGenericDescPtr->dims[2] + 7) >> 3) / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[1] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];

            Rpp32u partialSumArrLength = globalThreads_x * globalThreads_y * globalThreads_z;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
    }
}

void set_kernel_launch_config_3d(RpptGenericDescPtr srcGenericDescPtr,
                                 int &globalThreads_x,
                                 int &globalThreads_y,
                                 int &globalThreads_z,
                                 int &localThreads_x,
                                 int &localThreads_y,
                                 int &localThreads_z,
                                 Rpp32u axisMask,
                                 Rpp32f *partialSumArr,
                                 rpp::Handle& handle)
{
    switch (axisMask)
    {
        // compute along Z direction
        case 1:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[3] / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[2] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along Y direction
        case 2:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[3] / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[1] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along YZ direction
        case 3:
        {
            localThreads_x = 1;
            localThreads_y = 16;
            localThreads_z = 16;
            globalThreads_x = srcGenericDescPtr->dims[0] * srcGenericDescPtr->dims[3];
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[2] / localThreads_y));
            globalThreads_z = static_cast<int> (ceil((float)srcGenericDescPtr->dims[1] / localThreads_z));

            Rpp32u partialSumArrLength = globalThreads_x * globalThreads_y * globalThreads_z;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
        // compute along X direction
        case 4:
        {
            localThreads_x = 32;
            localThreads_y = 1;
            localThreads_z = 1;
            globalThreads_x = srcGenericDescPtr->dims[2];
            globalThreads_y = srcGenericDescPtr->dims[1];
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along XZ direction
        case 5:
        {
            localThreads_x = 32;
            localThreads_y = 1;
            localThreads_z = 1;
            globalThreads_x = srcGenericDescPtr->dims[2];
            globalThreads_y = 1;
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along XY direction
        case 6:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[3] / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[2] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0] * srcGenericDescPtr->dims[1];

            Rpp32u partialSumArrLength = globalThreads_x * globalThreads_y * globalThreads_z;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
        // compute along XYZ direction
        case 7:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[3] / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[2] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0] * srcGenericDescPtr->dims[1];

            Rpp32u partialSumArrLength = globalThreads_x * globalThreads_y * globalThreads_z;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
    }
}

RppStatus hip_exec_compute_mean_tensor(Rpp32f *srcPtr,
                                       RpptGenericDescPtr srcGenericDescPtr,
                                       Rpp32f *meanTensor,
                                       Rpp32u *roiTensor,
                                       Rpp32u axisMask,
                                       Rpp32u numDims,
                                       Rpp32u maxParamVolume,
                                       rpp::Handle& handle)
{
    Rpp32f *partialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    Rpp32u partialSumArrLength, partialSumBlocksPerSample;

    int globalThreads_x, globalThreads_y, globalThreads_z;
    int localThreads_x, localThreads_y, localThreads_z;
    // based on number of dimensions call the corresponding kernel
    if (numDims == 2)
    {
        // set the block and grid configuration based on axisMask
        set_kernel_launch_config_2d(srcGenericDescPtr, globalThreads_x, globalThreads_y, globalThreads_z,
                                    localThreads_x, localThreads_y, localThreads_z, axisMask,
                                    partialSumArr, handle);

        hipLaunchKernelGGL(compute_mean_2d_hip_tensor,
                           dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1]),
                           meanTensor,
                           partialSumArr,
                           roiTensor,
                           maxParamVolume,
                           axisMask);

        if(axisMask == 2)
        {
            partialSumBlocksPerSample = globalThreads_x;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), ceil((float)globalThreads_y), ceil((float)globalThreads_z)),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               roiTensor,
                               axisMask,
                               numDims);
        }
        else if(axisMask == 3)
        {
            partialSumBlocksPerSample = globalThreads_x * globalThreads_y;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), 1, globalThreads_z),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               roiTensor,
                               axisMask,
                               numDims);
        }
    }
    else if (numDims == 3)
    {
        // set the block and grid configuration based on axisMask
        set_kernel_launch_config_3d(srcGenericDescPtr, globalThreads_x, globalThreads_y, globalThreads_z,
                                    localThreads_x, localThreads_y, localThreads_z, axisMask,
                                    partialSumArr, handle);

        hipLaunchKernelGGL(compute_mean_3d_hip_tensor,
                           dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                           meanTensor,
                           roiTensor,
                           partialSumArr,
                           maxParamVolume,
                           axisMask);

        // perform final reduction on block wise sums for below cases
        // reduce on YZ partial sums
        if(axisMask == 3)
        {
            partialSumBlocksPerSample = globalThreads_y * globalThreads_z;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), srcGenericDescPtr->dims[3], srcGenericDescPtr->dims[0]),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               roiTensor,
                               axisMask,
                               numDims);
        }
        // reduce on XY partial sums
        if(axisMask == 6)
        {
            partialSumBlocksPerSample = globalThreads_x * globalThreads_y;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), srcGenericDescPtr->dims[1], srcGenericDescPtr->dims[0]),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               roiTensor,
                               axisMask,
                               numDims);
        }
        // reduce on XYZ block partial sums
        else if(axisMask == 7)
        {
            partialSumBlocksPerSample = globalThreads_x * globalThreads_y * srcGenericDescPtr->dims[1];
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), 1, srcGenericDescPtr->dims[0]),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               roiTensor,
                               axisMask,
                               numDims);
        }
    }

    return RPP_SUCCESS;
}

RppStatus hip_exec_normalize_tensor(Rpp32f *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u axisMask,
                                    Rpp32f *meanTensor,
                                    Rpp32f *stdDevTensor,
                                    Rpp32u computeMean,
                                    Rpp32u computeStdDev,
                                    Rpp32f scale,
                                    Rpp32f shift,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle)
{
    Rpp32u batchSize = srcGenericDescPtr->dims[0];
    Rpp32u numDims = srcGenericDescPtr->numDims - 1; // exclude batchsize from input dims

    // create buffer for paramShape and paramStride
    Rpp32u *paramShape, *paramStrides;
    paramShape =  handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem;
    paramStrides = handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem + (batchSize * numDims);

    // do initial preprocessing and fill the values for paramShape and paramStrides
    Rpp32u maxParamVolume;
    normalize_setup(roiTensor, batchSize, numDims, axisMask,
                    paramShape, paramStrides, maxParamVolume);

    if((computeMean == 0) && (computeStdDev == 0))
        maxParamVolume = 0;

    // if computeMean is set compute mean values by processing over input based on axisMask values
    if(computeMean)
        hip_exec_compute_mean_tensor(srcPtr, srcGenericDescPtr, meanTensor, roiTensor, axisMask, numDims, maxParamVolume, handle);

    // based on number of dimensions call the corresponding kernel
    if (numDims == 2)
    {
        // NHW
        int globalThreads_x = dstGenericDescPtr->dims[2];
        int globalThreads_y = dstGenericDescPtr->dims[1];
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(normalize_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1]),
                           dstPtr,
                           make_uint2(dstGenericDescPtr->strides[0], dstGenericDescPtr->strides[1]),
                           meanTensor,
                           stdDevTensor,
                           scale,
                           shift,
                           roiTensor,
                           paramShape,
                           paramStrides,
                           maxParamVolume);
    }
    else if (numDims == 3)
    {
        // NDHW
        int globalThreads_x = dstGenericDescPtr->dims[3];
        int globalThreads_y = dstGenericDescPtr->dims[2];
        int globalThreads_z = dstGenericDescPtr->dims[1];

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(normalize_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               &meanTensor[batchCount * maxParamVolume],
                               &stdDevTensor[batchCount * maxParamVolume],
                               scale,
                               shift,
                               &roiTensor[batchCount * 6 + 3],
                               &paramShape[batchCount * 3],
                               &paramStrides[batchCount * 3],
                               maxParamVolume);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        int globalThreads_x = dstGenericDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        // allocate tensor for src strides
        Rpp32u *srcMaxDims = handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem + (2 * batchSize * numDims);
        Rpp32u *srcStrides = handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem + (3 * batchSize * numDims);
        memcpy(srcMaxDims, &srcGenericDescPtr->dims[1], numDims * sizeof(Rpp32u));
        memcpy(srcStrides, &srcGenericDescPtr->strides[1], numDims * sizeof(Rpp32u));

        hipLaunchKernelGGL(normalize_nd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y), ceil((float)globalThreads_z)),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcMaxDims,
                           srcStrides,
                           dstPtr,
                           meanTensor,
                           stdDevTensor,
                           scale,
                           shift,
                           roiTensor,
                           paramShape,
                           paramStrides,
                           maxParamVolume,
                           numDims);
    }

    return RPP_SUCCESS;
}