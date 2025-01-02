#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pln1_hip(T *srcPtr,
                                uint2 srcStridesNH,
                                U *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln1_hip<Rpp8u, Rpp32u>(Rpp8u *srcPtr,
                                uint2 srcStridesNH,
                                Rpp32u *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln1_hip<Rpp8s, Rpp32s>(Rpp8s *srcPtr,
                                uint2 srcStridesNH,
                                Rpp32s *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pln3_hip(T *srcPtr,
                                uint3 srcStridesNCH,
                                U *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln3_hip<Rpp8u, Rpp32u>(Rpp8u *srcPtr,
                                uint3 srcStridesNH,
                                Rpp32u *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln3_hip<Rpp8s, Rpp32s>(Rpp8s *srcPtr,
                                uint3 srcStridesNH,
                                Rpp32s *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pkd3_hip(T *srcPtr,
                                uint2 srcStridesNH,
                                U *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pkd3_hip<Rpp8u, Rpp32u>(Rpp8u *srcPtr,
                                uint2 srcStridesNH,
                                Rpp32u *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pkd3_hip<Rpp8s, Rpp32s>(Rpp8s *srcPtr,
                                uint2 srcStridesNH,
                                Rpp32s *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc);

// Handle f16/f32 datatype
template <typename T, typename U>
RppStatus hip_exec_tensor_sum(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *tensorSumArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle);

template<>
RppStatus hip_exec_tensor_sum<Rpp8u, Rpp64u>(Rpp8u *srcPtr,
                              RpptDescPtr srcDescPtr,
                              Rpp64u *tensorSumArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle);

template<>
RppStatus hip_exec_tensor_sum<Rpp8s, Rpp64s>(Rpp8s *srcPtr,
                              RpptDescPtr srcDescPtr,
                              Rpp64s *tensorSumArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle);