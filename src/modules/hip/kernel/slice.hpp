#include "slice.hpp"

template <typename T>
RppStatus hip_exec_slice_tensor(T *srcPtr,
                                RpptGenericDescPtr srcGenericDescPtr,
                                T *dstPtr,
                                RpptGenericDescPtr dstGenericDescPtr,
                                Rpp32s *anchorTensor,
                                Rpp32s *shapeTensor,
                                T *fillValue,
                                bool enablePadding,
                                Rpp32u *roiTensor,
                                rpp::Handle& handle);
