#include "rpp.h"

void checkEqualBatchSize(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND) {
    if(src1DescriptorPtrND->dims[0] != src2DescriptorPtrND->dims[0]) {
        printf("Batch Size of Inputs must be equal\n");
        exit(0);
    }
}

Rpp32s getMaxNumDims(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND) {
    if(src1DescriptorPtrND->numDims > src2DescriptorPtrND->numDims)
        return src1DescriptorPtrND->numDims;
    return src2DescriptorPtrND->numDims;
}

void UpdateDstDimension(RpptGenericDescPtr srcDescriptorPtr, RpptGenericDescPtr dstDescriptorPtr, Rpp32s idx) {
    int srcDimension;
    int src_ndim = srcDescriptorPtr->numDims;
    int dst_ndim = dstDescriptorPtr->numDims;
    if(idx >= src_ndim - 1)
        srcDimension = 1;
    else
        srcDimension = srcDescriptorPtr->dims[src_ndim - 1 - idx];
    if (srcDimension != 1 && srcDimension != dstDescriptorPtr->dims[dst_ndim - 1 - idx])
        if (dstDescriptorPtr->dims[dst_ndim - 1 - idx] != 1) {
            printf("Incompatible dimension and leads to broadcasting failure\n");
            exit(0);
        }
    if (srcDimension != dstDescriptorPtr->dims[dst_ndim - 1 - idx] && dstDescriptorPtr->dims[dst_ndim - 1 - idx] == 1)
        dstDescriptorPtr->dims[dst_ndim - 1 - idx] = srcDimension;
}

void CopyDescriptorForBroadcasting(RpptGenericDescPtr descPtrND, RpptGenericDescPtr broadcastDescPtrND) {
    RpptGenericDesc descND = *descPtrND;
    RpptGenericDesc broadcastDescND = descND;
    broadcastDescPtrND = &broadcastDescND;
}

void BroadcastDstShape(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND) {
    int ndim = getMaxNumDims(src1DescriptorPtrND, src2DescriptorPtrND);
    dstDescriptorPtrND->numDims = ndim;
    dstDescriptorPtrND->dims[0] = src1DescriptorPtrND->dims[0];
    for(int i = 0; i < ndim - 1; i++) {
        dstDescriptorPtrND->dims[ndim - 1 - i] = 1;
        UpdateDstDimension(src1DescriptorPtrND, dstDescriptorPtrND, i);
        UpdateDstDimension(src2DescriptorPtrND, dstDescriptorPtrND, i);
        printf("Dest Shape at index %d after calculation is %d\n", ndim - 1 - i, dstDescriptorPtrND->dims[ndim - 1 - i]);
    }
}

inline int GetShapeAtIndex(RpptGenericDescPtr descriptorPtrND, int index, int ndim) {
    index -= ndim - descriptorPtrND->numDims;
    if(index < 1)
        return 1;
    return descriptorPtrND->dims[index];
}

// In DALI they check for both inputs and outputs - RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND,
// But ideally checking dst Shape is enough 
inline bool SkipIndexForBroadcasting(RpptGenericDescPtr dstDescriptorPtrND, int index) {
    if(GetShapeAtIndex(dstDescriptorPtrND, index, dstDescriptorPtrND->numDims) != 1)
        return false;
    return true;
}

inline bool CanCollapse(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND, int index, int group_start, int* volumes) {
    int ndim = dstDescriptorPtrND->numDims;
    int dst_index_shape = GetShapeAtIndex(dstDescriptorPtrND, index, ndim);
    int dst_start_shape = GetShapeAtIndex(dstDescriptorPtrND, group_start, ndim);
    bool dst_index_flag = dst_index_shape == 1;
    bool dst_start_flag = dst_start_shape == 1;
    if(dst_index_flag != dst_start_flag)
        return false;
    int src1_index_shape = GetShapeAtIndex(src1DescriptorPtrND, index, ndim);
    int src1_start_shape = GetShapeAtIndex(src1DescriptorPtrND, group_start, ndim);
    bool src1_index_flag = src1_index_shape == 1;
    bool src1_start_flag = src1_start_shape == 1;
    if(src1_index_flag != src1_start_flag)
        return false;
    int src2_index_shape = GetShapeAtIndex(src2DescriptorPtrND, index, ndim);
    int src2_start_shape = GetShapeAtIndex(src2DescriptorPtrND, group_start, ndim);
    bool src2_index_flag = src2_index_shape == 1;
    bool src2_start_flag = src2_start_shape == 1;
    if(src2_index_flag != src2_start_flag)
        return false;
    volumes[0] *= src1_index_shape;
    volumes[1] *= src2_index_shape;
    volumes[2] *= dst_index_shape;
    return true;
}

void AddGroupDimension(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND, int* volumes, int index, int n_groups, std::vector<int>& updated_dims) {
    int ndim = dstDescriptorPtrND->numDims;
    updated_dims.push_back(volumes[0]);
    updated_dims.push_back(volumes[1]);
    updated_dims.push_back(volumes[2]);
    volumes[0] = GetShapeAtIndex(src1DescriptorPtrND, index, ndim);
    volumes[1] = GetShapeAtIndex(src2DescriptorPtrND, index, ndim);
    volumes[2] = GetShapeAtIndex(dstDescriptorPtrND, index, ndim);
}

void GroupShapes(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND) {
    int ndim = dstDescriptorPtrND->numDims;
    int d = 1;
    int group_start;
    int n_groups = 1;

    int volumes[3];
    for(int i = 0; i < 3; i++)
        volumes[i] = 1;
    std::vector<int> updated_dims;

    for(; d < ndim; d++) {
        if (!SkipIndexForBroadcasting(dstDescriptorPtrND, d))
            break;
    }

    if(d < ndim) {
        group_start = d;
        volumes[0] *= GetShapeAtIndex(src1DescriptorPtrND, d, ndim);
        volumes[1] *= GetShapeAtIndex(src2DescriptorPtrND, d, ndim);
        volumes[2] *= GetShapeAtIndex(dstDescriptorPtrND, d, ndim);

        for (d++; d < ndim; d++) {
            if (SkipIndexForBroadcasting(dstDescriptorPtrND, d)) {
                continue;
            }
            if (CanCollapse(src1DescriptorPtrND, src2DescriptorPtrND, dstDescriptorPtrND, d, group_start, volumes)) {
                continue;
            }
            AddGroupDimension(src1DescriptorPtrND, src2DescriptorPtrND, dstDescriptorPtrND, volumes, d, n_groups, updated_dims);
            group_start = d;
            n_groups++;
        }
        updated_dims.push_back(volumes[0]);
        updated_dims.push_back(volumes[1]);
        updated_dims.push_back(volumes[2]);
        n_groups++;
        src1DescriptorPtrND->numDims = n_groups;
        src2DescriptorPtrND->numDims = n_groups;
        dstDescriptorPtrND->numDims = n_groups;

        int idx = 0;
        for(d = 1; d < n_groups; d++) {
            src1DescriptorPtrND->dims[d] = updated_dims[idx++];
            src2DescriptorPtrND->dims[d] = updated_dims[idx++];
            dstDescriptorPtrND->dims[d] = updated_dims[idx++];
        }

        compute_strides(src1DescriptorPtrND->strides, src1DescriptorPtrND->dims, src1DescriptorPtrND->numDims);
        compute_strides(src2DescriptorPtrND->strides, src2DescriptorPtrND->dims, src2DescriptorPtrND->numDims);
        compute_strides(dstDescriptorPtrND->strides, dstDescriptorPtrND->dims, dstDescriptorPtrND->numDims);
    }
}

void StridesForBroadcasting(RpptGenericDescPtr srcDescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND) {
    int ndim = dstDescriptorPtrND->numDims;
    for(int i = 0; i < ndim - 1; i++) {
        if((srcDescriptorPtrND->dims[ndim - 1 - i] != dstDescriptorPtrND->dims[ndim - 1 - 1]) && (srcDescriptorPtrND->dims[ndim - 1 - i] == 1)) {
            srcDescriptorPtrND->strides[ndim - 1 - i] = 0;
        }
    }
}