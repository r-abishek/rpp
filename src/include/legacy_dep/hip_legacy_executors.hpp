#include "rpp.h"
#include "handle.hpp"
#include "rpp_hip_load_store.hpp"

// -------------------- fisheye --------------------

RppStatus fisheye_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

// -------------------- snow --------------------

RppStatus snow_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

// -------------------- hueRGB --------------------

RppStatus hueRGB_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);

// -------------------- saturationRGB --------------------

RppStatus saturationRGB_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel);
