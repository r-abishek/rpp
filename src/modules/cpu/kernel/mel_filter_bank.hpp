#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

struct BaseMelScale
{
    public:
        virtual Rpp32f hz_to_mel(Rpp32f hz) = 0;
        virtual Rpp32f mel_to_hz(Rpp32f mel) = 0;
        virtual ~BaseMelScale() = default;
};

struct HtkMelScale : public BaseMelScale
{
    Rpp32f hz_to_mel(Rpp32f hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); }
    Rpp32f mel_to_hz(Rpp32f mel) { return 700.0f * (std::exp(mel / 1127.0f) - 1.0f); }
    public:
        ~HtkMelScale() {};
};

struct SlaneyMelScale : public BaseMelScale
{
    const Rpp32f freqLow = 0;
    const Rpp32f fsp = 66.6666667f; // 200.0f / 3.0f
    const Rpp32f stepLog = 0.068751777f;  // Equivalent to std::log(6.4) / 27.0;
    const Rpp32f minLogHz = 1000.0;
    const Rpp32f minLogMel = (minLogHz - freqLow) / fsp;
    const Rpp32f invFsp = 1.0f / fsp;
    const Rpp32f invMinLogHz = 0.001f;
    const Rpp32f invMinLogMel = 1.0f / stepLog;

    Rpp32f hz_to_mel(Rpp32f hz)
    {
        if (hz >= minLogHz)
            return minLogMel + std::log(hz *invMinLogHz) * invMinLogMel;
        else
            return (hz - freqLow) * invFsp;
    }

    Rpp32f mel_to_hz(Rpp32f mel)
    {
        if (mel >= minLogMel)
            return minLogHz * std::exp(stepLog * (mel - minLogMel));
        else
            return freqLow + mel * fsp;
    }
    public:
        ~SlaneyMelScale() {};
};

inline void compute_mel_filter_bank(Rpp32f *srcRowPtr, Rpp32f *dstPtrTemp, RpptDescPtr dstDescPtr, Rpp32s numFrames, Rpp32s filter,
                                    Rpp32f weight, std::vector<Rpp32f> &normFactors, Rpp32u alignedLength, Rpp32u vectorIncrement, bool normalize)
{
    Rpp32f *dstRowPtrTemp = dstPtrTemp + filter * dstDescPtr->strides.hStride;
    Rpp32f *srcRowPtrTemp = srcRowPtr;

    if (normalize)
        weight *= normFactors[filter];
    __m256 pWeight = _mm256_set1_ps(weight);

    int vectorLoopCount = 0;
    for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
    {
        __m256 pSrc, pDst;
        pSrc = _mm256_loadu_ps(srcRowPtrTemp);
        pSrc = _mm256_mul_ps(pSrc, pWeight);
        pDst = _mm256_loadu_ps(dstRowPtrTemp);
        pDst = _mm256_add_ps(pDst, pSrc);
        _mm256_storeu_ps(dstRowPtrTemp, pDst);
        dstRowPtrTemp += vectorIncrement;
        srcRowPtrTemp += vectorIncrement;
    }
    for (; vectorLoopCount < numFrames; vectorLoopCount++)
    {
        (*dstRowPtrTemp) += weight * (*srcRowPtrTemp);
        dstRowPtrTemp++;
        srcRowPtrTemp++;
    }
}

RppStatus mel_filter_bank_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr srcDims,
                                      Rpp32f maxFreq,
                                      Rpp32f minFreq,
                                      RpptMelScaleFormula melFormula,
                                      Rpp32s numFilter,
                                      Rpp32f sampleRate,
                                      bool normalize)
{
    BaseMelScale *melScalePtr;
    switch(melFormula)
    {
        case RpptMelScaleFormula::HTK:
            melScalePtr = new HtkMelScale;
            break;
        case RpptMelScaleFormula::SLANEY:
        default:
            melScalePtr = new SlaneyMelScale();
            break;
    }

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        // Extract nfft, numFrames, numBins
        Rpp32s nfft = (srcDims[batchCount].height - 1) * 2;
        Rpp32s numBins = nfft / 2 + 1;
        Rpp32s numFrames = srcDims[batchCount].width;

        if(maxFreq == 0.0f)
            maxFreq = sampleRate / 2;

        // Convert lower, higher freqeuncies to mel scale
        Rpp64f melLow = melScalePtr->hz_to_mel(minFreq);
        Rpp64f melHigh = melScalePtr->hz_to_mel(maxFreq);
        Rpp64f melStep = (melHigh - melLow) / (numFilter + 1);
        Rpp64f hzStep = static_cast<Rpp64f>(sampleRate) / nfft;
        Rpp64f invHzStep = 1.0 / hzStep;

        Rpp32s fftBinStart = std::ceil(minFreq * invHzStep);
        Rpp32s fftBinEnd = std::floor(maxFreq * invHzStep);
        fftBinEnd = std::min(fftBinEnd, numBins - 1);

        std::vector<Rpp32f> weightsDown, normFactors;
        weightsDown.resize(numBins);
        normFactors.resize(numFilter, 1.0f);

        std::vector<Rpp32s> intervals;
        intervals.resize(numBins, -1);

        Rpp32s fftBin = fftBinStart;
        Rpp64f mel0 = melLow, mel1 = melLow + melStep;
        Rpp64f f = fftBin * hzStep;
        for (int interval = 0; interval < numFilter + 1; interval++, mel0 = mel1, mel1 += melStep)
        {
            Rpp64f f0 = melScalePtr->mel_to_hz(mel0);
            Rpp64f f1 = melScalePtr->mel_to_hz(interval == numFilter ? melHigh : mel1);
            Rpp64f slope = 1. / (f1 - f0);

            if (normalize && interval < numFilter)
            {
                Rpp64f f2 = melScalePtr->mel_to_hz(mel1 + melStep);
                normFactors[interval] = 2.0 / (f2 - f0);
            }

            for (; fftBin <= fftBinEnd && f < f1; fftBin++, f = fftBin * hzStep)
            {
                weightsDown[fftBin] = (f1 - f) * slope;
                intervals[fftBin] = interval;
            }
        }

        // Set all values in dst buffer to 0.0
        memset(dstPtrTemp, 0.0f, (size_t)(numFilter * numFrames * sizeof(Rpp32f)));

        Rpp32u vectorIncrement = 8;
        Rpp32u alignedLength = numFrames & ~7;
        __m256 pSrc, pDst;
        Rpp32f *srcRowPtr = srcPtrTemp + fftBinStart * numFrames;
        for (int64_t fftBin = fftBinStart; fftBin <= fftBinEnd; fftBin++)
        {
            auto filterUp = intervals[fftBin];
            auto weightUp = 1.0f - weightsDown[fftBin];
            auto filterDown = filterUp - 1;
            auto weightDown = weightsDown[fftBin];

            if (filterDown >= 0)
                compute_mel_filter_bank(srcRowPtr, dstPtrTemp, dstDescPtr, numFrames, filterDown, weightDown, normFactors, alignedLength, vectorIncrement, normalize);
            if (filterUp >= 0 && filterUp < numFilter)
                compute_mel_filter_bank(srcRowPtr, dstPtrTemp, dstDescPtr, numFrames, filterUp, weightUp, normFactors, alignedLength, vectorIncrement, normalize);

            srcRowPtr += srcDescPtr->strides.hStride;
        }
    }
    delete melScalePtr;

    return RPP_SUCCESS;
}
