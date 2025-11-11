#ifndef KEYFRAMEDETECTOR_H
#define KEYFRAMEDETECTOR_H
#include "keyframe_det_inferer.h"

class KeyframeDetector
{
public:
    KeyframeDetector();

    int doInference(cv::Mat& src, std::string viewName, std::string mode, std::vector<PeakInfo>& vPeaks);

    int clearFeatMemory()
    {
        m_a2cInferer->clearFeatMemory();
        m_a3cInferer->clearFeatMemory();
        m_a5cInferer->clearFeatMemory();
        m_a4cInferer->clearFeatMemory();
        m_plaxInferer->clearFeatMemory();

        return 1;
    }

private:
    //KeyframeDetInferer *m_a4cInferer;
    KeyframeLGTAInferer* m_a4cInferer;

    KeyframeLGTAInferer* m_a3cInferer;

    KeyframeLGTAInferer* m_a5cInferer;

    KeyframeLGTAInferer* m_a2cInferer;

    KeyframeDetInferer* m_plaxInferer;
};

#endif // KEYFRAMEDETECTOR_H
