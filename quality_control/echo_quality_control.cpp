#include "echo_quality_control.h"

EchoQualityControl::EchoQualityControl()

{
    // EchoQualityAssessmentParams psaxgv;
    // m_psaxgvQualityControl = new EchoQualityAssessmentPSAXGV(psaxgv);

}

EchoQualityControl::~EchoQualityControl()
{
    // delete m_psaxgvQualityControl;
}

int EchoQualityControl::qualityAssess(std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyframeIdxes, std::string& viewName, float fRadius, s_f_map& videoResult)
{
    std::vector<cv::Mat> inputVideoClips;
    if (!keyframeIdxes.empty() || vVideoClips.size() >= 30)
    {
        if (keyframeIdxes.size() >= 2)
        {
            cropSingleEchoCycle(vVideoClips, inputVideoClips, keyframeIdxes);
        }
        else
        {
            inputVideoClips = std::vector<cv::Mat>(vVideoClips.begin(), vVideoClips.begin() + vVideoClips.size() / 2);
        }
    }
    
    else
        inputVideoClips = vVideoClips;

    videoResult = doQualityAssessVideo(viewName, inputVideoClips, keyframeIdxes, fRadius);
    //if (videoFinalResult.singleResults.empty())
    //{
    //    return 0;
    //}
    //videoResult = videoFinalResult.singleResults[0];

    return 1;
}

int EchoQualityControl::cropSingleEchoCycle(std::vector<cv::Mat>& vVideoClips, std::vector<cv::Mat>& sampledVideoClips, std::vector<int>& keyframeIdxes)
{
    std::sort(keyframeIdxes.begin(), keyframeIdxes.end());
    auto last = std::unique(keyframeIdxes.begin(), keyframeIdxes.end());
    keyframeIdxes.erase(last, keyframeIdxes.end());
    
    if (keyframeIdxes.size() >= 2)
    {
        int start = keyframeIdxes[0];
        int end = keyframeIdxes[1];
        if (start < vVideoClips.size() && end < vVideoClips.size())
        {
            sampledVideoClips = std::vector<cv::Mat>(vVideoClips.begin() + start, vVideoClips.begin() + end);
        }
    }
    else
    {
        sampledVideoClips = std::vector<cv::Mat>(vVideoClips.begin(), vVideoClips.begin() + vVideoClips.size() / 2);
    }

    return 1;
}
