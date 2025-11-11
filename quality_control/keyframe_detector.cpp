#include "keyframe_detector.h"

KeyframeDetector::KeyframeDetector()
    //: m_a4cInferer(new KeyframeDetInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\A4C\\a4c_backbone.engine",
    //                                      "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\A4C\\a4c_sgta.engine",
    //                                      5, 15, false, true, { 0.485, 0.456, 0.406 }, {0.229, 0.224, 0.225}))
    : m_a4cInferer(new KeyframeLGTAInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\lgta_backbone.engine",
        "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\lgta.engine",
        5, 5, false, true, { 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }))
    , m_a2cInferer(new KeyframeLGTAInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\lgta_backbone.engine",
        "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\lgta.engine",
        5, 5, false, true, { 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }))
    , m_a3cInferer(new KeyframeLGTAInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\lgta_backbone.engine",
        "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\lgta.engine",
        5, 5, false, true, { 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }))
    , m_a5cInferer(new KeyframeLGTAInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\lgta_backbone.engine",
        "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\lgta.engine",
        5, 5, false, true, { 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }))

    //, m_a2cInferer(new KeyframeDetInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\A2C\\a2c_backbone.engine",
    //                                      "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\A2C\\a2c_sgta.engine",
    //                                      5, 15, false, true, { 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }))
    , m_plaxInferer(new KeyframeDetInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\PLAX\\plax_backbone.engine",
                                           "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\PLAX\\plax_sgta.engine",
                                            5, 15))
    //, m_a3cInferer(new KeyframeDetInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\A3C\\a3c_backbone.engine",
    //                                      "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\A3C\\a3c_sgta.engine",
    //                                       5, 15))
    //, m_a5cInferer(new KeyframeDetInferer("D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\A5C\\a5c_backbone.engine",
    //                                      "D:\\Resources\\20240221\\quality_control_models\\keyframe_models\\A5C\\a5c_sgta.engine",
    //                                       5, 15))
{
    qInfo() << "[I] KeyframeDetInferer init success.";
}

int KeyframeDetector::doInference(cv::Mat& src, std::string viewName, std::string mode, std::vector<PeakInfo>& vPeaks)
{
    cv::Mat inferFrame = src.clone();
    if (viewName == "A4C")
    {
        m_a4cInferer->doInference(inferFrame, mode, vPeaks);
    }

    else if (viewName == "PLAX")
    {
        m_plaxInferer->doInference(inferFrame, mode, vPeaks);
    }

    else if (viewName == "A2C")
    {
        //m_a2cInferer->doInference(inferFrame, mode, vPeaks);
        m_a4cInferer->doInference(inferFrame, mode, vPeaks);
    }

    else if (viewName == "A3C")
    {
        m_a4cInferer->doInference(inferFrame, mode, vPeaks);
    }

    else if (viewName == "A5C")
    {
        m_a4cInferer->doInference(inferFrame, mode, vPeaks);
    }
    return 1;
}
