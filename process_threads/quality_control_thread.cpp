#include "quality_control_thread.h"

QualityControlThread::QualityControlThread(QObject *parent)
    : QThread(parent)
    //, m_qualityControler(new EchoQualityControl())
    // , m_keyframeDetector(new KeyframeDetector())
    // , m_viewClsInferer(new ViewClsInferer(
    //       "D:\\Resources\\20240221\\view_classification\\20240321\\viewcls_backbone.engine",
    //       "D:\\Resources\\20240221\\view_classification\\20240321\\viewcls_swinhead.engine"))
    // , m_roiDetector(new ROIDetection("D:\\Resources\\20240221\\quality_control_models\\roi_detection_models\\roi_detection.engine",
    //                                  cv::Size(640, 640)))
    , m_currInferFrame(cv::Mat())
    , m_vCurrVideoClips(std::vector<cv::Mat>())
{
    qInfo() << "[I] Quality Control Thread initialized.";
}

// int QualityControlThread::roiDetection()
// {
//     // ================ ROI Detection ===================
//     cv::Rect roiCropRect;
//     // float fRadius = 0.0f;

//     if (m_frameCounter == 0 || m_frameCounter % 30 == 0)
//     {
//         std::vector<Object> roiDetectResults = m_roiDetector->doInference(m_currInferFrame);
//         if (roiDetectResults.empty())
//             return 0;

//         roiCropRect = roiDetectResults[0].rect;
//         QVariant qRect;
//         qRect.setValue(roiCropRect);
//         emit sigRoIRect(qRect);
//         // fRadius = roiCropRect.height;
//         // qDebug() << "[I] RoI rect: " << roiCropRect.x << " " << roiCropRect.y << " " << roiCropRect.height;
//     }

//     // cv::Mat frameCropped;
//     // if (roiCropRect.empty())
//     //     frameCropped = m_currInferFrame.clone();
//     // else
//     // {
//     //     frameCropped = m_currInferFrame(roiCropRect).clone();
//     //     frameCropped.convertTo(frameCropped, CV_8UC3);
//     // }
//     // ==================================================
//     return 1;
// }

void QualityControlThread::run()
{
    while (!isInterruptionRequested())
    {
        // int bRoiRet = roiDetection();
        if (m_bIsVideoClipsUpdate)
        {
            std::string viewName = m_currViewName.toStdString();
            s_f_map currVideoResult;
            
            if (m_vCurrVideoClips.empty())
                continue;

            if (m_vKeyframesIdxes.empty())
            {
                m_vKeyframesIdxes.push_back(0);
			}

            for (auto& idx : m_vKeyframesIdxes)
            {
                if (idx >= m_vCurrVideoClips.size())
                {
					qDebug() << "[E] Keyframe index out of range!";
					idx = m_vKeyframesIdxes.size() - 1;
				}
			}

            // 删除es的负值
            m_vKeyframesIdxes.erase(
                std::remove_if(m_vKeyframesIdxes.begin(), m_vKeyframesIdxes.end(), [](int idx) {
                    return idx < 0;
                    }),
                m_vKeyframesIdxes.end()
            );

            m_qualityControler->qualityAssess(m_vCurrVideoClips, m_vKeyframesIdxes, viewName, m_fRadius, currVideoResult);

            qualityControlSignalSend(currVideoResult);

            m_bIsVideoClipsUpdate = false;

            m_vCurrVideoClips.clear();
            m_vKeyframesIdxes.clear();
        }
    }
}


int QualityControlThread::qualityControlSignalSend(s_f_map& qualityControlResult)
{
    QVariant qualityVar;
    QVector<float> qGrades(10, 0.0f);

    if (qualityControlResult.empty())
    {
        qGrades = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    }
    else
    {
        //int counter = 0;
        //for (auto& pair : qualityControlResult)
        //{
        //    qGrades[counter] = pair.second;
        //    ++counter;
        //}

        qGrades[0] = qualityControlResult["gain_score"];
        qGrades[1] = qualityControlResult["scaling_score"];
        qGrades[2] = qualityControlResult["structure_score"];
        if (qualityControlResult.count("mandrel_score"))
        {
            qGrades[3] = qualityControlResult["mandrel_score"];
        }

    }

    qualityVar.setValue(qGrades);
    emit sigVideoResult(qualityVar);

    return 1;
}

void QualityControlThread::setQualityInput(QString qViewName, QVariant qVideoClips, QVariant qKeyframeIdxes, float fRadius)
{
    auto tempValue = qVideoClips.value<QVector<cv::Mat>>();
    m_vCurrVideoClips.clear();
    for (auto mat : tempValue)
    {
        m_vCurrVideoClips.push_back(mat);
    }
    //m_vCurrVideoClips.resize(tempValue.size());
    //std::copy(tempValue.begin(), tempValue.end(), m_vCurrVideoClips.begin());

    m_currViewName = qViewName;

    auto tempIdx = qKeyframeIdxes.value<QVector<int>>();
    if (tempIdx.empty())
    {
        qDebug() << "[E] Keyframe indexes is empty!";
        //return;
    }
    //m_vKeyframesIdxes.resize(tempIdx.size());
    //std::copy(tempIdx.begin(), tempIdx.end(), m_vKeyframesIdxes.begin());
    m_vKeyframesIdxes.clear();
    for (auto idx : tempIdx)
    {
		m_vKeyframesIdxes.push_back(idx);
	}

    m_fRadius = fRadius;

    m_bIsVideoClipsUpdate = true;
}
