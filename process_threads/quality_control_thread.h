#ifndef QUALITYCONTROLTHREAD_H
#define QUALITYCONTROLTHREAD_H

#include <QObject>
#include <QWidget>
#include <QThread>
#include <QVariant>

#include "quality_control/echo_quality_control.h"
// #include "quality_control/roi_detection.h"
#include "quality_control/keyframe_detector.h"
#include "view_cls_inferer.h"
#include "type_registering.h"
//Q_DECLARE_METATYPE(cv::Rect)
//Q_DECLARE_METATYPE(QVector<float>)
//Q_DECLARE_METATYPE(QVector<int>)
//Q_DECLARE_METATYPE(QVector<cv::Mat>)

class QualityControlThread : public QThread
{
    Q_OBJECT
public:
    QualityControlThread(QObject *parent = nullptr);
    ~QualityControlThread()
    {
        exitThread();
    }

    void exitThread()
    {
        this->requestInterruption();
        this->quit();
        this->wait();
    }

    void run() override;

signals:
    void sigRoIRect(QVariant qRect);

    void sigVideoResult(QVariant qResult);

private:
    // int roiDetection();

    //int keyframeDetection();

    int qualityControlSignalSend(s_f_map& qualityControlResult);

public slots:
    void setQualityInput(QString qViewName, QVariant qVideoClips, QVariant qKeyframeIdxes, float fRadius);

private:
    EchoQualityControl *m_qualityControler;

    //KeyframeDetector *m_keyframeDetector;

    //ViewClsInferer *m_viewClsInferer;

    //ROIDetection *m_roiDetector;

    int m_frameCounter = 0;

    // quality control members
    cv::Mat m_currInferFrame;
    QString m_currViewName;
    std::vector<cv::Mat> m_vCurrVideoClips;
    std::vector<int> m_vKeyframesIdxes;
    float m_fRadius;

    bool m_bIsVideoClipsUpdate = false;
};

#endif // QUALITYCONTROLTHREAD_H
