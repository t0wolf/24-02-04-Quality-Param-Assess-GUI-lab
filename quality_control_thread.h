#ifndef QUALITYCONTROLTHREAD_H
#define QUALITYCONTROLTHREAD_H

#include <QObject>
#include <QThread>
#include <QString>
#include <QVariant>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "quality_control/echo_quality_control.h"

class QualityControlThread : public QThread
{
    Q_OBJECT
public:
    QualityControlThread(QObject *parent = nullptr);

    void run() override;

    void exitThread()
    {
        this->requestInterruption();
        this->quit();
        this->wait();
    }

signals:
    void sigQualityScoresAvailable(QVariant qVResult);

private:
    EchoQualityControl *m_qualityControler;

private:
    std::vector<cv::Mat> m_currVideoClips;
    std::vector<int> m_currKeyframeIdxes;
    QString m_currViewName;

};

#endif // QUALITYCONTROLTHREAD_H
