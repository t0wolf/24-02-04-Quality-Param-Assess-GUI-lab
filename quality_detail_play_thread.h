#ifndef QUALITYDETAILPLAYTHREAD_H
#define QUALITYDETAILPLAYTHREAD_H

#include <QObject>
#include <QWidget>
#include <QThread>
#include <QVector>

#include "opencv2/opencv.hpp"
#include "general_utils.h"

class QualityDetailPlayThread : public QThread
{
    Q_OBJECT
public:
    QualityDetailPlayThread(QObject *parent = nullptr) : QThread(parent) {}
    ~QualityDetailPlayThread()
    {
        exitThread();
    }

    void run() override;

    void startStream();
    void exitThread();

    inline void setVideoBuffer(QVector<cv::Mat>& videoBuffer)
    {
        m_videoBuffer = videoBuffer;
    }

signals:
    void frameAvailable(QImage image);

private:
    QVector<cv::Mat> m_videoBuffer;

    bool stopFlag = false;

// signals:
//     void frameAvailable(const QImage &image);

//     void cvFrameAvailable(const cv::Mat &image);

};

#endif // QUALITYDETAILPLAYTHREAD_H
