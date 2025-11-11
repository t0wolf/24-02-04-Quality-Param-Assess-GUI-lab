#pragma once

#include <QThread>
#include <opencv2/opencv.hpp>
#include <QMutex>
#include <QQueue>
#include <QPair>
#include <QDebug>
#include <QVector>
#include <QDir>
#include <QDateTime>
#include "view_cls_inferer.h"
#include "general_utils.h"
#include "quality_control/keyframe_detector.h"
#include "ViewClsKeyframeInferThread.h"
#include "QtLogger.h"


struct KeyframeData
{
    cv::Mat inferFrame;
    QString strViewName;
    QString strInferMode;
};


class KeyframeDataBuffer
{
private:
    std::vector<PeakInfo> m_peakResult;
    bool m_hasNewInfo;

    QQueue<KeyframeData> m_keyDetInfoQueue;
    QQueue<QVector<PeakInfo>> m_peakResultsQueue;

    QMutex m_imageBufferMutex;


public:
    KeyframeDataBuffer()
        : m_hasNewInfo(false)
    {

    }

    void updateKeyDetInfo(cv::Mat& frame, QString& viewName, QString& inferMode)
    {
        QMutexLocker locker(&m_imageBufferMutex);

        m_keyDetInfoQueue.enqueue(KeyframeData{ frame.clone(), viewName, inferMode });
    }

    KeyframeData getKeyDetInfo()
    {
        QMutexLocker locker(&m_imageBufferMutex);
        auto currKeyDetInfo = m_keyDetInfoQueue.dequeue();

        return currKeyDetInfo;
    }

    bool hasNewKeyDetInfo() const
    {
        return !m_keyDetInfoQueue.isEmpty();
    }

    void addPeakResult(const std::vector<PeakInfo>& vPeaks)
    {
        QMutexLocker locker(&m_imageBufferMutex);
        //m_peakResultsQueue.enqueue(vPeaks);
        m_peakResult = vPeaks;
    }

    std::vector<PeakInfo> getPeakResult()
    {
        QMutexLocker locker(&m_imageBufferMutex);
        //return m_peakResultsQueue.dequeue();
        std::vector<PeakInfo> tempPeakResults = m_peakResult;
        m_peakResult.clear();
        return tempPeakResults;
    }

    bool hasNewPeakResult()
    {
        return !m_peakResult.empty();
    }

    void clearKeyframeQueue()
    {
        m_keyDetInfoQueue.clear();
    }
};


class KeyframeInferThread : public QThread
{
    Q_OBJECT

public:
    KeyframeInferThread(QObject* parent = nullptr, KeyframeDataBuffer* keyDataBuffer = nullptr);
    ~KeyframeInferThread()
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

    int clearAllMemoryCache();

private:
    int parseKeyframeInferResult(int viewResultIdx);

    int saveInferenceImage(QString& viewName, std::vector<PeakInfo>& vecKeyframePeakInfo, std::vector<cv::Mat>& img);

    inline QString generateDate()
    {
        return QDateTime::currentDateTime().toString("yyyyMMddhhmmss");
    }

    QString m_videoSaveRootPath = "D:/Data/Saved-Images/B-Mode";

private:
    KeyframeDataBuffer* m_keyDataBuffer;
    KeyframeDetector* m_keyframeDetector;

    std::vector<cv::Mat> m_vImagesBuffer;

    //std::vector<QString> m_viewNameMapping = { "A2C", "A3C" ,"A4C" , "A5C", "OTHER", "PLAX", "PSAXA", "PSAXGV", "PSAXMV", "PSAXPM" };

    //QDateTime m_lastSaveTime;
    //int m_saveIntervalSeconds = 10;
};
