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
#include "config_parse.h"

class ImageBuffer 
{
private:
    cv::Mat m_viewClsBuffer;
    QString m_viewClsResult;
    bool m_hasNewFrame = false;

    cv::Mat m_keyDetBuffer;
    QPair<QString, QString> m_keyDetInfoBuffer;
    std::vector<PeakInfo> m_peakResult;
    bool m_hasNewInfo = false;

    QQueue<QPair<QString, QString>> m_infoQueue;
    QQueue<QString> m_viewResultQueue;
    QQueue<QVector<PeakInfo>> m_peakResultsQueue;

    QMutex m_imageBufferMutex;


public:
    ImageBuffer()
        : m_hasNewFrame(false)
        , m_hasNewInfo(false)
        , m_viewClsResult(QString(""))
        , m_keyDetInfoBuffer(qMakePair(QString(""), QString("")))
    {
        m_hasNewFrame = false;
    }

    void updateViewClsImage(const cv::Mat& newImage) {
        QMutexLocker locker(&m_imageBufferMutex);
        m_viewClsBuffer = newImage.clone();
        m_hasNewFrame = true;
    }

    cv::Mat getViewClsImage() {
        QMutexLocker locker(&m_imageBufferMutex);
        m_hasNewFrame = false;
        return m_viewClsBuffer.clone();
    }

    bool newImageAvailable() const {
        //QMutexLocker locker(&m_imageBufferMutex);
        return m_hasNewFrame;
    }

    bool hasViewClsResult() const {
        //return !m_resultQueue.isEmpty();
        return !m_viewClsResult.isEmpty();
    }

    void addViewClsResult(const QString& result) {
        QMutexLocker locker(&m_imageBufferMutex);
        //m_resultQueue.enqueue(result);
        m_viewClsResult = result;
    }

    QString getViewClsResult() {
        QMutexLocker locker(&m_imageBufferMutex);
        //return m_resultQueue.dequeue();
        QString tempViewClsResult = m_viewClsResult;
        m_viewClsResult.clear();
        return tempViewClsResult;
    }

    void updateKeyDetInfo(cv::Mat& frame, QString& viewName, QString& inferMode)
    {
        QMutexLocker locker(&m_imageBufferMutex);
        //m_infoQueue.enqueue(qMakePair(viewName, inferMode));
        m_keyDetInfoBuffer = qMakePair(viewName, inferMode);
        m_keyDetBuffer = frame.clone();
        m_hasNewInfo = true;
    }

    cv::Mat getKeyDetInfo(QPair<QString, QString>& infoPair)
    {
        QMutexLocker locker(&m_imageBufferMutex);
        m_hasNewInfo = false;
        infoPair = m_keyDetInfoBuffer;
        m_keyDetInfoBuffer = qMakePair(QString(""), QString(""));
        return m_keyDetBuffer;
    }

    bool hasNewKeyDetInfo() const
    {
        return m_hasNewInfo;
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

    void clearResultBuffer()
    {
        m_viewClsResult.clear();
    }
};


class ViewClsInferBuffer
{
private:
    cv::Mat m_inferImageBuffer;
    ViewClsResult m_viewClsResult;
    bool m_hasNewFrame = false;

    QQueue<ViewClsResult> m_viewResultQueue;

    QMutex m_imageBufferMutex;

public:
    ViewClsInferBuffer()
        : m_hasNewFrame(false)
        , m_viewClsResult({ QString(""), -10000.0f })
    {
    }

    void updateViewClsImage(const cv::Mat& newImage) 
    {
        QMutexLocker locker(&m_imageBufferMutex);
        m_inferImageBuffer = newImage.clone();
        m_hasNewFrame = true;
    }

    cv::Mat getViewClsImage() 
    {
        QMutexLocker locker(&m_imageBufferMutex);
        m_hasNewFrame = false;
        return m_inferImageBuffer.clone();
    }

    bool newImageAvailable() const {
        return m_hasNewFrame;
    }

    bool hasViewClsResult() {
        return !m_viewClsResult.isEmpty();
    }

    void addViewClsResult(const ViewClsResult& result) {
        QMutexLocker locker(&m_imageBufferMutex);
        //m_resultQueue.enqueue(result);
        m_viewClsResult = result;
    }

    ViewClsResult getViewClsResult() {
        QMutexLocker locker(&m_imageBufferMutex);
        //return m_resultQueue.dequeue();
        ViewClsResult tempViewClsResult = m_viewClsResult;
        m_viewClsResult.clear();
        return tempViewClsResult;
    }

    void clearResultBuffer()
    {
        m_viewClsResult.clear();
    }
};


class ViewClsKeyframeInferThread  : public QThread
{
	Q_OBJECT

public:
	ViewClsKeyframeInferThread(QObject *parent = nullptr, ViewClsInferBuffer* clsDataBuffer = nullptr, ConfigParse* config = nullptr);
    ~ViewClsKeyframeInferThread()
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

    int getViewClassClipSize();

private:
    int parseViewClassInferResult(int viewResultIdx);

    int parseViewClassInferResult(int viewResultIdx, float fQualityScore);

    int parseViewClassInferResult(int viewResultIdx, float fQualityScore, bool bIsSwitch);

    //int saveInferenceImage(QString& viewName, std::vector<cv::Mat>& img);

    //inline QString generateDate()
    //{
    //    return QDateTime::currentDateTime().toString("yyyyMMddhhmmss");
    //}

    //QString m_videoSaveRootPath = "D:/Data/Saved-Images/B-Mode";

private:
    ViewClsInferBuffer* m_clsDataBuffer;
    //ImageBuffer* m_keyDataBuffer;
    ViewClsInferer* m_viewClsInferer;
    //KeyframeDetector* m_keyframeDetector;

    std::vector<cv::Mat> m_vImagesBuffer;

    std::vector<QString> m_viewNameMapping = { "A2C", "A3C", "A4C", "A5C", "OTHER", "PLAX", "PSAXA", "PSAXGV", "PSAXMV", "PSAXPM" };

    //QDateTime m_lastSaveTime;
    //int m_saveIntervalSeconds = 10;
};
