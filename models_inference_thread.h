#ifndef MODELSINFERENCETHREAD_H
#define MODELSINFERENCETHREAD_H

#include <QWidget>
#include <QThread>
#include <QMutex>
#include <QMetaType>
#include <QVector>
#include <qmap.h>
#include <opencv2/opencv.hpp>
#include <QElapsedTimer>
// #include "view_classification_inferer.h"
#include "view_cls_inferer.h"
#include "general_utils.h"
#include "param_assessment/image_process.h"
//#include "param_assesser.h"
#include "quality_control/EchoQuality.h"
#include "quality_control/keyframe_detector.h"
// #include "quality_control/keyframe_det_inferer.h"
#include "quality_control/echo_quality_control.h"
#include "progress_super_thread.h"
#include "quality_control/roi_detection.h"
#include "process_threads/quality_control_thread.h"
#include "process_threads/ParamAssessThread.h"
#include "process_threads/ViewClsKeyframeInferThread.h"
#include "process_threads/KeyframeInferThread.h"
#include "info_extraction/InfoExtractor.h"

#include "DataBuffer.h"
#include "type_conversion.h"
#include "config_parse.h"

//Q_DECLARE_METATYPE(QVariant)
// Q_DECLARE_METATYPE(QVector<int>)
// Q_DECLARE_METATYPE(QVector<cv::Mat>)

class ModelsInferenceThread : public QThread
{
    Q_OBJECT
public:
    ModelsInferenceThread(QObject *parent = nullptr, 
        ProgressSuperThread *progressThread = nullptr, 
        DataBuffer* dataBuffer = nullptr, 
        ConfigParse* config = nullptr);

    ~ModelsInferenceThread()
    {
        this->exitThread();
    }

    void exitThread();

    void run() override;//覆盖重载

    int inputVideoFrame(cv::Mat& frame);

    void setModelInferThreadClear();

public slots:
    void getVideoFrame(const QImage frame);

    void setQualityControlScores(QVariant qResult);

    void setParamsAssessValues(QString viewName, QVariant qResult, QVariant qPremiums);

    //void setScaleInfo(QVariant qROIScaleInfo, QVariant qScaleInfo, QVariant qModeInfo);

    void setScaleInfo(QVariant qScaleInfo);

    void setScaleModeInfo(QVariant qScaleInfo, QVariant qModeInfo);

    void setROIScaleInfo(QVariant qROIScaleInfo);

    // void setCropRect(QVariant qRect);

signals:
    void imageProcessed(const QImage &image);

    void viewNameProcessed(const QString& viewName);

    void viewNameImageAvailable(const QString& viewName, QImage qImage);

    void sigReinitailizeLabel(const QString& viewName);

    void viewNameVideoAvailable(const QString& viewName, QVariant qVar);

    void sigViewNameImageVideoAvailable(const QString& viewName, QVariant qVideoClips, QVariant qQualityScores);

    void qualityScoresAvailable(QString viewName, QVariant result);

    void sigQualityInput(QString qViewName, QVariant qVideoClips, QVariant qKeyframeIdxes, float fRadius);

    void sigParamInput(QString viewName, QVariant videoClip, QVariant keyframeIdxes, QVariant modeInfo);

    void sigPatientInfo(QString patientName, QString patientID);

    void sigDebugText(QString debugText);

// Parameters Assessment signals
signals:
    void sigParamsAvailable(QString viewName, QVariant paramValues, QVariant paramPremium);

    void sigStructParamsAvailable(QVariant paramValues, QVariant paramPremium);

    void structureParamsAvailable(QString viewName, QVariant paramValues);

    // void spectrumParamsAvailable(QMap<QString, float> paramValues);

    // void funcParamsAvailable(QMap<QString, float> paramValues);

    void sigScaleInfo(QVariant qScaleInfo);

    void sigViewNameQualityScoreToTransmitter(QString strViewName, QString strQualityScore);

    void setimagesShow(const std::vector<cv::Mat>& images); // new

private:
    int parseConfigFile(ConfigParse* config);

    int processFrame(cv::Mat& frame, ScaleInfo& currScaleInfo, ModeInfo& currModeInfo, RoIScaleInfo& currROIScaleInfo);

    int handleBlankViewMode();

    int handleViewClassificationProcess(cv::Mat& inferFrame, RoIScaleInfo& assessROIScaleInfo);

    std::vector<cv::Mat> handleViewClassificationProcess(cv::Mat& inferFrame);

    int handleKeyframeSampling(cv::Mat& frameCropped, QVector<int>& keyframeIdxes, std::vector<PeakInfo>& vPeaks);

    int handleKeyframeSampling(std::vector<cv::Mat>& vecVideoBuffer, QVector<int>& keyframeIdxes, std::vector<PeakInfo>& vPeaks);

    int handleQualityControl(RoIScaleInfo& assessROIScaleInfo);

    int handleParamAssess(cv::Mat& inferFrame, ScaleInfo& scaleInfo);

    int handleParamAssess(QString& strCurrViewName, cv::Mat& inferFrame, std::vector<cv::Mat>& vecInferImages, ScaleInfo& scaleInfo);

    int doQualityAssess();

    int blankModeProcess();

    void clearDataCache();

    int updateRoIData(RoIScaleInfo& roiScaleInfo);

    int updateScaleInfo(ScaleInfo& scaleInfo, ModeInfo& modeInfo);

    int updateRoIScale(cv::Mat& frame, cv::Mat frameCropped, RoIScaleInfo& roiScaleInfo);

    bool isQualityParamRunning();

    inline void processPeakInfos(const std::vector<PeakInfo>& vPeakInfos, QVector<int>& keyframeIdx)
    {
        for (auto& peak : vPeakInfos) {
            //int currKeyframeIdx = adjustIndex(peak.index, m_isA4CSampleFlag ? m_a4cSampleNum : m_currKeyframeSampleNum);
            //keyframeIdx.push_back(currKeyframeIdx);
            keyframeIdx.push_back(peak.index);
        }
    }

    inline int adjustIndex(int index, int maxSampleNum)
    {
        if (abs(index) > maxSampleNum) {
            return index < 0 ? -(maxSampleNum - 1) : maxSampleNum - 1;
        }
        return index;
    }

    int frameCropping(cv::Mat& src, cv::Mat& dst, RoIScaleInfo& currRoIScaleInfo);

    int currViewModeJudgement(ModeInfo& currModeInfo, RoIScaleInfo& currROIScaleInfo);

    //int currViewModeJudgement(RoIScaleInfo& currROIScaleInfo);

    //int parseViewClassInferResult(cv::Mat& currInferFrame, int viewResultIdx);

    int parseViewClassInferResult(const QString& currViewName);

    int parseViewClassInferResult(const QString& currViewName, std::vector<cv::Mat>& vecViewClassImages);

    int parseViewClassInferResult(const QString& currViewName, const float fQualityScore, std::vector<cv::Mat>& vecViewClassImages);

    int parseKeyframeDetResult(cv::Mat& currInferFrame, QVector<int>& keyframeIdx, std::vector<PeakInfo>& vPeakInfos);

    int parseKeyframeDetResult(cv::Mat& currInferFrame, QVector<int>& keyframeIdx, QVector<PeakInfo>& vPeakInfos);

    int sendQualityControlBeginSignal(RoIScaleInfo& currROIScaleInfo);

    int sendQualityControlEndSignal();

    int sendImageSamplingInterruptSignal();

    int updateQualityScore(const QString& viewName, float currentScore);

    bool judgeQualityScoreToParam(QString& strViewName, float fCurrQualityScore);

    bool isCurrentScoreHigher(const QString& viewName, float currentScore);

    int sendParamAssessBeginSignal(cv::Mat& originFrame, ScaleInfo& currScaleInfo);

    int sendParamAssessBeginSignal(QString& strCurrViewName, cv::Mat& originFrame, std::vector<cv::Mat>& vecInferImages, ScaleInfo& currScaleInfo);

    int sendParamAssessEndSignal(ScaleInfo& currScaleInfo);

    int checkoutNormalParam();

    int paramValueScaling(ScaleInfo& currScaleInfo);

    int eraseTDINoUseParam();

    int computeEDevideA();

    float computeEDevideTDI();

    int computeEDevideTDIToParam();

    // 删除所有的异常值，即0
    int removeZeroNumParam();

    // 删除所有的离群点
    int removeOutlierParamEvent();

    void addParamToHistory();

    float getSpecMiddleValue(QVector<float>& vecParamValues);

    int getMaxHistParamValue(QString strValueName);

    int compareHistParamWithIncrementUpdate();

    int compareHistParamWithAllParamValid();

    int compareHistParamWithThresh();

    bool compareSpecParamWithThresh();

    int plotParamScaleOnPremium(ScaleInfo& currScaleInfo);

    int groupStructParam(QMap<QString, QMap<QString, QString>>& currStructParamValues, QMap<QString, QImage>& currStructPremiums);

    int sendParamToDisplay();

    int sendPLAXParamToDisplay();

    int sendNormalParamToDisplay();

    int checkVideoBufferSize(int maxSize = 200);

    int sendScaleInfo(ScaleInfo& currScaleInfo);

    int saveInferenceImage(QString& viewName, std::vector<cv::Mat>& vImgs);

    inline bool checkCurrentScaleModeMatch(QString& currEventName, ScaleInfo& currScaleInfo)
    {
        if (m_specParamEvents.indexOf(currEventName) != -1)  // Doppler
        {
            if ((currScaleInfo.fSpecScaleRange != -10000.0f && currScaleInfo.length == -10000.0f))
                return true;
            else
                return false;
        }
        else  // B
        {
            if ((currScaleInfo.length != -10000.0f && currScaleInfo.fSpecScaleRange == -10000.0f))
                return true;
            else
                return false;
        }
    }

    float getAvgHeight(std::vector<Object>& objects)
    {
        if (objects.empty())
            return 0.0f;

        float fAvgHeight = 0.0f;
        int counter = 0;

        for (auto& object : objects)
        {
            fAvgHeight += object.rect.height;
            ++counter;
        }

        return fAvgHeight / static_cast<float>(counter);
    }

    inline QString generateDate()
    {
        return QDateTime::currentDateTime().toString("yyyyMMddhhmmss");
    }

private:
    cv::Mat m_nextFrame;
    QMutex m_mutex, m_videoBufferMutex, m_keyframeBufferMutex, m_paramCommMutex;
    bool m_hasNewFrame = false;

    int m_videoBufferLength = 64;
    std::vector<QString> m_viewNameMapping = { "A2C", "A3C" ,"A4C" , "A5C", "OTHER", "PLAX", "PSAXA", "PSAXGV", "PSAXMV", "PSAXPM" };

    std::map<QString, float> m_mapViewFullScore = {
        {"A2C", 10.0f},
        {"A3C", 10.0f},
        {"A4C", 10.0f},
        {"A5C", 10.0f},
        {"PLAX", 10.0f},
        {"PSAXA", 6.0f},
        {"PSAXGV", 10.0f},
        {"PSAXMV", 10.0f},
        {"PSAXPM", 10.0f}
    };

    QString m_videoSaveRootPath = "D:/Data/Saved-Images/B-Mode";
    QDateTime m_lastSaveTime;
    int m_saveIntervalSeconds = 10;

    int m_frameCounter = 0;

    int m_blankFrameCounter = 0;

    int m_waitForScaleCounter = 0;

    int m_bIsKeyframeSampling = false;
    int m_nonKeyframeSampleNum = 10;
    int m_keyframeSampleNum = 25;
    int m_a4cSampleNum = m_keyframeSampleNum;
    int m_currKeyframeSampleNum = 0;

    float m_fQualityScoreThresh = 0.4f;
    float m_fDiffRatioThresh = 0.1f;
    float m_fStructAortaDiffRatioThresh = 0.03f;
    float m_fSpecDiffRatioThresh = 0.2f;

    bool m_bBestQualityScoreOnly = false;
    bool m_bAllParamsDiffValid = false;
    bool m_bLVIDSelectMaxValue = false;
    bool m_bAADSelectMaxValue = false;
    int m_primaryQualityFrameLen = 3;
    int m_primaryQualitySampleStride = 2;

    QVector<QString> m_specParamEvents = {
        "Ao", "VIT",
        "PA", "E", "A", "TR",
        "JGs", "JGe", "JGa",
        "CBs", "CBe", "CBa"
    };

    QVector<QString> m_structParamEvents = {
        "IVSTd",
        "LVDd",
        "LVPWTd",
        "AoD",
        "ASD",
        "SJD",
        "AAD",
        "LAD"
    };

    QVector<QString> m_vecStructParamPremiums = {
        "IVSTd",
        "AAD",
        "LAD"
    };

    QMap<QString, QVector<QString>> m_mapStructPremiumToEvent = {
        { "IVSTd", { "IVSTd", "LVDd", "LVPWTd" }},
        { "AAD", { "ASD", "SJD", "AAD" }},
        { "LAD", { "LAD" }}
    };

    QVector<QString> m_specialSpecEvents = {
        "MV", "TDIMVIVS", "TDIMVLW"
    };

    QVector<QString> m_vecTDISpecEvents = {
        "JGe", "CBe"
    };

    QVector<QString> m_vecSupportQualityControlViews = {
        "A2C", "A4C", "PLAX"
    };

private:
    std::vector<cv::Mat> m_vVideoBuffer, m_vecViewClassBuffer, m_vecKeyframeBuffer;
    std::vector<cv::Mat> m_vOriginVideoBuffer;
    std::vector<float*> mFeaturesBuffer;

    QString m_currViewName, m_currParamReturnViewName, m_currQualityControlViewName;
    QImage m_currQualityPremiumFrame;
    float m_fCurrQualityScore;
    QMap<QString, float> m_mapHistoryQualityScore;  // 存储当前患者的历史质量分，键名为切面名称，值为质量分

    cv::Rect m_roiRect;
    QVector<int> m_currKeyframeIdxes;
    QVector<float> m_currResult;
    QMap<QString, QVector<float>> m_currParamValues;
    QMap<QString, QVector<float>> m_histParamValues;
    QMap<QString, QString> m_currSignalParamValues, m_currUpdateParamValues;
    QMap<QString, QImage> m_currPremiums;

    ScaleInfo m_currScaleInfo;
    PatientInfo m_currPatientInfo;
    ModeInfo m_currModeInfo;
    ModeInfo m_prevModeInfo;
    RoIScaleInfo m_currROIScaleInfo;

    bool m_paramOnlyMode = true;

    bool m_isSpecViewMode = false;
    bool m_isBViewMode = false;
    bool m_isBlankMode = false;

    bool m_isViewRecognized = false;
    bool m_isKeyframeDetected = false;
    bool m_isScaleInfoUpdated = false;
    bool m_isROIScaleInfoUpdated = false;
    bool m_bQualityScoresUpdateFlag = false;
    bool m_bParamValuesUpdateFlag = false;

    bool m_isA4CSampleFlag = false;

    bool m_isQualityControlled = false;
    bool m_isQualityControlThreadRunning = false;
    bool m_isPrimaryQualityControlled = false;
    bool m_isPrimaryQualityControlRunning = false;
    bool m_isParamAssessed = false;
    bool m_isParamAssessThreadRunning = false;

public:
    ViewClsInferer *m_viewClsInferer;

    KeyframeDetector *m_keyframeDetector;

    // EchoQualityControl *m_qualityControler;

    ProgressSuperThread *m_progressThread;

    ViewClsKeyframeInferThread* m_viewClsKeyframeInferThread;
    KeyframeInferThread* m_keyframeInferThread;

    ViewClsInferBuffer* m_clsDataBuffer;
    KeyframeDataBuffer* m_keyDataBuffer;

    //ROIDetection *m_roiDetector;

    InfoExtractor* m_infoExtractor;

    QualityControlThread *m_qualityControlThread;

    ParamAssessThread* m_paramAssessThread;

    DataBuffer* m_roiDataBuffer;
};

#endif // MODELSINFERENCETHREAD_H
