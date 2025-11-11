#ifndef PARAMASSESSER_H
#define PARAMASSESSER_H

#include <iostream>
#include <QWidget>
#include <QMap>
#include <opencv2/opencv.hpp>
#include <QDateTime>
#include <QDir>

#include "param_assessment/plax_params_assess.h"
#include "param_assessment/DoSpecParamsAssess.h"
#include "param_assessment/DoFuncParamsAssess.h"
#include "info_extraction/InfoExtractor.h"
#include "spec_classification_inferer.h"
#include "SpecUnionCls.h"
#include "type_define.h"
#include "config_parse.h"
#include "QtLogger.h"
#include "quality_control/gaininfer.h"
//#include "ocr_inference/OCRInferer.h"
//#include "scale_extraction/ScaleExtractor.h"

class GainAssesser
{
public:
    GainAssesser(std::string& strGainModelsRootPath);
    ~GainAssesser();

    int doInference(std::vector<cv::Mat>& vecInputImages, std::string& strViewName, std::vector<int>& vecGainScoreResults);

protected:

    GainInfer* selectSpecifiedGainModel(std::string& strViewName);

    GainInfer* initializeGainModel(std::string strModelPath, std::string strViewName);

protected:
    GainInfer* m_a2cGainInferer;

    GainInfer* m_a4cGainInferer;

    GainInfer* m_plaxGainInferer;

    std::vector<std::string> m_vecViewNames = { "a2c", "a4c", "plax" };
};

class ParamAssesser
{
public:
    ParamAssesser(ConfigParse* config);

    ~ParamAssesser()
    {
        // delete m_ivsPWAssesser;
        // delete m_aaoAssesser;
        if (m_plaxParamsAssesser != nullptr)
        {
            delete m_plaxParamsAssesser;
            m_plaxParamsAssesser = nullptr;
        }

        if (m_specModeInferer != nullptr)
        {
            delete m_specModeInferer;
            m_specModeInferer = nullptr;
        }

        if (m_funcParamsAssesser != nullptr)
        {
            delete m_funcParamsAssesser;
            m_funcParamsAssesser = nullptr;
        }
    }

    int doInferece(std::string& currViewName, std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex, ModeInfo& modeInfo);

    int doInferece(std::string& currViewName, std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex, ModeInfo& modeInfo, RoIScaleInfo& roiScaleInfo);

    inline bool isParamEnable(std::string viewName)
    {
        return m_paramEnableMap[viewName];
    }

    inline void clearLVEFDataCache()
    {
        m_funcParamsAssesser->clearLVEFDataCache();
    }

    inline QMap<QString, QVector<float>> getParamValues()
    {
        return m_currValueMap;
    }

    inline QMap<QString, QImage> getParamPremiums()
    {
        return m_currPremiumsMap;
    }

    inline QMap<QString, QVector<float>> getSpecParamValues()
    {
        return m_currSpecValueMap;
    }

    inline QMap<QString, QImage> getSpecParamPremiums()
    {
        return m_currSpecPremiumsMap;
    }

    inline QMap<QString, QVector<float>> getFuncParamValues()
    {
        return m_currFuncValueMap;
    }

    inline QMap<QString, QImage> getFuncParamPremiums()
    {
        return m_currFuncPremiumsMap;
    }

    inline bool isFuncBiplaneMode()
    {
        return m_funcParamsAssesser->isFuncBiplaneMode();
    }

    inline int clearFuncParams()
    {
        m_currFuncValueMap.clear();
        m_currFuncPremiumsMap.clear();
        return 1;
    }

    inline int clearSpecParams()
    {
        m_currSpecValueMap.clear();
        m_currSpecPremiumsMap.clear();
        return 1;
    }

    inline int clearStructParams()
    {
        m_currValueMap.clear();
        m_currPremiumsMap.clear();
        return 1;
    }

    inline int clearAllParams()
    {
        clearStructParams();
        clearFuncParams();
        clearSpecParams();
        return 1;
    }

public:
    QMap<QString, float> m_plaxStructureParams;

    PLAXParamsAssess* m_plaxParamsAssesser;

    DoFuncParamsAssess* m_funcParamsAssesser;

private:
    int doQualityAssesser(std::string& viewName, std::vector<cv::Mat>& vVideoClips, std::vector<cv::Mat>& inputVideoClip, std::vector<int>& keyFrameIndex);

    int doPLAXAssesser(std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex);

    int doSpecModeClass(cv::Mat& src, std::map<std::string, int>& classResults);

    int doSpecUnionModeClass(cv::Mat& src, std::map<std::string, int>& classResults);

    int doSpecAssesser(std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex, std::map<std::string, int>& classResults);

    int doFuncAssesser(std::vector<cv::Mat>& vVideoClips);

    int doFuncAssesser(std::string& viewName, std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex);

    int deleteHistoryQualityScore(std::string& strViewName);

    int cropSingleEchoCycle(std::vector<cv::Mat>& vVideoClips, std::vector<cv::Mat>& sampledVideoClips, std::vector<int>& keyframeIdxes);

    int saveSpecInferenceImage(QString& eventName, cv::Mat& img, bool isPremium = false);

    int saveStructInferenceImage(QString viewName, 
        std::vector<cv::Mat>& vInferVideoClip, 
        std::vector<int>& vKeyframeIndexes, 
        QMap<QString, QImage>& qmPremiums);

    inline QString generateDate()
    {
        return QDateTime::currentDateTime().toString("yyyyMMddHHmmss");
    }

    void clearPatientHistoryRecord();

public:
    int setPatientName(QString patientName);


private:
    //OCRInferer* m_ocrInferer;
    //ScaleExtractor* m_scaleExtractor;

    float m_realScale;

    QString m_patientName, m_prevPatientName;

    DoSpecParamsAssess* m_specParamsAssesser;

    SPECClassInferer* m_specModeInferer;

    SpecUnionCls* m_specUnionInferer;

    InfoExtractor* m_infoExtractor;

    GainAssesser* m_gainAssesser;

    QMap<QString, QVector<float>> m_currValueMap, m_currSpecValueMap, m_currFuncValueMap;
    QMap<QString, QImage> m_currPremiumsMap, m_currSpecPremiumsMap, m_currFuncPremiumsMap;

    int m_nLVEFCycleLength = 8;
    int m_nPLAXCycleLength = 5;

    std::map<std::string, int> m_mapMaxQualityScore;

    std::map<std::string, bool> m_paramEnableMap {
                                                  {"A2C", true},
                                                  {"A3C", false},
                                                  {"A4C", true},
                                                  {"A5C", false},
                                                  {"OTHER", false},
                                                  {"PLAX", true},
                                                  {"PSAXGV", false},
                                                  {"PSAXPM", false},
                                                  {"PSAXMV", false},
                                                  {"PSAXA", false},
                                              };

    QString m_specImageSaveRootPath = "D:/Data/Saved-Images/Doppler-Mode";

    QString m_structImageSaveRootPath = "D:/Data/Saved-Images/B-Mode";

    QDateTime m_lastSpecSaveTime, m_lastStructSaveTime;
    int m_saveIntervalSeconds = 5;
};

#endif // PARAMASSESSER_H
