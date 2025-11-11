#include "param_assesser.h"

GainAssesser::GainAssesser(std::string& strGainModelsRootPath)
{
    m_a2cGainInferer = initializeGainModel(strGainModelsRootPath, "a2c");
    m_a4cGainInferer = initializeGainModel(strGainModelsRootPath, "a4c");
    m_plaxGainInferer = initializeGainModel(strGainModelsRootPath, "plax");
}

GainAssesser::~GainAssesser()
{
    for (auto& strViewName : m_vecViewNames)
    {
        GainInfer* pCurrGainModel = selectSpecifiedGainModel(strViewName);
        if (pCurrGainModel != nullptr)
            delete pCurrGainModel;
    }
}

int GainAssesser::doInference(std::vector<cv::Mat>& vecInputImages, std::string& strViewName, std::vector<int>& vecGainScoreResults)
{
    GainInfer* pGainInfer = selectSpecifiedGainModel(strViewName);

    pGainInfer->doInference(vecInputImages, vecGainScoreResults);
    return std::accumulate(vecGainScoreResults.begin(), vecGainScoreResults.end(), 0);
}

GainInfer* GainAssesser::selectSpecifiedGainModel(std::string& strViewName)  // 所有切面名称均为小写，因为要与文件名配合
{
    if (strViewName == "a2c")
        return m_a2cGainInferer;

    else if (strViewName == "a4c")
        return m_a4cGainInferer;

    else if (strViewName == "plax")
        return m_plaxGainInferer;
}

GainInfer* GainAssesser::initializeGainModel(std::string strModelPath, std::string strViewName)
{
    GainInfer* pCurrGainModel = selectSpecifiedGainModel(strViewName);
    std::fstream _file;

    // 载入模型
    std::string strCurrModelPath = strModelPath + "/" + strViewName + "/" + "gain_" + strViewName + ".engine";

    _file.open(strCurrModelPath.c_str(), std::ios::in);
    assert(_file), "Cannot find gain_classification model model!";
    _file.close();

    const int inputSize = 112;                                     // 模型输入图像尺寸（由于TensorRT特性，更改后仅输入尺寸改变，模型并不会变化）
    std::vector<float> vecMeans = { 0.485f, 0.456f, 0.406f };         // 归一化均值
    std::vector<float> vecStds = { 0.229f, 0.224f, 0.225f };          // 归一化标准差

    cv::Size sizeInputSize{ inputSize, inputSize };

    pCurrGainModel = new GainInfer(strCurrModelPath, sizeInputSize, vecMeans, vecStds);
    return pCurrGainModel;
}

ParamAssesser::ParamAssesser(ConfigParse* config)
    : m_plaxParamsAssesser(new PLAXParamsAssess(config))
    , m_specParamsAssesser(new DoSpecParamsAssess(config))
    , m_funcParamsAssesser(new DoFuncParamsAssess(config))
    , m_lastSpecSaveTime(QDateTime::currentDateTime().addSecs(-30))
    , m_lastStructSaveTime(QDateTime::currentDateTime().addSecs(-30))
    , m_patientName("Unknown")
{
    std::string specEnginePath = "";
    bool ret = config->getSpecifiedNode("SPEC_CLASS_PATH", specEnginePath);
    //m_specModeInferer = new SPECClassInferer(specEnginePath);
    m_specUnionInferer = new SpecUnionCls(specEnginePath);

    std::string strGainModelsRoot = "D:/Resources/20240221/quality_control_models/gain_classification";
    ret = config->getSpecifiedNode("GAIN_CLASS_PATH", strGainModelsRoot);
    m_gainAssesser = new GainAssesser(strGainModelsRoot);
}

int ParamAssesser::doInferece(std::string& currViewName, std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex, ModeInfo& modeInfo)  // 现在常用的重载
{
    if (keyFrameIndex.empty())
        keyFrameIndex.push_back(0);

    if (vVideoClips.empty())
        return 0;

    if (modeInfo.mode == "B-Mode")
    {
        if (currViewName == "PLAX")
        {
            std::cout << "[I] Param: " << currViewName << std::endl;
            std::vector<std::string> vTexts;

            if (keyFrameIndex.empty() || vVideoClips.empty())
            {
                return 0;
            }
            doPLAXAssesser(vVideoClips, keyFrameIndex);
        }

        else if (currViewName == "A4C" || currViewName == "A2C")
        {
            if (keyFrameIndex.empty() || vVideoClips.empty() || keyFrameIndex.size() < 1)
            {
                return 0;
            }
            doFuncAssesser(currViewName, vVideoClips, keyFrameIndex);

            //std::vector<cv::Mat> inputVideoClips;
            //doFuncAssesser(inputVideoClips);
        }
    }

    else if (modeInfo.mode == "Doppler-Mode")
    {
        std::map<std::string, int> classResults;
        
        int keyframeIdx = 0;
        if (keyFrameIndex.back() >= vVideoClips.size())
            keyframeIdx = vVideoClips.size() - 1;

        //doSpecModeClass(vVideoClips[keyframeIdx], classResults);
        doSpecUnionModeClass(vVideoClips[keyframeIdx], classResults);
        std::string viewName = m_specUnionInferer->m_viewNameMap[classResults["view"]];
        //QtLogger::instance().logMessage(QString("[I] Spec Mode: %1").arg(QString::fromStdString(viewName)));

        //std::string viewName = m_specModeInferer->m_viewNameMap[classResults["view"]];
        //std::string modeName = m_specModeInferer->m_specModeMap[classResults["mode"]];

        //qDebug() << "View: " << QString::fromStdString(viewName);
        //qDebug() << "Mode: " << QString::fromStdString(modeName);
        
        //doSpecAssesser(vVideoClips, keyFrameIndex, classResults);
        doSpecAssesser(vVideoClips, keyFrameIndex, classResults);
        currViewName = viewName;
    }

    return 1;
}

// 20250305更新：将关键帧检测逻辑从各函数中移到param_assesser中，并且计算质量得分
int ParamAssesser::doQualityAssesser(std::string& viewName, std::vector<cv::Mat>& vVideoClips, std::vector<cv::Mat>& inputVideoClip, std::vector<int>& keyFrameIndex)
{
    int nCurrCycleLength = 0;
    if (viewName == "A2C" || viewName == "A4C")
        nCurrCycleLength = m_nLVEFCycleLength;
    else if (viewName == "PLAX")
        nCurrCycleLength = m_nPLAXCycleLength;

    ParamsAssessUtils::parseKeyframes(keyFrameIndex, vVideoClips, inputVideoClip, nCurrCycleLength);
    if (inputVideoClip.empty())
    {
        QtLogger::instance().logMessage(QString("[I] %1 Assessing Fail, input video size %2, keyframe size %3").arg(QString::fromStdString(viewName)).arg(vVideoClips.size()).arg(keyFrameIndex.size()));
        return 0;
    }

    std::vector<int> vecGainScoreResults;
    std::string strLowerViewName = viewName;
    std::transform(strLowerViewName.begin(), strLowerViewName.end(), strLowerViewName.begin(),
        [](unsigned char c) { return std::tolower(c); });
    int nQualityScore = m_gainAssesser->doInference(inputVideoClip, strLowerViewName, vecGainScoreResults);
    if (m_mapMaxQualityScore.find(viewName) != m_mapMaxQualityScore.end())
    {
        int nHistoryQualityScore = m_mapMaxQualityScore[viewName];
        if (nHistoryQualityScore >= nQualityScore)
        {
            QtLogger::instance().logMessage(QString("[I] Won't Compute %1, Current Quality Score %2, Max History Score %3")
                .arg(QString::fromStdString(viewName)).arg(nQualityScore).arg(nHistoryQualityScore));
            return 0;
        }
        else
        {
            QtLogger::instance().logMessage(QString("[I] Ready to Compute %1, Current Quality Score %2, Max History Score %3, Override History Score to %4")
                .arg(QString::fromStdString(viewName)).arg(nQualityScore).arg(nHistoryQualityScore).arg(nQualityScore));
            m_mapMaxQualityScore[viewName] = nQualityScore;
            return 1;
        }
    }
    else
    {
        QtLogger::instance().logMessage(QString("[I] Ready to Compute %1, Current Quality Score %2").arg(QString::fromStdString(viewName)).arg(nQualityScore));
        m_mapMaxQualityScore[viewName] = nQualityScore;
        return 1;
    }

    //return 1;
}


int ParamAssesser::doInferece(std::string& currViewName, std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex, ModeInfo& modeInfo, RoIScaleInfo& roiScaleInfo)
{
    if (keyFrameIndex.empty())
        keyFrameIndex.push_back(0);

    if (vVideoClips.empty())
        return 0;

    if (modeInfo.mode == "B-Mode")
    {
        if (currViewName == "PLAX")
        {
            std::cout << "[I] Param: " << currViewName << std::endl;
            std::vector<std::string> vTexts;

            if (keyFrameIndex.empty() || vVideoClips.empty())
            {
                return 0;
            }

            doPLAXAssesser(vVideoClips, keyFrameIndex);
            std::cout << "[I] Param assessed.\n";
        }

        else if (currViewName == "A4C" || currViewName == "A2C")
        {
            if (keyFrameIndex.empty() || vVideoClips.empty() || keyFrameIndex.size() < 2)
            {
                return 0;
            }

            std::vector<cv::Mat> inputVideoClips = vVideoClips;
            //cropSingleEchoCycle(vVideoClips, inputVideoClips, keyFrameIndex);
            doFuncAssesser(inputVideoClips);
            //doFuncAssesser(currViewName, vVideoClips, keyFrameIndex);
        }
    }

    else if (modeInfo.mode == "Doppler-Mode")
    {
        std::map<std::string, int> classResults;

        int keyframeIdx = 0;
        if (keyFrameIndex.back() >= vVideoClips.size())
            keyframeIdx = vVideoClips.size() - 1;

        doSpecUnionModeClass(vVideoClips[keyframeIdx], classResults);
        //doSpecModeClass(vVideoClips[keyframeIdx], classResults);
        //std::string viewName = m_specModeInferer->m_viewNameMap[classResults["view"]];
        //std::string modeName = m_specModeInferer->m_specModeMap[classResults["mode"]];

        doSpecAssesser(vVideoClips, keyFrameIndex, classResults);
    }
    return 1;
}

int ParamAssesser::doPLAXAssesser(std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex)
{
    if (vVideoClips.size() < m_nPLAXCycleLength)
        return 0;

    // 20250305更新：将关键帧检测逻辑从各函数中移到param_assesser中，并且计算质量得分
    std::vector<cv::Mat> inputVideoClips;
    std::string strViewName = "PLAX";
    //int ret = doQualityAssesser(strViewName, vVideoClips, inputVideoClips, keyFrameIndex);
    //if (!ret || inputVideoClips.empty())
    //    return 0;
    ParamsAssessUtils::parseKeyframes(keyFrameIndex, vVideoClips, inputVideoClips, m_nPLAXCycleLength);
    if (inputVideoClips.empty())
    {
        QtLogger::instance().logMessage(QString("[I] %1 Assessing Fail, input video size %2, keyframe size %3").arg(QString::fromStdString(strViewName)).arg(vVideoClips.size()).arg(keyFrameIndex.size()));
        return 0;
    }

    QMap<QString, QVector<float>> qParamsValues;
    QMap<QString, QImage> qResultImages;
    if (vVideoClips.size() < 5)
        return 0;

    m_plaxParamsAssesser->getStrucParamsRst(inputVideoClips, keyFrameIndex, qParamsValues, qResultImages);

    // 如果计算不成功，则本次计算的质量分不记录，下次还会测
    //if (!ret)
    //{
    //    deleteHistoryQualityScore(strViewName);
    //}

    QDateTime currentTime = QDateTime::currentDateTime();
    if (m_lastStructSaveTime.secsTo(currentTime) >= m_saveIntervalSeconds)
    {
        saveStructInferenceImage(QString("PLAX"), vVideoClips, keyFrameIndex, qResultImages);
        m_lastSpecSaveTime = currentTime;
    }

    m_currValueMap = qParamsValues;
    m_currPremiumsMap = qResultImages;

    return 1;
}

int ParamAssesser::doSpecModeClass(cv::Mat& src, std::map<std::string, int>& classResults)
{
    m_specModeInferer->doInference(src, classResults);
    return 1;
}

int ParamAssesser::doSpecUnionModeClass(cv::Mat& src, std::map<std::string, int>& classResults)
{
    m_specUnionInferer->doInference(src, classResults);
    return 1;
}

int ParamAssesser::doSpecAssesser(std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex, std::map<std::string, int>& classResults)
{
    if (vVideoClips.empty())
        return 0;

    if (keyFrameIndex.empty())
        keyFrameIndex.push_back(0);

    std::map<std::string, std::vector<float>> values;
    std::map<std::string, cv::Mat> resultPics;

    int keyframeIdx = 0;
    //if (keyFrameIndex.back() >= vVideoClips.size())
    //    keyframeIdx = vVideoClips.size() - 1;

    cv::Mat inferFrame = *vVideoClips.begin();
    int ret = m_specParamsAssesser->getSpecParamsRstV2(inferFrame, values, resultPics, classResults);

    if (ret)
    {
        for (auto& pair : values)
        {
            QString eventName = QString::fromStdString(pair.first);
            cv::Mat eventPremiumImage = resultPics[eventName.toStdString()];
            std::vector<float> eventValues = pair.second;
            QVector<float> qEventValues(eventValues.begin(), eventValues.end());

            m_currSpecValueMap.insert(eventName, qEventValues);

            QDateTime currentTime = QDateTime::currentDateTime();
            if (m_lastSpecSaveTime.secsTo(currentTime) >= m_saveIntervalSeconds)
            {
                saveSpecInferenceImage(eventName, inferFrame, false);
                //saveSpecInferenceImage(eventName, eventPremiumImage, true);
                m_lastSpecSaveTime = currentTime;
            }
            
        }

        for (auto& pair : resultPics)
        {
            //cv::imshow("test", pair.second);
            //cv::waitKey(0);
            m_currSpecPremiumsMap.insert(QString::fromStdString(pair.first), GeneralUtils::matToQImage(pair.second));
        }
    }
    else
    {
        QString eventName = "OTHER";
        QDateTime currentTime = QDateTime::currentDateTime();
        if (m_lastSpecSaveTime.secsTo(currentTime) >= m_saveIntervalSeconds)
        {
            //saveSpecInferenceImage(eventName, inferFrame, false);

            m_lastSpecSaveTime = currentTime;
        }
    }

    return 1;
}

int ParamAssesser::doFuncAssesser(std::vector<cv::Mat>& vVideoClips)
{
    std::map<std::string, std::vector<float>> values;
    std::map<std::string, cv::Mat> resultPics;

    m_funcParamsAssesser->getFuncParamsRst(vVideoClips, values, resultPics);

    for (auto& pair : values)
    {
        QString eventName = QString::fromStdString(pair.first);
        std::vector<float> eventValues = pair.second;
        QVector<float> qEventValues(eventValues.begin(), eventValues.end());

        m_currFuncValueMap.insert(eventName, qEventValues);
    }

    for (auto& pair : resultPics)
    {
        //cv::imshow("test", pair.second);
        //cv::waitKey(0);
        m_currFuncPremiumsMap.insert(QString::fromStdString(pair.first), GeneralUtils::matToQImage(pair.second));
    }
    return 1;
}

int ParamAssesser::doFuncAssesser(std::string& viewName, std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex)
{
    std::map<std::string, std::vector<float>> values;
    std::map<std::string, cv::Mat> resultPics;

    // 20250305更新：将关键帧检测逻辑从各函数中移到param_assesser中，并且计算质量得分
    std::vector<cv::Mat> inputVideoClips;
    std::string strViewName = viewName;
    // 20250318更新：去除在参数测量param_assesser中计算质量分
    //int ret = doQualityAssesser(strViewName, vVideoClips, inputVideoClips, keyFrameIndex);
    //if (!ret || inputVideoClips.empty())
    //    return 0;
    ParamsAssessUtils::parseKeyframes(keyFrameIndex, vVideoClips, inputVideoClips, m_nLVEFCycleLength);
    if (inputVideoClips.empty())
    {
        QtLogger::instance().logMessage(QString("[I] %1 Assessing Fail, input video size %2, keyframe size %3").arg(QString::fromStdString(viewName)).arg(vVideoClips.size()).arg(keyFrameIndex.size()));
        return 0;
    }

    int ret = m_funcParamsAssesser->getFuncParamsRst(viewName, inputVideoClips, keyFrameIndex, values, resultPics);

    // 如果计算不成功，则本次计算的质量分不记录，下次还会测
    if (!ret)
    {
        deleteHistoryQualityScore(viewName);
    }

    for (auto& pair : values)
    {
        QString eventName = QString::fromStdString(pair.first);
        std::vector<float> eventValues = pair.second;
        QVector<float> qEventValues(eventValues.begin(), eventValues.end());

        m_currFuncValueMap.insert(eventName, qEventValues);
    }

    for (auto& pair : resultPics)
    {
        //cv::imshow("test", pair.second);
        //cv::waitKey(0);
        m_currFuncPremiumsMap.insert(QString::fromStdString(pair.first), GeneralUtils::matToQImage(pair.second));
    }
    return ret;
}

int ParamAssesser::deleteHistoryQualityScore(std::string& strViewName)
{
    //std::string strLowerViewName = strViewName;
    //std::transform(strLowerViewName.begin(), strLowerViewName.end(), strLowerViewName.begin(),
    //    [](unsigned char c) { return std::tolower(c); });

    QtLogger::instance().logMessage(QString("Compute fail, Delete %1 History Quality Score.").arg(QString::fromStdString(strViewName)));

    m_mapMaxQualityScore.erase(strViewName);

    //auto it = m_mapMaxQualityScore.find(strLowerViewName);
    //if (it != m_mapMaxQualityScore.end()) {
    //    m_mapMaxQualityScore.erase(it);
    //}
    return 1;
}

int ParamAssesser::cropSingleEchoCycle(std::vector<cv::Mat>& vVideoClips, std::vector<cv::Mat>& sampledVideoClips, std::vector<int>& keyframeIdxes)
{
    int start = keyframeIdxes[0];
    int end = keyframeIdxes[1];
    if (start < vVideoClips.size() && end < vVideoClips.size())
    {
        if (start - 2 < 0)
        {
            int padNum = 2 - start;
            for (int i = 0; i < padNum; i++)
            {
                sampledVideoClips.push_back(vVideoClips[0]);
            }
        }
        sampledVideoClips = std::vector<cv::Mat>(vVideoClips.begin() + std::max(start - 2, 0), vVideoClips.begin() + end);
    }
    return 1;
}

int ParamAssesser::saveSpecInferenceImage(QString& eventName, cv::Mat& img, bool isPremium)
{
    if (eventName.indexOf("JG") != -1)
        eventName = "TDIMVIVS";
    else if (eventName.indexOf("CB") != -1)
        eventName = "TDIMVLW";
    else if (eventName.indexOf("VTI") != -1)
        return 0;

    if (img.empty())
        return 0;

    QString fileName = "";
    if (isPremium)
        fileName  = generateDate() + "_premium_" + ".jpg";
    else
        fileName = generateDate() + ".jpg";

    QString folderPath = m_specImageSaveRootPath + "/" + eventName;
    QDir().mkpath(folderPath);

    QString filePath = folderPath + "/" + fileName;
    cv::imwrite(filePath.toStdString(), img);
    return 1;
}

int ParamAssesser::saveStructInferenceImage(QString viewName, 
    std::vector<cv::Mat>& vInferVideoClip,
    std::vector<int>& vKeyframeIndexes, 
    QMap<QString, QImage>& qmPremiums)
{
    QDateTime currentDataTime = QDateTime::currentDateTime();
    QString currentDate = currentDataTime.toString("yyyyMMdd"); // 当天日期，格式：年月日
    QString currentTime = currentDataTime.toString("HHmmss"); // 当前时间，格式：时分秒

    QString currDateTime = generateDate();
    QString videoSaveFolderName = m_structImageSaveRootPath + "/origin_videos/" + viewName + "/" + currDateTime;
    //QString premiumSaveFolderName = m_structImageSaveRootPath + "/premiums/" + viewName + "/" + currDateTime;
    QString premiumSaveFolderName = m_structImageSaveRootPath + "/premiums/" + viewName + "/" + currentDate + "/" + m_patientName + "/" + currentTime;
    QDir().mkpath(videoSaveFolderName);
    QDir().mkpath(premiumSaveFolderName);

    int counter = 0;
    for (auto& frame : vInferVideoClip)
    {
        QString currFrameSavePath = "";
        if (std::find(vKeyframeIndexes.begin(), vKeyframeIndexes.end(), counter) != vKeyframeIndexes.end())
            currFrameSavePath = videoSaveFolderName + "/" + QString("image_%1_keyframe.jpg").arg(counter, 5, 10, QChar('0'));
        else
            currFrameSavePath = videoSaveFolderName + "/" + QString("image_%1.jpg").arg(counter, 5, 10, QChar('0'));
        cv::imwrite(currFrameSavePath.toStdString(), frame);
        ++counter;
    }

    for (auto& eventName : qmPremiums.keys())
    {
        QString currFrameSavePath = premiumSaveFolderName + "/" + QString("premium_%1.jpg").arg(eventName);
        cv::Mat currPremium = GeneralUtils::qImage2cvMat(qmPremiums[eventName]);
        cv::imwrite(currFrameSavePath.toStdString(), currPremium);
    }

    return 1;
}

void ParamAssesser::clearPatientHistoryRecord()
{
    m_mapMaxQualityScore.clear();
}

int ParamAssesser::setPatientName(QString patientName)
{
    if (patientName != m_prevPatientName)
    {
        QtLogger::instance().logMessage(QString("Patient's name changed, Previous: %1, Current: %2. Clear all Quality Scores.").arg(m_prevPatientName).arg(m_patientName));
        clearPatientHistoryRecord();
    }

    m_prevPatientName = m_patientName;

    if (patientName.isEmpty())
        m_patientName = "Unknown";
    else
        m_patientName = patientName;
    return 1;
}