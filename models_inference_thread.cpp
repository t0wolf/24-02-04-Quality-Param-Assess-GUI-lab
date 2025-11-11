#include "models_inference_thread.h"

ModelsInferenceThread::ModelsInferenceThread(QObject *parent, ProgressSuperThread *progressThread, DataBuffer* dataBuffer, ConfigParse* config)
    : QThread(parent)
    , m_paramOnlyMode(true)
    //, m_keyframeDetector(new KeyframeDetector())
    , m_progressThread(progressThread)
    , m_infoExtractor(new InfoExtractor())
    , m_qualityControlThread(new QualityControlThread(this))
    , m_paramAssessThread(new ParamAssessThread(this, config))
    , m_roiDataBuffer(dataBuffer)
    , m_lastSaveTime(QDateTime::currentDateTime().addSecs(-30))
    , m_fCurrQualityScore(-10000.0f)
{
    parseConfigFile(config);
    m_clsDataBuffer = new ViewClsInferBuffer();
    m_keyDataBuffer = new KeyframeDataBuffer();
    m_viewClsKeyframeInferThread = new ViewClsKeyframeInferThread(this, m_clsDataBuffer, config);
    m_keyframeInferThread = new KeyframeInferThread(this, m_keyDataBuffer);
    
    m_qualityControlThread->start();
    m_paramAssessThread->start();
    m_viewClsKeyframeInferThread->start();
    m_keyframeInferThread->start();
}

int ModelsInferenceThread::parseConfigFile(ConfigParse* config)
{
    std::string paramOnlyMode = "";
    bool ret = config->getSpecifiedNode("PARAM_ONLY_MODE", paramOnlyMode);
    if (ret)
    {
        try
        {
            m_paramOnlyMode = static_cast<bool>(std::stoi(paramOnlyMode));
        }
        catch (std::invalid_argument& e)
        {
            m_paramOnlyMode = true;
        }
    }

    std::string strQualityScoreThresh = "";
    ret = config->getSpecifiedNode("QUALITY_SCORE_THRESH", strQualityScoreThresh);
    if (ret)
    {
        try
        {
            m_fQualityScoreThresh = std::stof(strQualityScoreThresh);
        }
        catch (std::invalid_argument& e)
        {
            m_fQualityScoreThresh = 0.4f;
        }
    }

    std::string strParamDiffRatioThresh = "";
    ret = config->getSpecifiedNode("PARAM_DIFF_RATIO_THRESH", strParamDiffRatioThresh);
    if (ret)
    {
        try
        {
            m_fDiffRatioThresh = std::stof(strParamDiffRatioThresh);
        }
        catch (std::invalid_argument& e)
        {
            m_fDiffRatioThresh = 0.05f;
        }
    }

    std::string strStructAortaDiffRatioThresh = "";
    ret = config->getSpecifiedNode("STRUCT_AORTA_DIFF_RATIO_THRESH", strStructAortaDiffRatioThresh);
    if (ret)
    {
        try
        {
            m_fStructAortaDiffRatioThresh = std::stof(strStructAortaDiffRatioThresh);
        }
        catch (std::invalid_argument& e)
        {
            m_fStructAortaDiffRatioThresh = 0.03f;
        }
    }

    std::string strSpecParamDiffRatioThresh = "";
    ret = config->getSpecifiedNode("SPEC_PARAM_DIFF_RATIO_THRESH", strSpecParamDiffRatioThresh);
    if (ret)
    {
        try
        {
            m_fSpecDiffRatioThresh = std::stof(strSpecParamDiffRatioThresh);
        }
        catch (std::invalid_argument& e)
        {
            m_fSpecDiffRatioThresh = 0.05f;
        }
    }

    std::string strBestQualityScoreOnly = "";
    ret = config->getSpecifiedNode("BEST_QUALITY_SCORE_ONLY", strBestQualityScoreOnly);
    if (ret)
    {
        try
        {
            m_bBestQualityScoreOnly = static_cast<bool>(std::stoi(strBestQualityScoreOnly));
        }
        catch (std::invalid_argument& e)
        {
            m_bBestQualityScoreOnly = false;
        }
    }

    std::string strAllParamsDiffValid = "";
    ret = config->getSpecifiedNode("ALL_PARAMS_DIFF_VALID", strAllParamsDiffValid);
    if (ret)
    {
        try
        {
            m_bAllParamsDiffValid = static_cast<bool>(std::stoi(strAllParamsDiffValid));
        }
        catch (std::invalid_argument& e)
        {
            m_bAllParamsDiffValid = false;
        }
    }

    std::string strLVIDSelectMaxValue = "";
    ret = config->getSpecifiedNode("LVID_SELECT_MAX", strLVIDSelectMaxValue);
    if (ret)
    {
        try
        {
            m_bLVIDSelectMaxValue = static_cast<bool>(std::stoi(strLVIDSelectMaxValue));
        }
        catch (std::invalid_argument& e)
        {
            m_bLVIDSelectMaxValue = false;
        }
    }

    std::string strAADSelectMaxValue = "";
    ret = config->getSpecifiedNode("AAD_SELECT_MAX", strAADSelectMaxValue);
    if (ret)
    {
        try
        {
            m_bAADSelectMaxValue = static_cast<bool>(std::stoi(strAADSelectMaxValue));
        }
        catch (std::invalid_argument& e)
        {
            m_bAADSelectMaxValue = false;
        }
    }

    return 1;
}

void ModelsInferenceThread::run()
{
    ScaleInfo currScaleInfo;
    ModeInfo currModeInfo;
    RoIScaleInfo currROIScaleInfo;

    while (!this->isInterruptionRequested())
    {
        cv::Mat frame;
        std::vector<PeakInfo> vPeaks;
        QVector<int> keyframeIdxes;

        // roi检测线程发送给RoI信息、Scale的信息
        {
            QMutexLocker locker(&m_mutex);
            if (m_hasNewFrame)
            {
                frame = m_nextFrame.clone();
                m_hasNewFrame = false;
            }

            updateRoIData(currROIScaleInfo);
            updateScaleInfo(currScaleInfo, currModeInfo);
        }

        if (!frame.empty())
        {
            QVector<int> keyframeIdx;
            QString currViewName = "";
            cv::Mat inferFrame/* = frame.clone()*/;
            cv::Mat originFrame = frame.clone();
            RoIScaleInfo assessROIScaleInfo = currROIScaleInfo;

            // ================ ROI Detection ===================
            cv::Mat frameCropped;
            if (m_frameCounter % 30 == 0 || m_currROIScaleInfo.roiRect.empty())
                m_currROIScaleInfo = currROIScaleInfo;

            frameCropping(frame, frameCropped, assessROIScaleInfo);

            if (currScaleInfo.length != -10000.0f && currScaleInfo.fPixelPerUnit == -10000.0f && !currScaleInfo.unit.empty() && currROIScaleInfo.fRadius > 0.0f)
            {
                if (currROIScaleInfo.fRadius > frame.rows / 2)
                    currScaleInfo.fPixelPerUnit = currROIScaleInfo.fRadius / currScaleInfo.length;
            }
            // ==================================================

            // ============= Spec Classification ===============
            //currViewModeJudgement(currROIScaleInfo);
            // ***********************************************
            // 判断图像是否为彩图（是否扫血流）
            // ***********************************************
            if (ImageProcess::isColorJudge(frameCropped))
            {
                currModeInfo.bIsColorMode = true;
            }
            currViewModeJudgement(currModeInfo, currROIScaleInfo);

            // 从频谱切换到B超模式时，清空原有采集缓存
            if (m_currModeInfo.mode == "Doppler-Mode" && m_prevModeInfo.mode == "B-Mode" && !m_vVideoBuffer.empty() && m_isViewRecognized && !m_currViewName.isEmpty())
                blankModeProcess();
            m_prevModeInfo = m_currModeInfo;
            // =================================================

            //m_vOriginVideoBuffer.push_back(originFrame);

            // blank mode process
            if (m_blankFrameCounter >= 20)
            {
                //sigDebugText(QString("[I] Blank mode."));
                sigDebugText(QString("[I] 检查暂停。"));
                blankModeProcess();
            }

            if (m_currModeInfo.mode == "Doppler-Mode" || m_currModeInfo.mode == "B-Mode")
            {
                inferFrame = frameCropped.clone();
                m_vVideoBuffer.push_back(frameCropped.clone());
            }

            else
            {
                if (m_currModeInfo.mode == "Color-Mode")
                {
                    ++m_blankFrameCounter;
                    continue;
                }
            }

            //if ((m_currModeInfo.mode == "Doppler-Mode" || m_currModeInfo.mode == "B-Mode") 
            //    && !m_isQualityControlThreadRunning
            //    && !m_isParamAssessThreadRunning)
            //{
            //    inferFrame = frameCropped.clone();
            //    m_vVideoBuffer.push_back(frameCropped.clone());
            //}

            //else
            //{
            //    if (!m_isQualityControlThreadRunning && !m_isParamAssessThreadRunning && m_currModeInfo.mode == "Color-Mode")
            //    {
            //        ++m_blankFrameCounter;
            //        continue;
            //    }
            //}

            //if (m_frameCounter % 5 == 0)
            //    currROIScaleInfo.clear();

            m_blankFrameCounter = 0;
            m_isBlankMode = false;
            m_vOriginVideoBuffer.push_back(originFrame);

            //handleViewClassificationProcess(inferFrame, currROIScaleInfo);

            //handleKeyframeSampling(inferFrame, keyframeIdxes, vPeaks);

            //handleQualityControl(assessROIScaleInfo);

            //handleParamAssess(inferFrame, currScaleInfo);

            //checkVideoBufferSize();

            std::vector<cv::Mat> vecViewClassImages = handleViewClassificationProcess(inferFrame);
            QString strCurrViewName;
            float fCurrQualityScore = -10000.0f;

            if (!vecViewClassImages.empty() || m_isSpecViewMode)
            {
                strCurrViewName = m_currViewName;
                fCurrQualityScore = m_fCurrQualityScore;

                bool bIsReadyToParam = false;

                bIsReadyToParam = judgeQualityScoreToParam(strCurrViewName, fCurrQualityScore);
                if (bIsReadyToParam || m_isSpecViewMode)
                {
                    handleKeyframeSampling(vecViewClassImages, keyframeIdxes, vPeaks);

                    handleParamAssess(strCurrViewName, inferFrame, vecViewClassImages, currScaleInfo);
                }
            }

            if (m_isParamAssessThreadRunning && m_bParamValuesUpdateFlag)
            {
                int ret = sendParamAssessEndSignal(currScaleInfo);

                m_bParamValuesUpdateFlag = false;
                m_isParamAssessThreadRunning = false;

                //m_isViewRecognized = false;
                m_isKeyframeDetected = false;
                m_isA4CSampleFlag = false;
            }
            //handleQualityControl(assessROIScaleInfo);

            checkVideoBufferSize();
            ++m_frameCounter;
        }
    }
}

//int ModelsInferenceThread::processFrame(cv::Mat& frame, ScaleInfo& currScaleInfo, ModeInfo& currModeInfo, RoIScaleInfo& currROIScaleInfo)
//{
//    cv::Mat inferFrame = frame.clone();
//    cv::Mat originFrame = frame.clone();
//    RoIScaleInfo assessROIScaleInfo = currROIScaleInfo;
//
//    cv::Mat frameCropped;
//    updateRoIScale(frame, frameCropped, currROIScaleInfo);
//
//    if (isScaleInfoValid(currScaleInfo, currROIScaleInfo)) {
//        currScaleInfo.fPixelPerUnit = computePixelPerUnit(currScaleInfo, currROIScaleInfo, frame.rows);
//    }
//
//    processModes(inferFrame, originFrame, currScaleInfo, currModeInfo, currROIScaleInfo, assessROIScaleInfo);
//    bufferManagement(originFrame);
//
//    checkVideoBufferSize();
//    ++m_frameCounter;
//
//    return 1;
//}

void ModelsInferenceThread::exitThread()
{
    this->requestInterruption();
    this->quit();
    this->wait();
    m_currROIScaleInfo.clear();
}

int ModelsInferenceThread::blankModeProcess()
{
    m_blankFrameCounter = 0;

    {
        QMutexLocker locker(&m_videoBufferMutex);
        m_vVideoBuffer.clear();
        m_vecViewClassBuffer.clear();
    }

    cv::Mat emptyFrame;
    m_clsDataBuffer->updateViewClsImage(emptyFrame);

    //m_keyframeDetector->clearFeatMemory();
    m_keyframeInferThread->clearAllMemoryCache();

    sendImageSamplingInterruptSignal();

    m_isViewRecognized = false;
    m_isKeyframeDetected = false;
    m_isQualityControlled = false;
    m_isA4CSampleFlag = false;
    m_isBlankMode = true;

    //m_vVideoBuffer.clear();
    //m_vOriginVideoBuffer.clear();

    return 1;
}

void ModelsInferenceThread::clearDataCache()
{
    //m_keyframeDetector->clearFeatMemory();
    m_paramAssessThread->clearLVEFDataCache();
    m_histParamValues.clear();
    m_mapHistoryQualityScore.clear();
}

int ModelsInferenceThread::updateRoIData(RoIScaleInfo& roiScaleInfo)
{
    if (m_roiDataBuffer->checkRoIDataUpdate()) {
        m_roiDataBuffer->getData(roiScaleInfo);
    }
    return 1;
}

int ModelsInferenceThread::updateScaleInfo(ScaleInfo& scaleInfo, ModeInfo& modeInfo)
{
    if (m_isScaleInfoUpdated) {
        scaleInfo = m_currScaleInfo;
        modeInfo = m_currModeInfo;
        m_isScaleInfoUpdated = false;
    }
    return 1;
}

int ModelsInferenceThread::updateRoIScale(cv::Mat& frame, cv::Mat frameCropped, RoIScaleInfo& roiScaleInfo)
{
    if (m_frameCounter % 30 == 0 || m_currROIScaleInfo.roiRect.empty()) {
        m_currROIScaleInfo = roiScaleInfo;
    }

    frameCropping(frame, frameCropped, roiScaleInfo);
    return 1;
}

bool ModelsInferenceThread::isQualityParamRunning()
{
    if (m_isQualityControlThreadRunning || m_isParamAssessThreadRunning)
        return true;
    return false;
}

int ModelsInferenceThread::handleBlankViewMode()
{
    return 1;
}

int ModelsInferenceThread::handleViewClassificationProcess(cv::Mat& inferFrame, RoIScaleInfo& assessROIScaleInfo)
{
    int viewIdx = -1;
    QString currViewName = "OTHER";

    if (!m_isSpecViewMode && m_isBViewMode && !m_isViewRecognized) 
    {
        m_vecViewClassBuffer.push_back(inferFrame);
        m_clsDataBuffer->updateViewClsImage(inferFrame);
    }

    if (m_clsDataBuffer->hasViewClsResult()) 
    {
        ViewClsResult viewClsResult = m_clsDataBuffer->getViewClsResult();
        QString strCurrViewName = viewClsResult.strViewName;
        viewIdx = 1;
        //sigDebugText(QString("[I] Get View Classification Result, View Name: %1, viewClsBuffer's Size: %2").arg(strCurrViewName).arg(m_vecViewClassBuffer.size()));
        sigDebugText(QString("[I] 获取切面分类结果，切面名称 : %1，切面视频的大小 : %2").arg(strCurrViewName).arg(m_vecViewClassBuffer.size()));
    }

    if (viewIdx >= 0
        && !m_isViewRecognized
        && !m_isSpecViewMode 
        && m_isBViewMode 
        //&& isQualityParamRunning()
        ) 
    {
        //sigDebugText(QString("[I] View, %1 %2").arg(currViewName).arg(m_vVideoBuffer.size()));

        parseViewClassInferResult(currViewName);
        if (m_currViewName == "A4C" && !m_isA4CSampleFlag) {
            m_isA4CSampleFlag = true;
        }

        //if (!isQualityParamRunning())
        //{
        //    m_vVideoBuffer.clear();
        //    m_vOriginVideoBuffer.clear();
        //    //m_keyframeInferThread->clearAllMemoryCache();
        //}
    }

    return 1;
}

std::vector<cv::Mat> ModelsInferenceThread::handleViewClassificationProcess(cv::Mat& inferFrame)  // 20250318常用重载
{
    int viewIdx = -1;
    QString currViewName = "OTHER";
    float fCurrQualityScore = -10000.0f;
    bool bIsSwitch = false;

    if (m_isSpecViewMode)
    {
        m_currViewName = "OTHER";
        m_fCurrQualityScore = -10000.0f;
        return std::vector<cv::Mat>{ inferFrame };
    }

    if (!m_isSpecViewMode 
        && m_isBViewMode 
        && !m_isViewRecognized 
        && !m_isBlankMode
        && !isQualityParamRunning()
        )
    {
        if (m_vecViewClassBuffer.size() < m_viewClsKeyframeInferThread->getViewClassClipSize())
        {
            m_vecViewClassBuffer.push_back(inferFrame);
            m_clsDataBuffer->updateViewClsImage(inferFrame);
        }
        else
        {
            QElapsedTimer timer;
            timer.start();

            while (true)
            {
                // 检查是否已经等待超过2秒
                if (timer.elapsed() > 1500) // 1500毫秒 = 1.5秒
                {
                    //sigDebugText(QString("[I] Timeout: Waiting for View Classification Result exceeded 1.5 seconds."));
                    sigDebugText(QString("[I] 超时：等待视图分类结果超过 1.5 秒。"));
                    m_vecViewClassBuffer.clear();

                    cv::Mat emptyImage;
                    m_clsDataBuffer->updateViewClsImage(emptyImage);  // 清空切面分类的缓存
                    break; // 超过2秒，跳出循环
                }

                if (m_clsDataBuffer->hasViewClsResult())
                {
                    ViewClsResult viewClsResult = m_clsDataBuffer->getViewClsResult();
                    currViewName = viewClsResult.strViewName;
                    fCurrQualityScore = viewClsResult.fQualityScore;
                    bIsSwitch = viewClsResult.bIsSwitch;

                    viewIdx = 1;
                    sigDebugText(QString("[I] 得到切面分类结果, 切面名称 : %1, 切面视频帧数 : %2, 质量 : %3, 是否发生切面切换 : %4")
                        .arg(currViewName)
                        .arg(m_vecViewClassBuffer.size())
                        .arg(fCurrQualityScore)
                        .arg(bIsSwitch));
                    QtLogger::instance().logMessage(
                        QString("[I] Get View Classification Result, View Name: %1, viewClsBuffer's Size: %2, Quality: %3, Switch: %4")
                            .arg(currViewName)
                            .arg(m_vecViewClassBuffer.size())
                            .arg(fCurrQualityScore)
                            .arg(bIsSwitch));
                    emit setimagesShow(m_vecViewClassBuffer);
                    break;
                }
                sigDebugText(QString("[I] 等待切面分类结果, %1").arg(m_vecViewClassBuffer.size()));
            }
        }
    }

    if (viewIdx >= 0
        && !m_isViewRecognized
        && !bIsSwitch
        //&& !m_isSpecViewMode
        //&& m_isBViewMode
        //&& isQualityParamRunning()
        )
    {
        //sigDebugText(QString("[I] View, %1 %2").arg(currViewName).arg(m_vVideoBuffer.size()));

        //parseViewClassInferResult(currViewName);
        parseViewClassInferResult(currViewName, fCurrQualityScore, m_vecViewClassBuffer);
        if (m_currViewName == "A4C" && !m_isA4CSampleFlag) {
            m_isA4CSampleFlag = true;
        }
        if (!m_isViewRecognized)
            return std::vector<cv::Mat>();

        std::vector<cv::Mat> vecViewClassVideo(m_vecViewClassBuffer.begin(), m_vecViewClassBuffer.end());
        m_vecViewClassBuffer.clear();
        m_isViewRecognized = false;
        return vecViewClassVideo;
    }

    return std::vector<cv::Mat>();
}

int ModelsInferenceThread::handleKeyframeSampling(cv::Mat& frameCropped, QVector<int>& keyframeIdxes, std::vector<PeakInfo>& vPeaks)
{
    if ((m_isViewRecognized 
        && !m_isSpecViewMode 
        && m_isBViewMode
        && m_vVideoBuffer.size() < m_currKeyframeSampleNum
        && !m_isQualityControlThreadRunning
        && !m_isParamAssessThreadRunning
        )
        || (m_isA4CSampleFlag && m_vVideoBuffer.size() < m_a4cSampleNum))
    {
        sigDebugText(QString("[I] Keying - %1 %2").arg(m_currViewName).arg(m_vVideoBuffer.size()));

        QString strTempViewName = QString(m_currViewName);
        QString strTempInferMode = QString("backbone");
        m_keyDataBuffer->updateKeyDetInfo(frameCropped, strTempViewName, strTempInferMode);
    }

    else if ((m_isViewRecognized 
        && !m_isSpecViewMode 
        && m_isBViewMode 
        && !m_isKeyframeDetected
        && !m_vVideoBuffer.empty()
        && m_vVideoBuffer.size() >= m_currKeyframeSampleNum
        && !isQualityParamRunning()
        ))
    {
        // ============= Keyframe Detection ================
        parseKeyframeDetResult(frameCropped, keyframeIdxes, vPeaks);
        //sigDebugText(QString("[I] Keyed - %1 %2 %3").arg(m_currViewName).arg(m_vVideoBuffer.size()).arg(keyframeIdxes.size()));
        // =================================================
    }

    return 1;
}

// 20250312更新，替换原有的关键帧检测和后处理逻辑，直接处理切面分类输出的结果，不再额外采样
int ModelsInferenceThread::handleKeyframeSampling(std::vector<cv::Mat>& vecVideoBuffer, QVector<int>& keyframeIdxes, std::vector<PeakInfo>& vPeaks)
{
    if (!vecVideoBuffer.empty())
    {
        QString strTempViewName = QString(m_currViewName);
        QString strBackboneInferMode = QString("backbone");
        QString strSGTAInferMode = QString("sgta");

        if (!m_progressThread->isKeyframeEnable(m_currViewName.toStdString()))
        {
            parseKeyframeDetResult(vecVideoBuffer[0], keyframeIdxes, vPeaks);
            {
                QMutexLocker locker(&m_keyframeBufferMutex);
                if (!m_isKeyframeDetected)
                    m_vecKeyframeBuffer = vecVideoBuffer;
            }
            return 0;
        }
        int nCounter = 0;
        for (auto& frame : vecVideoBuffer)
        {
            // backbone inference
            if (nCounter < vecVideoBuffer.size() - 1)
            {
                m_keyDataBuffer->updateKeyDetInfo(frame, strTempViewName, strBackboneInferMode);
                sigDebugText(QString("[I] Keying - %1 %2").arg(m_currViewName).arg(m_vVideoBuffer.size()));
            }

            // aggregation inference
            else
            {
                parseKeyframeDetResult(frame, keyframeIdxes, vPeaks);
                //sigDebugText(QString("[I] Keyed - %1 %2 %3").arg(m_currViewName).arg(m_vVideoBuffer.size()).arg(keyframeIdxes.size()));
            }

            nCounter++;
        }

        {
            QMutexLocker locker(&m_keyframeBufferMutex);
            if (!m_isKeyframeDetected)
                m_vecKeyframeBuffer = vecVideoBuffer;
        }
        return 1;
    }
    return 0;
}

int ModelsInferenceThread::handleQualityControl(RoIScaleInfo& assessROIScaleInfo)
{
    if (m_isViewRecognized && m_isKeyframeDetected && !m_isQualityControlThreadRunning && !m_isParamAssessThreadRunning && !m_vVideoBuffer.empty())
    {
        sigDebugText(QString("[I] QCing, %1 %2 %3").arg(m_currViewName).arg(m_vVideoBuffer.size()).arg(m_currKeyframeIdxes.size()));
        //saveInferenceImage(m_currViewName, m_vVideoBuffer);

        if (!m_paramOnlyMode) {
            sendQualityControlBeginSignal(assessROIScaleInfo);
            m_isQualityControlThreadRunning = true;
        }
        else {
            m_isQualityControlled = true;
        }

        m_isViewRecognized = false;
        //m_isKeyframeDetected = false;
        m_isA4CSampleFlag = false;
    }

    else if (m_bQualityScoresUpdateFlag && m_isQualityControlThreadRunning)
    {
        int ret = sendQualityControlEndSignal();
        sigDebugText(QString("[I] QCed, %1 %2").arg(m_currViewName).arg(m_vVideoBuffer.size()));
        
        if (ret)
            m_isQualityControlled = true;

        else
        {
            m_vVideoBuffer.clear();
            m_vOriginVideoBuffer.clear();
        }

        m_bQualityScoresUpdateFlag = false;
        m_isQualityControlThreadRunning = false;
    }
    return 1;
}

int ModelsInferenceThread::handleParamAssess(cv::Mat& inferFrame, ScaleInfo& scaleInfo)
{
    if ((m_isQualityControlled && m_isBViewMode && !m_isParamAssessThreadRunning && !m_isQualityControlThreadRunning && m_isKeyframeDetected)
        || (m_isSpecViewMode && !m_isParamAssessThreadRunning))
    {
        //sigDebugText(QString("[I] Paraming - %1 %2 %3").arg(m_currViewName).arg(m_vVideoBuffer.size()).arg(m_currKeyframeIdxes.size()));
        sigDebugText(QString("[I] 测值结果 - %1 %2 %3").arg(m_currViewName).arg(m_vVideoBuffer.size()).arg(m_currKeyframeIdxes.size()));
        sendParamAssessBeginSignal(inferFrame, scaleInfo);
    }

    if (m_isParamAssessThreadRunning && m_bParamValuesUpdateFlag)
    {
        int ret = sendParamAssessEndSignal(scaleInfo);

        m_bParamValuesUpdateFlag = false;
        m_isParamAssessThreadRunning = false;

        m_isViewRecognized = false;
        m_isKeyframeDetected = false;
        m_isA4CSampleFlag = false;
    }
    return 1;
}

int ModelsInferenceThread::handleParamAssess(QString& strCurrViewName, cv::Mat& inferFrame, std::vector<cv::Mat>& vecInferImages, ScaleInfo& scaleInfo)
{
    if ((m_isBViewMode && !m_isParamAssessThreadRunning && !m_isQualityControlThreadRunning && m_isKeyframeDetected)
        || (m_isSpecViewMode && !m_isParamAssessThreadRunning))
    {
        //sigDebugText(QString("[I] Paraming - %1 %2 %3").arg(m_currViewName).arg(vecInferImages.size()).arg(m_currKeyframeIdxes.size()));
        sigDebugText(QString("[I] 测值结果 - %1 %2 %3").arg(m_currViewName).arg(vecInferImages.size()).arg(m_currKeyframeIdxes.size()));
        sendParamAssessBeginSignal(strCurrViewName, inferFrame, vecInferImages, scaleInfo);
    }

    //if (m_isParamAssessThreadRunning && m_bParamValuesUpdateFlag)
    //{
    //    int ret = sendParamAssessEndSignal(scaleInfo);

    //    m_bParamValuesUpdateFlag = false;
    //    m_isParamAssessThreadRunning = false;

    //    //m_isViewRecognized = false;
    //    m_isKeyframeDetected = false;
    //    m_isA4CSampleFlag = false;
    //}
    return 1;
}

int ModelsInferenceThread::frameCropping(cv::Mat& src, cv::Mat& dst, RoIScaleInfo& currRoIScaleInfo)
{
    if (currRoIScaleInfo.specScaleRect.empty())
    {
        if (!m_currROIScaleInfo.roiRect.empty())
        {
            // B-Mode
            dst = src(m_currROIScaleInfo.roiRect).clone();
            //cv::imshow("dst", dst);
            //cv::waitKey(0);
        }
        else
            dst = src.clone();

    }
    else
    {
        // Doppler-Mode
        cv::Rect currScaleRect = currRoIScaleInfo.specScaleRect;
        int imgWidth = src.cols;

        int rightMargin = imgWidth - currScaleRect.width - currScaleRect.x;
        if (rightMargin > 0)
        {
            int leftMargin = rightMargin - 20;

            cv::Rect cropRect;
            int width = imgWidth - rightMargin - leftMargin;
            if (width < 0)
            {
                dst = src.clone();
                return 0;
            }

            cropRect.width = imgWidth - rightMargin - leftMargin;
            cropRect.height = src.rows;
            cropRect.y = 0;
            cropRect.x = leftMargin;

            if (cropRect.x < 0 || cropRect.y < 0)
            {
                cropRect.x = (std::max)(cropRect.x, 0);
                cropRect.y = (std::max)(cropRect.y, 0);
            }

            if (cropRect.height + cropRect.y >= src.rows)
            {
                cropRect.height = src.rows - 1 - cropRect.y;
            }

            if (cropRect.width + cropRect.x >= src.cols)
            {
                cropRect.width = src.cols - 1 - cropRect.x;
            }

            dst = src(cropRect).clone();
            
            //cv::imshow("src", src);
            //cv::imshow("dst", dst);
            //cv::waitKey(0);
        }
    }
    return 1;
}

//int ModelsInferenceThread::currViewModeJudgement(ModeInfo& currModeInfo, RoIScaleInfo& currROIScaleInfo)
//{
//    if (currModeInfo.mode == "Doppler-Mode")
//    {
//        m_isSpecViewMode = true;
//        m_isBViewMode = false;
//    }
//    else if (currModeInfo.mode == "B-Mode")
//    {
//        m_isSpecViewMode = false;
//        m_isBViewMode = true;
//    }
//    else
//    {
//        m_isSpecViewMode = false;
//        m_isBViewMode = false;
//    }
//
//    if (m_isSpecViewMode)
//        m_currModeInfo.mode = "Doppler-Mode";
//    else if (m_isBViewMode)
//        m_currModeInfo.mode = "B-Mode";
//    else
//        m_currModeInfo.mode = "Blank";
//
//    return 1;
//}

int ModelsInferenceThread::currViewModeJudgement(ModeInfo& currModeInfo, RoIScaleInfo& currROIScaleInfo)
{
    //if (currModeInfo.mode == "Color-Mode")
    //{
    //    m_currModeInfo.mode = "Color-Mode";
    //    m_isSpecViewMode = false;
    //    m_isBViewMode = false;
    //    return 1;
    //}

    if (!currROIScaleInfo.specScaleRect.empty())  // Doppler-Mode
    {
        if (currROIScaleInfo.roiRect.empty())
        {
            m_isSpecViewMode = true;
            m_isBViewMode = false;
        }
        else
        {
            //m_isSpecViewMode = false;
            //m_isBViewMode = true;
            if (m_currROIScaleInfo.specScaleRect.height > m_currROIScaleInfo.roiRect.height)
            {
                m_isSpecViewMode = true;
                m_isBViewMode = false;
            }
            else
            {
                m_isSpecViewMode = false;
                m_isBViewMode = true;
            }
        }
    }
    else if (currROIScaleInfo.specScaleRect.empty() && !currROIScaleInfo.roiRect.empty())  // B-Mode
    {
        if (currModeInfo.bIsColorMode)
        {
            m_currModeInfo.mode = "Color-Mode";
            m_isSpecViewMode = false;
            m_isBViewMode = false;
            //sigDebugText(QString("[I] Color Mode."));
            sigDebugText(QString("[I] 彩超模式。"));
        }
        else
        {
            m_isSpecViewMode = false;
            m_isBViewMode = true;
        }
    }
    else
    {
        m_isSpecViewMode = false;
        m_isBViewMode = false;
    }

    if (m_isSpecViewMode)
        m_currModeInfo.mode = "Doppler-Mode";
    else if (m_isBViewMode)
        m_currModeInfo.mode = "B-Mode";
    else
        m_currModeInfo.mode = "Blank";

    return 1;
}

int ModelsInferenceThread::parseViewClassInferResult(const QString& currViewName)
{
    m_currViewName = currViewName;
    if (!m_currViewName.isEmpty() && !m_vecViewClassBuffer.empty() && m_currViewName != "OTHER")
    {
        m_isViewRecognized = true;
        qDebug() << "[I] View recognized.";

        std::string viewName = m_currViewName.toStdString();

        cv::Mat showFrame = *(m_vVideoBuffer.begin() + m_vVideoBuffer.size() / 2);
        showFrame.convertTo(showFrame, CV_8UC3);
        QImage qImage = GeneralUtils::matToQImage(showFrame);
        m_currQualityPremiumFrame = qImage;

        if (m_progressThread->isKeyframeEnable(m_currViewName.toStdString()))
        {
            if (m_currViewName == "A4C")
                m_currKeyframeSampleNum = m_a4cSampleNum;
            else
                m_currKeyframeSampleNum = m_keyframeSampleNum;
        }

        else
            m_currKeyframeSampleNum = m_nonKeyframeSampleNum;

        emit viewNameImageAvailable(m_currViewName, m_currQualityPremiumFrame);
    }
    return 1;
}

int ModelsInferenceThread::parseViewClassInferResult(const QString& currViewName, std::vector<cv::Mat>& vecViewClassImages)
{
    m_currViewName = currViewName;
    if (m_currViewName == "OTHER")
    {
        m_vecViewClassBuffer.clear();
        return 0;
    }

    if (!m_currViewName.isEmpty() && !m_vecViewClassBuffer.empty() && m_currViewName != "OTHER")
    {
        m_isViewRecognized = true;
        std::string viewName = m_currViewName.toStdString();

        cv::Mat showFrame = *(vecViewClassImages.begin() + vecViewClassImages.size() / 2);
        showFrame.convertTo(showFrame, CV_8UC3);
        QImage qImage = GeneralUtils::matToQImage(showFrame);
        m_currQualityPremiumFrame = qImage;

        if (m_progressThread->isKeyframeEnable(m_currViewName.toStdString()))
        {
            if (m_currViewName == "A4C")
                m_currKeyframeSampleNum = m_a4cSampleNum;
            else
                m_currKeyframeSampleNum = m_keyframeSampleNum;
        }

        else
            m_currKeyframeSampleNum = m_nonKeyframeSampleNum;

        emit viewNameImageAvailable(m_currViewName, m_currQualityPremiumFrame);
    }
    return 1;
}

int ModelsInferenceThread::parseViewClassInferResult(const QString& currViewName, const float fQualityScore, std::vector<cv::Mat>& vecViewClassImages)
{
    m_currViewName = currViewName;
    m_fCurrQualityScore = fQualityScore;

    if (m_currViewName == "OTHER")
    {
        m_vecViewClassBuffer.clear();
        return 0;
    }

    if (!m_currViewName.isEmpty() && !m_vecViewClassBuffer.empty() && m_currViewName != "OTHER" && fQualityScore != -10000.0f && !vecViewClassImages.empty())
    {
        m_isViewRecognized = true;
        std::string viewName = m_currViewName.toStdString();

        if (!vecViewClassImages.empty())
        {
            cv::Mat showFrame = *vecViewClassImages.begin();
            showFrame.convertTo(showFrame, CV_8UC3);
            QImage qImage = GeneralUtils::matToQImage(showFrame);
            m_currQualityPremiumFrame = qImage;
        }

        if (m_progressThread->isKeyframeEnable(m_currViewName.toStdString()))
        {
            m_currKeyframeSampleNum = m_keyframeSampleNum;
        }

        else
            m_currKeyframeSampleNum = m_nonKeyframeSampleNum;

        emit viewNameImageAvailable(m_currViewName, m_currQualityPremiumFrame);
    }
    return 1;
}

int ModelsInferenceThread::parseKeyframeDetResult(cv::Mat& currInferFrame, QVector<int>& keyframeIdx, std::vector<PeakInfo>& vPeakInfos)
{
    if (m_progressThread->isKeyframeEnable(m_currViewName.toStdString()))
    {
        QString strTempViewName = QString(m_currViewName);
        QString strTempInferMode = QString("sgta");
        m_keyDataBuffer->updateKeyDetInfo(currInferFrame, strTempViewName, strTempInferMode);
        QtLogger::instance().logMessage(QString("[I] Ready to detect keyframe for %1.").arg(m_currViewName));

        QElapsedTimer timer;
        timer.start();

        while (true)
        {
            if (timer.elapsed() > 2500) // 1500毫秒 = 1.5秒
            {
                //sigDebugText(QString("[I] Timeout: Waiting for Keyframe Detection exceeded 2.5 seconds."));
                sigDebugText(QString("[I] 超时：等待关键帧检测超过 2.5 秒。"));
                QtLogger::instance().logMessage(QString("[I] Timeout: Waiting for %1 Keyframe Detection Result exceeded 2.5 seconds.").arg(m_currViewName));
                m_vecViewClassBuffer.clear();
                break; // 超过5秒，跳出循环
            }

            if (m_keyDataBuffer->hasNewPeakResult())
            {
                vPeakInfos = m_keyDataBuffer->getPeakResult();
                if (!vPeakInfos.empty())
                    break;
            }

            //sigDebugText(QString("[I] Waiting for %1 keyframe det results...").arg(strTempViewName));
            sigDebugText(QString("[I] 正在等待 %1 关键帧检测结果...").arg(strTempViewName));
        }

        if (vPeakInfos.empty() || vPeakInfos[0].index == -10000)  // 未检测到关键帧
        {
            m_vVideoBuffer.clear();
            m_vOriginVideoBuffer.clear();
            //m_isViewRecognized = false;
            m_isKeyframeDetected = false;
            m_isA4CSampleFlag = false;
            sigDebugText(QString("[E] %1 No Keyframe Detected.").arg(strTempViewName));
        }

        else  // 检测到关键帧
        {
            processPeakInfos(vPeakInfos, keyframeIdx);
        }
    }

    else  // 该切面无需处理关键帧
    {
        keyframeIdx.push_back(-10000);
    }

    if (!keyframeIdx.empty())
    {
        m_currKeyframeIdxes = keyframeIdx;
        m_isKeyframeDetected = true;
        //QString strKeyframeIdxes = GeneralUtils::formatVector(m_currKeyframeIdxes);
        //QtLogger::instance().logMessage(QString("[I] %1 Keyframe detected results: %2.").arg(m_currViewName).arg(strKeyframeIdxes));
    }
    else
    {
        m_currKeyframeIdxes = QVector<int>{ -10000 };
    }
    return 1;
}

int ModelsInferenceThread::parseKeyframeDetResult(cv::Mat& currInferFrame, QVector<int>& keyframeIdx, QVector<PeakInfo>& vPeakInfos)
{
    //if (m_progressThread->isKeyframeEnable(m_currViewName.toStdString()))
    //{
    //    if (m_isA4CSampleFlag && m_vVideoBuffer.size() < m_a4cSampleNum)
    //    {
    //        return 0;
    //    }

    //    QString inferMode = "sgta";
    //    m_keyDataBuffer->updateKeyDetInfo(currInferFrame, m_currViewName, inferMode);

    //    if (!m_keyDataBuffer->hasNewPeakResult())
    //        return 0;
    //    else
    //    {
    //        vPeakInfos = m_keyDataBuffer->getPeakResult();
    //    }
    //    
    //    if (vPeakInfos.empty())
    //    {
    //        m_vVideoBuffer.clear();
    //        m_vOriginVideoBuffer.clear();
    //        m_isViewRecognized = false;
    //        m_isKeyframeDetected = false;
    //    }

    //    else
    //    {
    //        if (m_isA4CSampleFlag && vPeakInfos.size() >= 2)
    //        {
    //            for (auto& peak : vPeakInfos)
    //            {
    //                m_currKeyframeIdxes.push_back(peak.index);
    //            }
    //        }

    //        for (auto& peak : vPeakInfos)
    //        {
    //            keyframeIdx.push_back(peak.index);
    //            qDebug() << "[I] Keyframe detected: " << peak.index;
    //        }
    //    }

    //}
    //else
    //{
    //    keyframeIdx.push_back(0);
    //}

    //if (!keyframeIdx.empty())
    //{
    //    m_currKeyframeIdxes = keyframeIdx;
    //    m_isKeyframeDetected = true;
    //}
    return 1;
}

int ModelsInferenceThread::sendQualityControlBeginSignal(RoIScaleInfo& currROIScaleInfo)
{
    if (m_currKeyframeIdxes.empty() && m_progressThread->isKeyframeEnable(m_currViewName.toStdString()))
    {
        m_currKeyframeIdxes.push_back(0);
    }
    //qDebug() << "[I] Stage 4, buffer length: " << m_vVideoBuffer.size() << " , keyframe length: " << m_currKeyframeIdxes.size();

    if (!m_vVideoBuffer.empty())
    {
        QVariant qVideoClips, qKeyframeIdxes;
        QVector<cv::Mat> qVVideos(m_vVideoBuffer.begin(), m_vVideoBuffer.end());
        qVideoClips.setValue(qVVideos);

        qKeyframeIdxes.setValue(m_currKeyframeIdxes);
        emit sigQualityInput(m_currViewName, qVideoClips, qKeyframeIdxes, currROIScaleInfo.fRadius);
    }

    return 1;
}

int ModelsInferenceThread::sendQualityControlEndSignal()
{
    QVector<float> currResult = m_currResult;
    if (!currResult.empty())
    {
        float fTotalScore = std::accumulate(currResult.begin(), currResult.end(), 0.0f);
        int nNormalScore = static_cast<int>(fTotalScore / m_mapViewFullScore[m_currQualityControlViewName] * 100.0f);
        QString strNormalScore = QString::number(nNormalScore);

        // QVector<float> qVResults(currResult.grades.begin(), currResult.grades.end());
        QVariant qualityVar, videoVar;
        QVector<cv::Mat> qVVideos;
        qualityVar.setValue(currResult);
        //emit qualityScoresAvailable(m_currViewName, qualityVar);

        if (!m_vOriginVideoBuffer.empty())
        {
            for (int i = 0; i < m_vOriginVideoBuffer.size(); i += 2)
                qVVideos.push_back(m_vOriginVideoBuffer[i]);
        }

        //QVariant videoVar;
        videoVar.setValue(qVVideos);
        //emit viewNameVideoAvailable(m_currViewName, videoVar);

        emit sigViewNameImageVideoAvailable(m_currViewName, videoVar, qualityVar);
        //m_currViewName = "";
    }
    else return 0;

    return 1;
}

int ModelsInferenceThread::sendImageSamplingInterruptSignal()
{
    if (!m_isViewRecognized)
        return 0;

    emit sigReinitailizeLabel(m_currViewName);
    m_vVideoBuffer.clear();
    return 1;
}

int ModelsInferenceThread::updateQualityScore(const QString& viewName, float currentScore)
{
    // 如果当前质量分高于历史最高分，则更新
    if (currentScore > m_mapHistoryQualityScore.value(viewName, -10000.0f)) 
    {
        m_mapHistoryQualityScore[viewName] = currentScore;
        return 1;
    }
    else 
    {
        return 0;
    }
}

bool ModelsInferenceThread::judgeQualityScoreToParam(QString& strViewName, float fCurrQualityScore)
{
    if (strViewName == "OTHER")
        return false;
    bool bIsReadyToParam = false;
    if (m_vecSupportQualityControlViews.contains(strViewName))
    {
        if (m_bBestQualityScoreOnly)
        {
            if (updateQualityScore(strViewName, fCurrQualityScore))
            {
                bIsReadyToParam = true;
                //sigDebugText(QString("%1 Max Quality Score Updated! Quality Score: %2").arg(strViewName).arg(fCurrQualityScore));
                sigDebugText(QString("%1 切面最高质量分数更新！质量得分： %2").arg(strViewName).arg(fCurrQualityScore));
                QtLogger::instance().logMessage(QString("%1 Max Quality Score Updated! Quality Score: %2").arg(strViewName).arg(fCurrQualityScore));
            }
            else
            {
                //sigDebugText(QString("%1 Won't Update Quality Score! Quality Score: %2").arg(strViewName).arg(fCurrQualityScore));
                sigDebugText(QString("%1 切面质量分数！未更新质量得分: %2").arg(strViewName).arg(fCurrQualityScore));
                QtLogger::instance().logMessage(QString("%1 Won't Update Quality Score! Quality Score: %2").arg(strViewName).arg(fCurrQualityScore));
            }
        }
        else
        {
            //if (updateQualityScore(strCurrViewName, fCurrQualityScore))
            if (fCurrQualityScore > m_fQualityScoreThresh)
            {
                bIsReadyToParam = true;
                //sigDebugText(QString("%1 Quality Score Passed: %2").arg(strViewName).arg(fCurrQualityScore));
                sigDebugText(QString("%1 质量评分达标: %2").arg(strViewName).arg(fCurrQualityScore));
                QtLogger::instance().logMessage(QString("%1 Max Quality Score Updated! Quality Score: %2").arg(strViewName).arg(fCurrQualityScore));
            }
            else
            {
                //sigDebugText(QString("%1 Quality Score NOT Passed: %2").arg(strViewName).arg(fCurrQualityScore));
                sigDebugText(QString("%1 质量评分未达标: %2").arg(strViewName).arg(fCurrQualityScore));
                QtLogger::instance().logMessage(QString("%1 Won't Update Quality Score! Quality Score: %2").arg(strViewName).arg(fCurrQualityScore));
            }
        }
    }
    else
        bIsReadyToParam = true;

    return bIsReadyToParam;
}

bool ModelsInferenceThread::isCurrentScoreHigher(const QString& viewName, float currentScore)
{
    // 获取历史最高分
    float historicalScore = m_mapHistoryQualityScore.value(viewName, -10000.0f);

    // 如果当前分数超过历史最高分，返回true，否则返回false
    return currentScore > historicalScore;
}

int ModelsInferenceThread::sendParamAssessBeginSignal(cv::Mat& originFrame, ScaleInfo& currScaleInfo)
{
    std::string viewName = m_currViewName.toStdString();
    if ((m_isQualityControlled && m_paramAssessThread->isParamEnable(m_currViewName)) 
        || m_isSpecViewMode 
        || (m_paramOnlyMode && m_isViewRecognized && m_isKeyframeDetected && m_paramAssessThread->isParamEnable(m_currViewName)))
    {
        QVector<cv::Mat> tempVideoClip;
        sendScaleInfo(currScaleInfo);
        if (m_vVideoBuffer.empty() || m_isSpecViewMode)
            tempVideoClip = { originFrame };
        else
            tempVideoClip = QVector<cv::Mat>(m_vVideoBuffer.begin(), m_vVideoBuffer.end());

        QVariant qVideoClip, qKeyframeIdx, qModeInfo;
        qVideoClip.setValue(tempVideoClip);
        qKeyframeIdx.setValue(m_currKeyframeIdxes);
        qModeInfo.setValue(m_currModeInfo);

        //emit sigParamInput(m_currViewName, qVideoClip, qKeyframeIdx, qModeInfo);
        m_paramAssessThread->inputParamAssess(m_currViewName, qVideoClip, qKeyframeIdx, qModeInfo);
        m_isParamAssessThreadRunning = true;
    }

    if (m_isParamAssessThreadRunning)
    {
        m_vVideoBuffer.clear();
        m_vOriginVideoBuffer.clear();
        m_currKeyframeIdxes.clear();

        m_isViewRecognized = false;
        m_isQualityControlled = false;
    }

    return 1;
}

int ModelsInferenceThread::sendParamAssessBeginSignal(QString& strCurrViewName, cv::Mat& originFrame, std::vector<cv::Mat>& vecInferImages, ScaleInfo& currScaleInfo)
{
    if (m_isSpecViewMode
        || (m_paramOnlyMode && !vecInferImages.empty() && m_isKeyframeDetected && m_paramAssessThread->isParamEnable(strCurrViewName)))
    {
        QVector<cv::Mat> tempVideoClip;
        sendScaleInfo(currScaleInfo);
        if (m_vVideoBuffer.empty() || m_isSpecViewMode)
            tempVideoClip = { originFrame };
        else
        {
            std::vector<cv::Mat> tempConcatInferImages;
            if (!m_vecKeyframeBuffer.empty())
            {
                QMutexLocker locker(&m_keyframeBufferMutex);
                GeneralUtils::concatVectors(m_vecKeyframeBuffer, vecInferImages, tempConcatInferImages);
                m_vecKeyframeBuffer.clear();
            }
            else
                tempConcatInferImages = vecInferImages;
            tempVideoClip = QVector<cv::Mat>(tempConcatInferImages.begin(), tempConcatInferImages.end());
        }

        QString strKeyframeIdxes = GeneralUtils::formatVector(m_currKeyframeIdxes);
        //QtLogger::instance().logMessage(QString("Ready to paraming: %1, Spectrum Mode: %2, Indices: %3, Video Length: %4").arg(strCurrViewName).arg(m_isSpecViewMode).arg(strKeyframeIdxes).arg(tempVideoClip.size()));
        QVariant qVideoClip, qKeyframeIdx, qModeInfo;
        qVideoClip.setValue(tempVideoClip);
        qKeyframeIdx.setValue(m_currKeyframeIdxes);
        qModeInfo.setValue(m_currModeInfo);

        //emit sigParamInput(m_currViewName, qVideoClip, qKeyframeIdx, qModeInfo);
        m_paramAssessThread->inputParamAssess(strCurrViewName, qVideoClip, qKeyframeIdx, qModeInfo);
        m_isParamAssessThreadRunning = true;
    }

    if (m_isParamAssessThreadRunning)
    {
        m_vVideoBuffer.clear();
        m_vOriginVideoBuffer.clear();
        m_currKeyframeIdxes.clear();

        //m_isViewRecognized = false;
        m_isQualityControlled = false;
    }

    return 1;
}

int ModelsInferenceThread::sendParamAssessEndSignal(ScaleInfo& currScaleInfo)
{
    {
        QMutexLocker locker(&m_paramCommMutex);
    }
    if (m_currParamValues.empty())
        return 1;
    
    eraseTDINoUseParam();

    paramValueScaling(currScaleInfo);

    //plotParamScaleOnPremium(currScaleInfo);
    //sendScaleInfo(currScaleInfo);
    if (m_currSignalParamValues.empty())
        return 0;

    checkoutNormalParam();
    if (m_currUpdateParamValues.isEmpty())
    {
        m_currSignalParamValues.clear();
        return 0;
    }

    computeEDevideA();  // 计算E/A参数

    computeEDevideTDIToParam();  // 计算E/e'参数

    //sigDebugText(QString("[I] Paramed - %1 %2 %3").arg(m_currParamValues.firstKey()).arg(m_currParamValues.first()[0]).arg(m_currKeyframeIdxes.size()));

    //if (m_currParamReturnViewName == QString("PLAX"))
    //{

    //}

    //QVariant valueVar, premiumsVar;
    //valueVar.setValue(m_currUpdateParamValues);
    //premiumsVar.setValue(m_currPremiums);
    //emit sigParamsAvailable(m_currParamReturnViewName, valueVar, premiumsVar);

    sendParamToDisplay();

    addParamToHistory();

    m_currSignalParamValues.clear();
    m_currUpdateParamValues.clear();
    return 1;
}

int ModelsInferenceThread::checkoutNormalParam()
{
    if (m_currSignalParamValues.empty())
        return 1;

    if (m_histParamValues.empty())
    {
        removeZeroNumParam();
        m_currUpdateParamValues = m_currSignalParamValues;
        return 1;
    }

    else
    {
        removeZeroNumParam();
        if (m_currSignalParamValues.empty())
            return 1;

        compareHistParamWithThresh();

        //bool bIsClose = compareSpecParamWithThresh();
        //if (bIsClose)
        //{
        //    m_currParamValues.clear();
        //    m_currPremiums.clear();
        //    m_currSignalParamValues.clear();

        //    return 0;
        //}
    }

    return 1;
}

int ModelsInferenceThread::paramValueScaling(ScaleInfo& currScaleInfo)
{
    float fPixelPerCm = currScaleInfo.fPixelPerUnit;
    int unitPosY = currScaleInfo.unitPositionY;
    QString strCurrUnit = QString::fromStdString(currScaleInfo.unit);

    //if (fPixelPerCm == 0 || fPixelPerCm == -10000.0f)
    //    return 0;

    for (auto& key : m_currParamValues.keys())
    {
        std::stringstream ss;
        if (m_currParamValues[key].size() > 1)  // 频谱参数多个值处理，只取第2个峰的值
            m_currParamValues[key] = QVector<float>{ m_currParamValues[key][1] };

        if (m_specParamEvents.contains(key))
        {
            for (auto& value : m_currParamValues[key])
            {
                if (strCurrUnit.isEmpty())
                    return 0;

                if (QString::fromStdString(currScaleInfo.unit).contains("cm"))
                    value /= 100.0f;
            }
        }

        for (auto& value : m_currParamValues[key])
        {
            if (!key.contains("EF") && !key.contains("EDV") && !key.contains("ESV") && !m_structParamEvents.contains(key))  // 不包含EF或者结构参数才需要处理单位换算
            {
                if (fPixelPerCm != 0 && fPixelPerCm != -10000.0f)
                {
                    bool matchRet = checkCurrentScaleModeMatch(key, currScaleInfo);
                    if (matchRet)
                    {
                        //if (key.indexOf("EDV") != std::string::npos || key.indexOf("ESV") != std::string::npos)
                        //    value = value / fPixelPerCm / fPixelPerCm / fPixelPerCm;
                        if (key.indexOf("VTI") != std::string::npos)
                            value = value / fPixelPerCm / fPixelPerCm;
                        else
                            value = value / fPixelPerCm;
                    }
                    else
                    {
                        value = -10000.0f;
                    }
                }
            }

            if (value != -10000.0f)
            {
                if (m_specParamEvents.contains(key))
                    ss << value << " ";
                else
                    ss << value;
            }

            //m_currParamValues[key] = QString::number(currValue / fPixelPerCm);
        }
        QString tempValueStr = QString::fromStdString(ss.str());
        m_currSignalParamValues.insert( key, tempValueStr );
    }
    return 1;
}

int ModelsInferenceThread::eraseTDINoUseParam()
{
    for (auto it = m_currParamValues.begin(); it != m_currParamValues.end();)
    {
        QString key = it.key();
        if (key.contains("CB") || key.contains("JG"))  // 过滤掉JGa, JGs, CBa, CBs
        {
            if (key.contains("a") || key.contains("s"))
            {
                it = m_currParamValues.erase(it);
                continue;
            }
            else
                ++it;
        }
        else
            ++it;
    }
    return 1;
}

int ModelsInferenceThread::computeEDevideA()
{
    bool bIsEAParam = false;
    for (auto it = m_currUpdateParamValues.begin(); it != m_currUpdateParamValues.end(); ++it)
    {
        if (it.key() == QString("E") || it.key() == QString("A"))
        {
            bIsEAParam = true;
            break;
        }
    }

    if (!bIsEAParam)
        return 0;
    else
    {
        //if (!m_currUpdateParamValues.contains("E") || !m_currUpdateParamValues.contains("A"))
        //    return 0;
        //if (m_currSignalParamValues["A"] == QString("0.0"))
        //    return 0;
        float fE = 0.0f, fA = 0.0f;
        if (!m_currUpdateParamValues.contains("A"))
        {
            fE = m_currUpdateParamValues["E"].toFloat();
            if (!m_histParamValues.contains("A"))
                return 0;
            if (m_histParamValues["A"].empty())
                return 0;

            fA = m_histParamValues["A"].last();
        }
        else if (!m_currUpdateParamValues.contains("E"))
        {
            if (!m_histParamValues.contains("E"))
                return 0;
            if (m_histParamValues["E"].empty())
                return 0;
            fA = m_currUpdateParamValues["A"].toFloat();
            fE = m_histParamValues["E"].last();
        }
        else
        {
            fE = m_currUpdateParamValues["E"].toFloat();
            fA = m_currUpdateParamValues["A"].toFloat();
        }

        if (fA == 0.0f)
            return 0;

        float fEDivideA = fE / fA;
        m_currUpdateParamValues.insert("E/A", QString::number(fEDivideA));
    }
    return 1;
}

float ModelsInferenceThread::computeEDevideTDI()
{
    if (m_currUpdateParamValues.contains("E"))
    {
        float fCurrE = m_currUpdateParamValues["E"].toFloat();
        if (!m_histParamValues.contains("CBe") || !m_histParamValues.contains("JGe"))
            return 0;

        if (m_histParamValues["CBe"].isEmpty() || m_histParamValues["JGe"].isEmpty())
            return 0;

        float fHistCBe = m_histParamValues["CBe"].last();
        float fHistJGe = m_histParamValues["JGe"].last();
        float meanTDI = (fHistCBe + fHistJGe) / 2.0f;

        return fCurrE / (meanTDI + 0.000001f);
    }
    else if (m_currUpdateParamValues.contains("JGe") || m_currUpdateParamValues.contains("CBe"))
    {
        if (!m_histParamValues.contains("E"))
            return 0;
        if (m_histParamValues["E"].isEmpty())
            return 0;

        float fCurrTDIValue = 0.0f, fHistTDIValue = 0.0f, fMeanTDIValue = 0.0f, fHistEValue =m_histParamValues["E"].last();
        if (m_currUpdateParamValues.contains("JGe"))
        {
            if (!m_histParamValues.contains("CBe"))
                return 0;

            if (m_histParamValues["CBe"].isEmpty())
                return 0;

            fCurrTDIValue = m_currUpdateParamValues["JGe"].toFloat();
            fHistTDIValue = m_histParamValues["CBe"].last();
        }
        else if (m_currUpdateParamValues.contains("CBe"))
        {
            if (!m_histParamValues.contains("JGe"))
                return 0;

            if (m_histParamValues["JGe"].isEmpty())
                return 0;

            fCurrTDIValue = m_currUpdateParamValues["CBe"].toFloat();
            fHistTDIValue = m_histParamValues["JGe"].last();
        }

        if (fCurrTDIValue == 0.0f || fHistTDIValue == 0.0f)
            return 0;

        fMeanTDIValue = (fCurrTDIValue + fHistTDIValue) / 2.0f;
        return fHistEValue / fMeanTDIValue;
    }

    return 0; 
}

int ModelsInferenceThread::computeEDevideTDIToParam()
{
    float fCurrEDevideTDI = computeEDevideTDI();
    if (fCurrEDevideTDI == 0.0f)
        return 0;

    // 20250514更新: E/e'的参数名更新为Ee
    m_currUpdateParamValues["Ee"] = QString::number(fCurrEDevideTDI);
    return 1;
}

int ModelsInferenceThread::removeZeroNumParam()
{
    for (auto it = m_currSignalParamValues.begin(); it != m_currSignalParamValues.end();)
    {
        const QString eventName = it.key();
        QString strValue = it.value();
        float fCurrValue = strValue.toFloat();
        if (fCurrValue == 0.0f)
        {
            it = m_currSignalParamValues.erase(it);
            continue;
        }
        ++it;

        //QVector<float> vecEventValues = it.value();
        ////sigDebugText(QString("[I] Paramed - %1 %2 %3").arg(m_currParamValues.firstKey()).arg(m_currParamValues.first()[0]).arg(m_currKeyframeIdxes.size()));
        ////sigDebugText(QString("[I] Paramed - %1 %2 %3").arg(eventName).arg(vecEventValues[0]).arg(m_currKeyframeIdxes.size()));

        //if (std::any_of(vecEventValues.begin(), vecEventValues.end(), [](float value) { return value == 0.0f; }))
        //{
        //    //sigDebugText(QString("[I] Current Value: %1").arg(it->first()));
        //    it = m_currParamValues.erase(it);
        //    continue;
        //}
        //++it;
    }

    return 1;
}

int ModelsInferenceThread::removeOutlierParamEvent()
{
    for (auto it = m_currParamValues.begin(); it != m_currParamValues.end();)
    {
        QString strEventName = it.key();
        QVector<float> vecEventValues = it.value();

        for (auto& value : vecEventValues)
        {
            bool ret = GeneralUtils::isWithinRange(value, m_histParamValues[strEventName], 1.5f);
            if (!ret)
            {
                it = m_currParamValues.erase(it);
                continue;
            }

            m_histParamValues[strEventName].push_back(value);
            ++it;
        }
    }
    return 1;
}

void ModelsInferenceThread::addParamToHistory()
{
    //if (m_currParamValues.isEmpty())
    //    return;

    //for (auto it = m_currParamValues.begin(); it != m_currParamValues.end(); ++it)
    //{
    //    QString strEventName = it.key();
    //    QVector<float> vecEventValues = it.value();
    //    float fMidValues = getSpecMiddleValue(vecEventValues);

    //    m_histParamValues[strEventName].push_back(fMidValues);
    //}
    if (m_currSignalParamValues.isEmpty())
        return;

    for (auto it = m_currSignalParamValues.begin(); it != m_currSignalParamValues.end(); ++it)
    {
        QString strEventName = it.key();
        if (strEventName == QString("E/A"))  // 永远不加入E/A，每次都要更新这个值
            continue;
        QString strParamValue = it.value();
        float fMidValues = strParamValue.toFloat();

        m_histParamValues[strEventName].push_back(fMidValues);
    }
}

float ModelsInferenceThread::getSpecMiddleValue(QVector<float>& vecParamValues)
{
    if (vecParamValues.isEmpty())
        return -10000.0f;

    int nMidIndex = static_cast<int>(vecParamValues.size() / 2);
    return vecParamValues[nMidIndex];
}

int ModelsInferenceThread::getMaxHistParamValue(QString strValueName)
{
    if (!m_histParamValues.contains(strValueName) || m_histParamValues[strValueName].isEmpty())
    {
        m_currUpdateParamValues = m_currSignalParamValues;
        return 0;
    }

    float fCurrValue = m_currSignalParamValues[strValueName].toFloat();
    float fMaxValue = *std::max_element(m_histParamValues[strValueName].begin(), m_histParamValues[strValueName].end());
    if (fCurrValue > fMaxValue)
    {
        m_currUpdateParamValues = m_currSignalParamValues;
        return 1;
    }

    return 0;
}

int ModelsInferenceThread::compareHistParamWithIncrementUpdate()
{
    for (auto it = m_currSignalParamValues.constBegin(); it != m_currSignalParamValues.constEnd(); ++it)
    {
        const QString& key = it.key();

        //QVector<float> vecCurrValues = it.value();

        //if (!m_specParamEvents.contains(key))
        //{
        //    return false;
        //}

        //if (vecCurrValues.isEmpty())
        //    continue;

        //float fCurrValue = getSpecMiddleValue(vecCurrValues);

        QString strCurrParamValue = it.value();
        float fCurrValue = strCurrParamValue.toFloat();

        if (m_histParamValues.contains(key))  // 含有历史测值，则比较是否差距较大
        {
            QVector<float> vecHistoricalValues = m_histParamValues.value(key);
            if (vecHistoricalValues.isEmpty())
                continue;

            float fHistoricalValue = vecHistoricalValues.last();
            float difference = fabs((fCurrValue - fHistoricalValue) / fHistoricalValue);

            float fCurrDiffRatioThresh = 0.0f;
            if (m_specParamEvents.contains(key))
                fCurrDiffRatioThresh = m_fSpecDiffRatioThresh;
            else if (m_mapStructPremiumToEvent["AAD"].contains(key))
                fCurrDiffRatioThresh = m_fStructAortaDiffRatioThresh;
            else
                fCurrDiffRatioThresh = m_fDiffRatioThresh;

            if (difference >= m_fDiffRatioThresh)
            {
                m_currUpdateParamValues[key] = strCurrParamValue;
            }
        }
        else  // 无历史测值，直接更新
        {
            m_currUpdateParamValues[key] = strCurrParamValue;
        }
    }

    return 0;
}

int ModelsInferenceThread::compareHistParamWithAllParamValid()
{
    bool allParamsValid = true;  // 标志位，默认假设所有参数都符合条件

    // 临时存储有效更新的参数值
    QMap<QString, QString> tempUpdateParamValues;
    
    // 20250415添加：如果LVID的值比之前的历史值要大，则直接上报
    if (m_bLVIDSelectMaxValue && getMaxHistParamValue(QString("LVDd")))
        return 1;

    //if (m_bLVIDSelectMaxValue && m_currSignalParamValues.contains("LVDd"))
    //{
    //    if (!m_histParamValues.contains("LVDd") || m_histParamValues["LVDd"].isEmpty())
    //    {
    //        m_currUpdateParamValues = m_currSignalParamValues;
    //        return 1;
    //    }

    //    float fCurrLVIDValue = m_currSignalParamValues["LVDd"].toFloat();
    //    float fLVIDMaxValue = *std::max_element(m_histParamValues["LVDd"].begin(), m_histParamValues["LVDd"].end());
    //    if (fCurrLVIDValue > fLVIDMaxValue)
    //    {
    //        m_currUpdateParamValues = m_currSignalParamValues;
    //        return 1;
    //    }
    //}

    // 20250507添加：如果AAD的值比之前的历史值要大，则直接上报
    QString strCurrAADValueName = "AoD";
    if (m_currSignalParamValues.contains("AAD"))
        strCurrAADValueName = "AAD";

    if (m_bAADSelectMaxValue && getMaxHistParamValue(strCurrAADValueName))
    {
        return 1;
    }
    //float fCurrAADValue = -10000.0f;
    //if (m_currSignalParamValues.contains("AoD"))
    //if (m_bAADSelectMaxValue && m_currSignalParamValues.contains("AoD"))
    //{
    //    if (!m_histParamValues.contains("AoD") || m_histParamValues["AoD"].isEmpty())
    //    {
    //        m_currUpdateParamValues = m_currSignalParamValues;
    //        return 1;
    //    }

    //    float fCurrAADValue = m_currSignalParamValues["AoD"].toFloat();
    //    float fAADMaxValue = *std::max_element(m_histParamValues["AoD"].begin(), m_histParamValues["AoD"].end());
    //    if (fCurrAADValue > fAADMaxValue)
    //    {
    //        m_currUpdateParamValues = m_currSignalParamValues;
    //        return 1;
    //    }
    //}

    for (auto it = m_currSignalParamValues.constBegin(); it != m_currSignalParamValues.constEnd(); ++it)
    {
        const QString& key = it.key();
        QString strCurrParamValue = it.value();
        float fCurrValue = strCurrParamValue.toFloat();

        if (m_histParamValues.contains(key))  // 含有历史测值，则比较是否差距较大
        {
            QVector<float> vecHistoricalValues = m_histParamValues.value(key);
            if (vecHistoricalValues.isEmpty())
                continue;

            float fHistoricalValue = vecHistoricalValues.last();
            float difference = fabs((fCurrValue - fHistoricalValue) / fHistoricalValue);

            float fCurrDiffRatioThresh = 0.0f;
            if (m_specParamEvents.contains(key))
                fCurrDiffRatioThresh = m_fSpecDiffRatioThresh;
            else if (m_mapStructPremiumToEvent["AAD"].contains(key))
                fCurrDiffRatioThresh = m_fStructAortaDiffRatioThresh;
            else
                fCurrDiffRatioThresh = m_fDiffRatioThresh;

            if (difference < fCurrDiffRatioThresh)
            {
                // 若发现某个参数不符合条件，则设置标志位为 false
                allParamsValid = false;
                break;  // 直接中断，不再继续检查
            }
            else
            {
                // 若符合条件，暂存当前参数值
                tempUpdateParamValues[key] = strCurrParamValue;
            }
        }
        else  // 无历史测值，直接更新
        {
            tempUpdateParamValues[key] = strCurrParamValue;
        }
    }

    // 只在所有参数都符合条件时更新
    if (allParamsValid)
    {
        m_currUpdateParamValues = tempUpdateParamValues;  // 批量更新
    }

    return allParamsValid ? 1 : 0;  // 返回 0 表示成功，-1 表示有参数不符合条件
}

int ModelsInferenceThread::compareHistParamWithThresh()
{
    int ret = -1;
    if (m_bAllParamsDiffValid)
        ret = compareHistParamWithAllParamValid();
    else
        ret = compareHistParamWithIncrementUpdate();

    return ret;
}

bool ModelsInferenceThread::compareSpecParamWithThresh()
{
    for (auto it = m_currSignalParamValues.constBegin(); it != m_currSignalParamValues.constEnd(); ++it)
    {
        const QString& key = it.key();
        //QVector<float> vecCurrValues = it.value();

        //if (!m_specParamEvents.contains(key))
        //{
        //    return false;
        //}

        //if (vecCurrValues.isEmpty())
        //    continue;

        //float fCurrValue = getSpecMiddleValue(vecCurrValues);

        QString strCurrParamValue = it.value();
        float fCurrValue = strCurrParamValue.toFloat();

        if (m_histParamValues.contains(key)) 
        {
            QVector<float> vecHistoricalValues = m_histParamValues.value(key);
            if (vecHistoricalValues.isEmpty())
                continue;

            float fHistoricalValue = vecHistoricalValues.last();
            float difference = fabs((fCurrValue - fHistoricalValue) / fHistoricalValue);

            float fCurrDiffRatioThresh = 0.0f;
            if (m_specParamEvents.contains(key))
                fCurrDiffRatioThresh = m_fDiffRatioThresh;
            else
                fCurrDiffRatioThresh = m_fSpecDiffRatioThresh;

            if (difference < fCurrDiffRatioThresh)
                return true;
            else
                return false;
        }
    }

    return false;
}

int ModelsInferenceThread::plotParamScaleOnPremium(ScaleInfo& currScaleInfo)
{
    QString unit = QString::fromStdString(currScaleInfo.unit);
    QString length = QString("Length: %1 %2").arg(currScaleInfo.length).arg(unit);
    QString pixelPerCM = "Pixel per Unit: " + QString::number(currScaleInfo.fPixelPerUnit);
    QString valueRange = "Value range: " + QString::number(currScaleInfo.fSpecScaleRange);

    QFont font("Arial", 20);
    QColor color(Qt::white);

    for (auto& event : m_currPremiums.keys())
    {
        //QImage premium = m_currPremiums[event];
        
        QPoint position(20, 50);
        GeneralUtils::drawTextOnImage(m_currPremiums[event], length, position, font, color);

        position = QPoint(20, 80);
        GeneralUtils::drawTextOnImage(m_currPremiums[event], pixelPerCM, position, font, color);

        position = QPoint(20, 110);
        GeneralUtils::drawTextOnImage(m_currPremiums[event], valueRange, position, font, color);
    }
    return 1;
}

int ModelsInferenceThread::groupStructParam(QMap<QString, QMap<QString, QString>>& currStructParamValues, QMap<QString, QImage>& currStructPremiums)
{
    for (auto strPremiumEvent : m_vecStructParamPremiums)
    {
        QVector<QString> vecCurrParamEvents = m_mapStructPremiumToEvent[strPremiumEvent];
        QMap<QString, QString> mapTempPremiumEvent;
        if (!m_currPremiums.contains(strPremiumEvent))
            continue;

        currStructPremiums[strPremiumEvent] = m_currPremiums[strPremiumEvent];

        for (auto strParamEvent : vecCurrParamEvents)
        {
            if (m_currUpdateParamValues.contains(strParamEvent))
            {
                mapTempPremiumEvent[strParamEvent] = m_currUpdateParamValues[strParamEvent];
            }
        }
        currStructParamValues[strPremiumEvent] = mapTempPremiumEvent;
    }
    return 1;
}

int ModelsInferenceThread::sendParamToDisplay()
{
    if (m_currParamReturnViewName == QString("PLAX"))
        sendPLAXParamToDisplay();
    else
        sendNormalParamToDisplay();
    return 1;
}

int ModelsInferenceThread::sendPLAXParamToDisplay()
{
    QMap<QString, QMap<QString, QString>> tempStructParamValues;
    QMap<QString, QImage> tempStructPremiums;

    groupStructParam(tempStructParamValues, tempStructPremiums);
    
    QVariant valueVar, premiumsVar;
    valueVar.setValue(tempStructParamValues);
    premiumsVar.setValue(tempStructPremiums);
    emit sigStructParamsAvailable(valueVar, premiumsVar);
    return 1;
}

int ModelsInferenceThread::sendNormalParamToDisplay()
{
    QVariant valueVar, premiumsVar;
    valueVar.setValue(m_currUpdateParamValues);
    premiumsVar.setValue(m_currPremiums);
    emit sigParamsAvailable(m_currParamReturnViewName, valueVar, premiumsVar);
    return 1;
}

int ModelsInferenceThread::checkVideoBufferSize(int maxSize)
{
    {
        //QMutexLocker locker(&m_mutex);
        if (m_vVideoBuffer.size() >= maxSize)
            m_vVideoBuffer.clear();

        if (m_vOriginVideoBuffer.size() >= maxSize)
            m_vOriginVideoBuffer.clear();
    }

    return 1;
}

void ModelsInferenceThread::getVideoFrame(const QImage frame)
{
    QMutexLocker locker(&m_mutex);
    cv::Mat cvFrame = GeneralUtils::qImage2cvMat(frame);
    m_nextFrame = cvFrame.clone();
    m_hasNewFrame = true;
}

int ModelsInferenceThread::inputVideoFrame(cv::Mat& frame)
{
    QMutexLocker locker(&m_mutex);
    m_nextFrame = frame.clone();
    m_hasNewFrame = true;
    return 1;
}

void ModelsInferenceThread::setQualityControlScores(QVariant qResult)
{
    QMutexLocker locker(&m_mutex);
    m_currResult = qResult.value<QVector<float>>();
    m_bQualityScoresUpdateFlag = true;
}

void ModelsInferenceThread::setParamsAssessValues(QString viewName, QVariant qResult, QVariant qPremiums)
{
    QMutexLocker locker(&m_paramCommMutex);
	m_currParamValues = qResult.value<QMap<QString, QVector<float>>>();
    m_currPremiums = qPremiums.value<QMap<QString, QImage>>();
    m_currParamReturnViewName = viewName;
	m_bParamValuesUpdateFlag = true;
}

//void ModelsInferenceThread::setScaleInfo(QVariant qROIScaleInfo, QVariant qScaleInfo, QVariant qModeInfo)
//{
//    QMutexLocker locker(&m_mutex);
//    m_currScaleInfo = qScaleInfo.value<ScaleInfo>();
//    m_currModeInfo = qModeInfo.value<ModeInfo>();
//    m_currROIScaleInfo = qROIScaleInfo.value<RoIScaleInfo>();
//    m_isScaleInfoUpdated = true;
//}

void ModelsInferenceThread::setScaleInfo(QVariant qScaleInfo)
{
    QMutexLocker locker(&m_mutex);
    m_currScaleInfo = qScaleInfo.value<ScaleInfo>();
    m_isScaleInfoUpdated = true;
}

void ModelsInferenceThread::setScaleModeInfo(QVariant qScaleInfo, QVariant qModeInfo)
{
    QMutexLocker locker(&m_mutex);
    m_currScaleInfo = qScaleInfo.value<ScaleInfo>();
    m_currModeInfo = qModeInfo.value<ModeInfo>();
    m_isScaleInfoUpdated = true;
}

void ModelsInferenceThread::setROIScaleInfo(QVariant qROIScaleInfo)
{
    QMutexLocker locker(&m_mutex);
    m_currROIScaleInfo = qROIScaleInfo.value<RoIScaleInfo>();
    m_isROIScaleInfoUpdated = true;
}

void ModelsInferenceThread::setModelInferThreadClear()
{
    QMutexLocker locker(&m_mutex);
    blankModeProcess();
    clearDataCache();
}

// void ModelsInferenceThread::setCropRect(QVariant qRect)
// {
//     QMutexLocker locker(&m_mutex);
//     m_roiRect = qRect.value<cv::Rect>();
//     m_bROIUpdateFlag = true;
// }

int ModelsInferenceThread::sendScaleInfo(ScaleInfo& currScaleInfo)
{
    QVariant qVar;
    qVar.setValue(currScaleInfo);
    emit sigScaleInfo(qVar);
    return 1;
}

int ModelsInferenceThread::saveInferenceImage(QString& viewName, std::vector<cv::Mat>& vImgs)
{
    QDateTime currentTime = QDateTime::currentDateTime();
    if (m_lastSaveTime.secsTo(currentTime) >= m_saveIntervalSeconds)
    {
        QString subFolderName = generateDate();
        QString folderPath = m_videoSaveRootPath + "/" + viewName;
        QDir().mkpath(folderPath);

        QString saveFolderPath = folderPath + "/" + subFolderName;
        QDir().mkpath(saveFolderPath);

        for (int i = 0; i < vImgs.size(); ++i)
        {
            QString filePath = saveFolderPath + "/" + QString("image_%1.jpg").arg(i);
            cv::imwrite(filePath.toStdString(), vImgs[i]);
        }
        m_lastSaveTime = currentTime;

        return 1;
    }

    return 0;
}