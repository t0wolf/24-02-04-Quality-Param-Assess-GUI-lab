#include "BiplaneLVEFAssesser.h"

BiplaneLVEFAssesser::BiplaneLVEFAssesser(std::string& lvefSegEnginePath)
	: m_lvefSegEnginePath(lvefSegEnginePath)
	, m_lvefAssesser(new LVEFAssesser(lvefSegEnginePath))
	, m_viewFrameLength(8)
    , m_bIsBiplaneMode(false)
    , m_fPixPerUnit(-10000.0f)
    , m_fLength(-10000.0f)
{

}

BiplaneLVEFAssesser::~BiplaneLVEFAssesser()
{
	if (m_lvefAssesser != nullptr)
		delete m_lvefAssesser;
}

int BiplaneLVEFAssesser::doInference(std::string& currViewName, std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex,
    std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
    if (vVideoClips.empty())
        return 0;

    std::vector<cv::Mat> inputVideoClips = vVideoClips;

    //ParamsAssessUtils::parseKeyframes(keyFrameIndex, vVideoClips, inputVideoClips, m_viewFrameLength);
    if (inputVideoClips.empty())
    {
        QtLogger::instance().logMessage(QString("[I] Compute LVEF Fail, input video size %1, keyframe size %2").arg(vVideoClips.size()).arg(keyFrameIndex.size()));
        return 0;
    }

    QString qstrViewName = QString::fromStdString(currViewName);
    QString qstrPatientName = QString("Unknown");
    cv::Size originSize = inputVideoClips[0].size();

    //GeneralUtils::saveInferenceImage(m_videoSaveRootPath, qstrViewName, qstrPatientName, inputVideoClips);
    //for (auto& mat : inputVideoClips)
    //{
    //    cv::imshow("outputVideo", mat);
    //    cv::waitKey(0);
    //}
    //parseKeyframes(keyFrameIndex, vVideoClips, inputVideoClips, m_viewFrameLength);
    
    std::vector<cv::Mat> a2cInferVideo, a4cInferVideo;
    std::string strCurrLVEFMode = "(BP)";

    m_bIsBiplaneMode = true;

    if (currViewName == "A2C")
    {
        m_a2cVideo = inputVideoClips;
        a2cInferVideo = inputVideoClips;
        
        if (m_a4cVideo.empty())
        {
            a4cInferVideo = inputVideoClips;
            strCurrLVEFMode = "(A2C)";
            m_bIsBiplaneMode = false;
        }

        else
            a4cInferVideo = m_a4cVideo;
    }

    else if (currViewName == "A4C")
    {
        m_a4cVideo = inputVideoClips;
        a4cInferVideo = inputVideoClips;

        if (m_a2cVideo.empty())
        {
            a2cInferVideo = inputVideoClips;
            strCurrLVEFMode = "(A4C)";
            m_bIsBiplaneMode = false;
        }

        else
            a2cInferVideo = m_a2cVideo;
    }

    else
        return 0;

    std::vector<std::vector<cv::Mat>> vViewPhaseMasks;
    //for (auto& mat: a4cInferVideo)
    //{
    //    cv::imshow("image", mat);
    //    cv::waitKey(0);
    //}

    m_lvefAssesser->doInference(a2cInferVideo, a4cInferVideo, vViewPhaseMasks);
    for (auto& vecMask : vViewPhaseMasks)
    {
        cv::resize(vecMask[0], vecMask[0], originSize);
    }
    std::map<std::string, float> edVolumeValue, esVolumeValue;
    std::map<std::string, cv::Mat> edVolumeVis, esVolumeVis;
    std::pair<VolumeInfo, VolumeInfo> currVolumeInfo, histVolumeInfo;

    {
        QMutexLocker locker(&m_scaleMutex);
        if (m_fPixPerUnit != 0.0f && m_fPixPerUnit != -10000.0f)
        {
            currVolumeInfo.first.fPixPerUnit = m_fPixPerUnit;
            currVolumeInfo.second.fPixPerUnit = m_fPixPerUnit;
        }
    }

    std::string strPlaneMode = "SP";
    if (m_bIsBiplaneMode)
        strPlaneMode = "BP";

    std::vector<cv::Mat> vecViewMasks;
    if (currViewName == "A2C")
    {
        vecViewMasks = { vViewPhaseMasks[0][0], vViewPhaseMasks[1][0] };  // ED ES
        histVolumeInfo = m_a4cHistVolumeInfo;
    }

    else
    {
        vecViewMasks = { vViewPhaseMasks[2][0], vViewPhaseMasks[3][0] };
        histVolumeInfo = m_a2cHistVolumeInfo;
    }

    int edRet = SimpsonCalculation::doSimsponCalc(inputVideoClips[0], vecViewMasks[0], histVolumeInfo.first, currVolumeInfo.first, edVolumeValue, edVolumeVis, strPlaneMode, currViewName);
    int esRet = SimpsonCalculation::doSimsponCalc(inputVideoClips.back(), vecViewMasks[1], histVolumeInfo.second, currVolumeInfo.second, esVolumeValue, esVolumeVis, strPlaneMode, currViewName);

    float fSPLVEF = 0.0f;
    float fBPLVEF = 0.0f;
    float fSPEDVolume = 0.0f;
    float fSPESVolume = 0.0f;
    float fBPEDVolume = 0.0f;
    float fBPESVolume = 0.0f;

    if (edRet && esRet)
    {
        fSPEDVolume = std::max(edVolumeValue["LV Volume"], esVolumeValue["LV Volume"]);
        fSPESVolume = std::min(edVolumeValue["LV Volume"], esVolumeValue["LV Volume"]);

        fBPEDVolume = std::max(edVolumeValue["BP LV Volume"], esVolumeValue["BP LV Volume"]);
        fBPESVolume = std::min(edVolumeValue["BP LV Volume"], esVolumeValue["BP LV Volume"]);

        fSPLVEF = (fSPEDVolume - fSPESVolume) / fSPEDVolume * 100.0f;
        fBPLVEF = (fBPEDVolume - fBPESVolume) / fBPEDVolume * 100.0f;

        int nIsSPInRange = checkLVEFResultsInRange(fSPLVEF);
        int nIsBPInRange = checkLVEFResultsInRange(fBPLVEF);
        
        if (nIsSPInRange && nIsBPInRange)
        {
            QtLogger::instance().logMessage(QString("[I] Compute LVEF Success, LVEF: %1, EDV: %2, ESV: %3, Current Scale: %4, History Scale: %5")
                .arg(fBPLVEF).arg(fBPEDVolume).arg(fBPESVolume).arg(currVolumeInfo.first.fPixPerUnit).arg(histVolumeInfo.first.fPixPerUnit));
            //values.insert({ "EDV " + strCurrLVEFMode, std::vector<float>{ fBPEDVolume } });
            //values.insert({ "ESV " + strCurrLVEFMode, std::vector<float>{ fBPESVolume } });
            values.insert({ "EF " + strCurrLVEFMode, std::vector<float>{ fSPLVEF } });
            updateHistoryVolumeInfo(currViewName, currVolumeInfo);

            std::vector<cv::Mat> vecCurrPremiums{ edVolumeVis[currViewName], esVolumeVis[currViewName] };
            updateHistoryPremiums(currViewName, vecCurrPremiums);

            std::vector<cv::Mat> vecHistPremiums = getHistoryPremiums(currViewName);
            std::vector<cv::Mat> vecPremiums;

            if (vecHistPremiums.empty() && vecHistPremiums.size() != 2)
                vecPremiums = { edVolumeVis[currViewName], esVolumeVis[currViewName], edVolumeVis[currViewName], esVolumeVis[currViewName] };
            else
                vecPremiums = { edVolumeVis[currViewName], esVolumeVis[currViewName], vecHistPremiums[0], vecHistPremiums[1] };
            cv::Size targetSize(1200, 1200);
            cv::Mat combinedImage = concatMultiImages(vecPremiums, targetSize);
            resultPics.insert({ "EF " + strCurrLVEFMode, combinedImage });

            //if (!m_bIsBiplaneMode)
            //{
            //    std::vector<cv::Mat> vecPremiums{ edVolumeVis[currViewName], esVolumeVis[currViewName], edVolumeVis[currViewName], esVolumeVis[currViewName] };
            //    cv::Size targetSize(1200, 600);
            //    cv::Mat combinedImage = concatMultiImages(vecPremiums, targetSize);
            //}
            //else
            //{
            //    std::vector<cv::Mat> vecPremiums{ edVolumeVis["A2C"], esVolumeVis["A2C"], edVolumeVis["A4C"], esVolumeVis["A4C"]};
            //    cv::Size targetSize(1200, 1200);
            //    cv::Mat combinedImage = concatMultiImages(vecPremiums, targetSize);
            //}
        }
        else
        {
            QtLogger::instance().logMessage(QString::fromStdString("[I] Compute LVEF fail, Not in Range. Current SP: %1, BP: %2, Clear %3 History Value")
                .arg(QString::number(fSPLVEF)).arg(QString::number(fBPLVEF)).arg(QString::fromStdString(currViewName)));
            clearHistoryBuffers(currViewName);
            clearHistoryPremiums(currViewName);
            return 0;
        }
    }
    else
    {
        QtLogger::instance().logMessage(QString::fromStdString("[I] Compute LVEF fail, Simpson Computing Error. Clear %3 History Value")
            .arg(QString::fromStdString(currViewName)));
        clearHistoryBuffers(currViewName);
        clearHistoryPremiums(currViewName);
        return 0;
    }

    // vMasks = { v2CEdMasks, v2CEsMasks, v4CEdMasks, v4CEsMasks }
    //std::map<std::string, float> edVolumeValue, esVolumeValue;
    //std::map<std::string, cv::Mat> edVolumeVis, esVolumeVis;
    //
    //int edRet = SimpsonCalculation::doSimsponCalc(a2cInferVideo[0], vViewPhaseMasks[0][0], a4cInferVideo[0], vViewPhaseMasks[2][0], edVolumeValue, edVolumeVis);
    //int esRet = SimpsonCalculation::doSimsponCalc(a2cInferVideo.back(), vViewPhaseMasks[1][0], a4cInferVideo.back(), vViewPhaseMasks[3][0], esVolumeValue, esVolumeVis);

    //float lvef = 0.0f;
    //float fEDVolume = 0.0f;
    //float fESVolume = 0.0f;

    //if (edRet && esRet)
    //{
    //    fEDVolume = std::max(edVolumeValue["LV Volume"], esVolumeValue["LV Volume"]);
    //    fESVolume = std::min(edVolumeValue["LV Volume"], esVolumeValue["LV Volume"]);
    //    lvef = (fEDVolume - fESVolume) / fEDVolume;
    //}

    //values.insert({ "EDV " + strCurrLVEFMode, std::vector<float>{ fEDVolume }});
    //values.insert({ "ESV " + strCurrLVEFMode, std::vector<float>{ fESVolume } });
    //values.insert({ "EF " + strCurrLVEFMode, std::vector<float>{ lvef * 100.0f } });

    //QtLogger::instance().logMessage(QString::fromStdString("[I] Biplane LVEF Mode: " + strCurrLVEFMode));

    //bool bIsAllInRange = checkFinalResultsInRange(values);
    //if (!bIsAllInRange)
    //{
    //    QtLogger::instance().logMessage(QString::fromStdString("[I] Compute LVEF fail, not in range. Current Value: %1").arg(QString::number(lvef * 100.0f)));
    //    values.clear();
    //    if (currViewName == "A2C")
    //        m_a2cVideo.clear();
    //    else if (currViewName == "A4C")
    //        m_a4cVideo.clear();

    //    return 0;
    //}

    //std::vector<cv::Mat> vecPremiums{ edVolumeVis["A2C"], esVolumeVis["A2C"], edVolumeVis["A4C"], esVolumeVis["A4C"]};
    //cv::Size targetSize(1200, 1200);
    //cv::Mat combinedImage = concatMultiImages(vecPremiums, targetSize);

    //resultPics.insert({ "EF " + strCurrLVEFMode, combinedImage });
    //QtLogger::instance().logMessage(QString::fromStdString("[I] Compute LVEF Success: %1").arg(lvef));
    ////resultPics.insert({ "ESV " + strCurrLVEFMode, esVolumeVis["A4C"] });

	return 1;
}


int BiplaneLVEFAssesser::biplaneSimpson(cv::Mat& a2cEdMask, cv::Mat& a2cEsMask, cv::Mat& a4cEdMask, cv::Mat& a4cEsMask)
{
    return 0;
}

int BiplaneLVEFAssesser::parseKeyframes(std::vector<int>& keyframeIdx, std::vector<cv::Mat>& inputVideo, std::vector<cv::Mat>& outputVideo, int length)
{
    if (inputVideo.empty()) {
        return 0;
    }

    size_t videoSize = inputVideo.size();
    std::vector<std::pair<int, int>> selectedIdx;

    if (keyframeIdx.size() == 1)  // 只有一个es/ed索引的情况
    {
        if (abs(keyframeIdx.back()) > videoSize)
        {
            outputVideo.insert(outputVideo.end(), inputVideo.end() - length, inputVideo.end());
            return 0;
        }
    }
    else   // 多个索引值的情况
    {
        std::sort(keyframeIdx.begin(), keyframeIdx.end(), [](int a, int b) {
            return std::abs(a) < std::abs(b);
            });
        for (size_t i = 0; i < keyframeIdx.size() - 1; i++)
        {
            std::pair<int, int> tempIdx;

            if ((abs(keyframeIdx[i]) > videoSize - 1) || (abs(keyframeIdx[i + 1]) > videoSize - 1))
                continue;

            if (keyframeIdx[i] >= 0 && keyframeIdx[i + 1] < 0)
            {
                tempIdx.first = keyframeIdx[i];
                tempIdx.second = keyframeIdx[i + 1];
                selectedIdx.push_back(tempIdx);
            }
            else
                continue;
        }
    }

    if (!selectedIdx.empty())
    {
        std::pair<int, int> indexPair = selectedIdx.back();

        int edIndex = indexPair.first;
        int esIndex = abs(indexPair.second);
        if (esIndex - edIndex + 1 >= length)  // 两个索引（包含）之间存在length长度帧
        {
            double interval = static_cast<double>(esIndex - edIndex) / (length - 1);
            for (size_t i = 0; i < length; ++i)
            {
                size_t index = static_cast<size_t>(std::round(edIndex + i * interval));
                outputVideo.push_back(inputVideo[index]);
            }
        }
        else   // 小于length
        {
            for (int i = edIndex; i <= esIndex; i++)
            {
                outputVideo.push_back(inputVideo[i]);
            }

            while (outputVideo.size() < length)
            {
                outputVideo.insert(outputVideo.begin(), inputVideo[edIndex]);
            }
        }

    }
    else  // 没有符合条件的ed和es以及keyframeIdx只有单个索引值的情况
    {
        int keyframeIndex = keyframeIdx.back();
        if (keyframeIndex >= 0)  // ed
        {
            if (keyframeIndex >= videoSize)
                keyframeIndex = videoSize - 1;
            for (size_t i = keyframeIndex; i < keyframeIndex + length; ++i)
            {
                if (i <= videoSize - 1)
                {
                    outputVideo.push_back(inputVideo[i]);
                }
                else
                {
                    outputVideo.insert(outputVideo.begin(), inputVideo[keyframeIndex]);
                }
            }
        }
        else   // es
        {
            if (abs(keyframeIndex) + 1 >= length)  // 索引应该是从0开始 
            {
                for (size_t i = abs(keyframeIndex) - length + 1; i < abs(keyframeIndex); ++i)
                {
                    outputVideo.push_back(inputVideo[i]);
                }
            }
            else
            {
                for (size_t i = 0; i < length; ++i)
                {
                    if (i <= abs(keyframeIndex))
                    {
                        outputVideo.push_back(inputVideo[i]);
                    }
                    else
                    {
                        outputVideo.insert(outputVideo.begin(), inputVideo[0]);
                    }
                }
            }

        }
    }

    for (auto& mat : outputVideo)
    {
        cv::imshow("outputVideo", mat);
        cv::waitKey(0);
    }

    return 1;

}

cv::Mat BiplaneLVEFAssesser::concatMultiImages(std::vector<cv::Mat>& vecImages, cv::Size& targetSize)
{
    if (vecImages.size() != 4) {
        throw std::invalid_argument("Exactly four images are required.");
    }

    // 检查所有图像是否有相同的尺寸
    cv::Size imageSize = vecImages[0].size();
    for (auto& img : vecImages) {
        if (img.size() != imageSize) {
            cv::resize(img, img, imageSize);
        }
    }

    // 创建一个大图像，大小为2x2图片拼接
    int combinedWidth = imageSize.width * 2;
    int combinedHeight = imageSize.height * 2;
    cv::Mat combinedImage(combinedHeight, combinedWidth, vecImages[0].type());

    // 将四张图片复制到大图像中
    vecImages[0].copyTo(combinedImage(cv::Rect(0, 0, imageSize.width, imageSize.height)));
    vecImages[1].copyTo(combinedImage(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height)));
    vecImages[2].copyTo(combinedImage(cv::Rect(0, imageSize.height, imageSize.width, imageSize.height)));
    vecImages[3].copyTo(combinedImage(cv::Rect(imageSize.width, imageSize.height, imageSize.width, imageSize.height)));

    // 调整拼接后的图像大小以适应目标尺寸
    cv::Mat resizedCombinedImage;
    cv::resize(combinedImage, resizedCombinedImage, cv::Size(combinedWidth / 4, combinedHeight / 4));

    return resizedCombinedImage;
}

bool BiplaneLVEFAssesser::checkFinalResultsInRange(std::map<std::string, std::vector<float>>& values)
{
    bool bIsAllInRange = false;

    for (auto& paramEventValues : values)
    {
        std::string strParamName = paramEventValues.first;
        std::string strParamEventName = strSplitSpace(strParamName)[0];
        float fParamValue = paramEventValues.second[0];

        if (m_referRange.find(strParamEventName) != m_referRange.end())
        {
            if (fParamValue >= m_referRange[strParamEventName].first && fParamValue <= m_referRange[strParamEventName].second)
                bIsAllInRange = true;
        }
    }

    return bIsAllInRange;
}

bool BiplaneLVEFAssesser::checkLVEFResultsInRange(float fValue)
{
    bool bIsAllInRange = false;
    if (fValue >= m_referRange["EF"].first && fValue <= m_referRange["EF"].second)
        bIsAllInRange = true;
    return bIsAllInRange;
}

std::vector<std::string> BiplaneLVEFAssesser::strSplitSpace(std::string& strInput)
{
    std::vector<std::string> result;
    std::istringstream stream(strInput);
    std::string word;

    while (stream >> word) {
        result.push_back(word);
    }

    return result;
}

int BiplaneLVEFAssesser::clearHistoryBuffers(std::string& strCurrViewName)
{
    if (strCurrViewName == "A2C")
        m_a2cVideo.clear();
    else if (strCurrViewName == "A4C")
        m_a4cVideo.clear();

    return 1;
}

int BiplaneLVEFAssesser::updateHistoryVolumeInfo(std::string& strCurrViewName, std::pair<VolumeInfo, VolumeInfo>& currVolumeInfo)
{
    if (strCurrViewName == "A2C")
        m_a2cHistVolumeInfo = currVolumeInfo;
    else if (strCurrViewName == "A4C")
        m_a4cHistVolumeInfo = currVolumeInfo;
    return 1;
}

int BiplaneLVEFAssesser::updateHistoryPremiums(std::string& strCurrViewName, std::vector<cv::Mat>& vecPremiums)
{
    if (strCurrViewName == "A4C")
        m_a4cHistPremiums = vecPremiums;
    else
        m_a2cHistPremiums = vecPremiums;
    return 1;
}

std::vector<cv::Mat> BiplaneLVEFAssesser::getHistoryPremiums(std::string& strCurrViewName)
{
    if (strCurrViewName == "A4C")
        return m_a2cHistPremiums;
    else
        return m_a4cHistPremiums;
    return std::vector<cv::Mat>();
}

int BiplaneLVEFAssesser::clearHistoryPremiums(std::string& strCurrViewName)
{
    if (strCurrViewName == "A4C")
        m_a4cHistPremiums = std::vector<cv::Mat>();
    else
        m_a2cHistPremiums = std::vector<cv::Mat>();
    return 1;
}

int BiplaneLVEFAssesser::clearAllHistoryRecords()
{
    m_a2cHistPremiums.clear();
    m_a4cHistPremiums.clear();
    m_a2cHistVolumeInfo = std::pair<VolumeInfo, VolumeInfo>{};
    m_a4cHistVolumeInfo = std::pair<VolumeInfo, VolumeInfo>{};

    return 1;
}
