#include "DoSpecParamsAssess.h"


// 频谱参数
DoSpecParamsAssess::DoSpecParamsAssess()
{
	std::string mveaEnginePathDetect = "D:\\Resources\\20240221\\param_assess_models\\MVEA_Objdetect_0612.engine";
	std::string mveaEnginePathKeypoint = "D:\\Resources\\20240221\\param_assess_models\\MVEA_Keypoint_0612.engine";
	m_pMveaAssesser = new MVEAAssess(mveaEnginePathDetect, mveaEnginePathKeypoint);
	QtLogger::instance().logMessage("[I] Spec MVEA Model Loaded");

	std::string pvEnginePathDetect = "D:\\Resources\\20240221\\param_assess_models\\PV_Objdetect_0612.engine";
	std::string pvEnginePathKeypoint = "D:\\Resources\\20240221\\param_assess_models\\PV_Keypoint_0612.engine";
	m_pPvAssesser = new PVAssess(pvEnginePathDetect, pvEnginePathKeypoint);
	QtLogger::instance().logMessage("[I] Spec PV Model Loaded");

	std::string ivsEnginePathDetect = "D:\\Resources\\20240221\\param_assess_models\\TDI_IVS_Objdetect_0801.engine";
	std::string ivsEnginePathKeypoint = "D:\\Resources\\20240221\\param_assess_models\\TDI_IVS_Keypoint_0729.engine";
	m_pTdiivsAssesser = new TDIIVSAssess(ivsEnginePathDetect, ivsEnginePathKeypoint);
	QtLogger::instance().logMessage("[I] Spec TDIIVS Model Loaded");

	std::string mvlwEnginePathDetect = "D:\\Resources\\20240221\\param_assess_models\\TDI_MVLW_Objdetect_0729.engine";
	std::string mvlwEnginePathKeypoint = "D:\\Resources\\20240221\\param_assess_models\\TDI_MVLW_Keypoint_0729.engine";
	m_pTdimvlwAssesser = new TDIMVLWAssess(mvlwEnginePathDetect, mvlwEnginePathKeypoint);
	QtLogger::instance().logMessage("[I] Spec TDIMVLW Model Loaded");

	std::string trEnginePathDetect = "D:\\Resources\\20240221\\param_assess_models\\TR_Objdetect_0328.engine";
	std::string trEnginePathKeypoint = "D:\\Resources\\20240221\\param_assess_models\\TR_Keypoint_0328.engine";
	m_pTrAssesser = new TRAssess(trEnginePathDetect, trEnginePathKeypoint);
	QtLogger::instance().logMessage("[I] Spec TR Model Loaded");

	std::string fvavEnginePath = "D:\\Resources\\20240221\\param_assess_models\\FVAV_0718.engine";
	m_pFvavAssesser = new FVAVAssess(fvavEnginePath);
	QtLogger::instance().logMessage("[I] Spec Ao Model Loaded");
}

DoSpecParamsAssess::DoSpecParamsAssess(ConfigParse* config)
{
    // 从配置文件中读取 MVEA 模型路径
    std::string mveaEnginePathDetect;
    std::string mveaEnginePathKeypoint;
    if (config->getSpecifiedNode("MVEA_DETECT_PATH", mveaEnginePathDetect) &&
        config->getSpecifiedNode("MVEA_KEYPOINT_PATH", mveaEnginePathKeypoint)) {

        // 检查路径是否存在
        if (GeneralUtils::fileExists(mveaEnginePathDetect) && GeneralUtils::fileExists(mveaEnginePathKeypoint)) {
            m_pMveaAssesser = new MVEAAssess(mveaEnginePathDetect, mveaEnginePathKeypoint);
            QtLogger::instance().logMessage(QString::fromStdString("[I] Spec MVEA Model Loaded"));
        }
        else {
            QtLogger::instance().logMessage(QString::fromStdString("[E] MVEA model paths do not exist: ") +
                QString::fromStdString(mveaEnginePathDetect) + ", " +
                QString::fromStdString(mveaEnginePathKeypoint));
        }
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load MVEA model paths from config"));
    }

    // 从配置文件中读取 PV 模型路径
    std::string pvEnginePathDetect;
    std::string pvEnginePathKeypoint;
    if (config->getSpecifiedNode("PV_DETECT_PATH", pvEnginePathDetect) &&
        config->getSpecifiedNode("PV_KEYPOINT_PATH", pvEnginePathKeypoint)) {

        // 检查路径是否存在
        if (GeneralUtils::fileExists(pvEnginePathDetect) && GeneralUtils::fileExists(pvEnginePathKeypoint)) {
            m_pPvAssesser = new PVAssess(pvEnginePathDetect, pvEnginePathKeypoint);
            QtLogger::instance().logMessage(QString::fromStdString("[I] Spec PV Model Loaded"));
        }
        else {
            QtLogger::instance().logMessage(QString::fromStdString("[E] PV model paths do not exist: ") +
                QString::fromStdString(pvEnginePathDetect) + ", " +
                QString::fromStdString(pvEnginePathKeypoint));
        }
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load PV model paths from config"));
    }

    // 从配置文件中读取 TDI IVS 模型路径
    std::string ivsEnginePathDetect;
    std::string ivsEnginePathKeypoint;
    if (config->getSpecifiedNode("TDI_IVS_DETECT_PATH", ivsEnginePathDetect) &&
        config->getSpecifiedNode("TDI_IVS_KEYPOINT_PATH", ivsEnginePathKeypoint)) {

        // 检查路径是否存在
        if (GeneralUtils::fileExists(ivsEnginePathDetect) && GeneralUtils::fileExists(ivsEnginePathKeypoint)) {
            m_pTdiivsAssesser = new TDIIVSAssess(ivsEnginePathDetect, ivsEnginePathKeypoint);
            QtLogger::instance().logMessage(QString::fromStdString("[I] Spec TDI IVS Model Loaded"));
        }
        else {
            QtLogger::instance().logMessage(QString::fromStdString("[E] TDI IVS model paths do not exist: ") +
                QString::fromStdString(ivsEnginePathDetect) + ", " +
                QString::fromStdString(ivsEnginePathKeypoint));
        }
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load TDI IVS model paths from config"));
    }

    // 从配置文件中读取 TDI MVLW 模型路径
    std::string mvlwEnginePathDetect;
    std::string mvlwEnginePathKeypoint;
    if (config->getSpecifiedNode("TDI_MVLW_DETECT_PATH", mvlwEnginePathDetect) &&
        config->getSpecifiedNode("TDI_MVLW_KEYPOINT_PATH", mvlwEnginePathKeypoint)) {

        // 检查路径是否存在
        if (GeneralUtils::fileExists(mvlwEnginePathDetect) && GeneralUtils::fileExists(mvlwEnginePathKeypoint)) {
            m_pTdimvlwAssesser = new TDIMVLWAssess(mvlwEnginePathDetect, mvlwEnginePathKeypoint);
            QtLogger::instance().logMessage(QString::fromStdString("[I] Spec TDI MVLW Model Loaded"));
        }
        else {
            QtLogger::instance().logMessage(QString::fromStdString("[E] TDI MVLW model paths do not exist: ") +
                QString::fromStdString(mvlwEnginePathDetect) + ", " +
                QString::fromStdString(mvlwEnginePathKeypoint));
        }
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load TDI MVLW model paths from config"));
    }

    // 从配置文件中读取 TR 模型路径
    std::string trEnginePathDetect;
    std::string trEnginePathKeypoint;
    if (config->getSpecifiedNode("TR_DETECT_PATH", trEnginePathDetect) &&
        config->getSpecifiedNode("TR_KEYPOINT_PATH", trEnginePathKeypoint)) {

        // 检查路径是否存在
        if (GeneralUtils::fileExists(trEnginePathDetect) && GeneralUtils::fileExists(trEnginePathKeypoint)) {
            m_pTrAssesser = new TRAssess(trEnginePathDetect, trEnginePathKeypoint);
            QtLogger::instance().logMessage(QString::fromStdString("[I] Spec TR Model Loaded"));
        }
        else {
            QtLogger::instance().logMessage(QString::fromStdString("[E] TR model paths do not exist: ") +
                QString::fromStdString(trEnginePathDetect) + ", " +
                QString::fromStdString(trEnginePathKeypoint));
        }
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load TR model paths from config"));
    }

    // 从配置文件中读取 FVAV 模型路径
    std::string fvavEnginePath;
    if (config->getSpecifiedNode("AO_SEG_PATH", fvavEnginePath)) {

        // 检查路径是否存在
        if (GeneralUtils::fileExists(fvavEnginePath)) {
            m_pFvavAssesser = new FVAVAssess(fvavEnginePath);
            QtLogger::instance().logMessage(QString::fromStdString("[I] Spec FVAV Model Loaded"));
        }
        else {
            QtLogger::instance().logMessage(QString::fromStdString("[E] FVAV model path does not exist: ") +
                QString::fromStdString(fvavEnginePath));
        }
    }
    else {
        QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load FVAV model path from config"));
    }
}

DoSpecParamsAssess::~DoSpecParamsAssess()
{
	delete m_pMveaAssesser;
	delete m_pPvAssesser;
	delete m_pTdiivsAssesser;
	delete m_pTdimvlwAssesser;
	delete m_pTrAssesser;
	delete m_pFvavAssesser;
}

int DoSpecParamsAssess::getSpecParamsRst(cv::Mat& frame, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, std::map<std::string, int> classResult)
{
	//int modeIdx = classResult["mode"];  // { "CW","PW","TDI","M" }
	//int viewIdx = classResult["view"];  // { "A4C", "A5C", "PLAX", "PSAXGV", "OTHER_VIEW" }

	int modeIdx = classResult["mode"];  // { "CW","PW","TDI","M" }
	int viewIdx = classResult["view"];  // { "A4C", "A5C", "PLAX", "PSAXGV", "TDIIVS", "TDIMV", "OTHER_VIEW" }

	if (modeIdx == 0 && viewIdx == 0)
	{
		m_pTrAssesser->trAssessment(frame, values, resultPics);
	}
	else if (modeIdx == 0 && viewIdx == 1)
	{
		m_pFvavAssesser->fvavAssessment(frame, values, resultPics);
	}
	else if (modeIdx == 1 && viewIdx == 0)
	{
		m_pMveaAssesser->mveaAssessment(frame, values, resultPics);
	}
	else if (modeIdx == 1 && viewIdx == 3)
	{
		m_pPvAssesser->pvAssessment(frame, values, resultPics);
	}
	else if (modeIdx == 2 && viewIdx == 4)
	{
		m_pTdiivsAssesser->tdiivsAssessment(frame, values, resultPics);
	}
	else if (modeIdx == 2 && viewIdx == 5)
	{
		m_pTdimvlwAssesser->tdimvlwAssessment(frame, values, resultPics);
	}
	else
	{
		return 0;
	}

	return 1;
}

int DoSpecParamsAssess::getSpecParamsRstV2(cv::Mat& frame, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, std::map<std::string, int> classResult)
{
	int viewIdx = classResult["view"];  // { "MV", "TR", "PA", "Ao", "TDIMVIVS", "TDIMVLW", "OTHER_VIEW" }

	if (viewIdx == 0)  // MV
	{
		m_pMveaAssesser->mveaAssessment(frame, values, resultPics);
	}
	else if (viewIdx == 1 || viewIdx == 4)  // TR
	{
		m_pTrAssesser->trAssessment(frame, values, resultPics);
	}
	else if (viewIdx == 3)  // PA
	{
		m_pPvAssesser->pvAssessment(frame, values, resultPics);
	}
	else if (viewIdx == 2)  // Ao
	{
		m_pFvavAssesser->fvavAssessment(frame, values, resultPics);
	}
	else if (viewIdx == 5)  // TDIMVIVS
	{
		m_pTdiivsAssesser->tdiivsAssessment(frame, values, resultPics);
	}
	else if (viewIdx == 6)  // TDIMVLW
	{
		m_pTdimvlwAssesser->tdimvlwAssessment(frame, values, resultPics);
	}
	else
	{
		return 0;
	}
	return 1;
}

