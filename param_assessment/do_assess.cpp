#include "do_assess.h"
// 结构参数
DoStrucParamsAssess::DoStrucParamsAssess()
	:m_pIvsandpwAssesser(NULL)
	,m_pAvadAssesser(NULL)
	,m_pAsdAndsjdAssesser(NULL)
	,m_pAadAssesser(NULL)
	,m_pLaadAssesser(NULL)
{
	m_pIvsandpwAssesser = new IVSAndPWAssess;

	std::string avadEnginePath = "../../extern/models/AVAD_HRNet_0229_170.engine";
	m_pAvadAssesser = new AVADAssess(avadEnginePath);

	std::string asdAndsjdEnginePath = "../../extern/models/AAO_0106.engine";
	m_pAsdAndsjdAssesser = new ASDAndSJDAssess(asdAndsjdEnginePath);

	std::string aadEnginePath = "../../extern/models/AAO_0106.engine";
	m_pAadAssesser = new AADAssess(aadEnginePath);

	std::string laadEnginePathLa = "../../extern/models/LAAD_LA_0220.engine";
	std::string laadEnginePathAv = "../../extern/models/LAAD_AV_0221.engine";
	m_pLaadAssesser = new LAADAssess(laadEnginePathLa, laadEnginePathAv);

}

DoStrucParamsAssess::~DoStrucParamsAssess()
{
	delete m_pIvsandpwAssesser;
	delete m_pAvadAssesser;
	delete m_pAsdAndsjdAssesser;
	delete m_pAadAssesser;
	delete m_pLaadAssesser;
}

int DoStrucParamsAssess::getStrucParamsRst(std::vector<cv::Mat>& video, std::vector<int>& keyframeIdx, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	int index = keyframeIdx.back();
	m_inferFrame = video[index];

	m_pIvsandpwAssesser->ivsAndPwAssessment(m_inferFrame, values, resultPics);

	m_pAvadAssesser->avadAssessment(m_inferFrame, values, resultPics);

	m_pAsdAndsjdAssesser->asdAndsjdAssessment(m_inferFrame, values, resultPics);

	m_pAadAssesser->aadAssessment(m_inferFrame, values, resultPics);

	m_pLaadAssesser->laadAssessment(m_inferFrame, values, resultPics);

	return 1;
}

// 功能参数
DoFuncParamsAssess::DoFuncParamsAssess()
{
	std::string efSegEnginePath = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\EF_seg_0306.engine";
	std::string efVideoEnginePath = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\EF_video_0306.engine";
	m_pEfAssesser = new EFAssess(efSegEnginePath, efVideoEnginePath);

	std::string lvvSegEnginePath = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\LVV_seg_0416.engine";
	m_pLvvAssesser = new LVVAssess(lvvSegEnginePath);
}

DoFuncParamsAssess::~DoFuncParamsAssess()
{
	delete m_pEfAssesser;
	delete m_pLvvAssesser;
}

int DoFuncParamsAssess::getFuncParamsRst(std::vector<cv::Mat>& video, std::map<std::string, 
	std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	// LVV-edv  // pred1, pred2, curr(edv)
	std::vector<std::vector<cv::Mat>> edvFramesVec = { {video[1], video[0], video[2]}, {video[video.size() - 2], video[video.size() - 3], video.back()} };

	float edv = 0.0f;
	for (auto& edvFrames : edvFramesVec) 
	{
		m_pLvvAssesser->lvvAssessment(edvFrames, values, resultPics);
		edv += values["LVV"][0];
	}
	std::vector<float> edvVec = { edv / static_cast<float>(edvFramesVec.size()) };
	values.insert({ "EDV", edvVec });
	resultPics.insert({ "EDV", resultPics["LVV"] });
	values.erase("LVV");
	resultPics.erase("LVV");

	//LVV-esv  // 先减去video开头的edv的前两帧，找到中间位置的esv
	size_t esvIdx = video.size() / 2 + 1;  
	std::vector<cv::Mat> esvFrames = { video[esvIdx - 1], video[esvIdx - 2], video[esvIdx] };
	m_pLvvAssesser->lvvAssessment(esvFrames, values, resultPics);
	values.insert({ "ESV", values["LVV"] });
	resultPics.insert({ "ESV", resultPics["LVV"] });
	values.erase("LVV");
	resultPics.erase("LVV");

	// EF
	m_pEfAssesser->efAssessment(video, values, resultPics);

	return 0;
}

// 频谱参数
DoSpecParamsAssess::DoSpecParamsAssess()
{
	std::string specEnginePath = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\Spec_Classification_0407.engine";
	m_pSpecInferer = new SPECClassInferer(specEnginePath);

	std::string mveaEnginePathDetect = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\MVEA_Objdetect_0321.engine";
	std::string mveaEnginePathKeypoint = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\MVEA_Keypoint_0326.engine";
	m_pMveaAssesser = new MVEAAssess(mveaEnginePathDetect, mveaEnginePathKeypoint);

	std::string pvEnginePathDetect = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\PV_Objdetect_0326.engine";
	std::string pvEnginePathKeypoint = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\PV_Keypoint_0326.engine";
	m_pPvAssesser = new PVAssess(pvEnginePathDetect, pvEnginePathKeypoint);

	std::string ivsEnginePathDetect = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\TDI_IVS_Objdetect_0327.engine";
	std::string ivsEnginePathKeypoint = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\TDI_IVS_Keypoint_0328.engine";
	m_pTdiivsAssesser = new TDIIVSAssess(ivsEnginePathDetect, ivsEnginePathKeypoint);

	std::string mvlwEnginePathDetect = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\TDI_MVLW_Objdetect_0328.engine";
	std::string mvlwEnginePathKeypoint = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\TDI_MVLW_Keypoint_0328.engine";
	m_pTdimvlwAssesser = new TDIMVLWAssess(mvlwEnginePathDetect, mvlwEnginePathKeypoint);

	std::string trEnginePathDetect = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\TR_Objdetect_0328.engine";
	std::string trEnginePathKeypoint = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\TR_Keypoint_0328.engine";
	m_pTrAssesser = new TRAssess(trEnginePathDetect, trEnginePathKeypoint);

	std::string fvavEnginePath = "C:\\Project\\QualityAssess\\23-12-18-Quality-Param-Assess\\extern\\models\\FVAV_0329.engine";
	m_pFvavAssesser = new FVAVAssess(fvavEnginePath);
}

DoSpecParamsAssess::~DoSpecParamsAssess()
{
	delete m_pSpecInferer;
	delete m_pMveaAssesser;
	delete m_pPvAssesser;
	delete m_pTdiivsAssesser;
	delete m_pTdimvlwAssesser;
	delete m_pTrAssesser;
	delete m_pFvavAssesser;
}

int DoSpecParamsAssess::getSpecParamsRst(cv::Mat& frame, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::map<std::string, int> classResult;
	m_pSpecInferer->doInference(frame, classResult);

	int modeIdx = classResult["mode"];  // { "CW","PW","TDI","M" }
	int viewIdx = classResult["view"];  // { "A4C", "A5C", "PLAX", "PSAXGV", "OTHER_VIEW" }

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
	else if (modeIdx == 2 && viewIdx == 0) // 无法区分的TDI-A4C切面
	{
		m_pTdiivsAssesser->tdiivsAssessment(frame, values, resultPics);
		m_pTdimvlwAssesser->tdimvlwAssessment(frame, values, resultPics);
	}
	else 
	{
		std::cout << "该切面不属于频谱参数检测切面" << std::endl;
		return 0;
	}
	return 1;
}

